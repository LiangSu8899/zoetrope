"""One recording, four visual languages, six colour systems.

Nothing here measures anything.  Every style below reads the *same*
`events.json` a stream recorder already wrote — the timestamp at which
each token actually arrived — and draws it a different way:

    curve    tokens delivered against time.  The slope is the rate, so the
             steeper line is the faster arm and no one has to be told.
    bars     one growing bar per arm, the plainest race there is.
    dots     one cell per token, lighting in arrival order.  Says *the
             same tokens* before it says anything about speed.
    ribbon   every arm's ticks laid on the same length of track, so the
             fast arm's ticks visibly bunch up.

They exist to be chosen between.  A film gets one idea, and which drawing
carries that idea is a real decision: `curve` argues about rate, `dots`
argues about identity, `bars` argues about finishing, `ribbon` argues
about density.  Render two, look at both, keep one.

    python -m demokit.compose.styles_compose --run examples/runs/stream \
        --style curve --palette paper --out curve.webm

    # every look at once, as one PNG, to pick from
    python -m demokit.compose.styles_compose --run examples/runs/stream \
        --sheet looks.png
"""

from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import subprocess
import tempfile

from PIL import Image, ImageDraw

from .canvas import Canvas, R_SM, appear, ease_out, font, _truetype
from .palettes import PALETTES, palette
from .race_compose import _ffmpeg

#: The contact sheet is a print, not a film: it is deliberately neutral so
#: that six palettes can sit on it without one of them owning the page.
SHEET_BG, SHEET_INK = (245, 245, 245), (28, 28, 28)

W, H = 1280, 720
PAD = 96
BODY_TOP, BODY_BOT = 150, 524
LEAD, TAIL = 0.7, 2.4


class Arm:
    """One recorded stream, asked questions at a wall-clock time."""

    def __init__(self, run_dir):
        d = pathlib.Path(run_dir)
        blob = json.loads((d / "events.json").read_text())
        self.dir = d
        self.meta = blob["meta"]
        self.label = self.meta["label"]
        self.sub = self.meta.get("sub", "")
        self.key = self.meta.get("color", "stock")
        self.ts = [float(e["t"]) for e in blob["events"]]
        self.n = int(self.meta.get("n_tokens") or len(self.ts))
        self.done = float(self.meta.get("done_s") or self.ts[-1])

    def upto(self, w):
        """Tokens delivered by wall time `w`."""
        return bisect.bisect_right(self.ts, w)

    def rate(self, w):
        """Tok/s over the decode so far — the number the meta ends on."""
        w = min(w, self.done)
        k = self.upto(w)
        if k < 2:
            return 0.0
        return (k - 1) / max(w - self.ts[0], 1e-6)


def load(run_dir=None, arms=()):
    dirs = [pathlib.Path(a) for a in arms]
    if run_dir:
        dirs += sorted(p for p in pathlib.Path(run_dir).iterdir()
                       if (p / "events.json").exists())
    if not dirs:
        raise SystemExit("no arms: pass --run DIR or --arm DIR")
    loaded = [Arm(d) for d in dirs]
    # slowest first, so the fastest arm is drawn last and sits on top
    loaded.sort(key=lambda a: -a.done)
    return loaded


# ---------------------------------------------------------------- chrome

def chrome(im, d, arms, w, pal, title, sub, style):
    d.rectangle((0, 0, W, H), fill=pal.bg)
    d.text((PAD, 44), title, font=font(31, True), fill=pal.ink)
    if sub:
        d.text((PAD, 86), sub, font=font(17), fill=pal.muted)
    clock = f"{min(w, max(a.done for a in arms)):.3f} s"
    f = font(17)
    d.text((W - PAD - d.textlength(clock, font=f), 88), clock, font=f,
           fill=pal.muted)
    d.line((PAD, 120, W - PAD, 120), fill=pal.line)

    y = 578
    for a in arms[::-1]:                       # fastest at the top, always
        ink = pal.role(a.key)
        d.rectangle((PAD, y + 7, PAD + 10, y + 21), fill=ink)
        d.text((PAD + 24, y + 2), a.label, font=font(19, True), fill=pal.ink)
        d.text((PAD + 24 + d.textlength(a.label, font=font(19, True)) + 14,
                y + 5), a.sub, font=font(15), fill=pal.muted)
        k, done = a.upto(w), w >= a.done
        right = (f"{a.n} tok · {a.done * 1e3:.0f} ms" if done
                 else f"{k} tok · {a.rate(w):.0f} tok/s")
        rf = font(19, True)
        d.text((W - PAD - d.textlength(right, font=rf), y + 2), right,
               font=rf, fill=ink if done else pal.muted)
        y += 34


def payoff(d, arms, w, pal):
    """Said only once every arm has stopped, and never before."""
    last = max(a.done for a in arms)
    if w < last:
        return
    slow, fast = arms[0], arms[-1]
    line = (f"the same {fast.n} tokens · "
            f"{slow.done * 1e3:.0f} ms → {fast.done * 1e3:.0f} ms · "
            f"{slow.done / fast.done:.2f}x")
    f = font(23, True)
    d.text(((W - d.textlength(line, font=f)) / 2, 536), line, font=f,
           fill=pal.role(fast.key))


# ---------------------------------------------------------------- styles

def _center(height):
    """Blocks sit in the middle of the body, never hard against the rule."""
    return BODY_TOP + max(0, (BODY_BOT - BODY_TOP - height)) / 2


def _axes(d, pal, x0, x1, y0, y1, tmax, nmax):
    for i in range(5):
        y = y1 - (y1 - y0) * i / 4
        d.line((x0, y, x1, y), fill=pal.grid)
        lab = f"{round(nmax * i / 4)}"
        d.text((x0 - 14 - d.textlength(lab, font=font(14)), y - 9), lab,
               font=font(14), fill=pal.muted)
    step = 0.1 if tmax <= 0.75 else 0.25 if tmax <= 2 else 1.0
    t = step
    while t < tmax:
        x = x0 + (x1 - x0) * t / tmax
        d.line((x, y0, x, y1), fill=pal.grid)
        lab = f"{t * 1e3:.0f} ms" if tmax <= 2 else f"{t:.0f} s"
        d.text((x - d.textlength(lab, font=font(14)) / 2, y1 + 10), lab,
               font=font(14), fill=pal.muted)
        t += step
    d.line((x0, y0, x0, y1), fill=pal.line)
    d.line((x0, y1, x1, y1), fill=pal.line)
    d.text((x0 - 14 - d.textlength("tokens", font=font(14)), y0 - 30),
           "tokens", font=font(14), fill=pal.muted)


def draw_curve(d, arms, w, pal):
    x0, x1 = PAD + 52, W - PAD - 128
    # the axis labels hang below y1, and the payoff line lands under them
    y0, y1 = BODY_TOP + 26, BODY_BOT - 26
    tmax = max(a.done for a in arms) * 1.02
    nmax = max(a.n for a in arms)
    _axes(d, pal, x0, x1, y0, y1, tmax, nmax)

    def pt(t, k):
        return (x0 + (x1 - x0) * t / tmax, y1 - (y1 - y0) * k / nmax)

    for a in arms:
        ink = pal.role(a.key)
        k = a.upto(w)
        pts = [pt(0, 0)] + [pt(a.ts[i], i + 1) for i in range(k)]
        if k and w < a.done:                   # hold the head at now
            pts.append(pt(w, k))
        if len(pts) > 1:
            d.line(pts, fill=ink, width=4, joint="curve")
        hx, hy = pts[-1]
        r = 6 if w < a.done else 8
        d.ellipse((hx - r, hy - r, hx + r, hy + r), fill=ink)
        if k:
            tag = (f"{a.rate(min(w, a.done)):.0f} tok/s" if w >= a.ts[0]
                   else "")
            d.text((min(hx + 16, x1 + 12), hy - 10), tag, font=font(17, True),
                   fill=ink)


def draw_bars(d, arms, w, pal):
    x0, x1 = PAD + 8, W - PAD - 150
    h, gap = 66, 40
    y = _center(len(arms) * h + (len(arms) - 1) * gap)
    for a in arms[::-1]:
        ink = pal.role(a.key)
        d.rounded_rectangle((x0, y, x1, y + h), R_SM, fill=pal.card)
        frac = a.upto(w) / a.n
        if frac:
            d.rounded_rectangle((x0, y, x0 + (x1 - x0) * frac, y + h), R_SM,
                                fill=ink)
        d.text((x0 + 16, y + h / 2 - 12), a.label, font=font(20, True),
               fill=pal.bg if frac > 0.34 else pal.ink)
        num = f"{a.upto(w)}"
        f = font(30, True)
        d.text((x1 + 22, y + h / 2 - 20), num, font=f,
               fill=ink if w >= a.done else pal.muted)
        if w >= a.done:
            d.text((x1 + 22 + d.textlength(num, font=f) + 10, y + h / 2 - 8),
                   f"{a.done * 1e3:.0f} ms", font=font(18), fill=pal.muted)
        y += h + gap


def draw_dots(d, arms, w, pal):
    n = max(a.n for a in arms)
    cols = min(n, 28)
    cell = min(34, (W - 2 * PAD) // cols - 6)
    rows = (n + cols - 1) // cols
    block = 28 + rows * (cell + 6)
    y = _center(len(arms) * block + (len(arms) - 1) * 44) + 28
    for a in arms[::-1]:
        ink = pal.role(a.key)
        d.text((PAD, y - 22), a.label, font=font(18, True), fill=pal.ink)
        k = a.upto(w)
        for i in range(n):
            cx = PAD + (i % cols) * (cell + 6)
            cy = y + (i // cols) * (cell + 6)
            box = (cx, cy, cx + cell, cy + cell)
            if i < k:
                d.rounded_rectangle(box, 4, fill=ink)
            else:
                d.rounded_rectangle(box, 4, fill=pal.card, outline=pal.line)
        y += block + 44


def draw_ribbon(d, arms, w, pal):
    """Every arm gets the same length of track: the ticks do the talking."""
    x0, x1 = PAD, W - PAD - 96
    tmax = max(a.done for a in arms)
    y = _center(len(arms) * 90 + (len(arms) - 1) * 26) + 30
    for a in arms[::-1]:
        ink = pal.role(a.key)
        d.text((PAD, y - 30), a.label, font=font(18, True), fill=pal.ink)
        d.line((x0, y + 30, x1, y + 30), fill=pal.line, width=2)
        for i, t in enumerate(a.ts):
            if t > w:
                break
            x = x0 + (x1 - x0) * t / tmax
            d.line((x, y + 6, x, y + 54), fill=ink, width=3)
        if w >= a.done:
            ex = x0 + (x1 - x0) * a.done / tmax
            d.text((ex + 12, y + 20), f"{a.done * 1e3:.0f} ms",
                   font=font(17, True), fill=ink)
        else:
            px = x0 + (x1 - x0) * min(w, tmax) / tmax
            d.line((px, y - 2, px, y + 62), fill=pal.muted)
        y += 116


STYLES = {"curve": draw_curve, "bars": draw_bars, "dots": draw_dots,
          "ribbon": draw_ribbon}


# ---------------------------------------------------------------- render

def frame(arms, w, pal, style, title, sub):
    d = Canvas(pal.bg)
    chrome(d, d, arms, w, pal, title, sub, style)
    STYLES[style](d, arms, w, pal)
    payoff(d, arms, w, pal)
    return d.image()


def render(arms, out, pal, style, title, sub, fps=30, seconds=7.0):
    race = max(a.done for a in arms)
    total = LEAD + seconds + TAIL
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="looks_"))
    for i in range(int(total * fps)):
        film = i / fps
        w = min(max(film - LEAD, 0.0) / seconds * race, race)
        frame(arms, w, pal, style, title, sub).save(tmp / f"{i:05d}.png")
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libvpx-vp9",
                    "-b:v", "0", "-crf", "24", "-row-mt", "1", "-pix_fmt", "yuv420p",
                    str(out)], check=True)
    return out


def sheet(arms, out, title, sub, at=0.62, scale=0.34):
    """Every style against every palette, one PNG, to choose from."""
    styles, pals = list(STYLES), list(PALETTES)
    tw, th = round(W * scale), round(H * scale)
    head = 34
    im = Image.new("RGB", (tw * len(pals), (th + head) * len(styles)),
                   SHEET_BG)
    d = ImageDraw.Draw(im)
    race = max(a.done for a in arms)
    for r, style in enumerate(styles):
        for c, name in enumerate(pals):
            pal = palette(name)
            tile = frame(arms, race * at, pal, style, title, sub)
            x, y = c * tw, r * (th + head)
            im.paste(tile.resize((tw, th), Image.LANCZOS), (x, y + head))
            # the sheet itself is a plain image, so it needs a real font
            d.text((x + 10, y + 9), f"{style} · {name}",
                   font=_truetype(17, True), fill=SHEET_INK)
    im.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a directory of arm directories")
    ap.add_argument("--arm", action="append", default=[],
                    help="one arm directory; repeatable, slowest to fastest")
    ap.add_argument("--style", default="curve", choices=sorted(STYLES))
    ap.add_argument("--palette", default="midnight", choices=sorted(PALETTES))
    ap.add_argument("--title", default=None)
    ap.add_argument("--sub", default=None)
    ap.add_argument("--seconds", type=float, default=7.0,
                    help="film seconds the recorded race is stretched over")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", help="write a film here")
    ap.add_argument("--frame", help="write one PNG here and stop")
    ap.add_argument("--at", type=float, default=0.62,
                    help="where in the race --frame and --sheet are taken")
    ap.add_argument("--sheet", help="write the style x palette sheet here")
    args = ap.parse_args()

    arms = load(args.run, args.arm)
    title = args.title or arms[0].meta.get("model_name") or "one request"
    sub = args.sub if args.sub is not None else arms[0].meta.get("prompt", "")
    if args.sheet:
        print(sheet(arms, args.sheet, title, sub, at=args.at))
    if args.frame:
        race = max(a.done for a in arms)
        frame(arms, race * args.at, palette(args.palette), args.style,
              title, sub).save(args.frame)
        print(args.frame)
    if args.out:
        print(render(arms, args.out, palette(args.palette), args.style,
                     title, sub, fps=args.fps, seconds=args.seconds))
    if not (args.sheet or args.frame or args.out):
        ap.error("nothing to write: pass --out, --frame or --sheet")


if __name__ == "__main__":
    main()
