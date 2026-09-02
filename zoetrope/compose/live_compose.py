"""The run demo: the answer arriving, with the rate drawn beside it.

The oldest film in this kit is also the one everybody reads without being
told anything — two arms writing the same answer on one wall clock, and one
of them finishes first. This module keeps that body and puts the shape of it
on the page as well: a strip under the panes where tokens delivered are
plotted against time, so the thing you feel in the text has a line you can
point at.

Everything is drawn from the recording, on the shared canvas, in whichever
colour system is asked for:

    zoetrope live examples/runs/stream --palette paper --out a.webm
    zoetrope live examples/runs/stream_batch --chart bars --palette phosphor \
        --out b.webm

Two run shapes, one painter. A `stream` arm is one request and its pane is
the answer; a `stream_batch` arm is a serving engine with several requests in
flight and its pane is one line per request. The rate strip does not care
which: it counts tokens that have arrived.
"""

from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import subprocess
import tempfile

from PIL import Image

from .canvas import Canvas, R_MD, R_SM, ease, ease_out, font
from .palettes import PALETTES, palette
from .race_compose import _ffmpeg, wrap

W, H = 1280, 720
PAD = 96
LEAD, TAIL = 0.8, 2.6

#: Under this, a run is too quick to watch and the film stretches it. The
#: page then says so: a clock the viewer cannot trust is worse than no clock.
REAL_TIME_FLOOR = 2.0


class Arm:
    def __init__(self, run_dir):
        d = pathlib.Path(run_dir)
        blob = json.loads((d / "events.json").read_text())
        self.dir = d
        self.meta = blob["meta"]
        self.events = blob["events"]
        self.label = self.meta["label"]
        self.sub = self.meta.get("sub", "")
        self.key = self.meta.get("color", "stock")
        self.batch = self.meta.get("kind") == "stream_batch"
        self.ts = [float(e["t"]) for e in self.events]
        self.done = float(self.meta.get("done_s") or self.ts[-1])
        self.streams = sorted({e.get("s", 0) for e in self.events})
        self.by_stream = {s: [] for s in self.streams}
        for e in self.events:
            self.by_stream[e.get("s", 0)].append((float(e["t"]), e["text"]))
        self.total = len(self.events)
        self.ttft_ms = self.meta.get("ttft_ms")
        self.final_rate = self.meta.get("decode_tok_s")
        self.note = self.meta.get("stream_note")
        img = d / "image.png"
        self.image = Image.open(img).convert("RGB") if img.exists() else None

    def n(self, t):
        return bisect.bisect_right(self.ts, t)

    #: One token is 5-15 ms, so a per-token rate is unreadable noise.
    WINDOW = 0.6

    def rate(self, t):
        if t >= self.done:
            if self.final_rate:
                return float(self.final_rate)
            return (self.total - 1) / max(self.ts[-1] - self.ts[0], 1e-6)
        n = self.n(t)
        if n < 2:
            return None
        lo = min(bisect.bisect_left(self.ts, t - self.WINDOW), n - 2)
        span = self.ts[n - 1] - self.ts[lo]
        return (n - 1 - lo) / span if span > 1e-6 else None

    def text(self, s, t):
        return "".join(x for u, x in self.by_stream[s] if u <= t)


def load(run_dir, order=None):
    root = pathlib.Path(run_dir)
    names = order or sorted(p.name for p in root.iterdir()
                            if (p / "events.json").exists())
    arms = [Arm(root / n) for n in names]
    if not arms:
        raise SystemExit(f"no arms under {run_dir}")
    return sorted(arms, key=lambda a: -a.done)


# ---------------------------------------------------------------- painting

def _pane(d, a, box, t, pal, ink):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, R_MD, fill=pal.card, outline=pal.line)
    d.rectangle((x0, y0, x0 + 4, y1), fill=ink)
    d.text((x0 + 18, y0 + 14), a.label, font=font(17, True), fill=pal.ink)
    d.text((x0 + 18, y0 + 36), a.sub, font=font(13), fill=pal.muted)

    body_top, fb = y0 + 66, font(15)
    if a.image is not None:
        # a vision model's pane has to show what it was looking at
        ih = 104
        iw = round(a.image.width * ih / a.image.height)
        d.image_at(a.image, (x0 + (x1 - x0 - iw) / 2, body_top), (iw, ih))
        d.rounded_rectangle((x0 + (x1 - x0 - iw) / 2, body_top,
                             x0 + (x1 - x0 + iw) / 2, body_top + ih), R_SM,
                            outline=pal.line)
        body_top += ih + 12
    n = a.n(t)
    if not n:
        d.text((x0 + 18, body_top + 6), "prefill...", font=font(14),
               fill=pal.muted)
    elif a.batch:
        rows = a.streams[:8]
        rh = 22
        for i, s in enumerate(rows):
            y = body_top + i * rh
            if y + rh > y1 - 112:
                break
            line = a.text(s, t).replace("\n", " ").strip()
            while line and d.textlength(line, font=font(13)) > x1 - x0 - 66:
                line = line[1:]
            d.text((x0 + 18, y), f"{s + 1}", font=font(12), fill=pal.muted)
            d.text((x0 + 40, y), line, font=font(13), fill=pal.ink)
    else:
        text = a.text(a.streams[0], t).lstrip()
        lines = wrap(d, text, fb, x1 - x0 - 36)
        lh = 21
        room = max(1, int((y1 - 116 - body_top) // lh))
        shown = lines[-room:]
        for j, ln in enumerate(shown):
            d.text((x0 + 18, body_top + j * lh), ln, font=fb, fill=pal.ink)
        if n < a.total:
            cx = x0 + 18 + d.textlength(shown[-1], font=fb)
            cy = body_top + (len(shown) - 1) * lh
            d.rectangle((cx + 3, cy + 1, cx + 10, cy + 17), fill=ink)

    live = a.rate(t)
    val = "--" if live is None else f"{live:.0f}"
    foot = y1 - (38 if a.note else 14)
    d.text((x0 + 18, foot - 58), val, font=font(38, True), fill=ink)
    vw = d.textlength(val, font=font(38, True))
    d.text((x0 + 26 + vw, foot - 28), "tok/s", font=font(15), fill=pal.muted)
    if a.ttft_ms and n:
        d.text((x0 + 26 + vw, foot - 52), f"TTFT {float(a.ttft_ms):.0f} ms",
               font=font(14), fill=pal.muted)
    right = (f"{n} of {a.total}" if t < a.done
             else f"done at {a.done:.2f} s")
    rf = font(15, True)
    d.text((x1 - 18 - d.textlength(right, font=rf), foot - 20), right,
           font=rf, fill=pal.muted if t < a.done else ink)
    # whether this arm wrote the same answer as the reference, and how far it
    # agreed: a page showing two different texts side by side owes the reader
    # that, and it is exactly the line a demo is tempted to leave out
    if a.note:
        ny = foot + 4
        for line in wrap(d, a.note, font(12), x1 - x0 - 36)[:2]:
            d.text((x0 + 18, ny), line, font=font(12), fill=pal.muted)
            ny += 15


def _curve(d, arms, box, t, pal, race):
    x0, y0, x1, y1 = box
    top = max(a.total for a in arms)
    d.line((x0, y1, x1, y1), fill=pal.line)
    for k in (0.5, 1.0):
        y = y1 - (y1 - y0) * k
        d.line((x0, y, x1, y), fill=pal.grid)
    d.text((x0, y0 - 20), "tokens delivered", font=font(13), fill=pal.muted)
    d.text((x1 - d.textlength(f"{top}", font=font(13)), y0 - 20), f"{top}",
           font=font(13), fill=pal.muted)

    def pt(u, k):
        return (x0 + (x1 - x0) * u / race, y1 - (y1 - y0) * k / top)

    for a in arms:
        ink = pal.role(a.key)
        n = a.n(t)
        # every stamp, not a sample of them: a batched engine delivers a
        # step's worth of tokens at once and then waits, and decimating that
        # turns a real staircase into a straight line that says nothing
        pts = [pt(0, 0)] + [pt(a.ts[i], i + 1) for i in range(n)]
        if n:
            pts.append(pt(min(t, a.done), n))
        if len(pts) > 1:
            d.line(pts, fill=ink, width=2.5, joint="curve")
            hx, hy = pts[-1]
            r = 5 if t < a.done else 6
            d.ellipse((hx - r, hy - r, hx + r, hy + r), fill=ink)
    px = x0 + (x1 - x0) * min(t, race) / race
    d.line((px, y0 - 6, px, y1 + 4), fill=pal.line)


def _bars(d, arms, box, t, pal, race):
    x0, y0, x1, y1 = box
    top = max(a.total for a in arms)
    rows = (y1 - y0) / len(arms)
    d.text((x0, y0 - 20), "tokens delivered", font=font(13), fill=pal.muted)
    for i, a in enumerate(arms[::-1]):
        ink = pal.role(a.key)
        y = y0 + i * rows
        n = a.n(t)
        w = (x1 - x0 - 120) * n / top
        d.rounded_rectangle((x0, y + 4, x0 + max(w, 2), y + rows - 12), R_SM,
                            fill=ink)
        d.text((x1 - 104, y + rows / 2 - 14), f"{n}", font=font(22, True),
               fill=ink)


CHARTS = {"curve": _curve, "bars": _bars, "none": None}


def frame(arms, t, pal, *, title, sub, chart="curve", race=1.0, stretch=None):
    d = Canvas(pal.bg)
    d.text((PAD, 40), title, font=font(30, True), fill=pal.ink)
    if sub:
        for i, line in enumerate(wrap(d, sub, font(16), W - 2 * PAD - 260)[:1]):
            d.text((PAD, 82), line, font=font(16), fill=pal.muted)
    clock = f"{min(t, race):.2f} s"
    cf = font(17)
    d.text((W - PAD - d.textlength(clock, font=cf), 82), clock, font=cf,
           fill=pal.muted)
    if stretch:
        note = f"slowed {stretch:.0f}x — one clock, both arms"
        d.text((W - PAD - d.textlength(note, font=font(13)), 44), note,
               font=font(13), fill=pal.muted)
    d.line((PAD, 116, W - PAD, 116), fill=pal.line)

    draw = CHARTS[chart]
    pane_bot = 660 if draw is None else 524
    gap = 20
    pw = (W - 2 * PAD - gap * (len(arms) - 1)) / len(arms)
    for i, a in enumerate(arms[::-1]):
        x = PAD + i * (pw + gap)
        _pane(d, a, (x, 140, x + pw, pane_bot), t, pal, pal.role(a.key))
    if draw is not None:
        draw(d, arms, (PAD, 580, W - PAD, 672), t, pal, race)
    return d.image()


def render(arms, out, pal, *, title, sub, chart="curve", seconds=None,
           fps=30):
    race = max(a.done for a in arms)
    film = seconds or (8.0 if race < REAL_TIME_FLOOR else race)
    stretch = film / race if film / race > 1.4 else None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="live_"))
    total = LEAD + film + TAIL
    for i in range(int(total * fps)):
        s = i / fps
        t = min(max(s - LEAD, 0.0) / film * race, race)
        frame(arms, t, pal, title=title, sub=sub, chart=chart, race=race,
              stretch=stretch).save(tmp / f"{i:05d}.png")
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libvpx-vp9",
                    "-b:v", "0", "-crf", "24", "-row-mt", "1",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--arms", help="comma-separated order; default: sorted")
    ap.add_argument("--palette", default="midnight", choices=sorted(PALETTES))
    ap.add_argument("--chart", default="curve", choices=sorted(CHARTS))
    ap.add_argument("--title"); ap.add_argument("--sub")
    ap.add_argument("--label", help="rename the first arm's pane — a film of "
                                    "one engine should not be labelled as if "
                                    "it were half of a comparison")
    ap.add_argument("--pane-sub", help="and its second line")
    ap.add_argument("--color", help="the role that arm is drawn in")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--at", type=float, default=0.6)
    ap.add_argument("--frame"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    arms = load(a.runs, a.arms.split(",") if a.arms else None)
    if a.label:
        arms[0].label = a.label
    if a.pane_sub:
        arms[0].sub = a.pane_sub
    if a.color:
        arms[0].key = a.color
    m = arms[0].meta
    title = a.title or m.get("model") or "one request"
    sub = a.sub if a.sub is not None else m.get("prompt", "")
    pal = palette(a.palette)
    race = max(x.done for x in arms)
    if a.frame:
        frame(arms, race * a.at, pal, title=title, sub=sub, chart=a.chart,
              race=race).save(a.frame)
        print(a.frame)
    if a.out:
        print(render(arms, a.out, pal, title=title, sub=sub, chart=a.chart,
                     seconds=a.seconds, fps=a.fps))
    if not (a.frame or a.out):
        ap.error("nothing to write: pass --frame or --out")


if __name__ == "__main__":
    main()
