"""Compose several arms into one side-by-side race on a shared wall clock.

`sim_compose.py` races robot arms, where a pane is a camera view and the
clock is a control loop.  The pipelines here have no control loop: a
diffusion pipeline produces one clip and stops, a language model emits
tokens as fast as it can.  What stays the same is the honesty rule — one
wall clock, every pane advancing at the rate that was actually recorded,
and a pane that finishes says when.

Two pane painters:

    video   a denoise step counter while the clip is being made, then the
            clip itself, played at its own frame rate
    stream  a text buffer filled at the timestamps the tokens really
            arrived, with a tok/s readout

Both arms record the same shape: `events.json` holding a meta block and a
list of events carrying the wall time, in seconds from the start of the
arm's own run, at which each thing became true.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/truetype/liberation"]

INK = (232, 236, 233)
MUTED = (147, 160, 154)
BG = (18, 25, 23)
CARD = (24, 33, 32)
LINE = (38, 50, 48)
ACCENT = (52, 194, 154)
STOCK = (171, 175, 164)
COMPILED = (140, 165, 200)
NATIVE = (226, 178, 96)

COLORS = {"stock": STOCK, "compiled": COMPILED, "accent": ACCENT,
          "native": NATIVE}


def _ffmpeg() -> str:
    """ffmpeg on PATH, or the one imageio-ffmpeg ships, or $FFMPEG."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "no ffmpeg: install it, pip install imageio-ffmpeg, or set "
            "$FFMPEG"
        ) from exc


def font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    for d in FONT_DIRS:
        for n in names:
            p = pathlib.Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def wrap(draw, text, f, width):
    lines, line = [], ""
    for word in text.replace("\n", " \n ").split(" "):
        if word == "\n":
            lines.append(line)
            line = ""
            continue
        trial = word if not line else line + " " + word
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class Arm:
    """One pane.  Subclasses own everything below the pane's own header."""

    def __init__(self, run_dir):
        d = pathlib.Path(run_dir)
        blob = json.loads((d / "events.json").read_text())
        self.dir = d
        self.meta = blob["meta"]
        self.events = blob["events"]
        self.label = self.meta["label"]
        self.sub = self.meta["sub"]
        self.color = COLORS.get(self.meta.get("color", "stock"), STOCK)
        self.done_t = float(self.meta["done_s"])

    def finished(self, t):
        return t >= self.done_t

    # -- to be provided by the painter ---------------------------------
    def paint_pane(self, im, d, x, y, pane, t):
        raise NotImplementedError

    def paint_readout(self, im, d, x, y, pane, t, fonts):
        raise NotImplementedError


class VideoArm(Arm):
    """A diffusion pipeline: N denoise steps, a VAE decode, then a clip."""

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.frames = np.load(self.dir / "frames.npy")
        self.steps = int(self.meta["steps"])
        self.step_t = [float(e["t"]) for e in self.events
                       if e["kind"] == "step"]
        self.decode_t = float(self.meta.get("decode_s", self.done_t))
        self.clip_fps = float(self.meta.get("clip_fps", 16.0))
        self.ms_per_step = float(self.meta["ms_per_step"])

    def steps_done(self, t):
        return int(np.searchsorted(self.step_t, t, side="right"))

    def paint_pane(self, im, d, x, y, pane, t):
        d.rectangle([x, y, x + pane - 1, y + pane - 1], fill=CARD,
                    outline=LINE)
        if not self.finished(t):
            f_big, f_small = font(46, True), font(14)
            k = self.steps_done(t)
            phase = ("decoding" if k >= self.steps else "denoising")
            txt = f"{k}/{self.steps}"
            d.text((x + pane // 2 - d.textlength(txt, font=f_big) / 2,
                    y + pane // 2 - 46), txt, self.color, font=f_big)
            d.text((x + pane // 2 - d.textlength(phase, font=f_small) / 2,
                    y + pane // 2 + 14), phase, MUTED, font=f_small)
            bar_y = y + pane // 2 + 44
            frac = min(t / self.done_t, 1.0)
            d.rectangle([x + 40, bar_y, x + pane - 40, bar_y + 5], fill=LINE)
            d.rectangle([x + 40, bar_y,
                         x + 40 + int((pane - 80) * frac), bar_y + 5],
                        fill=self.color)
            return
        # the clip, once this arm has actually produced it, in its own
        # aspect ratio — a letterbox is honest, a stretch is not
        i = int((t - self.done_t) * self.clip_fps) % len(self.frames)
        fr = self.frames[i]
        h, w = fr.shape[:2]
        cw = pane
        ch = max(1, int(round(pane * h / w)))
        if ch > pane:
            ch, cw = pane, max(1, int(round(pane * w / h)))
        img = Image.fromarray(fr).resize((cw, ch), Image.BILINEAR)
        im.paste(img, (x + (pane - cw) // 2, y + (pane - ch) // 2))
        d.rectangle([x, y, x + pane - 1, y + pane - 1], outline=LINE)

    def step_ms(self, t):
        """The step in force, or the paired median once the clip is done."""
        if self.finished(t):
            return self.ms_per_step
        k = self.steps_done(t)
        if k < 1:
            return None
        prev = self.step_t[k - 2] if k >= 2 else 0.0
        return (self.step_t[k - 1] - prev) * 1e3

    def paint_readout(self, im, d, x, y, pane, t, fonts):
        f_hz, f_unit, f_sub, f_small = fonts
        live = self.step_ms(t)
        val = "--" if live is None else f"{live:.0f}"
        d.text((x, y), val, self.color, font=f_hz)
        w = d.textlength(val, font=f_hz)
        d.text((x + w + 8, y + 30), "ms / step", MUTED, font=f_unit)
        d.text((x + w + 8, y + 6), f"{self.steps} steps", INK, font=f_sub)
        k = min(self.steps_done(t), self.steps)
        d.text((x, y + 62), f"{k} of {self.steps} denoise steps", MUTED,
               font=f_small)
        if self.finished(t):
            d.text((x, y + 80), f"clip done at {self.done_t:.1f} s",
                   self.color, font=f_small)


class StreamArm(Arm):
    """A language model: a prefill, then tokens at recorded timestamps."""

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.tok_t = [float(e["t"]) for e in self.events]
        self.pieces = [e["text"] for e in self.events]
        self.ttft_ms = float(self.meta["ttft_ms"])
        self.decode_tok_s = float(self.meta["decode_tok_s"])
        self.prompt = self.meta.get("prompt", "")
        img = self.dir / "image.png"
        self.image = Image.open(img).convert("RGB") if img.exists() else None

    def n_tokens(self, t):
        return int(np.searchsorted(self.tok_t, t, side="right"))

    #: seconds of history the live rate averages over. One token is
    #: 5-15 ms, so a per-token rate is unreadable; the robot pane can
    #: show a per-step rate because a control step is 20-100 ms.
    RATE_WINDOW = 0.6

    def rate(self, t):
        """tok/s over the trailing window, or the final median once done."""
        if self.finished(t):
            return self.decode_tok_s
        n = self.n_tokens(t)
        if n < 2:
            return None                      # still prefilling
        lo = np.searchsorted(self.tok_t, t - self.RATE_WINDOW, side="left")
        lo = min(int(lo), n - 2)
        span = self.tok_t[n - 1] - self.tok_t[lo]
        return (n - 1 - lo) / span if span > 1e-6 else None

    def paint_pane(self, im, d, x, y, pane, t):
        d.rectangle([x, y, x + pane - 1, y + pane - 1], fill=CARD,
                    outline=LINE)
        f_body, f_small = font(15), font(13)
        top = y
        if self.image is not None:
            # a VLM pane has to show what it was looking at
            ih = int(pane * 0.42)
            iw = int(self.image.width * ih / self.image.height)
            iw = min(iw, pane - 2)
            im.paste(self.image.resize((iw, ih), Image.BILINEAR),
                     (x + (pane - iw) // 2, y + 1))
            d.rectangle([x + (pane - iw) // 2, y + 1,
                         x + (pane - iw) // 2 + iw - 1, y + ih],
                        outline=LINE)
            top = y + ih + 4
        n = self.n_tokens(t)
        if n == 0:
            d.text((x + 16, top + 14), "reading the image…", MUTED,
                   font=f_small)
            return
        text = "".join(self.pieces[:n]).lstrip()
        lines = wrap(d, text, f_body, pane - 32)
        lh, lh0 = 21, top + 14
        vis = max(1, (y + pane - lh0 - 10) // lh)
        shown = lines[-vis:]
        for j, ln in enumerate(shown):
            d.text((x + 16, lh0 + j * lh), ln, INK, font=f_body)
        if n < len(self.tok_t):                      # a live caret
            cx = x + 16 + d.textlength(shown[-1], font=f_body)
            cy = lh0 + (len(shown) - 1) * lh
            d.rectangle([cx + 2, cy, cx + 9, cy + 16], fill=self.color)

    def paint_readout(self, im, d, x, y, pane, t, fonts):
        f_hz, f_unit, f_sub, f_small = fonts
        live = self.rate(t)
        val = "--" if live is None else f"{live:.0f}"
        d.text((x, y), val, self.color, font=f_hz)
        w = d.textlength(val, font=f_hz)
        d.text((x + w + 8, y + 30), "tok / s", MUTED, font=f_unit)
        # the first token is an event: it has not happened yet at t=0
        arrived = self.tok_t and t >= self.tok_t[0]
        d.text((x + w + 8, y + 6),
               (f"TTFT {self.ttft_ms:.0f} ms" if arrived else "prefill..."),
               INK if arrived else MUTED, font=f_sub)
        n = self.n_tokens(t)
        d.text((x, y + 62), f"{n} of {len(self.tok_t)} tokens", MUTED,
               font=f_small)
        if self.finished(t):
            d.text((x, y + 80), f"answer done at {self.done_t:.1f} s",
                   self.color, font=f_small)
        note = self.meta.get("stream_note")
        if note:
            d.text((x, y + 98), note, MUTED, font=f_small)


class StreamBatchArm(Arm):
    """A serving engine under load: N requests in flight at once.

    One row per request, each filling at the timestamps that request's
    tokens actually arrived. The big readout is aggregate throughput,
    because that is what a batch is for; per-request rate and TTFT sit
    beside it, because that is what a caller feels.
    """

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.n = int(self.meta["concurrency"])
        self.streams = [[] for _ in range(self.n)]
        for e in self.events:
            self.streams[int(e["s"])].append((float(e["t"]), e["text"]))
        for st in self.streams:
            st.sort(key=lambda x: x[0])
        self.times = [[t for t, _ in st] for st in self.streams]
        self.agg = float(self.meta["aggregate_tok_s"])
        self.per = float(self.meta["decode_tok_s_per_stream"])
        self.ttft = float(self.meta["ttft_ms_median"])
        self.total = sum(len(st) for st in self.streams)

    #: same reasoning as StreamArm.RATE_WINDOW, wider because the
    #: aggregate is the sum of N bursty streams.
    RATE_WINDOW = 1.0

    def done_tokens(self, t):
        return sum(int(np.searchsorted(ts, t, side="right"))
                   for ts in self.times)

    def rate(self, t):
        """Aggregate tok/s over the trailing window, final value once done."""
        if self.finished(t):
            return self.agg
        lo = t - self.RATE_WINDOW
        if lo < 0:
            lo = 0.0
        n = sum(int(np.searchsorted(ts, t, side="right"))
                - int(np.searchsorted(ts, lo, side="left"))
                for ts in self.times)
        span = t - lo
        return n / span if span > 1e-6 and n else None

    def paint_pane(self, im, d, x, y, pane, t):
        d.rectangle([x, y, x + pane - 1, y + pane - 1], fill=CARD,
                    outline=LINE)
        f_head = font(12)
        d.text((x + 12, y + 8), f"{self.n} concurrent request"
               + ("s" if self.n > 1 else ""), MUTED, font=f_head)
        top = y + 28
        rows = max(1, (y + pane - 10) - top)
        rh = min(26, rows // self.n)
        fs = max(9, min(15, rh - 5))
        f_body = font(fs)
        for k, st in enumerate(self.streams):
            ry = top + k * rh
            if ry + rh > y + pane:
                break
            n = int(np.searchsorted(self.times[k], t, side="right"))
            if n == 0:
                continue
            text = "".join(p for _, p in st[:n]).replace("\n", " ")
            # keep the tail visible: a request that is still going should
            # look like it is going
            while d.textlength(text, font=f_body) > pane - 24 and text:
                text = text[1:]
            live = n < len(st)
            d.text((x + 12, ry), text, INK if live else MUTED, font=f_body)
            if live:
                cx = x + 12 + d.textlength(text, font=f_body)
                d.rectangle([cx + 2, ry + 1, cx + 6, ry + fs],
                            fill=self.color)

    def paint_readout(self, im, d, x, y, pane, t, fonts):
        f_hz, f_unit, f_sub, f_small = fonts
        live = self.rate(t)
        val = "--" if live is None else f"{live:.0f}"
        d.text((x, y), val, self.color, font=f_hz)
        w = d.textlength(val, font=f_hz)
        d.text((x + w + 8, y + 30), "tok / s total", MUTED, font=f_unit)
        started = any(ts and t >= ts[0] for ts in self.times)
        d.text((x + w + 8, y + 6),
               (f"{self.per:.0f} per request · TTFT {self.ttft:.0f} ms"
                if started else "prefill..."),
               INK if started else MUTED, font=f_sub)
        d.text((x, y + 62),
               f"{self.done_tokens(t)} of {self.total} tokens", MUTED,
               font=f_small)
        if self.finished(t):
            d.text((x, y + 80), f"all {self.n} done at {self.done_t:.1f} s",
                   self.color, font=f_small)
        note = self.meta.get("stream_note")
        if note:
            d.text((x, y + 98), note, MUTED, font=f_small)


KINDS = {"video": VideoArm, "stream": StreamArm,
         "stream_batch": StreamBatchArm}


def _layout(n_panes, pane, footer):
    """Canvas size for one chapter, and its wrapped footer lines."""
    gap, pad, head = 26, 34, 168
    W = pad * 2 + pane * n_panes + gap * (n_panes - 1)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    lines = []
    for para in (footer or "").split("\n"):
        if para:
            lines.extend(wrap(probe, para, font(13), W - pad * 2))
    H = head + pane + 145 + 17 * len(lines) + 18
    return W, H, lines


def paint(chapter, t, W, H, pane, foot_lines):
    """One frame of one chapter, on the canvas the whole film shares."""
    gap, pad, head = 26, 34, 168
    arms = chapter["arms"]
    f_lab, f_sub, f_small = font(19, True), font(14), font(13)
    fonts = (font(52, True), font(15, True), f_sub, f_small)
    title, note = chapter.get("title"), chapter.get("note")

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    f_title = font(27, True)
    while (title and probe.textlength(title, font=f_title)
           > W - pad * 2 - 150 and f_title.size > 15):
        f_title = font(f_title.size - 1, True)
    f_note = f_small
    while (note and probe.textlength(note, font=f_note) > W - pad * 2
           and f_note.size > 9):
        f_note = font(f_note.size - 1)

    row = pad * 2 + pane * len(arms) + gap * (len(arms) - 1)
    x0 = pad + (W - row) // 2          # narrower chapters stay centred

    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if title:
        d.text((pad, 26), title, INK, font=f_title)
    if chapter.get("subtitle"):
        d.text((pad, 62), chapter["subtitle"], MUTED, font=f_sub)
    if note:
        d.text((pad, 84), note, MUTED, font=f_note)

    for idx, a in enumerate(arms):
        x = x0 + idx * (pane + gap)
        a.paint_pane(im, d, x, head, pane, t)
        d.text((x, head - 46), a.label, a.color, font=f_lab)
        d.text((x, head - 24), a.sub, MUTED, font=f_sub)
        a.paint_readout(im, d, x, head + pane + 12, pane, t, fonts)
        bar_y = head + pane + 132
        width = int(pane * min(t / a.done_t, 1.0))
        d.rectangle([x, bar_y, x + pane, bar_y + 5], fill=LINE)
        d.rectangle([x, bar_y, x + width, bar_y + 5], fill=a.color)

    d.text((W - pad - 96, 30), f"{t:5.1f} s", INK, font=f_lab)
    for j, ln in enumerate(foot_lines):
        d.text((pad, H - 14 - 17 * (len(foot_lines) - j)), ln, MUTED,
               font=f_small)
    return im


def render(chapters, out_path, fps=30, pane=400, speed=1.0):
    """Render every chapter onto one canvas and encode them as one film."""
    sized = []
    for c in chapters:
        W, H, lines = _layout(len(c["arms"]), pane, c.get("footer"))
        sized.append((W, H, lines))
    W = max(w for w, _, _ in sized)
    H = max(h for _, h, _ in sized)

    frames_dir = pathlib.Path(out_path).parent / "_race_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    k = 0
    for c, (_, _, lines) in zip(chapters, sized):
        span = (c["seconds"] if c.get("seconds") is not None
                else max(a.done_t for a in c["arms"]) + c.get("tail", 2.0))
        for i in range(int(span / speed * fps)):
            paint(c, i / fps * speed, W, H, pane, lines).save(
                frames_dir / f"{k:05d}.jpg", quality=88)
            k += 1

    subprocess.run([
        _ffmpeg(), "-y", "-v", "error", "-framerate", str(fps),
        "-i", str(frames_dir / "%05d.jpg"),
        "-c:v", "libvpx-vp9", "-b:v", "1400k", "-pix_fmt", "yuv420p",
        str(out_path)], check=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    frames_dir.rmdir()
    print(f"wrote {out_path} ({k} frames, {k / fps:.1f}s, "
          f"{len(chapters)} chapter(s))")


def load_chapter(spec):
    arms = []
    runs = pathlib.Path(spec["runs"])
    names = (spec["arms"].split(",") if isinstance(spec["arms"], str)
             else spec["arms"])
    for name in names:
        d = runs / str(name).strip()
        meta = json.loads((d / "events.json").read_text())["meta"]
        arms.append(KINDS[meta["kind"]](d))
    return dict(spec, arms=arms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None,
                    help="JSON file: {pane, fps, chapters:[{runs, arms, "
                         "title, subtitle, note, footer, tail, seconds}]}")
    ap.add_argument("--runs")
    ap.add_argument("--arms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pane", type=int, default=400)
    ap.add_argument("--tail", type=float, default=2.0)
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--footer", default=None)
    args = ap.parse_args()

    if args.spec:
        blob = json.loads(pathlib.Path(args.spec).read_text())
        chapters = [load_chapter(c) for c in blob["chapters"]]
        render(chapters, args.out, fps=blob.get("fps", args.fps),
               pane=blob.get("pane", args.pane),
               speed=blob.get("speed", args.speed))
        return

    if not (args.runs and args.arms):
        ap.error("give --spec, or both --runs and --arms")
    chapter = load_chapter({
        "runs": args.runs, "arms": args.arms, "title": args.title,
        "subtitle": args.subtitle, "note": args.note,
        "footer": args.footer, "tail": args.tail,
        "seconds": args.seconds})
    render([chapter], args.out, fps=args.fps, pane=args.pane,
           speed=args.speed)


if __name__ == "__main__":
    main()
