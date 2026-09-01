"""One idea: the same decision, and how many times each runtime goes to memory.

Every CUDA kernel reads its operands from device memory and writes its result
back. So a kernel launch is a trip to memory, and the count of them is a
number a person can hold in their head — unlike a kernel taxonomy, which is
a number a person closes the tab over.

This compositor draws exactly that and nothing else: two lanes, one shared
memory bar, two counters, and a race on the measured wall clock. The panes
elsewhere in this package describe; this one argues.

    python -m demokit.compose.trips_compose --runs runs/pi05_trips \
        --arms torch,fp8 --also fp16 --out why.webm
"""

import argparse
import json
import pathlib
import subprocess

from PIL import Image, ImageDraw

from demokit.compose.race_compose import (BG, CARD, INK, LINE, MUTED, COLORS,
                                          STOCK, _ffmpeg, font, wrap)

W, H = 1280, 720
LX, RX = 96, W - 96
MEM_Y, MEM_H = 360, 56
LEAD, TAIL, RACE = 2.0, 3.6, 9.0        # seconds: title, end card, the race


class Lane:
    def __init__(self, run_dir, above: bool):
        blob = json.loads((pathlib.Path(run_dir) / "events.json").read_text())
        m = blob["meta"]
        self.meta = m
        self.label = m["label"]
        self.sub = m["sub"]
        self.color = COLORS.get(m.get("color", "stock"), STOCK)
        self.trips = int(m["launches"])
        self.ms = float(m.get("decision_ms") or m["median_ms"])
        self.above = above

    def shown(self, t, slow_ms):
        """Trips completed, on the film clock, at this arm's measured rate."""
        if t <= LEAD:
            return 0
        span = RACE * self.ms / slow_ms
        return min(self.trips, int(self.trips * (t - LEAD) / span))

    def done(self, t, slow_ms):
        return self.shown(t, slow_ms) >= self.trips


def paint(lanes, t, slow_ms, peak, note=None, gate=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    m = lanes[0].meta
    d.text((LX, 54), "one decision", INK, font=font(46, True))
    head = (f'{m.get("model", "")}   ·   {m.get("steps", "")} flow steps   ·   '
            f'the same policy, the same instruction, both sides')
    d.text((LX, 112), head, MUTED, font=font(18))

    d.rectangle([LX, MEM_Y, RX, MEM_Y + MEM_H], fill=CARD, outline=LINE)
    cap = "GPU memory"
    f_mem = font(22, True)
    d.text((W // 2 - d.textlength(cap, font=f_mem) / 2, MEM_Y + 17), cap,
           MUTED, font=f_mem)

    every = max(1, peak // 300)          # one tick per N trips, both lanes
    for lane in lanes:
        n = lane.shown(t, slow_ms)
        top = MEM_Y - 118 if lane.above else MEM_Y + MEM_H + 62
        bar = MEM_Y - 46 if lane.above else MEM_Y + MEM_H + 46
        f_lab, f_num = font(24, True), font(60, True)

        # the trips themselves, one line each, thinned so they stay countable
        for i in range(0, n, every):
            x = LX + (RX - LX) * i / peak
            y0, y1 = (bar, MEM_Y) if lane.above else (bar, MEM_Y + MEM_H)
            d.line([x, y0, x, y1], fill=lane.color, width=1)

        # and the pile of them, on one scale so the two lanes compare
        w = (RX - LX) * n / peak
        h = 26
        y = bar - h if lane.above else bar
        d.rectangle([LX, y, LX + max(w, 1), y + h], fill=lane.color)

        ly = top - 74 if lane.above else top + 30
        d.text((LX, ly), lane.label, lane.color, font=f_lab)
        d.text((LX, ly + 30), lane.sub, MUTED, font=font(16))
        num = f"{n:,}"
        tx = RX - d.textlength(num, font=f_num)
        d.text((tx, ly - 14), num, lane.color, font=f_num)
        unit = "trips to memory"
        d.text((RX - d.textlength(unit, font=font(17)), ly + 52), unit,
               MUTED, font=font(17))
        # a lane that has finished says so and waits, the way a demo pane
        # does -- otherwise the long tail reads as the film having stalled
        if lane.done(t, slow_ms):
            fin = f"done in {lane.ms:.0f} ms"
            f_fin = font(19, True)
            d.text((RX - d.textlength(fin, font=f_fin), ly + 76), fin,
                   lane.color, font=f_fin)

    if t > LEAD + RACE + 0.4:
        fast = min(lanes, key=lambda a: a.ms)
        slow = max(lanes, key=lambda a: a.ms)
        line = f"{slow.ms:.0f} ms  \u2192  {fast.ms:.0f} ms"
        f_end = font(46, True)
        d.text((LX, H - 112), line, INK, font=f_end)
        # Two baselines, because one of them is the one that judges: the
        # host as shipped is what a reader recognises, and the compiled
        # host is what the claim has to survive.
        # the trips ratio is already on screen twice, in the counters
        tail = (f"{slow.ms / fast.ms:.1f}x over PyTorch as shipped"
                + (f"   \u00b7   {gate.ms / fast.ms:.1f}x over torch.compile"
                   if gate else ""))
        tx = LX + d.textlength(line, font=f_end) + 34
        f_tail = font(21)
        while (d.textlength(tail, font=f_tail) > RX - tx
               and f_tail.size > 12):
            f_tail = font(f_tail.size - 1)
        d.text((tx, H - 94), tail, MUTED, font=f_tail)
    if note:
        for j, ln in enumerate(wrap(d, note, font(14), RX - LX)[:2]):
            d.text((LX, H - 44 + j * 18), ln, MUTED, font=font(14))
    return im


def render(runs, out_path, *, fps=30, note=None, gate=None):
    """`gate` is the baseline the claim must survive, named but not drawn."""
    lanes = [Lane(r, above=(i == 0)) for i, r in enumerate(runs)]
    gate = Lane(gate, above=True) if gate else None
    slow_ms = max(a.ms for a in lanes)
    peak = max(a.trips for a in lanes)
    frames = pathlib.Path(out_path).parent / "_trip_frames"
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("*.jpg"):
        old.unlink()
    total = LEAD + RACE + TAIL
    k = 0
    for i in range(int(total * fps)):
        paint(lanes, i / fps, slow_ms, peak, note, gate).save(
            frames / f"{k:05d}.jpg", quality=92)
        k += 1
    subprocess.run([
        _ffmpeg(), "-y", "-v", "error", "-framerate", str(fps),
        "-i", str(frames / "%05d.jpg"),
        "-c:v", "libvpx-vp9", "-b:v", "1600k", "-pix_fmt", "yuv420p",
        str(out_path)], check=True)
    for old in frames.glob("*.jpg"):
        old.unlink()
    frames.rmdir()
    print(f"wrote {out_path} ({k} frames, {k / fps:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arms", required=True, help="slow first, fast second")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--note")
    ap.add_argument("--gate", help="the compiled baseline, named in the tail")
    a = ap.parse_args()
    root = pathlib.Path(a.runs)
    render([root / n.strip() for n in a.arms.split(",")], a.out, fps=a.fps,
           note=a.note, gate=(root / a.gate) if a.gate else None)


if __name__ == "__main__":
    main()
