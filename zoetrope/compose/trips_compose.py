"""One picture: the same work, cut into the trips to memory it really took.

Every CUDA kernel reads its operands from device memory and writes its result
back, so a kernel launch is a trip. Both arms compute the same decision, so
both rows are the same length; one is cut into five times as many pieces.

That is the whole argument, and it needs no legend: the same block of work,
one of them chopped fine. Fewer pieces for the same work means each piece
carries more of it, which is what fusing kernels means. The speed follows.

Colour is measured too -- how hard the chip was working during that stretch,
from Nsight Compute, weighted by each kernel's own duration.

    python -m zoetrope.compose.trips_compose --runs runs/pi05_trips \
        --arms torch,fp8 --gate compiled --out why.webm
"""

import argparse
import json
import pathlib
import subprocess

from PIL import Image, ImageDraw

from zoetrope.compose.race_compose import (BG, INK, LINE, MUTED, COLORS,
                                          STOCK, _ffmpeg, font, wrap)

W, H = 1280, 720
LX, RX = 96, W - 96
LEAD, TAIL, RACE = 1.6, 3.8, 9.0        # seconds: title, payoff, the race
PER_BLOCK = 100                         # trips per drawn block, both rows
ROW_A, ROW_B, STRIP_H = 176, 372, 56    # where each arm's row starts

#: How hard a part of the device was working, as a temperature. Cool is the
#: machine idling while the work waits on something else; hot is the work
#: being done. For a runtime that is the right way round -- the goal is a
#: hot GPU, not a calm one.
HEAT = ((0, (58, 170, 120)), (20, (96, 190, 96)), (50, (226, 196, 70)),
        (80, (232, 142, 58)), (100, (226, 74, 66)))
COLD = (38, 48, 45)


def _heat(pct):
    """The utilisation ramp: green, yellow, orange, red."""
    pct = max(0.0, min(100.0, float(pct)))
    for (a, ca), (b, cb) in zip(HEAT, HEAT[1:]):
        if pct <= b:
            k = 0.0 if b == a else (pct - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * k) for i in range(3))
    return HEAT[-1][1]


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
        self.heat = m.get("sm_slices") or []
        self.peak = m.get("sm_peak_pct")
        self.blocks = max(1, round(self.trips / PER_BLOCK))

    def span_s(self, slow_ms):
        """How long this row takes to fill, on the film clock."""
        return RACE * self.ms / slow_ms

    def at(self, t, slow_ms):
        """How far through its own decision this arm is, 0 to 1."""
        if t <= LEAD:
            return 0.0
        return min(1.0, (t - LEAD) / self.span_s(slow_ms))

    def shown(self, t, slow_ms):
        return int(self.trips * self.at(t, slow_ms))

    def done(self, t, slow_ms):
        return self.at(t, slow_ms) >= 1.0


def _row(d, lane, y, t, slow_ms):
    """One arm: its name, its work cut into pieces, and two rolling numbers."""
    at = lane.at(t, slow_ms)
    d.text((LX, y), lane.label, lane.color, font=font(25, True))
    d.text((LX, y + 31), lane.sub, MUTED, font=font(16))

    top = y + 62
    wide = RX - LX
    bw = wide / lane.blocks
    heat = lane.heat
    for k in range(int(lane.blocks * at)):
        x = LX + k * bw
        # no utilisation measured for this arm: the arm's own accent, not a
        # temperature. Green would say "measured, and cool", which is a
        # claim nobody made.
        ink = (_heat(heat[min(int(k * len(heat) / lane.blocks),
                              len(heat) - 1)]) if heat else lane.color)
        d.rectangle([x + 1, top, x + bw - 1, top + STRIP_H], fill=ink)
    d.rectangle([LX, top, RX, top + STRIP_H], outline=LINE)

    f_num = font(20, True)
    ms = f"{lane.ms * at:.0f} ms"
    d.text((LX, top + STRIP_H + 12), ms, lane.color, font=f_num)
    trips = f"{lane.shown(t, slow_ms):,} trips to memory"
    d.text((RX - d.textlength(trips, font=f_num), top + STRIP_H + 12), trips,
           lane.color, font=f_num)
    if lane.done(t, slow_ms):
        f_fin = font(20, True)
        d.text((RX - d.textlength("done", font=f_fin), y + 4), "done",
               lane.color, font=f_fin)


def paint(lanes, t, slow_ms, peak=None, note=None, gate=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    slow = max(lanes, key=lambda a: a.ms)
    fast = min(lanes, key=lambda a: a.ms)

    m = lanes[0].meta
    d.text((LX, 30), "one decision", INK, font=font(44, True))
    d.text((LX, 86), f'{m.get("model", "")}   ·   the same policy, the same '
           f'instruction, both sides', MUTED, font=font(18))
    d.text((LX, 128), f"cut into the trips to memory it really took, "
           f"one block per {PER_BLOCK}", INK, font=font(19, True))

    for lane in lanes:
        _row(d, lane, ROW_A if lane.above else ROW_B, t, slow_ms)

    if t > LEAD + RACE + 0.3:
        r = slow.trips / fast.trips
        d.text((LX, 556), f"the same work   ·   {r:.0f}x fewer trips   ·   "
               f"so each of ours carries {r:.0f}x as much", INK,
               font=font(25, True))
        line = f"{slow.ms:.0f} ms  →  {fast.ms:.0f} ms"
        f_end = font(42, True)
        d.text((LX, 604), line, INK, font=f_end)
        # Two baselines, because one of them is the one that judges: the host
        # as shipped is what a reader recognises, and the compiled host is
        # what the claim has to survive.
        tail = (f"{slow.ms / fast.ms:.1f}x over PyTorch as shipped"
                + (f"   ·   {gate.ms / fast.ms:.1f}x over torch.compile"
                   if gate else ""))
        tx = LX + d.textlength(line, font=f_end) + 32
        f_tail = font(21)
        while d.textlength(tail, font=f_tail) > RX - tx and f_tail.size > 12:
            f_tail = font(f_tail.size - 1)
        d.text((tx, 618), tail, MUTED, font=f_tail)
    if note:
        for j, ln in enumerate(wrap(d, note, font(14), RX - LX)[:2]):
            d.text((LX, H - 46 + j * 18), ln, MUTED, font=font(14))
    return im


def render(runs, out_path, *, fps=30, note=None, gate=None):
    """`gate` is the baseline the claim must survive, named but not drawn."""
    lanes = [Lane(r, above=(i == 0)) for i, r in enumerate(runs)]
    gate = Lane(gate, above=True) if gate else None
    slow_ms = max(a.ms for a in lanes)
    frames = pathlib.Path(out_path).parent / "_trip_frames"
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("*.jpg"):
        old.unlink()
    k = 0
    for i in range(int((LEAD + RACE + TAIL) * fps)):
        paint(lanes, i / fps, slow_ms, None, note, gate).save(
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
