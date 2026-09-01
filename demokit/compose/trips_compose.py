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

import numpy as np

from demokit.compose.race_compose import (BG, CARD, INK, LINE, MUTED, COLORS,
                                          STOCK, _ffmpeg, _kname, font, wrap)

W, H = 1280, 720
LX, RX = 96, W - 96
MEM_Y, MEM_H = 322, 56
LEAD, TAIL, RACE = 2.0, 3.6, 9.0        # seconds: title, end card, the race
PULSE = 0.62                            # how long one trip stays on screen
NAME = 1.1                              # how long a kernel name stays legible
SPRAY = 26                              # px of scatter, so trips stay countable
PER_TICK = 40                           # trips per mark in the filled bar

#: Both lanes span the full width, because both do the same job: one
#: decision. So length is the work, the sweep is the clock, and the density
#: of the marks is the trips it took -- which is the whole comparison, held
#: still, in one picture. A lane that stopped a fifth of the way across
#: would read as an arm that did less.

#: what the trip was for, in ink. The same palette on both sides, so the
#: colour of the storm is itself the comparison: the shipped host's is grey
#: -- operands being moved -- and the native one's is gold and green, which
#: is arithmetic and the quantizing that feeds it. No chart says this faster.
FAMILY_INK = {
    "gemm": (226, 178, 96),
    "attention": (156, 146, 224),
    "quantize": (52, 194, 154),
    "norm": (104, 178, 188),
    "elementwise": (150, 155, 145),
    "copy": (98, 104, 96),
    "layout": (140, 165, 200),
    "other": (118, 123, 116),
}
FAMILY_ORDER = ("gemm", "attention", "quantize", "norm", "elementwise",
                "copy", "layout", "other")


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

        # the recorded launches, in the order and with the spacing they
        # really had. A uniform stream would be a nicer animation and a
        # worse one: the bursts and the gaps are the shape of the run.
        legend = m.get("legend") or []
        fam = [FAMILY_INK.get(k.get("family", "other"), FAMILY_INK["other"])
               for k in legend]
        name = [_kname(k["name"]) for k in legend]
        ev = [e for e in blob["events"] if e.get("kind") == "launch"]
        span = float(m.get("done_s") or 1.0) or 1.0
        self.frac = np.array([e["t"] / span for e in ev]) if ev else \
            np.linspace(0, 1, max(self.trips, 1))
        idx = [int(e.get("k", 0)) for e in ev]
        self.ink = ([fam[i] for i in idx] if fam else
                    [FAMILY_INK["other"]] * len(self.frac))
        self.names = ([name[i] for i in idx] if name else
                      [""] * len(self.frac))
        self.mix = dict(m.get("families") or {})

    def span_s(self, slow_ms):
        """How long this lane's race lasts, on the film clock."""
        return RACE * self.ms / slow_ms

    def at(self, t, slow_ms):
        """Position in this lane's own run, 0 to 1, at film time t."""
        if t <= LEAD:
            return 0.0
        return min(1.0, (t - LEAD) / self.span_s(slow_ms))

    def shown(self, t, slow_ms):
        """Trips completed, on the film clock, at this arm's measured rate."""
        return int(self.trips * self.at(t, slow_ms))

    def done(self, t, slow_ms):
        return self.at(t, slow_ms) >= 1.0

    def in_flight(self, t, slow_ms, stride):
        """(index, age 0..1) for every trip still crossing the gap.

        `stride` is shared by both lanes, so what the eye compares between
        them -- how thick the traffic is for the same stretch of work -- is
        the measurement and not a drawing decision.
        """
        span = self.span_s(slow_ms)
        now = self.at(t, slow_ms)
        then = max(0.0, (t - LEAD - PULSE) / span)
        lo = int(np.searchsorted(self.frac, then, "left"))
        hi = int(np.searchsorted(self.frac, now, "right"))
        if hi <= lo:
            return []
        out = []
        for i in range(lo, hi, stride):
            age = (now - self.frac[i]) * span / PULSE
            if 0.0 <= age <= 1.0:
                out.append((i, age))
        return out

    def flashes(self, t, slow_ms, every):
        """The kernel names worth reading right now, newest last."""
        span = self.span_s(slow_ms)
        now = self.at(t, slow_ms)
        then = max(0.0, (t - LEAD - NAME) / span)
        lo = int(np.searchsorted(self.frac, then, "left"))
        hi = int(np.searchsorted(self.frac, now, "right"))
        out, seen = [], set()
        for i in range(hi - 1, lo - 1, -1):
            if i % every:
                continue
            if self.names[i] in seen:
                continue
            seen.add(self.names[i])
            out.append((i, (now - self.frac[i]) * span / NAME))
            if len(out) == 3:
                break
        return out


def paint(lanes, t, slow_ms, peak, note=None, gate=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    m = lanes[0].meta
    d.text((LX, 36), "one decision", INK, font=font(46, True))
    head = (f'{m.get("model", "")}   ·   {m.get("steps", "")} flow steps   ·   '
            f'the same policy, the same instruction, both sides')
    d.text((LX, 94), head, MUTED, font=font(18))

    d.rectangle([LX, MEM_Y, RX, MEM_Y + MEM_H], fill=CARD, outline=LINE)
    cap = "GPU memory"
    f_mem = font(22, True)
    d.text((RX - d.textlength(cap, font=f_mem) - 14, MEM_Y + 17), cap,
           INK, font=f_mem)

    # one stride for both lanes, so thickness is measured, not drawn
    stride = max(1, max(a.trips for a in lanes) // 3500)
    for lane in lanes:
        n = lane.shown(t, slow_ms)
        top = MEM_Y - 118 if lane.above else MEM_Y + MEM_H + 62
        bar = MEM_Y - 46 if lane.above else MEM_Y + MEM_H + 46
        edge = MEM_Y if lane.above else MEM_Y + MEM_H
        f_lab, f_num = font(24, True), font(60, True)

        # the trips in flight: down to memory for the operands, back up with
        # the result. A kernel does both, so one pulse makes the round trip.
        at = lane.at(t, slow_ms)
        wide = RX - LX
        for i, age in lane.in_flight(t, slow_ms, stride):
            x = min(RX, max(LX, LX + wide * i / lane.trips
                            + ((i * 2654435761) % 1000) / 1000 * SPRAY * 2
                            - SPRAY))
            leg = min(age * 2, 2 - age * 2)          # 0 at the ends, 1 mid
            y = bar + (edge - bar) * (age * 2 if age < 0.5 else 2 - age * 2)
            ink = _dim(lane.ink[i], 0.3 + 0.7 * leg)
            d.line([x, y, x, y + (11 if lane.above else -11)], fill=ink,
                   width=2)

        # the work done so far, hatched one mark per PER_TICK trips: the
        # two bars reach the same place and one of them is five times as
        # thick getting there
        h = 26
        y0 = bar - h if lane.above else bar
        w = wide * at
        d.rectangle([LX, y0, LX + max(w, 1), y0 + h], fill=(30, 38, 36),
                    outline=None)
        marks = int(lane.trips * at) // PER_TICK
        for k in range(marks):
            x = LX + wide * (k * PER_TICK) / lane.trips
            d.line([x, y0 + 2, x, y0 + h - 2], fill=lane.color, width=1)
        d.rectangle([LX, y0, LX + max(w, 1), y0 + h], outline=lane.color)

        # a name, now and then, so the storm is made of things with names
        every = max(1, lane.trips // 140)
        f_flash = font(15)
        head = LX + wide * at
        for j, (i, age) in enumerate(lane.flashes(t, slow_ms, every)):
            txt = lane.names[i]
            fy = (bar - h - 24 - j * 19) if lane.above else (bar + h + 8 + j * 19)
            fx = min(head + 14, RX - d.textlength(txt, font=f_flash))
            d.text((fx, fy), txt, _dim(lane.ink[i], 1.0 - 0.7 * age),
                   font=f_flash)

        ly = top - 74 if lane.above else top + 76
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

    _legend(d, lanes)

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


def _dim(ink, k):
    return tuple(int(BG[i] + (ink[i] - BG[i]) * max(0.0, min(1.0, k)))
                 for i in range(3))


def _legend(d, lanes):
    """What the colours are. Eight words, once, along the bottom."""
    seen = set()
    for lane in lanes:
        seen |= {f for f, c in lane.mix.items() if c}
    x, f = LX, font(13)
    for fam in FAMILY_ORDER:
        if fam not in seen:
            continue
        d.rectangle([x, MEM_Y + MEM_H // 2 - 4, x + 8,
                     MEM_Y + MEM_H // 2 + 4], fill=FAMILY_INK[fam])
        d.text((x + 13, MEM_Y + MEM_H // 2 - 8), fam, MUTED, font=f)
        x += 13 + d.textlength(fam, font=f) + 22


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
