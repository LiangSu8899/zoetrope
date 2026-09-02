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
    arch    the model's own module tree, lighting up in the order the
            forward hooks actually fired, with FlashRT's seats shown where
            they landed
    runtime every CUDA kernel the arm launched, in the order it launched
            them, bucketed by what it was for and whose code it was
    diagram the framework's own architecture, authored as a layout and lit
            by the run: a box turns on when its entry point is called

Both arms record the same shape: `events.json` holding a meta block and a
list of events carrying the wall time, in seconds from the start of the
arm's own run, at which each thing became true.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..frames import load_frames
from .canvas import Canvas, font, _truetype

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

#: `ours` is the name in the protocol; `accent` is the older spelling and
#: stays readable so published specs keep drawing.
COLORS = {"stock": STOCK, "compiled": COMPILED, "ours": ACCENT,
          "accent": ACCENT, "native": NATIVE}


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

    #: pixels this painter needs under its readout, beyond the common block
    readout_extra = 0

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
        self.frames = load_frames(self.dir)
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
        im.image_at(img, (x + (pane - cw) // 2, y + (pane - ch) // 2),
                    (cw, ch))
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
            im.image_at(self.image, (x + (pane - iw) // 2, y + 1), (iw, ih))
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



def _short(node: str) -> str:
    """`blocks.*.attn1` reads as `attn1`; the indent carries the rest."""
    parts = [p for p in node.split(".") if p != "*"]
    return parts[-1] if parts else node


class ArchArm(Arm):
    """The model's own module tree, drawn from a run rather than from source.

    Nodes come from `named_modules()`, their order from the order the forward
    hooks first fired, and — when the arm was recorded after an attach — each
    row carries what FlashRT put in that seat, or why it was left alone.

    This pane reports no performance figure. One forward pass is milliseconds,
    so the recorder stretched it for viewing and the pane says so.
    """

    readout_extra = 36

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.nodes = [n for n in self.meta["nodes"]]
        self.seats = self.meta.get("seats", {})
        self.stretch = float(self.meta.get("stretch", 1.0))
        self.root = self.nodes[0]["node"] if self.nodes else "<root>"
        self.spans = {n["node"]: [] for n in self.nodes}
        open_at = {}
        for e in self.events:
            key = (e["node"], e.get("idx", 0))
            if e["kind"] == "enter":
                open_at[key] = float(e["t"])
            elif key in open_at:
                self.spans.setdefault(e["node"], []).append(
                    (open_at.pop(key), float(e["t"])))
        self.rows = [n for n in self.nodes if n["node"] != self.root]

    # -- state of one node at time t -----------------------------------
    def _state(self, node, t):
        sp = self.spans.get(node, [])
        if not sp:
            return "pending", 0.0
        n_done = sum(1 for a, b in sp if b <= t)
        live = any(a <= t < b for a, b in sp)
        frac = n_done / len(sp)
        if live:
            return "active", frac
        return ("done", 1.0) if n_done == len(sp) else (
            ("pending", 0.0) if n_done == 0 else ("active", frac))

    def current(self, t):
        """The deepest node running right now, for the readout."""
        best = None
        for n in self.rows:
            if any(a <= t < b for a, b in self.spans.get(n["node"], [])):
                if best is None or n["depth"] >= best["depth"]:
                    best = n
        return best

    def _chip(self, node):
        """What FlashRT put in this seat, and what it turned down.

        Both, when both happened: a stack whose fused form was refused can
        still bind the pieces, and hiding either half of that would be the
        one thing this pane exists to prevent.
        """
        s = self.seats.get(node)
        if not s:
            return []
        out = []
        if s["bound"]:
            kinds = s["kinds"]
            txt = (kinds[0] if len(kinds) == 1 else
                   f"{kinds[0]} +{len(kinds) - 1}" if kinds else "bound")
            out.append((txt, NATIVE if s["fallbacks"] else self.color))
            if s["fallbacks"]:
                out.append((f"{s['fallbacks']} fell back", NATIVE))
        if s["refused"]:
            out.append((f"\u00d7{s['refused']} refused", MUTED))
        return out

    def reason(self, node):
        """Why a node was refused, in the ledger's own words."""
        s = self.seats.get(node) or {}
        why = [w for w in s.get("reasons", {})]
        return why[0] if why else None

    def paint_pane(self, im, d, x, y, pane, t):
        d.rectangle([x, y, x + pane - 1, y + pane - 1], fill=CARD,
                    outline=LINE)
        head = f'{self.meta.get("model_class", "")}   ' \
               f'{self.meta.get("n_modules", len(self.nodes))} modules'
        d.text((x + 12, y + 9), head, MUTED, font=font(12))
        top, bottom = y + 30, y + pane - 8
        rows = self.rows
        if not rows:
            return
        rh = max(13, min(30, (bottom - top) // len(rows)))
        fits = max(1, (bottom - top) // rh)
        # a tree taller than the pane scrolls to keep the running node in
        # view, the way a debugger follows a stack
        first = 0
        if len(rows) > fits:
            live = next((i for i, n in enumerate(rows)
                         if self._state(n["node"], t)[0] == "active"), None)
            if live is None:
                live = sum(1 for n in rows
                           if self._state(n["node"], t)[0] == "done") - 1
            first = min(max(0, live - fits // 2), len(rows) - fits)
        fs = max(9, min(14, rh - 6))
        f_row, f_chip = font(fs), font(max(8, fs - 2))
        for i, n in enumerate(rows[first:first + fits]):
            ry = top + i * rh
            ind = 10 + (n["depth"] - 1) * 12
            x0, x1 = x + ind, x + pane - 10
            state, frac = self._state(n["node"], t)
            if state == "pending":
                d.rectangle([x0, ry, x1, ry + rh - 3], outline=LINE)
                ink, bar = MUTED, LINE
            elif state == "active":
                d.rectangle([x0, ry, x1, ry + rh - 3], fill=LINE,
                            outline=self.color)
                ink, bar = INK, self.color
            else:
                d.rectangle([x0, ry, x1, ry + rh - 3], outline=self.color)
                ink, bar = INK, self.color
            d.rectangle([x0, ry, x0 + 2, ry + rh - 3], fill=bar)
            name = _short(n["node"])
            if n["repeat"] > 1:
                name += f'  \u00d7{n["repeat"]}'
            d.text((x0 + 8, ry + (rh - 3 - fs) // 2 - 1), name, ink,
                   font=f_row)
            cx = x1 - 8
            for txt, cc in reversed(self._chip(n["node"])):
                cx -= d.textlength(txt, font=f_chip)
                d.text((cx, ry + (rh - 3 - f_chip.size) // 2), txt,
                       cc if state != "pending" else MUTED, font=f_chip)
                cx -= 10
            if n["repeat"] > 1 and state == "active":
                d.rectangle([x0, ry + rh - 5,
                             x0 + int((x1 - x0) * frac), ry + rh - 4],
                            fill=self.color)
        if len(rows) > fits:
            d.text((x + pane - 40, y + 9), f"{first + fits}/{len(rows)}",
                   MUTED, font=font(11))

    def paint_readout(self, im, d, x, y, pane, t, fonts):
        f_hz, f_unit, f_sub, f_small = fonts
        if self.seats:
            val, unit = str(self.meta.get("bound", 0)), "seats bound"
            sub = f'{self.meta.get("refused", 0)} refused'
        else:
            val, unit = str(self.meta.get("n_modules", 0)), "modules"
            sub = f'depth {self.meta.get("depth", 2)} · ' \
                  f'{self.meta.get("n_groups", len(self.nodes))} groups'
        d.text((x, y), val, self.color, font=f_hz)
        w = d.textlength(val, font=f_hz)
        d.text((x + w + 8, y + 30), unit, MUTED, font=f_unit)
        d.text((x + w + 8, y + 6), sub, INK, font=f_sub)
        now = self.current(t)
        d.text((x, y + 62),
               (f'{now["node"]}   {now["cls"]}' if now else
                ("pass complete" if self.finished(t) else "entering")),
               INK if now else MUTED, font=f_small)
        # deliberately without the factor: a stretch is a duration, and a
        # duration on this pane would be a performance figure
        d.text((x, y + 80), "one forward pass, slowed for viewing", MUTED,
               font=f_small)
        fb = self.meta.get("fallbacks", 0)
        why = self.reason(now["node"]) if now else None
        if fb:
            d.text((x, y + 98), f"{fb} silent fall-back(s) in the ledger",
                   NATIVE, font=f_small)
        elif why:
            for j, ln in enumerate(wrap(d, why, f_small, pane)[:2]):
                d.text((x, y + 98 + j * 16), ln, MUTED, font=f_small)



#: fixed row order, so two panes line up and the eye can compare a row
#: across them rather than hunting for it
FAMILY_ROWS = ("gemm", "attention", "quantize", "norm", "elementwise",
               "copy", "layout", "other")

#: whose kernel it is, in ink. The green band is FlashRT's share of the GPU.
ORIGIN_INK = {"FlashRT": ACCENT, "inductor": COMPILED, "FA2": NATIVE,
              "cuBLAS/CUTLASS": (108, 116, 110), "PyTorch": STOCK,
              "cuDNN": (86, 94, 88), "memory op": LINE, "other": MUTED}
ORIGIN_ORDER = ("FlashRT", "inductor", "FA2", "cuBLAS/CUTLASS", "PyTorch",
                "cuDNN", "memory op", "other")


class RuntimeArm(Arm):
    """What this arm asked the GPU to do, one kernel launch at a time.

    The demo film says one arm finished sooner. This says what each spent
    the GPU on to get there: how many kernels ran, what they were for, and
    whose code they were.

    Read the rows across panes, not down one. And read them as composition,
    not as a score: an arm can launch more kernels than its baseline and
    still finish first, because it made each one cheaper. That is what the
    NVFP4 row is.
    """

    readout_extra = 36

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.legend = self.meta["legend"]
        self.launches = int(self.meta["launches"])
        self.distinct = int(self.meta["distinct"])
        #: filled in by load_chapter so every pane shares one bar scale
        self.scale = self.launches
        t = np.array([float(e["t"]) for e in self.events])
        k = np.array([int(e["k"]) for e in self.events])
        fam = np.array([e["family"] for e in self.legend])
        org = np.array([e["origin"] for e in self.legend])
        self._fam_t = {f: t[fam[k] == f] for f in FAMILY_ROWS}
        self._org_t = {o: t[org[k] == o] for o in ORIGIN_ORDER}
        self._t, self._k = t, k

    def _done(self, times, t):
        return int(np.searchsorted(times, t, side="right"))

    def counts(self, t):
        return {f: self._done(v, t) for f, v in self._fam_t.items()}

    def total(self, t):
        return self._done(self._t, t)

    def top_kernels(self, t, k=5):
        """The busiest kernels so far, as (symbol, count, origin)."""
        n = self.total(t)
        if not n:
            return []
        idx, counts = np.unique(self._k[:n], return_counts=True)
        rank = np.argsort(-counts)[:k]
        return [(self.legend[int(idx[i])]["name"], int(counts[i]),
                 self.legend[int(idx[i])]["origin"]) for i in rank]

    def newest(self, t):
        """The kernel that launched most recently, for the readout."""
        n = self.total(t)
        return self.legend[int(self._k[n - 1])] if n else None

    def paint_pane(self, im, d, x, y, pane, t):
        d.rectangle([x, y, x + pane - 1, y + pane - 1], fill=CARD,
                    outline=LINE)
        f_head, f_row, f_num = font(12), font(13), font(12, True)
        counts = self.counts(t)
        d.text((x + 12, y + 10), "kernel launches, by what they are for",
               MUTED, font=f_head)

        top, rh = y + 34, 30
        left = x + 12
        label_w = 86
        bar_x0 = left + label_w
        bar_w = pane - label_w - 74
        for i, fam in enumerate(FAMILY_ROWS):
            ry = top + i * rh
            n = counts[fam]
            d.text((left, ry + 4), fam, INK if n else MUTED, font=f_row)
            d.rectangle([bar_x0, ry + 3, bar_x0 + bar_w, ry + 17], fill=BG)
            if n:
                w = max(2, int(bar_w * n / max(self.scale, 1)))
                d.rectangle([bar_x0, ry + 3, bar_x0 + w, ry + 17],
                            fill=self.color)
            txt = str(n) if n else "-"
            d.text((bar_x0 + bar_w + 10, ry + 4), txt,
                   INK if n else MUTED, font=f_num)

        # whose kernel: one stacked band, and the legend under it
        band_y = top + len(FAMILY_ROWS) * rh + 14
        d.text((left, band_y), "whose kernel", MUTED, font=f_head)
        band_y += 18
        total = max(self.total(t), 1)
        cx, width = left, pane - 24
        for org in ORIGIN_ORDER:
            n = self._done(self._org_t[org], t)
            if not n:
                continue
            w = int(width * n / total)
            d.rectangle([cx, band_y, cx + w, band_y + 13],
                        fill=ORIGIN_INK.get(org, MUTED))
            cx += w
        d.rectangle([left, band_y, left + width, band_y + 13], outline=LINE)

        ly, lx, f_leg = band_y + 20, left, font(11)
        for org in ORIGIN_ORDER:
            n = self._done(self._org_t[org], t)
            if not n:
                continue
            chip = f"{org} {n}"
            w = d.textlength(chip, font=f_leg)
            if lx + w + 18 > left + width:
                lx, ly = left, ly + 15
            d.rectangle([lx, ly + 3, lx + 7, ly + 10],
                        fill=ORIGIN_INK.get(org, MUTED))
            d.text((lx + 11, ly), chip, MUTED, font=f_leg)
            lx += w + 24

        # and the kernels themselves, because a family is an argument and a
        # symbol is a fact the reader can go and look up
        ky = ly + 28
        d.text((left, ky), "the kernels doing the most of it", MUTED,
               font=f_head)
        ky += 18
        live = self.top_kernels(t, 5)
        for name, n, org in live:
            if ky + 15 > y + pane - 8:
                break
            d.rectangle([left, ky + 4, left + 7, ky + 11],
                        fill=ORIGIN_INK.get(org, MUTED))
            d.text((left + 11, ky), _kname(name), INK, font=f_leg)
            num = str(n)
            d.text((left + width - d.textlength(num, font=f_leg), ky),
                   num, MUTED, font=f_leg)
            ky += 15

    def paint_readout(self, im, d, x, y, pane, t, fonts):
        f_hz, f_unit, f_sub, f_small = fonts
        n = self.total(t)
        val = str(n)
        d.text((x, y), val, self.color, font=f_hz)
        w = d.textlength(val, font=f_hz)
        d.text((x + w + 8, y + 30), "kernel launches", MUTED, font=f_unit)
        d.text((x + w + 8, y + 6), f"{self.distinct} distinct kernels", INK,
               font=f_sub)
        prec = self.meta.get("precisions") or {}
        line = "  ".join(f"{k} x{v}" for k, v in list(prec.items())[:4])
        d.text((x, y + 62), line or "arithmetic not named in the symbols",
               MUTED, font=f_small)
        now = self.newest(t)
        if now:
            line = f'{now["origin"]}  {_kname(now["name"])}'
            while (d.textlength(line, font=f_small) > pane
                   and len(line) > 12):
                line = line[:-2]
            d.text((x, y + 80), line, INK, font=f_small)
        note = self.meta.get("runtime_note")
        if note:
            for j, ln in enumerate(wrap(d, note, f_small, pane)[:2]):
                d.text((x, y + 98 + j * 16), ln, MUTED, font=f_small)


#: identifiers that appear in every symbol and distinguish nothing
_NOISE = {
    "at", "native", "std", "c10", "cuda", "detail", "anonymous", "namespace",
    "gpu_kernel_impl", "gpu_kernel_impl_nocast", "array", "tuple", "pair",
    "TensorIterator", "TensorIteratorBase", "func_wrapper_t", "OpaqueType",
    "BFloat16", "Half", "float", "double", "int", "unsigned", "char", "bool",
    "true", "false", "void", "long", "short", "signed", "const", "Array",
    "operator", "type", "value_type", "__half", "__nv_bfloat16",
}
#: wrappers whose own name says nothing; the template argument is the kernel
_WRAPPERS = ("Kernel2", "device_kernel", "kernel_impl", "Kernel")


def _mangled_parts(sym: str) -> list[str]:
    """Every length-prefixed identifier in an Itanium-mangled symbol.

    Done by counting rather than by regex: an identifier may contain digits,
    so a pattern that reads "digits then word characters" swallows the whole
    symbol on the first match.
    """
    parts, i = [], 0
    while i < len(sym):
        if sym[i].isdigit():
            j = i
            while j < len(sym) and sym[j].isdigit():
                j += 1
            n = int(sym[i:j])
            if 0 < n <= len(sym) - j:
                parts.append(sym[j:j + n])
                i = j + n
                continue
        i += 1
    return parts


def _demangle_lite(sym: str) -> str | None:
    """`_ZN7cutlass13device_kernelI...` -> `cutlass::device_kernel`.

    Enough of the Itanium mangling to read a nested name. Anything harder
    than that is left alone rather than guessed at.
    """
    if not sym.startswith("_ZN"):
        return None
    parts, i = [], 3
    while i < len(sym) and sym[i].isdigit():
        j = i
        while j < len(sym) and sym[j].isdigit():
            j += 1
        n = int(sym[i:j])
        parts.append(sym[j:j + n])
        i = j + n
    return "::".join(parts) or None


def _kname(sym: str) -> str:
    """A CUDA symbol, shortened to the part a reader can act on.

    Two instantiations of `elementwise_kernel` are two different kernels,
    and printing both as `elementwise_kernel` would read as a bug. So the
    name carries the first template argument that actually distinguishes
    it, which is usually the functor that says what the kernel computes.
    """
    head = sym.replace("(anonymous namespace)::", "").split("(")[0]
    if head.startswith("void "):
        head = head[5:]
    nested = _demangle_lite(head)
    base = (nested or head).split("<")[0].strip().split("::")[-1]
    if "<" in sym:
        inner = next((w for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*",
                                            sym.split("<", 1)[1])
                      if w not in _NOISE and w != base), "")
    else:
        # a symbol that never demangled: the length-prefixed pieces are all
        # we have, and the longest of them is the one that names the kernel
        # Substitution codes (S_, S1_) break the length counting, so the
        # tail of a deeply mangled symbol yields debris. Keep only pieces
        # that read like a name a person wrote.
        pieces = [m for m in _mangled_parts(sym)
                  if m not in _NOISE and m != base and len(m) >= 4
                  and "EEE" not in m
                  and sum(c.isupper() for c in m) / len(m) < 0.5]
        inner = max(pieces, key=len) if pieces else ""
    # the origin chip beside the name already says whose kernel it is, so
    # the namespace is redundant and the template argument is not
    if base in _WRAPPERS and inner:
        return inner[:56]
    return (f"{base}  {inner}" if inner else base)[:56]



class DiagramArm(Arm):
    """The framework's own architecture, lit by a run.

    The layout is authored, because a picture of a system is a human
    judgement and pretending otherwise makes a worse picture. What lights up
    is not: a box turns on when the entry point it stands for is actually
    called, in the order it is called.

    Boxes that stay dark are half the message. An eager arm lights the host
    and nothing under it, and that is the honest picture of an eager arm.
    """

    readout_extra = 36

    def __init__(self, run_dir):
        super().__init__(run_dir)
        self.diagram = self.meta["diagram"]
        self.nodes = self.diagram["nodes"]
        self.by = {n["id"]: n for n in self.nodes}
        self.groups = {n["id"] for n in self.nodes
                       if any(m.get("group") == n["id"] for m in self.nodes)}
        self.lit = {e["node"]: e for e in self.events if e["kind"] == "lit"}
        self.order = [e for e in self.events if e["kind"] == "lit"]
        self.canvas = self.diagram["canvas"]

    def on(self, t):
        return {n for n, e in self.lit.items() if e["t"] <= t}

    # -- geometry ------------------------------------------------------
    def _fit(self, x, y, pane):
        cw, ch = self.canvas
        s = (pane - 20) / cw
        return s, x + 10, y + 8

    def _px(self, box, s, ox, oy):
        bx, by, bw, bh = box
        return [ox + bx * s, oy + by * s, ox + (bx + bw) * s,
                oy + (by + bh) * s]

    def paint_pane(self, im, d, x, y, pane, t):
        d.rectangle([x, y, x + pane - 1, y + pane - 1], fill=BG, outline=LINE)
        s, ox, oy = self._fit(x, y, pane)
        live = self.on(t)
        newest = self._newest(t)

        for a, b, lab in self.diagram["edges"]:
            ax0, ay0, ax1, ay1 = self._px(self.by[a]["box"], s, ox, oy)
            bx0, by0, bx1, by1 = self._px(self.by[b]["box"], s, ox, oy)
            hot = a in live and b in live
            ink = self.color if hot else LINE
            if by0 >= ay1 - 1:
                # the centre of the overlap, so an edge into a full-width
                # layer leaves from inside the box it starts in
                cx = (max(ax0, bx0) + min(ax1, bx1)) / 2
                _arrow(d, cx, ay1, cx, by0, ink)
                if lab:
                    d.text((cx + 7, (ay1 + by0) / 2 - 7), lab,
                           ink if hot else MUTED, font=font(10))
            else:
                cy = (ay0 + ay1) / 2
                _arrow(d, ax1, cy, bx0, cy, ink)

        for n in self.nodes:
            x0, y0, x1, y1 = self._px(n["box"], s, ox, oy)
            hot = n["id"] in live
            fresh = newest is not None and n["id"] == newest["node"]
            edge = INK if fresh else (self.color if hot else LINE)
            if n["id"] in self.groups or n.get("outside"):
                d.rectangle([x0, y0, x1, y1], fill=(20, 28, 26), outline=edge)
                f = font(max(11, int(15 * s / 0.8)), True)
            else:
                d.rectangle([x0, y0, x1, y1],
                            fill=_warm(self.color) if hot else CARD,
                            outline=edge)
                d.rectangle([x0, y0, x0 + 3, y1],
                            fill=self.color if hot else LINE)
                f = font(max(10, int(13 * s / 0.8)), True)
            ink = INK if hot else MUTED
            d.text((x0 + 10, y0 + 7), n["label"], ink, font=f)
            fs = font(max(8, int(10 * s / 0.8)))
            if n.get("sub"):
                d.text((x0 + 10, y0 + 9 + f.size), n["sub"], MUTED, font=fs)
            calls = self.lit.get(n["id"], {}).get("calls", 0)
            if hot and calls and n["id"] not in self.groups:
                txt = f"\u00d7{calls}"
                d.text((x1 - 8 - d.textlength(txt, font=fs),
                        y1 - 6 - fs.size), txt, self.color, font=fs)

        self._ledger(d, x, y, pane, oy + self.canvas[1] * s + 12, t)

    def _ledger(self, d, x, y, pane, top, t):
        """The boxes in the order they lit, which is the order they ran."""
        seen = [e for e in self.order if e["t"] <= t]
        f_h, f_l = font(11), font(12)
        d.text((x + 12, top), "in the order they were called", MUTED,
               font=f_h)
        top += 16
        rows = max(1, int((y + pane - 8 - top) // 15))
        seen = [e for e in seen if e["node"] not in self.groups]
        for i, e in enumerate(seen[-rows:]):
            n = self.by[e["node"]]
            ry = top + i * 15
            d.rectangle([x + 12, ry + 4, x + 18, ry + 10], fill=self.color)
            d.text((x + 24, ry), n["label"], INK, font=f_l)
            why = self.lit[n["id"]].get("why") or (
                f'\u00d7{e["calls"]}' if e["calls"] else "")
            if why:
                d.text((x + pane - 12 - d.textlength(why, font=f_l), ry),
                       why, MUTED, font=f_l)

    def _newest(self, t):
        seen = [e for e in self.order if e["t"] <= t]
        return seen[-1] if seen else None

    def paint_readout(self, im, d, x, y, pane, t, fonts):
        f_hz, f_unit, f_sub, f_small = fonts
        live = self.on(t)
        val = str(len(live))
        d.text((x, y), val, self.color, font=f_hz)
        w = d.textlength(val, font=f_hz)
        d.text((x + w + 8, y + 30), "boxes lit", MUTED, font=f_unit)
        d.text((x + w + 8, y + 6), f'of {self.meta["n_nodes"]} in the diagram',
               INK, font=f_sub)
        newest = self._newest(t)
        if newest:
            n = self.by[newest["node"]]
            calls = newest["calls"]
            d.text((x, y + 62),
                   f'{n["label"]}' + (f'   x{calls}' if calls else ""),
                   INK, font=f_small)
        prov = self.meta.get("providers", {})
        hub, ext = prov.get("hub") or [], prov.get("extension") or []
        line = (f"{len(hub)} kernel package(s) loaded" if hub else
                (f"{len(ext)} native extension(s) loaded" if ext else
                 "no FlashRT kernel provider in this process"))
        d.text((x, y + 80), line, MUTED, font=f_small)
        note = self.meta.get("diagram_note")
        if note:
            for j, ln in enumerate(wrap(d, note, f_small, pane)[:2]):
                d.text((x, y + 98 + j * 16), ln, MUTED, font=f_small)


def _warm(color, k=0.22):
    """The pane accent, dimmed to a fill a label still reads on."""
    return tuple(int(BG[i] + (color[i] - BG[i]) * k) for i in range(3))


def _arrow(d, x0, y0, x1, y1, ink, head=5):
    d.line([x0, y0, x1, y1], fill=ink, width=2)
    if y1 > y0:
        d.polygon([(x1, y1), (x1 - head, y1 - head), (x1 + head, y1 - head)],
                  fill=ink)
    elif x1 > x0:
        d.polygon([(x1, y1), (x1 - head, y1 - head), (x1 - head, y1 + head)],
                  fill=ink)


KINDS = {"video": VideoArm, "stream": StreamArm,
         "stream_batch": StreamBatchArm, "arch": ArchArm,
         "runtime": RuntimeArm, "diagram": DiagramArm}


#: One canvas width for the whole kit; the panes divide it.
CANVAS_W = 1280


def pane_width(n_panes, gap=26, pad=34):
    return (CANVAS_W - pad * 2 - gap * (n_panes - 1)) // n_panes


def _layout(n_panes, pane, footer, extra=0):
    """Canvas size for one chapter, and its wrapped footer lines.

    `extra` is room a painter asked for under its readout — an `arch` pane
    prints the ledger's refusal reason there.
    """
    gap, pad, head = 26, 34, 168
    # one canvas width for the whole kit, so a page of these films lines up
    W = CANVAS_W
    pane = pane or pane_width(n_panes)
    probe = Canvas(BG, k=1)
    lines = []
    for para in (footer or "").split("\n"):
        if para:
            lines.extend(wrap(probe, para, font(13), W - pad * 2))
    H = head + pane + 145 + extra + 17 * len(lines) + 18
    return W, H, pane, lines


def paint(chapter, t, W, H, pane, foot_lines, extra=0, speed=1.0):
    """One frame of one chapter, on the canvas the whole film shares."""
    gap, pad, head = 26, 34, 168
    arms = chapter["arms"]
    f_lab, f_sub, f_small = font(19, True), font(14), font(13)
    fonts = (font(52, True), font(15, True), f_sub, f_small)
    title, note = chapter.get("title"), chapter.get("note")

    probe = Canvas(BG, k=1)
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

    cv = Canvas(BG, size=(W, H))
    im = d = cv
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
        bar_y = head + pane + 132 + extra
        width = int(pane * min(t / a.done_t, 1.0))
        d.rectangle([x, bar_y, x + pane, bar_y + 5], fill=LINE)
        d.rectangle([x, bar_y, x + width, bar_y + 5], fill=a.color)

    d.text((W - pad - 96, 30), f"{t:5.1f} s", INK, font=f_lab)
    # the clock is model time; if the film is not running at model rate the
    # page has to say so, or the clock is a number nobody can trust
    if abs(speed - 1.0) > 1e-3:
        note = f"played at {speed:g}x"
        d.text((W - pad - 96 - 14 - probe.textlength(note, font=f_small), 36),
               note, MUTED, font=f_small)
    for j, ln in enumerate(foot_lines):
        d.text((pad, H - 14 - 17 * (len(foot_lines) - j)), ln, MUTED,
               font=f_small)
    return im.image()


def render(chapters, out_path, fps=30, pane=400, speed=1.0):
    """Render every chapter onto one canvas and encode them as one film."""
    # every chapter of a film shares one pane size — the one the busiest
    # chapter can afford — or a two-arm chapter would tower over a three-arm
    # one and leave the shorter chapters standing in a hole
    shared = pane_width(max(len(c["arms"]) for c in chapters))
    sized = []
    for c in chapters:
        room = max((a.readout_extra for a in c["arms"]), default=0)
        W, H, wide, lines = _layout(len(c["arms"]), shared, c.get("footer"),
                                    room)
        sized.append((W, H, wide, lines, room))
    W = max(w for w, _, _, _, _ in sized)
    H = max(h for _, h, _, _, _ in sized)

    frames_dir = pathlib.Path(out_path).parent / "_race_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    k = 0
    for c, (_, _, wide, lines, room) in zip(chapters, sized):
        span = (c["seconds"] if c.get("seconds") is not None
                else max(a.done_t for a in c["arms"]) + c.get("tail", 2.0))
        for i in range(int(span / speed * fps)):
            paint(c, i / fps * speed, W, H, wide, lines, room,
                  speed=speed).save(
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
    # runtime panes are only comparable on one bar scale: eager launching
    # 1900 kernels and a compiled arm launching 700 must look different
    peak = max([a.launches for a in arms if isinstance(a, RuntimeArm)],
               default=0)
    for a in arms:
        if isinstance(a, RuntimeArm):
            a.scale = peak
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
