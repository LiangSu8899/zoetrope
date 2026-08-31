"""Record when things happened. Nothing else.

This module does one job: attach to a host that is already running, and write
down the arrival time of each token, each denoise step, or each decision. It
does not accelerate anything, does not draw anything, and does not know what a
film is. What it produces is a run directory, and the compositors take it from
there.

Keeping it that narrow is what makes the films cheap to change: the drawing can
be reworked and every film redrawn without running a model again.

    from demokit import hook

    rec = hook.Recorder("stream", label="the host, as shipped", color="stock")
    with hook.on_tokens(rec, tok):
        model.generate(**inputs, streamer=rec.streamer, max_new_tokens=256)
    rec.write("runs/myfilm/eager")

See docs/PROTOCOL.md for what each `kind` needs in `meta`.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import statistics
import time
from typing import Any

__all__ = ["Recorder", "on_tokens", "on_denoiser", "on_calls"]

#: pane accents the compositors know
COLORS = ("stock", "compiled", "ours", "native")


def _now() -> float:
    return time.perf_counter()


def _sync() -> None:
    """Make the clock mean what it says on an async device."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:                                   # noqa: BLE001
        pass


class Recorder:
    """One arm of one film.

    ``kind`` picks the painter: ``stream`` (a language model), ``video`` (a
    diffusion pipeline), ``stream_batch`` (an engine under concurrency), or
    ``loop`` (a robot policy). ``label`` and ``sub`` are the pane header,
    ``color`` its accent.
    """

    def __init__(self, kind: str, *, label: str, sub: str = "",
                 color: str = "ours", **meta: Any):
        if color not in COLORS:
            raise ValueError(f"color must be one of {COLORS}, got {color!r}")
        self.kind = kind
        self.meta: dict[str, Any] = {
            "kind": kind, "label": label, "sub": sub, "color": color, **meta}
        self.events: list[dict[str, Any]] = []
        self._t0: float | None = None
        self._frames = None
        self._image = None

    # -- clock ---------------------------------------------------------
    def start(self) -> "Recorder":
        """Zero the arm's clock. Called for you on the first stamp."""
        _sync()
        self._t0 = _now()
        return self

    @property
    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else _now() - self._t0

    def stamp(self, **event: Any) -> float:
        """Record one event at the current time, and return that time."""
        if self._t0 is None:
            self.start()
        t = _now() - self._t0
        self.events.append({"t": round(t, 6), **event})
        return t

    def finish(self, **meta: Any) -> "Recorder":
        """Close the arm. `done_s` is the last event unless you pass one."""
        _sync()
        self.meta.update(meta)
        if "done_s" not in self.meta:
            self.meta["done_s"] = round(
                self.events[-1]["t"] if self.events else self.elapsed, 4)
        return self

    # -- what the painters read beyond the events ----------------------
    def frames(self, arr) -> "Recorder":
        """Pixels for a `video` or `loop` pane: uint8 [N, H, W, 3]."""
        self._frames = arr
        return self

    def image(self, img) -> "Recorder":
        """The still a VLM pane shows above its text."""
        self._image = img
        return self

    # -- derived meta, so a caller does not compute it by hand ----------
    def _derive(self) -> None:
        if self.kind == "stream" and self.events:
            n = len(self.events)
            self.meta.setdefault("n_tokens", n)
            self.meta.setdefault("ttft_ms", round(self.events[0]["t"] * 1e3, 1))
            span = self.events[-1]["t"] - self.events[0]["t"]
            if span > 0 and n > 1:
                self.meta.setdefault("decode_tok_s", round((n - 1) / span, 1))
        if self.kind == "video":
            steps = [e for e in self.events if e.get("kind") == "step"]
            self.meta.setdefault("steps", len(steps))
            if len(steps) > 1:
                gaps = [b["t"] - a["t"] for a, b in zip(steps, steps[1:])]
                self.meta.setdefault(
                    "ms_per_step", round(statistics.median(gaps) * 1e3, 1))
            self.meta.setdefault("clip_fps", 16.0)

    def write(self, out_dir: str | pathlib.Path) -> pathlib.Path:
        """Write the run directory the compositors read."""
        if "done_s" not in self.meta:
            self.finish()
        self._derive()
        d = pathlib.Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "events.json").write_text(json.dumps(
            {"meta": self.meta, "events": self.events}, indent=1))
        if self._frames is not None:
            import numpy as np
            np.save(d / "frames.npy",
                    np.asarray(self._frames, dtype="uint8"))
        if self._image is not None:
            self._image.save(d / "image.png")
        return d

    # -- transformers: a streamer that stamps instead of printing -------
    @property
    def streamer(self):
        """A `transformers` streamer that records arrival times."""
        rec = self

        class _Stamping:
            def put(self, value):
                # the first put is the prompt echo, not a generated token
                if not hasattr(self, "_seen_prompt"):
                    self._seen_prompt = True
                    rec.start()
                    return
                ids = value.reshape(-1).tolist() if hasattr(value, "reshape") \
                    else list(value)
                for i in ids:
                    _sync()
                    rec.stamp(i=len(rec.events), text=rec._decode(i))

            def end(self):
                rec.finish()

        return _Stamping()

    def _decode(self, token_id: int) -> str:
        tok = getattr(self, "_tokenizer", None)
        return tok.decode([token_id]) if tok is not None else ""


@contextlib.contextmanager
def on_tokens(rec: Recorder, tokenizer=None):
    """Arm `rec.streamer` with a tokenizer so events carry their text."""
    rec._tokenizer = tokenizer
    try:
        yield rec
    finally:
        rec.finish()


@contextlib.contextmanager
def on_denoiser(rec: Recorder, pipeline, attr: str | None = None):
    """Stamp every denoiser call on a diffusers pipeline.

    Wraps the module in place and restores it on exit, so the pipeline is
    left exactly as it was found.
    """
    names = ([attr] if attr else
             [a for a in dir(pipeline)
              if a.startswith(("transformer", "unet"))])
    patched = []
    for name in names:
        mod = getattr(pipeline, name, None)
        if mod is None or not hasattr(mod, "forward"):
            continue
        original = mod.forward

        def wrapped(*a, _o=original, **k):
            out = _o(*a, **k)
            _sync()
            rec.stamp(kind="step")
            return out

        mod.forward = wrapped
        patched.append((mod, original))
    if not patched:
        raise ValueError("no denoiser found on this pipeline; pass attr=")
    rec.start()
    try:
        yield rec
    finally:
        for mod, original in patched:
            mod.forward = original
        rec.finish()


@contextlib.contextmanager
def on_calls(rec: Recorder, module, event_kind: str = "step"):
    """Stamp every call of one module. The general form of `on_denoiser`."""
    original = module.forward
    rec.start()

    def wrapped(*a, **k):
        out = original(*a, **k)
        _sync()
        rec.stamp(kind=event_kind)
        return out

    module.forward = wrapped
    try:
        yield rec
    finally:
        module.forward = original
        rec.finish()
