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
import functools
import json
import pathlib
import sys
import statistics
import time
from typing import Any

__all__ = ["Recorder", "on_tokens", "on_denoiser", "on_calls",
           "on_tree", "seats", "on_kernels", "on_components",
           "load_diagram", "light", "kernel_origin", "kernel_family",
           "kernel_precision"]

#: pane accents the compositors know
COLORS = ("stock", "compiled", "ours", "native", "accent")


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


# ---------------------------------------------------------------------
# the module tree, as it actually ran
# ---------------------------------------------------------------------

def _fold_key(name: str) -> str:
    """`blocks.7.attn1` and `blocks.8.attn1` are one node drawn once.

    Every index segment collapses to `*`, so a repeated stack folds at any
    depth. Only indices fold: `encoder.layer_norm` keeps its own row.
    """
    return ".".join("*" if part.isdigit() else part
                    for part in name.split("."))


def _overlap(outer, inner) -> float:
    """Seconds of `inner` that fall inside `outer`. Both sorted by start."""
    total, j = 0.0, 0
    for a0, a1 in outer:
        while j < len(inner) and inner[j][1] <= a0:
            j += 1
        k = j
        while k < len(inner) and inner[k][0] < a1:
            total += max(0.0, min(a1, inner[k][1]) - max(a0, inner[k][0]))
            k += 1
    return total


class _Clock:
    """Marks on the device timeline, read once at the end.

    A forward hook that calls `cuda.synchronize()` measures the model plus
    the stall it just caused. CUDA events are recorded on the stream and
    only read after the run, so the hot path pays for the record alone.
    On CPU this is `perf_counter` and the distinction does not arise.
    """

    def __init__(self):
        self.cuda = False
        try:
            import torch
            self.cuda = torch.cuda.is_available()
            self._torch = torch
        except Exception:                               # noqa: BLE001
            pass

    def mark(self):
        if self.cuda:
            e = self._torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return _now()

    def settle(self) -> None:
        if self.cuda:
            self._torch.cuda.synchronize()

    def since(self, ref, mark) -> float:
        """Seconds from `ref` to `mark`."""
        if self.cuda:
            return ref.elapsed_time(mark) / 1e3
        return mark - ref


@contextlib.contextmanager
def on_tree(rec: Recorder, model, *, depth: int = 2, stretch: float | None = None,
            target_s: float = 8.0, root_name: str = "<root>"):
    """Record the module tree and one real pass through it.

    Nodes come from `named_modules()`; edges come from the order the forward
    hooks first fire, which is the order the model actually ran. A module
    that never fires did not run in this pass — that is observed, not
    guessed.

        rec = hook.Recorder("arch", label="Wan2.2, as shipped", color="stock")
        with hook.on_tree(rec, pipe.transformer):
            pipe(prompt=..., num_inference_steps=2)
        rec.write("runs/arch/stock")

    One forward pass is milliseconds, so the events are stretched to
    `target_s` seconds for viewing and the pane says by how much. Pass
    `stretch=` to fix the factor instead.

    Two things this gets right that a naive version does not:

    * **Containers never fire.** `ModuleList` and `Sequential` have no
      `forward`, so a parent link walks up to the nearest *recorded*
      ancestor. Otherwise every block's time lands in the root.
    * **The pass is the median one, not the first.** The first pass carries
      warm-up and would pace the animation wrongly.
    """
    clock = _Clock()
    spans: list[list] = []           # [name, pass_id, start_mark, end_mark]
    order: list[str] = []
    seen: set[str] = set()
    info: dict[str, dict] = {}
    passes = [0]
    handles = []

    def pre(name):
        def f(mod, args):
            spans.append([name, passes[0], clock.mark(), None])
            if name not in seen:
                seen.add(name)
                order.append(name)
        return f

    def post(name):
        def f(mod, args, out):
            mark = clock.mark()
            for s in reversed(spans):
                if s[0] == name and s[3] is None:
                    s[3] = mark
                    break
            if name == root_name:
                passes[0] += 1
        return f

    for name, mod in model.named_modules():
        d = 0 if name == "" else name.count(".") + 1
        if d > depth:
            continue
        key = name or root_name
        info[key] = {"cls": type(mod).__name__, "depth": d}
        handles.append(mod.register_forward_pre_hook(pre(key)))
        handles.append(mod.register_forward_hook(post(key)))

    ref = clock.mark()
    rec.start()
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()
        clock.settle()
        _resolve_tree(rec, clock, ref, spans, order, info, depth,
                      root_name, stretch, target_s,
                      type(model).__name__)


def _resolve_tree(rec, clock, ref, spans, order, info, depth, root_name,
                  stretch, target_s, model_class) -> None:
    """Turn recorded marks into a folded tree and one pass of events."""
    done = [s for s in spans if s[3] is not None]
    if not done:
        raise RuntimeError("no module fired: is this the module that runs?")

    # A container (`ModuleList`, `Sequential`) has no forward, so it never
    # fires and is not in `order`. Parent links walk up to the nearest
    # ancestor that did fire, or every child's time lands in the root.
    fired = set(order)
    for name in order:
        head, parent = name, None
        while "." in head:
            head = head.rsplit(".", 1)[0]
            if head in fired:
                parent = head
                break
        info[name]["parent"] = (
            None if name == root_name else (parent or root_name))

    # every span, on one timeline in seconds from the reference mark
    timed = [(n, p, clock.since(ref, a), clock.since(ref, b))
             for n, p, a, b in done]

    # the representative pass: the median root call, never the first
    roots = sorted((t1 - t0, p) for n, p, t0, t1 in timed if n == root_name)
    if roots:
        pick = roots[len(roots) // 2][1]
    else:                                     # no root span: take pass 0
        pick = min(p for _, p, _, _ in timed)
    pass_spans = [s for s in timed if s[1] == pick]
    base = min(t0 for _, _, t0, _ in pass_spans)
    span_s = max(t1 for _, _, _, t1 in pass_spans) - base
    if stretch is None:
        stretch = round(target_s / span_s, 2) if span_s > 1e-9 else 1.0

    # fold identical siblings, but only when they really are identical
    families: dict[str, list[str]] = {}
    for name in order:
        families.setdefault(_fold_key(name), []).append(name)
    fold: dict[str, str] = {}
    for key, members in families.items():
        classes = {info[m]["cls"] for m in members}
        if "*" in key.split(".") and len(classes) == 1 and len(members) > 1:
            for m in members:
                fold[m] = key
        else:
            for m in members:
                fold[m] = m

    calls: dict[str, int] = {}
    incl: dict[str, float] = {}
    windows: dict[str, list[tuple[float, float]]] = {}
    for n, _, t0, t1 in timed:
        k = fold[n]
        calls[k] = calls.get(k, 0) + 1
        incl[k] = incl.get(k, 0.0) + (t1 - t0)
        windows.setdefault(k, []).append((t0, t1))
    for w in windows.values():
        w.sort()

    nodes, place = [], {}
    for name in order:
        k = fold[name]
        if k in place:
            continue
        members = families[_fold_key(name)] if "*" in k.split(".") else [name]
        place[k] = len(nodes)
        parent = info[name]["parent"]
        nodes.append({
            "node": k,
            "cls": info[name]["cls"],
            "depth": info[name]["depth"],
            "parent": fold.get(parent, parent),
            "repeat": len(members),
            "calls": calls.get(k, 0),
        })
    for nd in nodes:
        # A child's time counts against its parent only where the two
        # really overlap. An engine that calls part of the tree from its
        # own entry point -- vLLM computes logits outside `forward` -- has
        # children that ran while the parent was not on the stack, and
        # subtracting those wholesale drives self time negative.
        own = windows.get(nd["node"], [])
        kids = sum(_overlap(own, windows.get(c["node"], []))
                   for c in nodes if c["parent"] == nd["node"])
        nd["incl_ms"] = round(incl.get(nd["node"], 0.0) * 1e3, 3)
        nd["self_ms"] = round(nd["incl_ms"] - kids * 1e3, 3)

    index = {}
    for key, members in families.items():
        for i, m in enumerate(members):
            index[m] = i
    for n, _, t0, t1 in sorted(pass_spans, key=lambda s: s[2]):
        k = fold[n]
        tag = {"idx": index[n]} if "*" in k.split(".") else {}
        rec.events.append({"t": round((t0 - base) * stretch, 6),
                           "kind": "enter", "node": k, **tag})
        rec.events.append({"t": round((t1 - base) * stretch, 6),
                           "kind": "exit", "node": k, **tag})
    rec.events.sort(key=lambda e: e["t"])

    rec.meta.update({
        "model_class": model_class,
        "depth": depth,
        "nodes": nodes,
        "n_modules": len(order),
        "n_groups": len(nodes),
        "passes": max(p for _, p, _, _ in timed) + 1,
        "pass_ms": round(span_s * 1e3, 3),
        "stretch": stretch,
        "_timing_note": "hook-perturbed, and it paces the animation. "
                        "Never quote it as a performance figure.",
    })
    rec.meta["done_s"] = round(span_s * stretch, 4)


def seats(rec: Recorder, report: dict, refused=None) -> Recorder:
    """Join a structures receipt onto the tree, by module path.

    Call after `on_tree` and before `write`. The keys of `handle.report()`
    *are* module paths, so this is a join rather than a guess, and it is the
    one thing a tool that reads source code cannot produce: which node was
    replaced, what it became, whether it truly ran, and — through `refused`
    — why a node was left alone.

        with hook.on_tree(rec, model):
            run()
        hook.seats(rec, handle.report(), refused=handle.refusals())

    `refused` takes a mapping of path to reason, a list of
    `{"path", "reason"}`, or a bare count when the ledger is not to hand.
    """
    nodes = rec.meta.get("nodes")
    if not nodes:
        raise RuntimeError("call on_tree before seats: there is no tree yet")
    names = {n["node"] for n in nodes}

    def owner(path: str) -> str | None:
        """The deepest drawn node this path sits under."""
        parts = path.split(".")
        for i in range(len(parts), 0, -1):
            head = ".".join(parts[:i])
            for cand in (head, _fold_key(head)):
                if cand in names:
                    return cand
        return rec.meta.get("_root", "<root>") if "<root>" in names else None

    seat: dict[str, dict] = {}

    def slot(node):
        return seat.setdefault(node, {
            "bound": 0, "refused": 0, "calls": 0, "fallbacks": 0,
            "detached": 0, "kinds": [], "reasons": {}})

    for key, entry in (report or {}).items():
        node = owner(str(key).split("::", 1)[0])
        if node is None:
            continue
        s = slot(node)
        s["bound"] += 1
        s["calls"] += int(entry.get("calls", 0) or 0)
        s["fallbacks"] += int(entry.get("fallbacks", 0) or 0)
        s["detached"] += 1 if entry.get("detached") else 0
        kind = entry.get("kind")
        if kind and kind not in s["kinds"]:
            s["kinds"].append(kind)
        why = entry.get("last_reason")
        if why:
            s["reasons"][str(why)] = s["reasons"].get(str(why), 0) + 1

    n_refused = 0
    if isinstance(refused, int):
        n_refused = refused
    elif refused:
        items = (refused.items() if isinstance(refused, dict) else
                 [(r.get("path", ""), r.get("reason", "")) for r in refused])
        for path, why in items:
            n_refused += 1
            node = owner(str(path).split("::", 1)[0])
            if node is None:
                continue
            s = slot(node)
            s["refused"] += 1
            why = str(why or "no reason recorded")
            s["reasons"][why] = s["reasons"].get(why, 0) + 1

    for s in seat.values():
        s["kinds"].sort()
    rec.meta["seats"] = seat
    rec.meta["bound"] = sum(s["bound"] for s in seat.values())
    rec.meta["refused"] = (n_refused if isinstance(refused, int)
                           else sum(s["refused"] for s in seat.values()))
    rec.meta["fallbacks"] = sum(s["fallbacks"] for s in seat.values())
    return rec


# ---------------------------------------------------------------------
# what the GPU was actually asked to do
# ---------------------------------------------------------------------

#: Who wrote the kernel that ran. The crispest axis there is: it answers
#: "whose code is on the GPU right now" with no interpretation at all.
ORIGIN = (
    ("FlashRT", ("flash_rt", "flashrt")),
    ("inductor", ("triton_",)),
    ("cuBLAS/CUTLASS", ("cutlass", "xmma", "ampere_", "sm80_", "sm90_",
                        "sm100_", "sm120_", "gemv", "splitkreduce")),
    ("cuDNN", ("cudnn",)),
    ("FA2", ("pytorch_flash", "flash_fwd", "fmha")),
    ("memory op", ("memcpy", "memset")),
    ("PyTorch", ("at::native", "at_cuda", "c10::")),
)

#: What the kernel is for. First match wins, so a GEMM is a GEMM before it
#: is "something with e2m1 in the name", and a quantizer is a quantizer
#: before it is "something with add in it".
FAMILY = (
    ("attention", ("flash_fwd", "fmha", "pytorch_flash", "mha_",
                   "attention", "flash::")),
    ("gemm", ("gemm", "cutlass", "xmma", "ampere_", "sm80_", "sm90_",
              "sm100_", "sm120_", "gemv", "matmul", "conv")),
    ("quantize", ("quantize", "quant_", "_quant", "dequant", "nvfp4",
                  "fp4_", "fp8_", "e4m3", "e2m1", "scale_max")),
    ("norm", ("norm", "reduce_kernel", "welford", "rms", "softmax")),
    ("layout", ("nchwtonhwc", "nhwctonchw", "transpose", "permute",
                "catarray", "index_", "gather", "scatter")),
    ("copy", ("memcpy", "memset", "copy_kernel", "fill", "zero_")),
    ("elementwise", ("elementwise", "silu", "gelu", "unary", "binary",
                     "add", "mul", "clamp", "rope")),
)

#: The arithmetic a kernel names in its own symbol. Absent for most, which
#: is why it is reported as a count and never as a share.
PRECISION = (
    ("nvfp4", ("e2m1", "nvfp4", "fp4")),
    ("fp8", ("e4m3", "e5m2", "fp8")),
    ("int8", ("int8", "s8_", "i8_")),
    ("bf16", ("bf16", "bfloat16")),
    ("fp16", ("fp16", "__half", "half_t")),
    ("fp32", ("float32", "_f32", "fp32")),
)


def _bucket(rules, name: str, fallback: str) -> str:
    low = name.lower()
    for label, keys in rules:
        if any(k in low for k in keys):
            return label
    return fallback


def kernel_origin(name: str) -> str:
    """Whose kernel this is."""
    return _bucket(ORIGIN, name, "other")


def kernel_family(name: str) -> str:
    """What the kernel is for."""
    return _bucket(FAMILY, name, "other")


def kernel_precision(name: str) -> str:
    """The arithmetic the kernel names in its own symbol, if it names one."""
    return _bucket(PRECISION, name, "")


@contextlib.contextmanager
def on_kernels(rec: Recorder, *, stretch: float | None = None,
               target_s: float = 8.0, top: int = 24):
    """Record every CUDA kernel that ran, and who asked for it.

    This is the runtime companion to a demo film. The demo shows that one
    arm finished sooner; this shows what each arm spent the GPU on to get
    there. One instrument covers all four forms, because a profiler records
    the kernel that ran and does not care who launched it: eager PyTorch,
    an inductor-generated Triton kernel, a FlashRT seam, or a FlashRT
    native pipeline that never enters torch at all.

        rec = hook.Recorder("runtime", label="eager PyTorch", color="stock")
        with hook.on_kernels(rec):
            model(**inputs)
        rec.write("runs/wan22_runtime/eager")

    Launch counts and kernel names are structural facts and go on the
    canvas. The timestamps are perturbed by the profiler and pace the
    animation only.

    Fewer launches is not the point, and the film should not imply it is:
    an arm can issue more kernels than its baseline and still finish first,
    because it made each one cheaper.
    """
    from torch.profiler import ProfilerActivity, profile

    _sync()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        rec.start()
        yield rec
        _sync()
    _collect_kernels(rec, prof, stretch, target_s, top)


def _collect_kernels(rec, prof, stretch, target_s, top) -> None:
    raw = []
    for e in prof.events():
        if getattr(e, "device_type", None) is None:
            continue
        if e.device_type.name != "CUDA":
            continue
        rng = getattr(e, "time_range", None)
        if rng is None:
            continue
        raw.append((e.name, rng.start / 1e6, rng.end / 1e6))
    if not raw:
        raise RuntimeError(
            "the profiler recorded no CUDA kernel: did anything run on the "
            "device inside the block?")
    raw.sort(key=lambda r: r[1])
    base = raw[0][1]
    span = max(r[2] for r in raw) - base
    if stretch is None:
        stretch = round(target_s / span, 3) if span > 1e-9 else 1.0

    order: dict[str, int] = {}
    kernels: list[dict[str, Any]] = []
    for name, _, _ in raw:
        if name not in order:
            order[name] = len(kernels)
            kernels.append({
                "name": name,
                "origin": kernel_origin(name),
                "family": kernel_family(name),
                "precision": kernel_precision(name),
                "count": 0,
            })
        kernels[order[name]]["count"] += 1

    for name, t0, _ in raw:
        rec.events.append({"t": round((t0 - base) * stretch, 6),
                           "kind": "launch", "k": order[name]})
    rec.events.sort(key=lambda e: e["t"])

    def tally(key):
        out: dict[str, int] = {}
        for k in kernels:
            if k[key]:
                out[k[key]] = out.get(k[key], 0) + k["count"]
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    rec.meta.update({
        "launches": len(raw),
        "distinct": len(kernels),
        "kernels": sorted(kernels, key=lambda k: -k["count"])[:top],
        "families": tally("family"),
        "origins": tally("origin"),
        "precisions": tally("precision"),
        "stretch": stretch,
        "_timing_note": "profiler-perturbed, and it paces the animation. "
                        "Never quote it as a performance figure.",
    })
    # the legend the events index into, in first-launch order
    rec.meta["legend"] = [{"name": k["name"], "origin": k["origin"],
                           "family": k["family"],
                           "precision": k["precision"]} for k in kernels]
    rec.meta["done_s"] = round(span * stretch, 4)


# ---------------------------------------------------------------------
# the framework's own architecture, lit by a run
# ---------------------------------------------------------------------

DIAGRAMS = pathlib.Path(__file__).resolve().parent / "diagrams"


def load_diagram(name_or_path: str) -> dict:
    """A diagram by name (`flashrt`) or by path."""
    p = pathlib.Path(name_or_path)
    if not p.exists():
        p = DIAGRAMS / f"{name_or_path}.json"
    return json.loads(p.read_text())


def _resolve(target: str):
    """`pkg.mod:name` or `pkg.mod:Class.method` -> (owner, attr, current)."""
    import importlib

    mod_name, _, path = target.partition(":")
    obj = importlib.import_module(mod_name)
    parts = path.split(".")
    for name in parts[:-1]:
        obj = getattr(obj, name)
    return obj, parts[-1], getattr(obj, parts[-1])


@contextlib.contextmanager
def on_components(rec: Recorder, diagram, *, target_s: float = 8.0,
                  pace: str = "order", kernels: bool = True):
    """Light an architecture diagram from a run.

    The layout of a diagram is authored — a picture of a system is a human
    judgement and pretending otherwise makes a worse picture. What lights up
    is not: each box names the entry point it stands for, and a box turns on
    when that entry point is actually called, in the order it is called.

        rec = hook.Recorder("diagram", label="+ FlashRT structures")
        with hook.on_components(rec, "flashrt"):
            attach_and_run()
        rec.write("runs/doors/attach")

    A box with no entry point of its own — the kernel providers, the
    hardware — is lit by evidence instead: which kernel package is loaded in
    this process, and whether any of its kernels reached the device.

    The default pacing is **call order**, not the clock. Attaching spends
    most of its wall time inside calibration, so a stretched clock would put
    six boxes on top of each other and one at the end. Order is the fact
    this pane has; a duration is not, and inventing an even one would read
    as a measurement. `pace="clock"` keeps the real intervals for anyone who
    wants them; the real seconds are recorded either way.

    Boxes that stay dark are the point. An eager run lights the host and
    nothing else, and that is the honest picture of it.
    """
    if isinstance(diagram, str):
        diagram = load_diagram(diagram)
    fired: dict[str, list[float]] = {}
    patched, missing = [], {}

    def arm(node_id, target):
        try:
            owner, attr, current = _resolve(target)
        except (ImportError, AttributeError) as exc:      # noqa: BLE001
            missing[target] = f"{type(exc).__name__}: {str(exc)[:80]}"
            return
        if isinstance(current, type):
            inner, hook_attr, holder = current.__init__, "__init__", current
        else:
            inner, hook_attr, holder = current, attr, owner

        @functools.wraps(inner)
        def watched(*a, **k):
            fired.setdefault(node_id, []).append(rec.elapsed)
            return inner(*a, **k)

        try:
            setattr(holder, hook_attr, watched)
        except (AttributeError, TypeError) as exc:        # noqa: BLE001
            missing[target] = f"{type(exc).__name__}: {str(exc)[:80]}"
            return
        patched.append((holder, hook_attr, inner))

    for node in diagram["nodes"]:
        for target in node.get("watch", ()):
            arm(node["id"], target)

    prof = None
    if kernels:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CUDA])
        prof.__enter__()
    _sync()
    rec.start()
    try:
        yield rec
    finally:
        _sync()
        for holder, attr, inner in reversed(patched):
            setattr(holder, attr, inner)
        if prof is not None:
            prof.__exit__(None, None, None)
        _light(rec, diagram, fired, missing, prof, pace, target_s)


def light(rec: Recorder, node: str, *, calls: int = 0, why: str = "",
          after: str | None = None) -> None:
    """Light a box from a receipt instead of from a wrapper.

    Some things a diagram box stands for are not one function call. A
    qualification that ends in a refusal is recorded in the plan's ledger by
    whichever site decided it, and wrapping every one of those would be a
    worse description of the box than the ledger it produced. So the caller
    hands over the receipt and says where in the authored order it belongs.

    Call inside the `on_components` block, or after it and before `write`.
    Receipt-lit boxes are named in `meta["by_receipt"]`, because "we watched
    this happen" and "we read this afterwards" are different claims.
    """
    rec.__dict__.setdefault("_receipts", {})[node] = {
        "calls": int(calls), "why": why, "after": after}


def _providers_loaded() -> dict[str, list[str]]:
    """Which kernel providers are in this process, by where they came from.

    A hub package lives in the kernels cache; a native extension is a
    compiled module inside `flash_rt`. Both answer to `flash_rt::` on the
    device, so the symbol cannot tell them apart — the import can.
    """
    hub, ext = [], []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if "--flashrt--" in f.lower():
            hub.append(f.lower().split("--flashrt--", 1)[1].split("/")[0])
        elif name.startswith(("flash_rt.flash_rt_", "flash_rt_")) and f:
            ext.append(name)
    return {"hub": sorted(set(hub)), "extension": sorted(set(ext))}


def _light(rec, diagram, fired, missing, prof, pace, target_s) -> None:
    by = {n["id"]: n for n in diagram["nodes"]}
    kids: dict[str, list[str]] = {}
    for n in diagram["nodes"]:
        if n.get("group"):
            kids.setdefault(n["group"], []).append(n["id"])

    origins: dict[str, int] = {}
    if prof is not None:
        for e in prof.events():
            dt = getattr(e, "device_type", None)
            if dt is not None and dt.name == "CUDA":
                o = kernel_origin(e.name)
                origins[o] = origins.get(o, 0) + 1

    providers = _providers_loaded()
    span = max((v[0] for v in fired.values()), default=rec.elapsed) or 1e-3

    lit: dict[str, dict[str, Any]] = {}
    for node in diagram["nodes"]:
        nid = node["id"]
        hits = fired.get(nid)
        if hits:
            lit[nid] = {"first_s": round(hits[0], 4), "calls": len(hits)}
        elif node.get("kernel_origin"):
            got = sum(origins.get(o, 0) for o in node["kernel_origin"])
            if got and providers["hub"]:
                lit[nid] = {"first_s": span, "calls": got,
                            "why": f"{len(providers['hub'])} package(s) loaded"}
        elif node.get("provider") == "extension" and providers["extension"]:
            lit[nid] = {"first_s": span, "calls": 0,
                        "why": ", ".join(providers["extension"][:3])}

    # A group is not a call site; it is lit by its children, a hair ahead
    # of the first of them so the container opens before what is inside it.
    for gid, children in kids.items():
        on = [lit[c]["first_s"] for c in children if c in lit]
        if on and gid not in lit:
            lit[gid] = {"first_s": min(on) - 1e-9, "calls": 0}
    # the host is what ran; it is lit before anything it reached into
    lit.setdefault("host", {"first_s": -1.0, "calls": 1})
    if origins.get("FlashRT"):
        lit.setdefault("hw", {"first_s": span, "calls": origins["FlashRT"]})

    for nid, r in getattr(rec, "_receipts", {}).items():
        if nid in lit or nid not in by:
            continue
        anchor = lit.get(r.get("after") or "", {}).get("first_s")
        lit[nid] = {"first_s": (anchor + 1e-6) if anchor is not None else span,
                    "calls": r["calls"], "why": r["why"], "receipt": True}
    for gid, children in kids.items():                # groups, again
        on = [lit[c]["first_s"] for c in children if c in lit]
        if on and gid not in lit:
            lit[gid] = {"first_s": min(on) - 1e-9, "calls": 0}

    ranked = sorted(lit.items(), key=lambda kv: kv[1]["first_s"])
    step = target_s / max(len(ranked), 1)
    for i, (nid, v) in enumerate(ranked):
        v["t"] = (round(i * step, 4) if pace == "order"
                  else round(v["first_s"] * target_s / span, 4))
        rec.events.append({"t": v["t"], "kind": "lit", "node": nid,
                           "calls": v["calls"]})
    rec.meta.update({
        "diagram": diagram,
        "lit": lit,
        "n_lit": len(lit),
        "n_nodes": len(diagram["nodes"]),
        "providers": providers,
        "kernel_origins": origins,
        "unarmed": missing,
        "by_receipt": sorted(n for n, v in lit.items() if v.get("receipt")),
        "pace": pace,
        "wall_s": round(span, 4),
        "_timing_note": "the pane paces by call order, not by the clock. "
                        "`first_s` is the real first-call time and is "
                        "wrapper-perturbed; neither is a performance figure.",
    })
    rec.meta["done_s"] = round(target_s, 4)
