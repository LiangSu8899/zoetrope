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

__all__ = ["Recorder", "on_tokens", "on_denoiser", "on_calls",
           "on_tree", "seats"]

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
