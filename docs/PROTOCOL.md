# The two protocols

Everything here produces one thing: a **run directory** holding `events.json`
(plus `frames.npy` or `image.png` when the painter needs pixels). A compositor
reads several of those and draws one film on one shared wall clock.

That is the whole contract. A recorder never draws, and a compositor never runs
a model.

---

## 1. The run directory

```
runs/<film>/<arm>/
    events.json      required
    frames.npy       uint8 [N, H, W, 3] — robot rollouts, and generated clips
    image.png        the VLM's input, when the pane shows one
```

### `events.json`

```json
{"meta": { ... }, "events": [ ... ]}
```

`meta` always carries what the pane header and readout need:

| key | meaning |
|---|---|
| `label` | pane title, e.g. `"+ FlashRT structures"` |
| `sub` | one line under it, e.g. `"auto_swaps + capture"` |
| `color` | `stock` \| `compiled` \| `ours` \| `native` — picks the pane accent |
| `done_s` | when this arm finished, on the shared clock |
| `kind` | which painter draws it: `video` \| `stream` \| `stream_batch` \| `loop` |

Everything else in `meta` belongs to the painter named by `kind`, and is listed
with that painter below.

`events` is a list, each with a **`t` in seconds on the arm's own clock**. That
timestamp is the entire point: the compositor replays it, it does not invent a
rate.

---

## 2. Painters, and what each one needs

### `kind: "stream"` — a language model
```json
{"i": 0, "t": 0.0325, "text": "The"}
```
`meta` adds: `ttft_ms`, `decode_tok_s`, `n_tokens`, `prompt`, optional
`stream_note`, and the parity fields the footer prints
(`teacher_forced_same_token`, `tokens_match_host`, …).
Reads `image.png` if present, and shows it above the text.

### `kind: "video"` — a diffusion pipeline
```json
{"kind": "step", "t": 0.31}
```
`meta` adds: `steps`, `ms_per_step`, `decode_s`, `clip_fps`.
Reads `frames.npy` and plays the clip once the arm is done.

### `kind: "stream_batch"` — one serving engine under concurrency
One row per in-flight request; `meta` adds the aggregate and per-request rates.

### `kind: "diagram"` — the framework's own architecture, lit by a run
```json
{"kind": "lit", "t": 1.85, "node": "s.discover", "calls": 2}
```
`meta` adds: `diagram` (the whole authored layout, carried with the run so a
film draws years later), `lit`, `n_lit`, `n_nodes`, `providers`,
`kernel_origins`, `unarmed`, `by_receipt`, `pace`, `wall_s`.

A diagram is a JSON layout — nodes with a box, a label, and the entry point
each box stands for:

```json
{"id": "s.bind", "label": "bind", "box": [366, 158, 108, 92],
 "group": "structures", "watch": ["flash_rt.structures.swap:attach"]}
```

The layout is **authored**: a picture of a system is a human judgement and
pretending otherwise makes a worse picture. What lights up is **not**. A box
turns on when its entry point is actually called, and the order is the order
they were called. Boxes with no entry point of their own — the kernel
providers, the hardware — are lit by evidence instead: which kernel package
is loaded in the process, and whether its kernels reached the device.

`hook.light(rec, node, after=...)` lights a box from a receipt where no
single function stands for it — a qualification that ends in a refusal is
recorded by whichever site decided it. Those boxes are named in
`by_receipt`, because "we watched this happen" and "we read this afterwards"
are different claims.

**The dark boxes are half the message.** An eager arm lights the host and
nothing under it, and drawing the rest of the diagram dark beside it is the
only way that reads as a fact rather than as an omission.

Pacing is by **call order**, not the clock: attaching spends most of its wall
time inside calibration, and a stretched clock would put six boxes on top of
each other and one at the end. `wall_s` and each box's `first_s` keep the
real seconds; neither is a performance figure.

### `kind: "runtime"` — what the arm asked the GPU to do
```json
{"kind": "launch", "t": 0.0041, "k": 12}
```
`meta` adds: `launches`, `distinct`, `legend`, `kernels`, `families`,
`origins`, `precisions`, `stretch`, optional `runtime_note`.

`k` indexes `legend`, which lists every distinct kernel in first-launch
order with its `origin` (whose code — PyTorch, inductor, cuBLAS/CUTLASS,
FA2, FlashRT, a memory op), its `family` (what it is for — gemm, attention,
quantize, norm, elementwise, copy, layout) and the `precision` its own
symbol names, when it names one.

This is the runtime companion to a demo film: the demo says which arm
finished first, this says what each spent the GPU on to get there. One
instrument covers every form, because a profiler records the kernel that
ran and does not care who launched it — eager PyTorch, an inductor kernel,
a FlashRT seam, or a native pipeline that never enters torch.

Launch counts and kernel names are structural and go on the canvas.
Timings do not: the profiler perturbs the run, and the timestamps are there
to pace the animation.

> **Fewer launches is not the goal**, and a pane must not imply it is. An
> arm can issue more kernels than its baseline and still finish first,
> because it made each of them cheaper. Composition, not count.

`compose/trips_compose.py` reads these runs too, and draws the one thing a
person takes away: both arms compute the same decision, so both rows are the
same length, and one is cut into five times as many pieces. It also reads
`decision_ms`, and — when Nsight Compute measured them — `sm_slices`,
`sm_pct`, `dram_pct` and `sm_peak_pct`, which colour the pieces by how hard
the chip was working. Those are optional: without them the film draws in the
arm's own accent instead of inventing a temperature.

### `kind: "arch"` — the model's own module tree
```json
{"kind": "enter", "t": 0.41, "node": "blocks.*", "idx": 12}
{"kind": "exit",  "t": 0.55, "node": "blocks.*", "idx": 12}
```
`meta` adds: `model_class`, `depth`, `nodes`, `n_modules`, `n_groups`,
`stretch`, and — when the run was recorded after an attach — `seats`, `bound`,
`refused`, `fallbacks`.

`nodes` is the tree in the order the forward hooks first fired, which is the
order the model actually ran. Each node carries `node` (its module path),
`cls`, `depth`, `parent`, `repeat` and `calls`. A module that never fires did
not run in this pass; that is observed, not inferred.

`seats` maps a node to what FlashRT put there — `bound`, `kinds`, `calls`,
`fallbacks`, `refused`, and `reasons`. The keys of `handle.report()` *are*
module paths, so this is a join, and it is the part a tool that reads source
code cannot produce.

The pane draws no performance figure. One forward pass is milliseconds, so
`stretch` records what the timestamps were multiplied by to make it watchable.
The node timings in `nodes` exist to pace that animation and are perturbed by
the hooks themselves — they are not a measurement of anything.

### `kind: "loop"` — a robot policy
```json
{"step": 0, "infer_ms": 22.25, "fresh": true, "action": [...]}
```
`meta` adds: `arm`, `host`, `suite`, `task`, `task_id`, `trial`, `steps`,
`median_infer_ms`, `success`, `replan`.
Reads `frames.npy` — the simulator's own observation, one frame per control
step.

> **Not yet unified.** `loop` is currently read by `compose/sim.py`, the other
> three by `compose/race.py`, and the `loop` recorders do not write `label` /
> `sub` / `color` / `done_s` / `kind` at all. Making them write those is the one
> change that would collapse the two compositors into one. See "What is left".

---

## 3. Recorder protocol — robot policies

This half is already a plugin system. `simloop.rollout(host, ...)` owns the
LIBERO loop, the timing and the event writing; a host adapter owns nothing but
the model. Eight of them exist in `demokit/hosts/`, 120–337 lines each.

A host adapter is a plain object with these methods — no base class, no
registration:

```python
class MyHost:
    def build(self): ...                      # load the checkpoint, arm the arm
    def set_task(self, text: str): ...        # the language instruction
    def observe(self, img, wrist, state): ... # one control step's observation
    def sync(self): ...                       # block until the device is done
    # plus whatever `_predict` needs internally
```

`record.py --host <name>` maps a name to one of these. To add a policy or a
host, write one adapter and add one branch to `build_host()`.

---

## 4. Recorder protocol — everything else

**There isn't one yet.** `demokit/pipelines/` holds four independent scripts,
153–595 lines each, that repeat the same shape by hand:

1. load the host
2. run the shipped form, capture its calls and its output
3. run the compiled form
4. attach structures, run again
5. time each arm paired, score parity against the *original* baseline
6. write one run directory per arm, plus a receipt

Steps 1 and 4 are host-specific. Steps 2, 3, 5 and 6 are the same every time and
should be a loop, the way `simloop.rollout` is for robots.

---

## What is left, in order

1. **Give the `loop` recorders the five common `meta` keys** so one compositor
   reads everything. Small, and it removes a whole file.
2. **Extract `pipelines/` into a `rollout()` plus an adapter**, mirroring
   `simloop.rollout(host)`. A new model then costs one adapter, not 400 lines.
3. **Name the adapter protocol in code** (a `typing.Protocol`), so an agent
   writing a new one gets checked rather than guessing from an example.
