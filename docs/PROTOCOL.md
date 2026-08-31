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
