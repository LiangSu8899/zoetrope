---
name: record-a-demo
description: Record an inference host as a side-by-side film — two or more arms of the same model on one wall clock, with live rate readouts. Use when someone wants to show what an optimization did, not just state it.
---

# Recording a demo film

You record **when** each thing happened; a compositor replays those timestamps
and draws every frame. Nothing is screen-captured. That split is what lets the
drawing change later without running the model again.

Install: `pip install -e /path/to/demo-kit` (or put it on `PYTHONPATH`).

## Before you run anything: is the data already there?

Running a model is expensive; drawing is not. **Recording and drawing are
separate, so a run someone else recorded draws just as well as one you made.**
A whole concurrency film in this repository was drawn from 960 KB of
`events.json` pulled off a Jetson, without the model ever running on the
machine that drew it.

So, in order:

1. **Look for existing runs.** If `events.json` exists for the arms you want,
   go straight to `demokit draw`. Nothing needs a GPU.
2. **Missing one arm?** Record only that arm. The others stand.
3. **Only then** record from scratch.

`examples/runs/` holds one real, complete film per painter kind. Use them to
see the schema, to check your environment draws at all, and as a template:

```bash
demokit check examples/runs/*/*                          # 9/9 ready to draw
demokit draw  examples/runs/video --out /tmp/check.webm  # should just work
```

`examples/specs/` holds the multi-chapter specs behind the published films —
the shortest way to see how chapters, arms, notes and footers fit together.

## The shape of the job

One arm per process. Never two — they would share allocator state and clocks.

```
run each arm  →  runs/<film>/<arm>/events.json   (+ frames.npy if it has pixels)
                 demokit check runs/<film>/*
                 demokit draw  runs/<film> --out film.webm
```

## Recording an arm

```python
from demokit import hook

rec = hook.Recorder(
    "video",                       # stream | video | stream_batch | loop | arch
    label="+ FlashRT structures",  # the pane header
    sub="auto_swaps, nvfp4_balance",
    color="ours")                  # stock | compiled | ours | native

with hook.on_denoiser(rec, pipe):  # stamps every denoiser call, restores on exit
    out = pipe(prompt=..., num_inference_steps=20)

rec.frames(np.asarray(out.frames[0])).write("runs/wan22/attach")
```

Four hooks, by what the host is:

| host | hook |
|---|---|
| a diffusers pipeline | `hook.on_denoiser(rec, pipe)` |
| any module you want one stamp per call from | `hook.on_calls(rec, module)` |
| a `transformers` generate | `with hook.on_tokens(rec, tokenizer): model.generate(..., streamer=rec.streamer)` |
| the model's own structure | `hook.on_tree(rec, model, depth=4)` |

Anything else: call `rec.stamp(...)` yourself at the moment the thing arrives.
That is all a hook does.

## Recording the architecture

`hook.on_tree` is the odd one out: it records structure, not speed. Nodes come
from `named_modules()`, edges from the order the forward hooks first fired.
After an attach, `hook.seats(rec, handle.report(), refused=...)` joins the
receipt onto that tree by module path, so each node says what FlashRT put in
it, whether it truly ran, and where a seam was turned down, why.

Record the same tree twice — once with no FlashRT in the process, once after
attach — and draw the two arms side by side.

Two rules for this pane. **No performance figure goes on it**, ever: a forward
pass is milliseconds, the recorder stretches it to make it watchable, and the
node timings exist only to pace that. And **a refusal is drawn, never
dropped** — it is the half of the picture that explains the other half.

`ttft_ms`, `decode_tok_s`, `ms_per_step` and friends are derived from the
timestamps on write — do not pass them unless you measured them a different way
and mean it.

## Choosing arms

A film is only worth drawing if the arms are honestly comparable:

- **The baseline is the production form.** The host as its authors ship it,
  *and* the host after `torch.compile` + graph capture. The second is the one a
  deployment actually feels; report both.
- **Same fixture in every arm.** Same prompt, same seed, same shapes, same
  task, same initial state.
- **Report TTFT and decode separately.** Never one blended end-to-end number.
- **The headline is a median.** min-of-N belongs in a footnote as a lower
  bound, never on the page.
- **Score parity against the original baseline**, never against an intermediate
  arm.
- **Real inputs.** Random tensors hide calibration bugs.
- **A refusal is a result.** A seam that does not pay gets recorded with its
  reason, not dropped.

## Drawing

```bash
demokit check runs/wan22/eager runs/wan22/attach     # says what would fail, and why
demokit draw  runs/wan22 --arms eager,attach --out wan22.webm \
    --title "Wan2.2 TI2V-5B" \
    --subtitle "480x480 · 33 frames · 20 steps · one prompt, one seed · RTX 5090" \
    --note "what a reader should notice"
```

Several chapters in one file (e.g. four concurrency levels) go through a spec:

```json
{"pane": 430, "fps": 30, "chapters": [
  {"runs": "runs/conc/c1", "arms": "base,attach", "tail": 1.6,
   "title": "...", "subtitle": "...", "note": "...", "footer": "..."}
]}
```
```bash
demokit draw --spec spec.json --out concurrency.webm
```

## If something looks wrong

- `demokit check` first. It names the missing `meta` key or the unsorted
  timestamps rather than letting the compositor fail obscurely.
- A pane stuck at `--` means no event arrived yet; that is correct during
  prefill.
- Redrawing never needs the model. If a readout or a layout is wrong, fix the
  compositor and draw the same runs again.

See `docs/PROTOCOL.md` for the full run-directory contract and the host-adapter
protocols.

## After the demo film

A demo film shows that one arm finished first. Explaining *why* is a
different job with a different rule -- one idea, no legend -- and the
versions that failed at it are worth reading before starting:
`.ai/skills/explain-a-speedup/SKILL.md`.
