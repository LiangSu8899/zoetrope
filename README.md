# demo-kit

Tools for recording what an optimization actually did, as a film.

Each arm runs the real model once and writes down **when** each thing happened.
A compositor then replays those timestamps and draws every frame. Nothing is
screen-captured: the robot pixels are the simulator's own observation, and every
number, bar and clock on the page is drawn from the recorded events.

That indirection is what makes the films possible at all. Arms that cannot share
a process — a serving engine bakes its module tree into a compiled decode path
at load time, so `attach` has to happen in a different process than the baseline
— still end up side by side on one wall clock.

```
demokit/
  simloop.py            the LIBERO closed loop, shared by every robot host
  record.py             one arm per process; maps --host to an adapter
  hosts/                8 host adapters (lerobot, openpi, Isaac-GR00T, native)
  pipelines/            4 recorders: Wan2.2, Qwen3-VL, Qwen3.6-35B, vLLM/SGLang
  compose/race.py       stream / video / stream_batch panes
  compose/sim.py        robot panes
adapters/
  diffusers_cli_hook.py attach inside `diffusers-cli run`, no diffusers edit
docs/PROTOCOL.md        the run-directory contract and the adapter protocols
```

## Recording one film

```bash
# one arm per process, never two
python -m demokit.record --host lerobot_pi05 --arm eager \
    --suite libero_spatial --task-id 0 --trial 0 --out runs/pi05_race/eager
python -m demokit.record --host lerobot_pi05 --arm attach ... --out runs/pi05_race/attach

# then draw them on one clock
python -m demokit.compose.sim --runs runs/pi05_race --out pi05_race.mp4
```

## Adding a model

Read [`docs/PROTOCOL.md`](docs/PROTOCOL.md). Short version:

- **a robot policy** — write one host adapter and one branch in
  `record.py:build_host()`. The loop, the timing and the event writing are
  already done.
- **anything else** — currently a new 300–500 line script, because that half has
  no shared loop yet. Fixing that is item 2 in the protocol doc.

## Rules the recorders follow

These came from getting them wrong first, and they are why the numbers hold up.

- **One arm per process.** Two arms in one process share allocator state and
  clocks.
- **Report TTFT and decode separately.** Never blend them into one end-to-end
  number.
- **Two baselines**: the host as shipped, and the host after `torch.compile` +
  graph capture. The gate is the compiled one.
- **Headline is a median.** min-of-N is a lower bound in a footnote, never the
  number on the page.
- **Score parity against the original baseline**, never against an intermediate
  arm.
- **Real inputs.** Random tensors hide calibration bugs.
- **A refusal is a result.** A seam that does not pay is recorded with its
  reason, not dropped.
