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
  compose/race_compose.py   stream / video / stream_batch / arch / runtime / diagram
  compose/trips_compose.py  the one-idea film: trips to memory, on one clock
  stubs.py                  satisfy an import a recording path never calls
  diagrams/                 authored architecture layouts, wired to entry points
  compose/sim_compose.py    robot panes
adapters/
  diffusers_cli_hook.py attach inside `diffusers-cli run`, no diffusers edit
docs/PROTOCOL.md        the run-directory contract and the adapter protocols
```

## Recording one film

Install with `pip install -e .`, then record each arm in its own process and
draw them together.

```python
from demokit import hook

rec = hook.Recorder("video", label="+ FlashRT structures",
                    sub="auto_swaps, nvfp4_balance", color="ours")
with hook.on_denoiser(rec, pipe):            # stamps every denoiser call
    out = pipe(prompt=..., num_inference_steps=20)
rec.frames(frames).write("runs/wan22/attach")
```

```bash
demokit check runs/wan22/eager runs/wan22/attach    # names what would fail
demokit draw  runs/wan22 --arms eager,attach --out wan22.webm \
    --title "Wan2.2 TI2V-5B" --subtitle "480x480 · 33 frames · 20 steps"
```

## One idea, for someone who will not read a chart

Every CUDA kernel reads its operands from device memory and writes its result
back, so a kernel launch is a trip to memory. That makes the launch count a
number a person can hold in their head, which a kernel taxonomy is not.

`compose/trips_compose.py` draws exactly that and nothing else: two lanes,
one shared memory bar, two counters, and a race on the measured wall clock.
The arms are ordinary `runtime` runs.

Both lanes span the full width, because both do the same job — one decision.
So **length is the work, the sweep is the clock, and the density of the marks
is the trips it took**. A lane that stopped a fifth of the way across would
read as an arm that did less, which is the opposite of the point.

Each trip crosses the gap as it really happened: the recorded launch
timestamps drive the animation, so the bursts and the gaps are the shape of
the run and not a loop. Colour says what the trip was for, on the same
palette both sides, which makes the storms themselves the comparison — the
shipped host's is grey, operands being moved; the native one's is gold and
green, arithmetic and the quantizing that feeds it.

```bash
python -m demokit.compose.trips_compose --runs examples/runs/trips \
    --arms torch,fp8 --gate compiled --out why.webm
```

On pi0.5, one decision on LIBERO, an RTX 5090:

| | trips to memory | one decision |
|---|---|---|
| PyTorch as shipped (LeRobot, eager, bf16) | 13,637 | 134.3 ms |
| the same host, `torch.compile` | 5,214 | 62.2 ms |
| FlashRT native, FP16 | 2,862 | 27.7 ms |
| FlashRT native, FP8 | **2,742** | **22.3 ms** |

Fewer trips and less time, in step, all the way down the ladder — `compile`
takes 2.6x off the trips and 2.2x off the clock, and the native pipeline takes
another 1.9x and 2.8x. That the two columns move together is what makes the
first one an explanation of the second rather than a coincidence beside it.

The film draws the top and bottom rows and names both baselines at the end,
because one of them is what a reader recognises and the other is what the
claim has to survive.

The other panes describe. This one argues, and it is the only one that puts a
speed figure on the canvas — because that is the one thing everybody reads.

## Lighting an architecture diagram

`hook.on_components` draws the framework itself — not the model — and lights
it from a run.

```python
rec = hook.Recorder("diagram", label="the same host, structures attached")
with hook.on_components(rec, "flashrt"):
    attach_and_run()
rec.write("runs/doors/attach")
```

The layout is authored, in `demokit/diagrams/*.json`: a picture of a system is
a human judgement and pretending otherwise makes a worse picture. What lights
up is not. Each box names the entry point it stands for, and turns on when
that entry point is actually called, in the order it is called.

Drawn beside a baseline, the dark boxes carry as much as the lit ones: an
eager arm lights the host and nothing under it, which is the honest picture of
an eager arm and is not something a bar chart can say.

## Recording what the runtime did

A demo film says which arm finished first. `hook.on_kernels` says what each
one asked the GPU to do to get there — every CUDA kernel that ran, in the
order it ran, bucketed by what it was for and whose code it was.

```python
rec = hook.Recorder("runtime", label="+ FlashRT structures", color="ours")
with hook.on_kernels(rec):
    model(**inputs)
rec.write("runs/wan22_runtime/attach")
```

One instrument covers every form, because a profiler records the kernel that
ran and does not care who launched it: eager PyTorch, an inductor-generated
Triton kernel, a FlashRT seam, or a native pipeline that never enters torch.
Drawn side by side, an eager arm spending most of the GPU moving operands and
a compiled arm that fused that away are visible without a word of commentary.

The pane carries counts and kernel names, which are structural. It carries no
timing: the profiler perturbs the run it watches, and the demo films are where
speed belongs. And it does not treat a smaller number as a better one — an arm
can launch more kernels than its baseline and still finish first, because it
made each of them cheaper.

## Recording the architecture

`hook.on_tree` records the model's own module tree and one real pass through
it: nodes from `named_modules()`, order from the order the forward hooks first
fired. Recorded after an attach, `hook.seats` joins the structures receipt onto
that tree by module path — so each node says what FlashRT put in it, whether it
truly ran, and, where a seam was turned down, the ledger's reason.

```python
rec = hook.Recorder("arch", label="the same tree, after attach", color="ours")
with hook.on_tree(rec, model, depth=4):
    run_one_pass()
hook.seats(rec, handle.report(),
           refused=[{"path": p, "reason": why} for p, why in refusals])
rec.write("runs/arch/attach")
```

Draw it beside the same tree recorded without FlashRT in the process and the
distribution layer is visible rather than described. The pane carries no
performance figure: one forward pass is milliseconds, so the recorder stretches
the timestamps to make it watchable and says so on its face.

Robot policies have their own path, because the LIBERO loop is shared:

```bash
python demokit/record.py --host lerobot_pi05 --arm eager \
    --suite libero_spatial --task-id 0 --trial 0 --out runs/pi05_race/eager
python demokit/compose/sim_compose.py --runs runs/pi05_race --out pi05_race.webm
```

## Installing the agent skill

```bash
demokit skills add                      # ~/.claude/skills and ~/.agents/skills
demokit skills add --project .          # into this repository instead
demokit skills add --agents-md ~/.codex/AGENTS.md   # also leave a pointer
demokit skills where                    # print the targets without writing
```

Agents disagree about where a skill lives. Claude Code reads
`~/.claude/skills/<name>/SKILL.md`; several others read
`~/.agents/skills/<name>/`; Codex reads `AGENTS.md`. `add` writes the skills
directories, and `--agents-md` appends a pointer to an `AGENTS.md` for the
agents that only read that. It is idempotent — a second run says the pointer is
already there rather than duplicating it.

## Examples

`examples/runs/` carries one real film per painter kind — the actual recorded
events, with pixels subsampled or truncated only where a file would otherwise
be too large for a repository (each says so in its own `_example_note`).

```bash
demokit check examples/runs/*/*                          # 20/20 ready to draw
demokit draw  examples/runs/stream       --out /tmp/a.webm   # a VLM, 3 arms
demokit draw  examples/runs/video        --out /tmp/b.webm   # Wan2.2, 2 arms
demokit draw  examples/runs/stream_batch --out /tmp/c.webm   # vLLM at batch 8
demokit draw  examples/runs/loop         --out /tmp/d.webm   # GR00T, 60 steps
demokit draw  examples/runs/arch         --out /tmp/e.webm   # the tree inside vLLM
demokit draw  examples/runs/runtime --arms eager,compiled,attach \
                                        --out /tmp/f.webm   # 3 runtimes, one call
demokit draw  examples/runs/doors   --arms eager,attach \
                                        --out /tmp/g.webm   # the diagram, lit
python -m demokit.compose.trips_compose --runs examples/runs/trips \
    --arms torch,fp8 --out /tmp/h.webm                      # why it is fast
```

They are also the answer to "do I need to run the model?" — usually not. A run
someone else recorded draws exactly as well as one you made.

`examples/specs/` holds the multi-chapter specs behind the published films.

## Adding a model

Read [`docs/PROTOCOL.md`](docs/PROTOCOL.md), or hand an agent
[`.ai/skills/record-a-demo/SKILL.md`](.ai/skills/record-a-demo/SKILL.md).

- **a diffusers / transformers host** — one `hook.Recorder` and one `with`
  block. No new script.
- **its architecture** — `hook.on_tree` takes any `nn.Module`. Containers,
  repeated stacks and engines that own their own model tree all read the same
  way.
- **a robot policy** — one host adapter plus one branch in
  `record.py:build_host()`; the loop, the timing and the event writing are done.
- **something with no hook yet** — call `rec.stamp(...)` where the thing
  arrives. That is all a hook does.

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
