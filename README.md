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
  compose/race_compose.py   stream / video / stream_batch / arch panes
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
demokit check examples/runs/*/*                          # 9/9 ready to draw
demokit draw  examples/runs/stream       --out /tmp/a.webm   # a VLM, 3 arms
demokit draw  examples/runs/video        --out /tmp/b.webm   # Wan2.2, 2 arms
demokit draw  examples/runs/stream_batch --out /tmp/c.webm   # vLLM at batch 8
demokit draw  examples/runs/loop         --out /tmp/d.webm   # GR00T, 60 steps
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
