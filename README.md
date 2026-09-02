# demokit

**Ask an agent for a video that explains a system — a model you just optimized,
a paper you just read, a framework you have to teach — and get one back where
every frame is drawn from a measurement or a citation.**

Text-to-video makes something plausible. This makes something checkable: the
numbers on the screen came off a profiler or out of a paper, the picture is
replayed from timestamps that were really recorded, and a claim with nothing
behind it is printed as `not measured` rather than drawn as a bar.

<p align="center">
  <img src="docs/gif/run_demo.gif" width="49%" alt="three arms writing the same answer on one clock">
  <img src="docs/gif/why_faster.gif" width="49%" alt="the same work, cut into fewer trips to memory">
  <img src="docs/gif/explain_radix.gif" width="49%" alt="SGLang's radix tree, with the number it produces beside it">
  <img src="docs/gif/explain_tiling.gif" width="49%" alt="FlashAttention's tiles, and the matrix that is never written">
</p>

---

## What you say, and what you get

The whole interface is a sentence. These are real prompts; each one names the
work the agent then does.

| Say this | You get |
|---|---|
| *"Make a demo film from the recordings in this repo."* | Two or more arms writing the same answer on one wall clock, with live tok/s. No GPU — it draws from `events.json`. |
| *"Record my model against `torch.compile` and draw it."* | One process per arm, both timed the same way, both baselines named, headline reported as a median. |
| *"Explain why my version is faster, in one idea."* | The one-idea film: the same work cut into a different number of trips to memory. Everything else cut. |
| *"Explain PagedAttention to someone who has never read the paper."* | A three-page explainer drawn from the paper's own mechanism, each page carrying its citation and the figure it earns. |
| *"Explain this repo's core idea the way you explained vLLM."* | A new spec in `demokit/explainers/`, same painters, redrawn in seconds. |
| *"Draw it in every colour system and let me pick."* | One contact sheet, 4 styles × 6 palettes, from the same recording. |
| *"Slower, and hold the ending."* | A pacing change. No model re-run — redrawing costs seconds. |

The last row is the important one. **Recording and drawing are separate.** A
run writes down *when* each thing happened; a compositor replays those
timestamps and paints every frame. Changing the picture never re-runs the
model, so ten drafts is normal and the conversation stays a conversation.

## What the agent will not do

This is the part that makes the output worth showing to other people.

- **It will not invent a number.** A pane with no measurement behind it prints
  `not measured`.
- **It will not report a best-of-N as a headline.** Medians, with min/max as a
  footnote.
- **It will not hide the number that did not move.** FlashRT's own explainer
  puts 17.8% against 20.0% utilisation on the page and says in as many words
  that the chip is not being driven harder — the win is that far less was
  asked of it.
- **It will not silently speed up a clock.** A run too short to watch is
  stretched, and the page says `slowed 18x` on its face.
- **It will not race someone else's framework to make a point.** Explainers
  explain; comparisons belong in a demo where the comparison *is* the subject.

## Install

```bash
pip install -e .
demokit skills add          # teach your agent: ~/.claude/skills, ~/.agents/skills
demokit skills add --agents-md ~/.codex/AGENTS.md    # Codex reads this instead
```

Two skills go in. `record-a-demo` covers making the film that shows an
optimization worked; `explain-a-speedup` covers the harder one — saying *why* —
and most of it is a record of the versions that failed and the reason each was
unreadable, because that judgement is what a first draft gets wrong.

Then, in Claude Code, Codex, or anything that reads a skills directory:

> Draw the recordings in `examples/runs/stream` as a demo film, in the paper
> palette, and put a tok/s curve under it.

## Two families

### Run demos — drawn from a run

Two arms writing the same answer on one clock. Everybody reads it without
being told anything, and `--chart` puts the shape of it on the page as well.

```bash
demokit live examples/runs/stream       --palette paper    --chart curve --out a.webm
demokit live examples/runs/stream_batch --palette phosphor --chart bars  --out b.webm
demokit live examples/runs/stream       --palette mono     --chart none  --out c.webm
```

<p align="center">
  <img src="docs/gif/serving.gif" width="70%" alt="a serving engine with eight requests in flight">
</p>

One painter covers both shapes: a `stream` arm is one request and its pane is
the answer; a `stream_batch` arm is a serving engine with several requests in
flight and its pane is one scrolling line each.

### Explainers — drawn from a paper

Four ship, and pointing this at a fifth framework is a JSON spec, not a patch.

```bash
demokit explain vllm_paged     --menu menu.png              # every panel
demokit explain flashattention --palette phosphor --film fa.webm
demokit explain sglang_radix   --panel 0 --sheet six.png    # in six palettes
```

| spec | the idea it draws | source |
|---|---|---|
| `vllm_paged` | the KV cache handed out in blocks instead of reserved whole | PagedAttention, SOSP 2023 |
| `sglang_radix` | what has been computed, kept in a tree | RadixAttention, NeurIPS 2024 |
| `flashattention` | the N×N matrix that is never written down | FlashAttention 1 and 2 |
| `flashrt_trips` | fewer, larger kernels — and the utilisation that *did not* move | measured here, RTX 5090 |

<p align="center">
  <img src="docs/gif/explain_result.gif" width="70%" alt="a results page: meters that fill and bars against a baseline">
</p>

Every mechanism page can carry a chart in its right-hand column, and that
chart is **derived from the diagram beside it** — the tokens in the drawn
tree, the queue in the drawn cards — so the picture and its number cannot
drift apart.

## The one rule

**One idea per film, and everything else is cut.**

The corollary is uncomfortable and holds anyway: *more accurate is often
harder to read, and readable wins.* Measure everything; draw one thing.

Five versions of the "why is it faster" film were built and rejected before
the one above. That record — what each version tried and why a person could
not read it — is in
[`.ai/skills/explain-a-speedup/SKILL.md`](.ai/skills/explain-a-speedup/SKILL.md),
and it is the most useful thing in this repository.

## Look

Six colour systems, addressed by role rather than by name, so one flag
restyles a whole film and no drawing code moves:

`midnight` · `paper` · `phosphor` · `blueprint` · `ember` · `mono`

Pages are drawn at twice the size and brought back down — PIL has no
anti-aliasing, and that alone is the difference between a page that looks
designed and one that looks emitted. Beats are eased, lists arrive item by
item, connectors leave and arrive along one axis and curve between, and films
cross-fade rather than cut.

```bash
demokit looks examples/runs/stream --sheet looks.png     # 4 styles x 6 palettes
```

![four styles against six palettes](docs/looks.png)

## Recording your own

```python
from demokit import hook

rec = hook.Recorder("stream", label="+ my change", sub="what it does",
                    color="ours")
with hook.on_tokens(rec, tokenizer):      # stamps each token as it arrives
    model.generate(**inputs, streamer=rec.streamer, max_new_tokens=256)
rec.write("runs/mine/attach")
```

```bash
demokit check runs/mine/attach     # names what would fail before you draw
demokit live  runs/mine --out mine.webm
```

Other hooks record what the run *did*: `hook.on_kernels` (every CUDA kernel,
via `torch.profiler`, whoever launched it), `hook.on_tree` (the model's own
module tree, and where an optimization's seams landed), `hook.on_components`
(an authored architecture diagram, lit by the entry points a run actually
called).

## Rules the recorders follow

These came from getting them wrong first.

- **One arm per process.** Two arms in one process share allocator state and
  clocks.
- **Warm before you profile.** The first call carries autotune, lazy init and
  graph capture; profiling it once reported 29,000 kernel launches where the
  settled answer was 2,742.
- **Time and profile in separate passes.** A profiler perturbs the run it
  watches.
- **Report TTFT and decode separately.** Never blended into one number.
- **Two baselines**: the host as shipped, and the host after `torch.compile` +
  graph capture. The gate is the compiled one.
- **Headline is a median.** min-of-N is a lower bound, in a footnote.
- **Score parity against the original baseline**, never an intermediate arm.
- **Real inputs.** Random tensors hide calibration bugs.
- **A refusal is a result.** A seam that does not pay is recorded with its
  reason, not dropped.
- **Never estimate what a profiler can measure.** Utilisation derived from
  tensor shapes is off by five to ten times.

## Layout

```
demokit/
  hook.py                   the recorders: tokens, kernels, module tree, diagram
  record.py                 one arm per process; --host picks an adapter
  hosts/                    8 host adapters (transformers, diffusers, robot policies)
  pipelines/                recorders for vLLM, SGLang, Wan2.2, Qwen3-VL
  compose/live_compose.py   the run demo: the answer arriving, and its rate
  compose/explain_compose.py a published mechanism, drawn from a spec
  compose/styles_compose.py one recording, four visual languages
  compose/race_compose.py   stream / video / arch / runtime / diagram panes
  compose/canvas.py         the supersampled surface, and the easing
  compose/palettes.py       six colour systems, addressed by role
  explainers/*.json         the paper specs
  diagrams/*.json           authored architecture layouts, wired to entry points
examples/runs/              20 real recordings; a first cut needs no GPU
.ai/skills/                 record-a-demo, explain-a-speedup
docs/PROTOCOL.md            the run-directory contract and the adapter protocols
```

## License

MIT.
