---
name: explain-a-speedup
description: Turn a measured speedup into a film that answers *why*, in one picture a person reads without a legend. Use after a demo film has shown that one arm finished first.
---

# Explaining a speedup

A demo film shows one arm finishing first. Everyone understands it, because
there is nothing to parse. This is the follow-up, and it is much harder:
saying **why**, and having that be as easy to read.

## The rule that decides everything

**One idea. Everything else is cut.**

Not "one idea per section" — one idea in the whole film. A viewer holds four
or five elements before they stop reading, and a chart is not one element,
it is a language they have to learn first.

The corollary is brutal and correct: a picture that is *more true* is often
*less readable*, and readable wins. Measure everything; draw one thing.

## What a human cannot read — learned the hard way

Five versions of this film were built and rejected. Each was more accurate
than the last, and four of them were unreadable. Do not rebuild them:

| version | why it failed |
|---|---|
| the model's module tree, lit by forward hooks | that is *the host's* architecture, not the runtime's — and a hand-written pipeline has no module tree at all |
| a kernel taxonomy: launches bucketed by family and origin | a better measurement, and still a chart. Bars are not a picture |
| the framework's own architecture diagram, boxes lighting up | reads as an org chart. The viewer learns the boxes, not the point |
| utilisation meters, % of peak | the measurement said the two arms drive the chip *equally hard* (18% and 20%). A fill bar would have implied otherwise |
| two heat maps of different area | area is a comparison the eye has to *measure*, and "smaller is better" is not intuitive |

What finally worked: **the same block of work, cut into pieces.** Both rows
the same length, one cut five times finer. No legend. Same work, fewer
trips, and therefore more work per trip — three claims from one picture, and
the third is a consequence of the first two rather than a fourth thing to
remember.

The pattern to reuse is not that picture. It is: **find the physical
metaphor the numbers already are.** A kernel launch really is a trip to
memory. Fusing kernels really is cutting the same work into fewer pieces.
When the metaphor is the mechanism, nothing needs explaining.

## Measure first, draw second

The numbers are load-bearing. Every one of these was wrong once:

- **Profile after the eighth call, never the first.** A cold call carries
  autotune, lazy init and graph capture. Profiling it reported 29,000
  launches where the truth was 2,742, and that number looked reasonable.
- **Take the clock with the profiler off**, in a separate loop, so the
  instrument is never inside the number it reports.
- **Settle before timing**: run until two consecutive windows of twenty
  agree within one percent *and* twenty seconds have passed, then take a
  median of fifty. Report p90 and min as footnotes, never as the headline.
- **Two baselines.** The host as shipped is what a reader recognises; the
  compiled host is what the claim has to survive. Name both.
- **Never estimate what a profiler can measure.** Utilisation derived from
  tensor shapes is off by five to ten times. Nsight Compute reports
  `sm__throughput` and `gpu__dram_throughput` as percent of peak; weight
  them by each kernel's own duration.
- **Say what you did not measure.** A pane with no measurement behind it
  prints "not measured", it does not draw a plausible bar.

## The templates, and what each answers

| painter | the question it answers |
|---|---|
| `compose/trips_compose.py` | why is it faster — the one-idea film |
| `compose/race_compose.py` `runtime` | what did each arm ask the GPU to do |
| `compose/race_compose.py` `arch` | where did the kernels land in the host's tree |
| `compose/race_compose.py` `diagram` | which door through the framework did this run take |
| `compose/race_compose.py` `video`/`stream`/`stream_batch` | the demo film itself |
| `compose/styles_compose.py` | the same stream recording, drawn four other ways |
| `compose/explain_compose.py` | a mechanism from a paper, when there is nothing to record |

Start from `trips_compose`. The others exist because they were built; they
answer narrower questions and they are charts.

## Choosing the drawing, and the ink

A recording does not imply a picture. The same `events.json` — the wall time
each token arrived — carries four different arguments depending on how it is
drawn, and picking one is part of the work rather than a detail after it:

| style | what it argues | reach for it when |
|---|---|---|
| `curve` | rate: tokens against time, the slope *is* the speed | the claim is tok/s |
| `bars` | finishing | the audience should not have to read an axis |
| `dots` | identity: one cell per token, the same cells in both arms | *the same output* is the claim, and speed is second |
| `ribbon` | density: the same ticks on the same track | a fixed amount of work fitting into less time |

Try more than one. It costs seconds:

```bash
demokit looks RUNDIR --sheet looks.png                # 4 styles x 6 palettes
demokit looks RUNDIR --style dots --palette paper --out a.webm
```

Then open the sheet and choose, the same way you would choose between two
sentences.

**Never name a colour in a painter.** Ask `palettes.py` for a role — the
ground, the rule, the reading text, `stock` / `compiled` / `ours` / `native`.
One flag then restyles the whole film, and the fix for a light palette needing
darker arm colours lands in one place. It also stops the failure this kit hit
once already: two modules disagreeing about whether the accent is called
`ours` or `accent`, and half a film drawing in the wrong ink.

Palettes carry an argument too. `paper` for slides and print, `phosphor` when
the subject is a runtime close to the metal, `blueprint` under a diagram,
`ember` for heat and throughput, `mono` when nothing should compete with the
one accent, `midnight` as the default. A palette that fights the subject is a
real cost even when every number on the page is right.


## When the source is a paper, not a run

Some things worth explaining were never yours to measure: how a system
allocates, decides, or caches. `explain_compose.py` draws those from an
authored spec in `demokit/explainers/`, and the discipline transfers exactly:

- the spec's `source` line prints on every panel, so a reader always knows
  the numbers are the paper's and not yours;
- nothing on the page is measured — if a claim needs a measurement, it needed
  a recording instead;
- a concrete scenario is labelled an illustration, and every count the panel
  prints is derived from the spec rather than typed into a caption.

Then **say what it came to, and say it where the mechanism is**. A film that
explains without landing the outcome feels like a preamble — but an outcome
on its own page is worse, because the reader has to carry the diagram in
their head to make sense of it. A big number is a token to read; a bar is a
length to see. So put a small chart in the mechanism panel's own column, and
give the results page bars with a baseline at 1x rather than three lonely
numbers.

Charts beside a diagram are **derived from that diagram** (`derive` in the
spec) — the tokens in the drawn tree, the queue in the drawn cards. A number
typed in beside a picture will eventually contradict it; one computed from it
cannot.

The discipline there is the same one as everywhere: **report the number that
did not move.** FlashRT's own explainer puts 17.8% against 20.0% utilisation
on the page and says in as many words that the chip is not being driven
harder — the win is that far less was asked of it. A results page that
quietly dropped that would be selling rather than explaining, and the
audience for these films can tell.

Draw the mechanism its own authors put at the centre of the design, and draw
it as *their* idea. A page that quietly turns into a comparison with your own
work is a different genre, and it is the one that makes people defensive
instead of interested.


## Work from the assets, not from a GPU

`examples/runs/` carries real recordings for every painter, including four
arms of pi0.5 with launch counts, wall times and Nsight utilisation. **A
first cut needs no model and no device.** Draw from those, get the shape
right, and only then record the run that is actually being explained.

Recording and drawing are separate on purpose: changing the picture costs
seconds, because `events.json` is the whole raw material and no model has to
run again. Use that. Ten drafts is normal.

## Look at what you drew

You can read images. Render a single frame to a PNG, open it, and ask what a
person who has never seen this project would take away from it. Then cut
something. That loop is what produced the version that worked, and it is
faster than reasoning about the layout in the abstract.

Two failure modes it catches immediately: text sitting on top of other text,
and an encoding that needed a legend you forgot you had invented.
