"""Record the reference implementation's answer, token by token.

The serving films race an engine against itself. This records the third
pane those films are read against: the same checkpoint under
`transformers`' own `generate`, bf16, nothing applied — the implementation
the weights were published for.

It writes the same `kind: "stream"` arm directory the serving recorder
writes at concurrency 1, with the same prompt through the same chat
template, so a film can put the three side by side and be comparing one
thing.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch

if os.environ.get("FRT_WORKTREE"):
    sys.path.insert(0, os.environ["FRT_WORKTREE"])

#: the serving recorder's first prompt, so the panes line up
PROMPT = ("Explain, in one paragraph, why a GPU is faster than a CPU at "
          "matrix multiplication.")


class Stamper:
    """A logits processor that says when each token arrived."""

    def __init__(self):
        self.t0 = None
        self.stamps: list[float] = []

    def arm(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        self.stamps = []
        return self

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.stamps.append(time.perf_counter() - self.t0)
        return scores


def split_rate(stamps):
    """TTFT first, then decode over the tail — never blended."""
    if len(stamps) < 3:
        return stamps[0] * 1e3, 0.0
    span = stamps[-1] - stamps[0]
    return stamps[0] * 1e3, (len(stamps) - 1) / span if span > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="the arm's directory")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--label", default="the reference implementation")
    ap.add_argument("--sub", default="transformers, bf16, nothing applied")
    ap.add_argument("--color", default="stock")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    # NOTE: on the Jetson AGX Thor the serving films were recorded on,
    # this load wants roughly 10 GB more than vLLM needs for the same
    # checkpoint, and it does not draw on the nvmap carveout that a big
    # previous run leaves behind — so it fits on a freshly booted board
    # and not on a busy one, while vLLM fits on either. Run it behind
    # `harness/memguard.sh` and see RECORDING-THOR.md §20-21 before
    # concluding anything from a failure here.
    from transformers import (AutoTokenizer,
                              Qwen3_5MoeForConditionalGeneration)

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    # placed straight onto the device, never loaded to host memory and
    # copied: this board's memory is unified, so `.to("cuda")` after a
    # host load would need 67 GB of weights twice at the same instant
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda:0"}, attn_implementation=args.attn).eval()
    print(f"loaded in {time.perf_counter() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on device",
          flush=True)

    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, return_tensors="pt",
        enable_thinking=False)
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to("cuda")
    L = int(ids.shape[1])
    print(f"prompt {L} tokens", flush=True)

    stamp = Stamper()

    def run(stamper=None):
        with torch.no_grad():
            return model.generate(
                ids, max_new_tokens=args.tokens, do_sample=False,
                logits_processor=[stamper] if stamper else None)

    times = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    clean = statistics.median(times)

    g = run(stamper=stamp.arm())
    stamps = stamp.stamps
    drift = abs(stamps[-1] - clean) / clean
    print(f"clean {clean * 1e3:.0f} ms, stamped {stamps[-1] * 1e3:.0f} ms, "
          f"drift {drift * 100:.1f}%", flush=True)

    gen = g[0, L:L + len(stamps)].tolist()
    pieces = [tok.decode([t], skip_special_tokens=True) for t in gen]
    ttft, rate = split_rate(stamps)
    d = pathlib.Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "stream", "label": args.label, "sub": args.sub,
            "color": args.color, "prompt": args.prompt,
            "ttft_ms": round(ttft, 1), "decode_tok_s": round(rate, 1),
            "done_s": round(stamps[-1], 4), "n_tokens": len(stamps),
            "e2e_ms": round(clean * 1e3, 1), "stamp_drift": round(drift, 4),
            "model": args.model, "attn_implementation": args.attn,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0)}
    (d / "events.json").write_text(json.dumps(
        {"meta": meta,
         "events": [{"i": i, "t": round(s, 4), "text": p}
                    for i, (s, p) in enumerate(zip(stamps, pieces))]},
        indent=1))
    (d / "tokens.json").write_text(json.dumps(
        {"prompt_ids": [ids[0].tolist()], "generated": [gen]}))
    print(f"wrote {d}  TTFT {ttft:.1f} ms  {rate:.1f} tok/s  "
          f"done {stamps[-1]:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
