"""The same recording, on SGLang instead of vLLM.

Two things differ from `serve_rollout_vllm.py`, and both are properties
of the engine rather than of the structure layer.

SGLang's scheduler is a **spawn** subprocess, so a hook installed in this
interpreter never arrives. The attach rides in through a `sitecustomize`
module written to a directory put on `PYTHONPATH`, so every child
interpreter installs it itself, patching the same-named
`ModelRunner.load_model` — after the weights load, before SGLang's own
CUDA graph capture.

And SGLang captures raw CUDA graphs rather than compiling with dynamo,
which is the interesting part: the same seats have to survive two
completely different graph mechanisms.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import tempfile
import time

SITE = '''
import os
if os.environ.get("FRT_ATTACH") == "1":
    import sys, importlib
    if os.environ.get("FRT_WORKTREE"):
        sys.path.insert(0, os.environ["FRT_WORKTREE"])
    import torch
    from torch import nn
    import sglang.srt.model_executor.model_runner as mr
    from flash_rt.structures import swap as swap_mod

    impl = importlib.import_module(
        "flash_rt.structures.impls.linear_proj."
        + os.environ.get("FRT_SCHEME", "nvfp4_dynamic"))
    SEATS = tuple(s for s in os.environ.get("FRT_SEATS", "").split(",") if s)
    ONEWAY = os.environ.get("FRT_ONEWAY") == "1"

    class Seat(nn.Module):
        def __init__(self, seam):
            super().__init__()
            self.seam = seam

        def forward(self, x, *a, **kw):
            return self.seam(x), None

    _orig = mr.ModelRunner.load_model

    def load_model(self, *a, **kw):
        _orig(self, *a, **kw)
        swaps, freed, refused = {}, 0, 0
        for name, mod in list(self.model.named_modules()):
            if not any(name.endswith(s) for s in SEATS):
                continue
            try:
                bound = impl.bind_proj_seam({"w": mod.weight.data})
            except Exception:
                refused += 1
                continue
            swaps[name] = Seat(bound[0] if isinstance(bound, tuple)
                               else bound)
            if ONEWAY:
                freed += mod.weight.numel() * mod.weight.element_size()
                mod.weight.data = torch.empty(0, device=mod.weight.device,
                                              dtype=mod.weight.dtype)
        torch.cuda.empty_cache()
        self.model.eval()
        swap_mod.attach(self.model, swaps)
        print(f"[attach] {len(swaps)} seats, {refused} refused, "
              f"{freed/1e9:.2f} GB freed", flush=True)
        with open(os.environ["FRT_ATTACH_REPORT"], "w") as f:
            json.dump({"seats": len(swaps), "refused": refused,
                       "freed_gb": round(freed / 1e9, 2)}, f)
    import json
    mr.ModelRunner.load_model = load_model
'''

PROMPT = ("Explain, in one paragraph, why a GPU is faster than a CPU at "
          "matrix multiplication.")


def inject(scheme, seats, oneway):
    d = pathlib.Path(tempfile.mkdtemp(prefix="frt_sglang_"))
    (d / "sitecustomize.py").write_text(SITE)
    report = d / "attach.json"
    os.environ["PYTHONPATH"] = f"{d}:{os.environ.get('PYTHONPATH', '')}"
    os.environ["FRT_ATTACH"] = "1"
    os.environ["FRT_SCHEME"] = scheme
    os.environ["FRT_SEATS"] = seats
    os.environ["FRT_ONEWAY"] = "1" if oneway else "0"
    os.environ["FRT_ATTACH_REPORT"] = str(report)
    return report


def stream_once(engine, prompt, tokens):
    """One request, stamped as SGLang hands each piece back."""
    sp = {"max_new_tokens": tokens, "temperature": 0.0}
    ev, seen, t0 = [], 0, time.perf_counter()
    for chunk in engine.generate(prompt, sp, stream=True):
        text = chunk["text"]
        n = chunk["meta_info"].get("completion_tokens", 0)
        piece = text[seen:] if isinstance(text, str) else ""
        if not piece and n <= len(ev):
            continue
        ev.append({"i": len(ev), "t": round(time.perf_counter() - t0, 5),
                   "text": piece})
        seen = len(text)
    ids = chunk["meta_info"].get("output_token_logprobs")
    return ev, time.perf_counter() - t0, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=("base", "attach"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seats", default="qkv_proj,o_proj,gate_up_proj,"
                                       "down_proj")
    ap.add_argument("--scheme", default="nvfp4_dynamic")
    ap.add_argument("--oneway", action="store_true")
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--raw", action="store_true",
                    help="send the prompt as raw text instead of through "
                         "the model's chat template")
    ap.add_argument("--label", default=None)
    ap.add_argument("--sub", default=None)
    args = ap.parse_args()

    report_path = None
    if args.arm == "attach":
        report_path = inject(args.scheme, args.seats, args.oneway)

    import sglang

    engine = sglang.Engine(model_path=args.model,
                           mem_fraction_static=args.mem_fraction,
                           disable_radix_cache=True,
                           log_level="error")
    prompt = PROMPT
    if not args.raw:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stream_once(engine, prompt, 8)                          # warm
    best = None
    for _ in range(args.repeats):
        ev, wall, chunk = stream_once(engine, prompt, args.tokens)
        if best is None or wall < best[1]:
            best = (ev, wall, chunk)
    ev, wall, chunk = best
    ttft = ev[0]["t"] * 1e3
    rate = (len(ev) - 1) / (ev[-1]["t"] - ev[0]["t"])
    meta_info = chunk["meta_info"]

    d = out / args.arm
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "stream", "color": "stock" if args.arm == "base"
        else "accent",
        "label": args.label or ("SGLang 0.5.13, as shipped"
                                if args.arm == "base"
                                else "+ FlashRT structures"),
        "sub": args.sub or ("raw CUDA graph capture" if args.arm == "base"
                            else "seats bound at load, inside SGLang's "
                                 "own captured graph"),
        "prompt": PROMPT, "arm": args.arm, "model": args.model,
        "ttft_ms": round(ttft, 1), "decode_tok_s": round(rate, 1),
        "done_s": round(ev[-1]["t"], 4), "wall_s": round(wall, 4),
        "n_tokens": len(ev),
        "completion_tokens": meta_info.get("completion_tokens"),
    }
    if report_path and report_path.exists():
        meta.update(json.loads(report_path.read_text()))
    (d / "events.json").write_text(json.dumps(
        {"meta": meta, "events": ev}, indent=1))
    (d / "text.json").write_text(json.dumps(chunk["text"]))
    print(f"[{args.arm}] TTFT {ttft:.1f} ms, decode {rate:.1f} tok/s, "
          f"{len(ev)} tokens, done {ev[-1]['t']:.3f} s", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
