"""Record one Qwen3-VL-8B answer per arm, with the wall time of every token.

Same image, same prompt, greedy — three ways of running the same
transformers checkpoint in one process:

    eager    AutoModelForImageTextToText.generate, as shipped
    static   the same generate with a static cache and a compiled,
             graph-captured decode step — the harder reference, and the
             form a deployment actually runs
    attach   the same host with FlashRT structures attached
             (auto_swaps over decoder_ffn + attention projections, the
             lm_head bound through the impl's public binder) and the
             whole decode step taken over by structures.decode_loop

TTFT and decode rate are reported separately: a blended tokens/second
hides which half moved.  All three token streams are compared; the film
only claims what the comparison shows.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

#: point $FRT_WORKTREE at a FlashRT checkout to run against it in place;
#: without it the installed package is used.
if os.environ.get("FRT_WORKTREE"):
    sys.path.insert(0, os.environ["FRT_WORKTREE"])

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
#: any image will do; $RACE_IMAGE points at the one to use.
IMAGE = os.environ.get("RACE_IMAGE")
PROMPT = ("Describe this scene in detail: what objects are on the table, "
          "and what is the robot arm doing?")


def real_image():
    from PIL import Image
    if not IMAGE:
        raise SystemExit("set $RACE_IMAGE to the image to ask about")
    return Image.open(IMAGE).convert("RGB")


class Stamper:
    """A logits processor is called once per generated token, so it is
    the host's own per-token clock — no wrapper around generate needed."""

    def __init__(self):
        self.t0 = None
        self.stamps = []

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
    """TTFT is the first stamp; the decode rate is the steady tail."""
    if len(stamps) < 3:
        return stamps[0] * 1e3, 0.0
    ttft = stamps[0]
    steps = np.diff(np.asarray(stamps))
    return ttft * 1e3, float(1.0 / np.median(steps))


def write_arm(out_dir, name, label, sub, color, stamps, pieces, meta_extra,
              image=None):
    d = pathlib.Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    if image is not None:
        image.save(d / "image.png")
    ttft, rate = split_rate(stamps)
    meta = {
        "kind": "stream", "label": label, "sub": sub, "color": color,
        "prompt": PROMPT,
        "ttft_ms": round(ttft, 1),
        "decode_tok_s": round(rate, 1),
        "done_s": round(stamps[-1], 4),
        "n_tokens": len(stamps),
    }
    meta.update(meta_extra)
    events = [{"i": i, "t": round(s, 4), "text": p}
              for i, (s, p) in enumerate(zip(stamps, pieces))]
    (d / "events.json").write_text(json.dumps(
        {"meta": meta, "events": events}, indent=1))
    print(f"  wrote {d}  TTFT {ttft:.1f} ms  decode {rate:.1f} tok/s  "
          f"done {stamps[-1]:.2f} s", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="directory to write one sub-directory per arm")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt", default=None,
                    help="override the question asked about the image")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    N = args.tokens
    global PROMPT
    if args.prompt:
        PROMPT = args.prompt

    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              CompileConfig)
    from flash_rt import structures
    from flash_rt.structures.impls.linear_proj import w8a16_static
    from flash_rt.structures.swap import attach as swap_attach

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    image = real_image()
    inputs = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image},
                                      {"type": "text", "text": PROMPT}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt")
    inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    L = int(inputs["input_ids"].shape[1])
    print(f"prompt {L} tokens (image included)", flush=True)

    stamp = Stamper()
    out = pathlib.Path(args.out)
    summary = {}

    def gen(**kw):
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=N,
                                  do_sample=False, **kw)

    def timed(fn, n):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    def first_diff(ids):
        """How many leading tokens this arm shares with the host."""
        a, b = ids[0, L:], base[0, L:]
        n = min(a.shape[0], b.shape[0])
        ne = (a[:n] != b[:n]).nonzero()
        return int(ne[0]) if ne.numel() else int(n)

    def pieces_of(ids):
        new = ids[0, L:].tolist()
        return [proc.tokenizer.decode([t], skip_special_tokens=True)
                for t in new]

    def record(name, label, sub, color, run, extra):
        """One clean timed median, then one stamped run whose own total
        is checked against it — the timeline is a recording, not a
        reconstruction, and this is the check that says so."""
        clean = timed(run, args.repeats)
        ids = run(stamper=stamp.arm())
        st = list(stamp.stamps)
        drift = abs(st[-1] - clean) / clean
        print(f"{name}: clean {clean*1e3:.1f} ms, stamped "
              f"{st[-1]*1e3:.1f} ms, drift {drift*100:.1f}%", flush=True)
        extra = dict(extra, e2e_ms=round(clean * 1e3, 1),
                     stamp_drift=round(drift, 4))
        summary[name] = write_arm(out, name, label, sub, color, st,
                                  pieces_of(ids), extra, image)
        return ids

    # ---- eager: the host as shipped ----------------------------------
    def run_eager(stamper=None):
        return gen(logits_processor=[stamper] if stamper else None)

    base = run_eager()
    assert torch.equal(base, run_eager()), "host is not deterministic"
    n_eff = int(base.shape[1] - L)
    print(f"host produced {n_eff} tokens of {N}", flush=True)
    if n_eff != N:
        # the host stopped at EOS; the loop arm has no EOS rule, so hold
        # every arm to the same count instead of letting one run on
        print(f"host stopped early; racing {n_eff} tokens", flush=True)
        N = n_eff
    record("eager", "Qwen3-VL-8B, as shipped", "eager PyTorch", "stock",
           run_eager, {"tokens_match_host": 1.0})

    # ---- static: static cache + compiled, graph-captured step ---------
    def run_static(stamper=None):
        return gen(cache_implementation="static",
                   compile_config=CompileConfig(mode="reduce-overhead"),
                   logits_processor=[stamper] if stamper else None)

    ids = run_static()                                     # warm/compile
    m_static = float((ids[0, L:L + n_eff]
                      == base[0, L:L + n_eff]).float().mean())
    print(f"static cache token match vs host: {m_static:.4f}", flush=True)
    record("static", "the same host, compiled",
           "static cache + graph capture", "compiled", run_static,
           {"tokens_match_host": round(m_static, 4),
            "agrees_for_first": first_diff(ids)})
    torch._dynamo.reset()

    # ---- attach: auto_swaps + the whole-step decode loop --------------
    def fwd():
        with torch.no_grad():
            model(**inputs)

    plan = structures.auto_swaps(model, fwd,
                                 structures=("decoder_ffn", "linear_proj"),
                                 scheme="w8a16_decode", verbose=False)
    swap_attach(model, plan.swaps, observe=plan.observed,
                revert=plan.revert)
    head = model.lm_head
    model.lm_head = w8a16_static.bind_proj_seam(
        {"w": head.weight.detach()}, original=head)
    print(f"seams bound: {len(plan.swaps)} + lm_head", flush=True)

    loop = structures.decode_loop(model, max_len=L + N + 32)
    loop.generate_from(inputs, N)                          # warm/capture

    def run_attach(stamper=None):
        if stamper is None:
            return loop.generate_from(inputs, N)
        tail = loop._decode_tail

        def stamped(max_new_tokens, toks):
            torch.cuda.synchronize()
            stamper.stamps.append(time.perf_counter() - stamper.t0)
            for _ in range(max_new_tokens - 1):
                loop._graph.replay()
                torch.cuda.synchronize()
                stamper.stamps.append(time.perf_counter() - stamper.t0)
                toks.append(loop._cur.clone())

        loop._decode_tail = stamped
        try:
            return loop.generate_from(inputs, N)
        finally:
            loop._decode_tail = tail

    # the protocol gate for an LLM arm: teacher-forced same-token rate.
    # A free run cascades — one tie-break at token k makes every token
    # after it incomparable — so agreement is judged step by step, on
    # the host's own sequence, and the free run is reported beside it.
    with torch.no_grad():
        loop.generate_from(inputs, 1)
        hits = 0
        for k in range(n_eff - 1):
            loop._cur.copy_(base[:, L + k:L + k + 1])
            loop._pos.fill_(L + k)
            lg = loop._step(loop._cur, loop._pos)
            hits += int(lg.float().argmax(-1).item()
                        == int(base[0, L + k + 1]))
    tf = hits / (n_eff - 1)
    print(f"teacher-forced same-token: {tf:.4f}", flush=True)

    reps = [loop.generate_from(inputs, N) for _ in range(3)]
    same = all(torch.equal(r, reps[0]) for r in reps)
    print(f"loop repeat identical (chain x3): {same}", flush=True)
    m_attach = float((reps[0][0, L:L + n_eff]
                      == base[0, L:L + n_eff]).float().mean())
    print(f"attach token match vs host: {m_attach:.4f}", flush=True)
    record("attach", "+ FlashRT structures", "auto_swaps + decode_loop",
           "accent", run_attach,
           {"tokens_match_host": round(m_attach, 4),
            "teacher_forced_same_token": round(tf, 4),
            "agrees_for_first": first_diff(reps[0]),
            "repeat_identical": bool(same), "seams": len(plan.swaps)})

    summary["_ladder"] = {
        "prompt_tokens": L, "new_tokens": N,
        "decode_vs_eager": round(summary["attach"]["decode_tok_s"]
                                 / summary["eager"]["decode_tok_s"], 3),
        "decode_vs_static": round(summary["attach"]["decode_tok_s"]
                                  / summary["static"]["decode_tok_s"], 3),
        "ttft_vs_eager": round(summary["eager"]["ttft_ms"]
                               / summary["attach"]["ttft_ms"], 3),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["_ladder"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
