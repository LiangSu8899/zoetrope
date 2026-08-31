"""Record one answer per arm from Qwen3.6-35B-A3B, token by token.

A 35B MoE in bf16 is 67 GB of weights.  A 32 GB consumer card cannot
hold it, so before anything can be raced the expert banks are adopted to
NVFP4 — `structures.quantize_on_adopt`, on the CPU, before the model
ever moves to the device.  That is the precondition, not the claim.  All
three arms then run the same adopted weights:

    host    the transformers host's own `generate`
    loop    + FlashRT structures: the gated-delta core fused, and the
            whole decode step compiled and captured by decode_loop
    mtp     + the checkpoint's own MTP draft head, running speculative
            decode on top of the same loop

The MTP arm emits a whole accepted run at once, which is what
speculative decoding actually does, so its pane fills in bursts and its
acceptance length is reported next to its rate.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import gc

import numpy as np
import torch

#: point $FRT_WORKTREE at a FlashRT checkout to run against it in place;
#: without it the installed package is used.
if os.environ.get("FRT_WORKTREE"):
    sys.path.insert(0, os.environ["FRT_WORKTREE"])

#: the local checkpoint directory; the MTP draft head is loaded from it
#: too, so it has to be the full repository and not just a model id.
P = os.environ.get("QWEN36_35B_PATH", "")
QUESTION = ("Explain, for someone who writes CUDA kernels, why a "
            "mixture-of-experts model can be large and still decode "
            "fast.")


class Stamper:
    def __init__(self):
        self.t0 = None
        self.stamps = []
        self.counts = []

    def arm(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        self.stamps, self.counts = [], []
        return self

    def mark(self, n=1):
        torch.cuda.synchronize()
        t = time.perf_counter() - self.t0
        self.stamps.append(t)
        self.counts.append(n)

    def __call__(self, input_ids, scores):     # a logits processor
        self.mark(1)
        return scores

    def expand(self):
        """One timestamp per token; a burst shares its round's time."""
        out = []
        for t, n in zip(self.stamps, self.counts):
            out.extend([t] * n)
        return out


class _StampGraph:
    """A CUDA graph that says when its replay finished."""

    def __init__(self, g, cb):
        object.__setattr__(self, "_g", g)
        object.__setattr__(self, "_cb", cb)

    def replay(self):
        self._g.replay()
        self._cb()

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_g"), k)


def split_rate(stamps):
    """TTFT first, then decode throughput over the tail.

    Not a median of per-token gaps: a speculative arm emits a whole
    accepted run at one instant, so most of its gaps are zero. Tokens
    after the first, divided by the time they took, is the rate that
    holds for both shapes."""
    if len(stamps) < 3:
        return stamps[0] * 1e3, 0.0
    span = stamps[-1] - stamps[0]
    return stamps[0] * 1e3, (len(stamps) - 1) / span if span > 0 else 0.0


def write_arm(out_dir, name, label, sub, color, stamps, pieces, extra):
    d = pathlib.Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    ttft, rate = split_rate(stamps)
    meta = {"kind": "stream", "label": label, "sub": sub, "color": color,
            "prompt": QUESTION, "ttft_ms": round(ttft, 1),
            "decode_tok_s": round(rate, 1),
            "done_s": round(stamps[-1], 4), "n_tokens": len(stamps)}
    meta.update(extra)
    (d / "events.json").write_text(json.dumps(
        {"meta": meta,
         "events": [{"i": i, "t": round(s, 4), "text": p}
                    for i, (s, p) in enumerate(zip(stamps, pieces))]},
        indent=1))
    print(f"  wrote {d}  TTFT {ttft:.1f} ms  {rate:.1f} tok/s  "
          f"done {stamps[-1]:.2f} s", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="directory to write one sub-directory per arm")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--prompt", default=None, help="override the question")
    ap.add_argument("--no-gdn", action="store_true",
                    help="skip the W4A4 gated-delta fusion; the experts "
                         "are still NVFP4, because otherwise nothing fits")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model emit its <think> block; off by "
                         "default, because a truncated thinking dump is "
                         "not an answer a pane can show")
    ap.add_argument("--ksweep", default="6,4,3,2")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    N = args.tokens
    global QUESTION
    if args.prompt:
        QUESTION = args.prompt

    from transformers import (AutoTokenizer,
                              Qwen3_5MoeForConditionalGeneration)
    from flash_rt import structures
    from flash_rt.structures.impls.decode_loop.mtp_speculative import (
        MtpDraftHead)
    from flash_rt.structures.impls.decode_loop.whole_step import _find_stack
    from flash_rt.structures.swap import attach as swap_attach

    if not P:
        raise SystemExit("set $QWEN36_35B_PATH to the checkpoint directory")
    tok = AutoTokenizer.from_pretrained(P)
    t0 = time.perf_counter()
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        P, dtype="auto", device_map=None, low_cpu_mem_usage=True)
    print(f"loaded in {time.perf_counter()-t0:.0f} s", flush=True)
    report = structures.quantize_on_adopt(model, "moe_experts_nvfp4")
    print(f"adopted: {report.summary()}", flush=True)
    model = model.to("cuda")
    print(f"on device: {torch.cuda.memory_allocated()/2**30:.1f} GiB",
          flush=True)

    enc = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        add_generation_prompt=True, return_tensors="pt",
        enable_thinking=bool(args.thinking))
    ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to("cuda")
    L = int(ids.shape[1])
    print(f"prompt {L} tokens", flush=True)

    stamp = Stamper()
    out = pathlib.Path(args.out)
    summary = {}

    def timed(fn, n):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    def pieces_of(g):
        return [tok.decode([t], skip_special_tokens=True)
                for t in g[0, L:L + N].tolist()]

    def record(name, label, sub, color, run, extra):
        clean = timed(run, args.repeats)
        g = run(stamper=stamp.arm())
        st = stamp.expand()
        drift = abs(st[-1] - clean) / clean
        print(f"{name}: clean {clean*1e3:.0f} ms, stamped "
              f"{st[-1]*1e3:.0f} ms, drift {drift*100:.1f}%", flush=True)
        summary[name] = write_arm(
            out, name, label, sub, color, st, pieces_of(g),
            dict(extra, e2e_ms=round(clean * 1e3, 1),
                 stamp_drift=round(drift, 4)))
        return g

    # ---- the host's own generate, on the adopted weights -------------
    def run_host(stamper=None):
        with torch.no_grad():
            return model.generate(
                ids, max_new_tokens=N, do_sample=False,
                logits_processor=[stamper] if stamper else None)

    base = run_host()
    assert torch.equal(base, run_host()), "host is not deterministic"
    n_eff = int(base.shape[1] - L)
    print(f"host produced {n_eff} tokens of {N}", flush=True)
    if n_eff != N:
        print(f"host stopped early; racing {n_eff} tokens", flush=True)
        N = n_eff
    # for the record only: this environment has no flash-linear-attention,
    # so the host's gated-delta layers are on the PyTorch fallback. It is
    # not the baseline the film uses — see below.
    t_plain = timed(run_host, args.repeats)
    print(f"host, plain: {t_plain*1e3:.0f} ms for {N} tokens", flush=True)

    # ---- fuse the gated-delta core, then set the baseline -------------
    # The baseline pane runs with the fused core already attached. That
    # gives away part of our own margin on purpose: without it the
    # baseline would be measured on a fallback kernel path this machine
    # simply has not installed, and the comparison would be against a
    # weakness of the environment rather than of the host.
    with torch.no_grad():
        pre = model(input_ids=ids, use_cache=True)
    step_ids = ids[:, -1:]

    def decode_step():
        with torch.no_grad():
            model(input_ids=step_ids, past_key_values=pre.past_key_values,
                  use_cache=True)

    if args.no_gdn:
        class _Empty:
            swaps = ()
            observed = ()
        plan = _Empty()
        print("gated-delta cores left with the host", flush=True)
    else:
        plan = structures.auto_swaps(model, decode_step,
                                     structures=("gated_delta_core",),
                                     scheme="w4a4_decode", verbose=False)
        swap_attach(model, plan.swaps, observe=plan.observed,
                    revert=plan.revert)
        print(f"gated-delta cores fused: {len(plan.observed)}", flush=True)
    del pre
    torch.cuda.empty_cache()

    base_g = run_host()
    # the fused core is a different arithmetic, so the host can stop at a
    # different token; race whichever count both of them reached
    n_g = int(base_g.shape[1] - L)
    if n_g != N:
        print(f"fused host stopped at {n_g}; racing {min(N, n_g)}",
              flush=True)
        N = n_eff = min(N, n_g)
    m_fused = float((base_g[0, L:L + N]
                     == base[0, L:L + N]).float().mean())
    print(f"fused core vs plain host stream: {m_fused:.4f}", flush=True)
    base = base_g          # the baseline pane is what everything else
                           # is compared against from here on
    record("host", "the host's own generate",
           "transformers loop, NVFP4 experts", "stock", run_host,
           {"tokens_match_host": 1.0,
            "plain_host_ms": round(t_plain * 1e3, 1),
            "fused_vs_plain_stream": round(m_fused, 4),
            "gdn_cores": len(plan.observed)})

    def build_loop(compile_step):
        lp = structures.decode_loop(model, max_len=L + N + args.k + 24,
                                    compile_step=compile_step,
                                    compile_prefill=False)
        lp.generate(ids, N)                                 # warm/capture
        return lp

    def loop_gate(lp):
        """Teacher-forced same-token rate over the host's own sequence,
        plus the free run beside it."""
        with torch.no_grad():
            lp.generate(base[:, :L], 1)
            hits = 0
            for k in range(n_eff - 1):
                lp._cur.copy_(base[:, L + k:L + k + 1])
                lp._pos.fill_(L + k)
                lg = lp._step(lp._cur, lp._pos)
                hits += int(lg.float().argmax(-1).item()
                            == int(base[0, L + k + 1]))
        g = lp.generate(ids, N)
        free_ = float((g[0, L:L + N] == base[0, L:L + N]).float().mean())
        same = all(torch.equal(lp.generate(ids, N), g) for _ in range(3))
        return hits / (n_eff - 1), free_, same, g

    def loop_runner(lp):
        def run_loop(stamper=None):
            if stamper is None:
                return lp.generate(ids, N)
            tail = lp._decode_tail

            def stamped(max_new_tokens, toks):
                stamper.mark(1)
                for _ in range(max_new_tokens - 1):
                    lp._graph.replay()
                    stamper.mark(1)
                    toks.append(lp._cur.clone())

            lp._decode_tail = stamped
            try:
                return lp.generate(ids, N)
            finally:
                lp._decode_tail = tail
        return run_loop

    # ---- the loop with an uncompiled step: slower, and token-exact ----
    lp0 = build_loop(False)
    tf0, free0, same0, _ = loop_gate(lp0)
    print(f"loop, uncompiled step: teacher-forced {tf0:.4f}, free-run "
          f"{free0:.4f}, repeat identical {same0}", flush=True)
    record("loop_exact", "+ FlashRT structures",
           "whole-step capture, step not compiled", "compiled",
           loop_runner(lp0),
           {"teacher_forced_same_token": round(tf0, 4),
            "tokens_match_host": round(free0, 4),
            "repeat_identical": bool(same0), "compiled_step": False})
    del lp0
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    loop = build_loop(True)

    run_loop = loop_runner(loop)
    tf, free_run, same, g_loop = loop_gate(loop)
    print(f"loop, compiled step: teacher-forced {tf:.4f}, free-run "
          f"{free_run:.4f}, repeat identical {same}", flush=True)
    record("loop", "+ FlashRT structures",
           "fused gated-delta + whole-step capture, step compiled",
           "accent", run_loop,
           {"teacher_forced_same_token": round(tf, 4),
            "tokens_match_host": round(free_run, 4),
            "repeat_identical": bool(same),
            "compiled_step": True,
            "gdn_cores": len(plan.observed)})

    # ---- + MTP: the checkpoint's own draft head ----------------------
    lm = _find_stack(model)
    head = MtpDraftHead(model, lm, P, len(lm.layers))
    loop.enable_mtp(head=head, default_k=args.k, verify_capture=True)

    # A draft chain's worth depends on how much of it the verify pass
    # keeps, and that is a property of this prompt, not of the model.
    # Sweep K, measure, and let the pane be earned: an arm that does not
    # beat the plain loop does not go in the film.
    sweep = [int(k) for k in str(args.ksweep).split(",")]
    trials = {}
    for K in sweep:
        g_k = loop.generate_speculative(ids, N, K=K)
        al_k = float(loop.last_acceptance)
        ex_k = torch.equal(g_k, g_loop)
        rep_k = all(torch.equal(loop.generate_speculative(ids, N, K=K),
                                g_k) for _ in range(2))
        t_k = timed(lambda: loop.generate_speculative(ids, N, K=K),
                    args.repeats)
        trials[K] = {"acceptance_length": round(al_k, 2),
                     "identical_to_greedy": bool(ex_k),
                     "repeat_identical": bool(rep_k),
                     "e2e_ms": round(t_k * 1e3, 1),
                     "tok_s": round(N / t_k, 1)}
        print(f"  K={K}: AL {al_k:.2f}, {N/t_k:.1f} tok/s, identical "
              f"{ex_k}, repeat {rep_k}", flush=True)
    summary["_mtp_sweep"] = trials
    best_k = max(trials, key=lambda k: trials[k]["tok_s"])
    loop_tok_s = summary["loop"]["decode_tok_s"]
    if trials[best_k]["tok_s"] <= loop_tok_s:
        print(f"MTP does not earn a pane: best K={best_k} at "
              f"{trials[best_k]['tok_s']} tok/s against the loop's "
              f"{loop_tok_s}", flush=True)
        summary["_ladder"] = {
            "prompt_tokens": L, "new_tokens": N,
            "loop_vs_host": round(loop_tok_s
                                  / summary["host"]["decode_tok_s"], 3),
            "loop_exact_vs_host": round(
                summary["loop_exact"]["decode_tok_s"]
                / summary["host"]["decode_tok_s"], 3),
            "mtp_verdict": "refused: net negative on this prompt",
        }
        (out / "metrics.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary["_ladder"], indent=1), flush=True)
        return
    args.k = best_k
    g_spec = loop.generate_speculative(ids, N, K=best_k)
    al = loop.last_acceptance
    exact = trials[best_k]["identical_to_greedy"]
    same_s = trials[best_k]["repeat_identical"]

    def run_mtp(stamper=None):
        if stamper is None:
            return loop.generate_speculative(ids, N, K=args.k)
        # the accepted count of a round is the one .item() the round
        # makes; the round's tokens all become known when its verify
        # replay lands, so they share that timestamp
        seen = []
        real_item = torch.Tensor.item

        def item(self):
            v = real_item(self)
            if self.dim() == 0 and self.dtype in (torch.int64, torch.int32):
                seen.append(int(v))
            return v

        vg = loop._vgraph
        if vg is None:
            raise RuntimeError("verify pass is not captured; the round "
                               "clock has nothing to hang on")
        loop._vgraph = _StampGraph(vg, lambda: stamper.mark(0))
        torch.Tensor.item = item
        try:
            g = loop.generate_speculative(ids, N, K=args.k)
        finally:
            torch.Tensor.item = real_item
            loop._vgraph = vg
        rounds = [j + 1 for j in seen]
        # a final round shorter than K takes the uncaptured path and so
        # never replays the verify graph; its tokens land when the call
        # returns, which is the stamp to give them
        while len(stamper.stamps) < len(rounds):
            stamper.mark(0)
        if len(rounds) == len(stamper.stamps):
            got = 0
            for i, n in enumerate(rounds):
                n = min(n, N - got)
                stamper.counts[i] = n
                got += n
            if got != N:                        # the tail round is short
                stamper.counts[-1] += N - got
        else:
            print(f"  round count {len(rounds)} != stamps "
                  f"{len(stamper.stamps)}; spreading evenly", flush=True)
            base_n = N // len(stamper.stamps)
            stamper.counts = [base_n] * len(stamper.stamps)
            stamper.counts[-1] += N - base_n * len(stamper.stamps)
        return g

    record("mtp", "+ MTP speculative decode",
           f"draft chain K={args.k}, verify captured", "native", run_mtp,
           {"acceptance_length": round(float(al), 2), "k": args.k,
            "identical_to_greedy": bool(exact),
            "repeat_identical": bool(same_s)})

    summary["_ladder"] = {
        "prompt_tokens": L, "new_tokens": N,
        "loop_vs_host": round(summary["loop"]["decode_tok_s"]
                              / summary["host"]["decode_tok_s"], 3),
        "loop_exact_vs_host": round(summary["loop_exact"]["decode_tok_s"]
                                    / summary["host"]["decode_tok_s"], 3),
        "compiled_step_costs": (
            "compiling the step buys "
            f"{summary['loop']['decode_tok_s'] / summary['loop_exact']['decode_tok_s']:.2f}x "
            f"and takes the teacher-forced rate from "
            f"{summary['loop_exact']['teacher_forced_same_token']} to "
            f"{summary['loop']['teacher_forced_same_token']}"),
        "mtp_vs_host": round(summary["mtp"]["decode_tok_s"]
                             / summary["host"]["decode_tok_s"], 3),
        "mtp_vs_loop": round(summary["mtp"]["decode_tok_s"]
                             / summary["loop"]["decode_tok_s"], 3),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["_ladder"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
