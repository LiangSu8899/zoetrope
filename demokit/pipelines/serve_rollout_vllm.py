"""Record a serving engine under concurrency, token by token.

vLLM's `LLM.generate` hands back a finished answer, which is no use for
a film: a race needs to know when each token of each request arrived.
So this drives `LLMEngine.step()` directly and stamps every token as the
engine emits it. One step normally produces one token per running
request, so the step boundary is the token's arrival.

Two arms, and they cannot share a process — a serving engine bakes its
model tree into a compiled, graph-captured decode path at load time, and
attaching after that fails by construction. So `base` and `attach` are
separate runs and the token streams are compared afterwards, from the
ids each run wrote down.

    --arm base      the engine in its own default production form
    --arm attach    the same engine with structure seats bound by a
                    hook that fires after load_model and before the
                    engine's first trace
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

# vLLM's V1 stack runs the worker in a subprocess by default, and a hook
# installed in this interpreter would never reach it.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# the compile cache key does not include the module tree, so a cached
# artefact from the other arm would be replayed against a swapped tree
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

if os.environ.get("FRT_WORKTREE"):
    sys.path.insert(0, os.environ["FRT_WORKTREE"])

PROMPTS = [
    "Explain, in one paragraph, why a GPU is faster than a CPU at "
    "matrix multiplication.",
    "Write a short function in C that reverses a string in place.",
    "In one paragraph: what is the difference between latency and "
    "throughput?",
    "Describe what happens when a CUDA kernel is launched.",
    "In one paragraph, explain what a page cache does.",
    "Write a short Python function that merges two sorted lists.",
    "Explain what memory bandwidth means for a language model.",
    "In one paragraph: why do neural networks use non-linearities?",
]


#: The grouped expert GEMV addresses one launch row per (token, expert)
#: pair through a grid dimension, and a CUDA grid dimension stops at
#: 65535. It is a decode kernel, so no decode step comes close — but the
#: engine profiles with its whole token budget, and 8192 tokens at top-8
#: is 65536 rows: one over. The symptom is `CUDA error: invalid
#: argument` surfacing asynchronously in whatever op runs next.
_SEAM_MAX_ROWS = 32768


def _seam_call(seam, x, top_k_index, top_k_weights):
    """Run the seam, in row-bounded chunks when the batch is large."""
    if x.shape[0] * top_k_index.shape[1] <= _SEAM_MAX_ROWS:
        return seam(x, top_k_index, top_k_weights)
    import torch

    step = max(1, _SEAM_MAX_ROWS // top_k_index.shape[1])
    return torch.cat(
        [seam(x[i:i + step], top_k_index[i:i + step],
              top_k_weights[i:i + step])
         for i in range(0, x.shape[0], step)], dim=0)


def _wire_routed(cls):
    """Route `forward_modular` through the seam, once per class.

    The boundary is narrow on purpose: `forward_modular` returns the
    **routed** half alone. The engine runs the shared experts either side
    of this call, on whichever stream it chose for itself, and its runner
    adds the two halves afterwards. `shared_experts` arrives here only so
    a fused kernel *could* overlap them; a seam that does not fuse them
    must leave the argument alone — touching it consumes an output the
    runner is about to read, and vLLM's own assertion catches that.

    Everything else stays the engine's: the router's expert selection,
    the scheduler, the KV manager, the compiled decode path this is
    bound inside of.
    """
    if getattr(cls, "_frt_wired", False):
        return
    original = cls.forward_modular

    def forward_modular(self, x, topk_weights, topk_ids,
                        shared_experts=None, shared_experts_input=None,
                        *a, **kw):
        seam = getattr(self, "_frt_seam", None)
        if seam is None:
            return original(self, x, topk_weights, topk_ids, shared_experts,
                            shared_experts_input, *a, **kw)
        return _seam_call(seam, x, topk_ids, topk_weights)

    cls.forward_modular = forward_modular
    cls._frt_wired = True


def install_expert_seats(model, oneway, report, variant):
    """Bind the routed expert bank, which is where a MoE keeps its weight.

    The projection seats reach `qkv_proj` and friends. On a routed
    mixture of experts those are the small half: Qwen3.5-35B-A3B keeps
    64.4 of its 70.2 GB in `mlp.experts.routed_experts`, as two 3-D
    stacks (`w13_weight` [E, 2I, H], `w2_weight` [E, H, I]) that no
    name-suffix rule reaches. Binding them is the whole lever.

    `variant` picks the launch: `w4a4` needs SM120/SM121 block-scaled
    MMA, so on SM110 (Thor) the family's W4A16 member is the one that
    binds — 4-bit weights, bf16 activations. Ask for the wrong one and
    the kernel says so at bind time rather than at run time.
    """
    import torch
    import torch.nn.functional as F
    import importlib

    bound, refused, freed, rel = 0, [], 0, []
    targets = [(name, mod) for name, mod in model.named_modules()
               if name.endswith("routed_experts")
               and hasattr(mod, "w13_weight")
               and hasattr(mod, "w2_weight")]
    impl = None
    if targets:
        module_variant = "dynamic" if variant == "w4a4" else variant
        impl = importlib.import_module(
            f"flash_rt.structures.impls.moe_experts.nvfp4_{module_variant}")
    for name, mod in targets:
        try:
            seam, quality = impl.bind_experts_seam(
                {"gate_up_proj": mod.w13_weight.data,
                 "down_proj": mod.w2_weight.data}, F.silu)
        except Exception as exc:                            # noqa: BLE001
            refused.append((name, repr(exc)[:70]))
            continue
        rel.append(max(quality.values()))
        if oneway:
            for attr in ("w13_weight", "w2_weight"):
                w = getattr(mod, attr)
                freed += w.numel() * w.element_size()
                w.data = torch.empty(0, device=w.device, dtype=w.dtype)
        mod._frt_seam = seam
        _wire_routed(type(mod))
        bound += 1
    torch.cuda.empty_cache()
    report.update(expert_banks=bound,
                  expert_banks_refused_count=len(refused),
                  expert_banks_refused=refused,
                  expert_variant=variant,
                  expert_freed_gb=round(freed / 1e9, 2),
                  expert_worst_pack_rel_l2=(round(max(rel), 5) if rel
                                            else None))
    print(f"[attach] {bound} expert banks ({variant}), "
          f"{len(refused)} refused, {freed / 1e9:.1f} GB freed", flush=True)


def install_adapter(report, verbose=True):
    """Use the library's own vLLM adapter instead of a hand-written list.

    The seat list further down was written against this engine by
    reading its module tree, and it converges on the same dense
    projections — but it stops there. The shipped adapter also binds the
    **LM head**, which vLLM consults through `quant_method.apply` rather
    than a module forward, and which is 1.02 GB of every decoded token
    on this checkpoint; and it dispatches by band inside a custom op, so
    decode rows go to the packed bank while **prefill rows stay on the
    retained host** instead of being pushed through a decode GEMV in
    chunks.

    Both of those are invisible to a seat list that matches on name
    suffixes, and the second is why the hand-rolled arm's TTFT grew with
    the batch instead of shrinking.
    """
    from flash_rt.structures.adapters import vllm_engine

    def on_attached(handle):
        try:
            rep = handle.report()
        except Exception:                                   # noqa: BLE001
            rep = {}
        summary = None
        try:
            summary = handle.summary()
        except Exception:                                   # noqa: BLE001
            pass
        notes = getattr(handle, "notes", {})
        refused = notes.get("refused", [])
        report.update(adapter="flash_rt.structures.adapters.vllm_engine",
                      adapter_report=rep if isinstance(rep, dict)
                      else str(rep)[:400],
                      adapter_summary=summary if isinstance(summary, dict)
                      else (str(summary)[:400] if summary else None),
                      adapter_seats=(summary or {}).get("seams", 0)
                      if isinstance(summary, dict) else None,
                      adapter_head_slabs=notes.get("head_slabs", 0),
                      adapter_refused_count=len(refused),
                      adapter_refused=refused)
        print(f"[adapter] {summary or rep}"[:400], flush=True)

    patched = vllm_engine.install_load_hook(on_attached=on_attached,
                                            verbose=verbose)
    report["adapter_patched"] = patched
    print(f"[adapter] hooked {patched}", flush=True)


def install_fusion(report, verbose=True):
    """Bind the structures discovery finds, at the last moment it can.

    The name-suffix seats below reach the projections and the expert
    banks, which is where the *bytes* are. They do not reach the
    structures the automatic path finds by shape — on this engine's tree,
    55 `norm_fused` and 10 `decoder_block` — and those cost launches and
    activation traffic rather than weight traffic, so a bytes-per-token
    roofline cannot see them at all. At batch 1 that is exactly the term
    that matters.

    Discovery is a pure structural pass, but binding calibrates from a
    real forward, and a load hook has none: the KV cache does not exist
    yet. `capture_model` does — it runs after the cache is allocated and
    immediately before the engine's first trace, which is the same window
    the seats use, and the engine's own `_dummy_run` is the forward.
    """
    import torch

    def bind(runner):
        from flash_rt import structures
        from flash_rt.structures import swap

        # The dummy run must not capture: its default mode dispatches a
        # piecewise CUDA graph, and calibration hooks running inside an
        # active capture invalidate it — `cudaErrorStreamCaptureInvalidated`,
        # which then poisons the engine's own capture that follows.
        # NONE is the mode vLLM documents for "warm up and profile run",
        # and it needs force_attention to build the metadata itself.
        from vllm.config import CUDAGraphMode

        def forward():
            with torch.inference_mode():
                try:
                    runner._dummy_run(
                        1, cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        force_attention=True, uniform_decode=True,
                        skip_eplb=True)
                except TypeError:                            # older runner
                    runner._dummy_run(1, uniform_decode=True)

        t0 = time.time()
        plan = structures.auto_swaps(runner.model, forward, verbose=verbose)
        if not plan.swaps:
            report.update(fusion_seats=0, fusion_note="nothing bound")
            print("[fusion] nothing bound", flush=True)
            return
        swap.attach(runner.model, plan.swaps, observe=plan.observed,
                    revert=plan.revert)
        fams = {}
        for path in plan.swaps:
            fams[path.rsplit(".", 1)[-1]] = fams.get(
                path.rsplit(".", 1)[-1], 0) + 1
        report.update(fusion_seats=len(plan.swaps),
                      fusion_refused=len(plan.notes.get("refused", [])),
                      fusion_bind_s=round(time.time() - t0, 1))
        print(f"[fusion] {len(plan.swaps)} seats, "
              f"{len(plan.notes.get('refused', []))} refused, "
              f"{report['fusion_bind_s']} s", flush=True)

    import vllm.v1.worker.gpu.model_runner as v2
    mods = [v2]
    try:
        import vllm.v1.worker.gpu_model_runner as v1
        mods.append(v1)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import vllm.v2.worker.gpu_model_runner as v2_runner
        mods.append(v2_runner)
    except Exception:                                       # noqa: BLE001
        pass
    for m in mods:
        orig = m.GPUModelRunner.capture_model

        def make(orig):
            def f(self, *a, **kw):
                try:
                    bind(self)
                except Exception as exc:                    # noqa: BLE001
                    report.update(fusion_error=repr(exc)[:160])
                    print(f"[fusion] refused: {exc!r}"[:200], flush=True)
                return orig(self, *a, **kw)
            return f

        m.GPUModelRunner.capture_model = make(orig)


def install_seats(seats, oneway, scheme, expert_variant=None):
    """Bind structure seats after the engine loads and before it traces."""
    import torch
    from torch import nn
    from flash_rt.structures import swap as swap_mod
    import importlib
    # the impls are submodules, not attributes of the package
    impl = importlib.import_module(
        f"flash_rt.structures.impls.linear_proj.{scheme}")

    class Seat(nn.Module):
        """vLLM's linear layers return (out, bias); the seam returns out."""

        def __init__(self, seam):
            super().__init__()
            self.seam = seam

        def forward(self, x, *a, **kw):
            return self.seam(x), None

    report = {}

    def attach(model):
        t0 = time.time()
        swaps, freed, refused = {}, 0, []
        for name, mod in list(model.named_modules()):
            if not any(name.endswith(s) for s in seats):
                continue
            try:
                weight = mod.weight.data
                if weight.dtype not in (torch.bfloat16, torch.float16,
                                        torch.float32):
                    raise ValueError(
                        "refused: prequantized ModelOpt packed storage "
                        f"({weight.dtype}) is not a dense weight accepted "
                        f"by linear_proj.{scheme}")
                bound = impl.bind_proj_seam({"w": weight})
                seam = bound[0] if isinstance(bound, tuple) else bound
            except Exception as exc:                        # noqa: BLE001
                refused.append((name, repr(exc)[:70]))
                continue
            swaps[name] = Seat(seam)
            if oneway:
                freed += mod.weight.numel() * mod.weight.element_size()
                mod.weight.data = torch.empty(0, device=mod.weight.device,
                                              dtype=mod.weight.dtype)
        torch.cuda.empty_cache()
        model.eval()
        if swaps:
            swap_mod.attach(model, swaps)
        report.update(scheme=scheme, seats_asked=",".join(seats),
                      seats=len(swaps), refused_count=len(refused),
                      refused=refused,
                      freed_gb=round(freed / 1e9, 2),
                      bind_s=round(time.time() - t0, 1))
        print(f"[attach] {len(swaps)} seats, {len(refused)} refused, "
              f"{report['freed_gb']} GB freed, {report['bind_s']} s",
              flush=True)
        if expert_variant:
            install_expert_seats(model, oneway, report, expert_variant)

    import vllm.v1.worker.gpu.model_runner as v2
    mods = [v2]
    try:
        import vllm.v1.worker.gpu_model_runner as v1
        mods.append(v1)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import vllm.v2.worker.gpu_model_runner as v2_runner
        mods.append(v2_runner)
    except Exception:                                       # noqa: BLE001
        pass
    for m in mods:
        orig = m.GPUModelRunner.load_model

        def make(orig):
            def f(self, *a, **kw):
                orig(self, *a, **kw)
                attach(self.model)
            return f

        m.GPUModelRunner.load_model = make(orig)
    return report


def race(engine, sp_cls, n, tokens, tok):
    """N requests at once, every token stamped as the engine emits it."""
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=tokens, temperature=0.0,
                        ignore_eos=True)
    seen = [0] * n
    ids = [[] for _ in range(n)]
    prompt_ids = [None] * n
    ev = []
    torch_sync()
    t0 = time.perf_counter()
    for i in range(n):
        engine.add_request(str(i), PROMPTS[i % len(PROMPTS)], sp)
    while engine.has_unfinished_requests():
        for out in engine.step():
            k = int(out.request_id)
            if prompt_ids[k] is None and out.prompt_token_ids:
                prompt_ids[k] = list(out.prompt_token_ids)
            new = list(out.outputs[0].token_ids)[seen[k]:]
            if not new:
                continue
            t = time.perf_counter() - t0
            for tid in new:
                ev.append({"s": k, "i": seen[k], "t": round(t, 5),
                           "text": tok.decode([tid],
                                              skip_special_tokens=True)})
                ids[k].append(int(tid))
                seen[k] += 1
    return ev, ids, time.perf_counter() - t0, prompt_ids


def torch_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def teacher_forced(llm, ref, tok):
    """Same-token rate against a reference stream, one step at a time.

    A free run cascades: one tie-break and every token after it is
    incomparable, which is why these two arms agree on 7% of a
    hundred-and-sixty-token answer and both are fluent. The gate is
    whether this arm, handed the reference's own prefix, would have
    chosen the reference's own next token. vLLM will answer that for a
    whole sequence in one prefill if you ask for prompt logprobs.
    """
    from vllm import SamplingParams

    hits = total = 0
    per_stream = []
    for pid, gen in zip(ref["prompt_ids"], ref["generated"]):
        full = list(pid) + list(gen)
        out = llm.generate([{"prompt_token_ids": full}],
                           SamplingParams(max_tokens=1, temperature=0.0,
                                          prompt_logprobs=1),
                           use_tqdm=False)[0]
        lp = out.prompt_logprobs
        h = t = 0
        # position i's logprobs predict the token at i; score only the
        # generated tail, where the reference actually made a choice
        for i in range(len(pid), len(full)):
            cand = lp[i]
            if not cand:
                continue
            top = max(cand.items(), key=lambda kv: kv[1].logprob)[0]
            h += int(int(top) == int(full[i]))
            t += 1
        hits += h
        total += t
        per_stream.append(round(h / t, 4) if t else None)
    return {"teacher_forced_same_token": round(hits / total, 4) if total
            else None, "scored_positions": total, "per_stream": per_stream}


def summarise(ev, ids, wall, n):
    firsts, rates = [], []
    for k in range(n):
        ts = [e["t"] for e in ev if e["s"] == k]
        if not ts:
            continue
        firsts.append(ts[0] * 1e3)
        if len(ts) > 2:
            rates.append((len(ts) - 1) / (ts[-1] - ts[0]))
    total = sum(len(x) for x in ids)
    return {
        "ttft_ms_median": round(statistics.median(firsts), 1),
        "ttft_ms_p90": round(sorted(firsts)[int(len(firsts) * 0.9) - 1], 1),
        "decode_tok_s_per_stream": round(statistics.median(rates), 1),
        "aggregate_tok_s": round(total / wall, 1),
        "n_tokens_total": total,
        "done_s": round(max(e["t"] for e in ev), 4),
        "wall_s": round(wall, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=("base", "attach"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", default="1,4,8,16")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompt", default=None,
                    help="one prompt for every request, instead of the "
                         "built-in set. A film shows the answer for its "
                         "whole length, so it needs one that sustains "
                         "--tokens without the model padding.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed passes; the last one is the one stamped")
    ap.add_argument("--seats", default="qkv_proj,o_proj,gate_up_proj,"
                                       "down_proj")
    ap.add_argument("--scheme", default="nvfp4_dynamic",
                    help="linear_proj impl for the seats: "
                         "nvfp4_dynamic (4-bit) or w8a16_static (8-bit)")
    ap.add_argument("--expert-variant", default=None,
                    choices=("w4a4", "w4a16"),
                    help="also bind the routed expert bank, which on a MoE "
                         "is most of the model. w4a4 needs SM120/121; "
                         "SM110 (Thor) takes w4a16")
    ap.add_argument("--adapter", action="store_true",
                    help="bind through the library's own vLLM adapter "
                         "(dense projections + expert banks + LM head, "
                         "with prefill kept on the host) instead of the "
                         "hand-written seat list")
    ap.add_argument("--fusion", action="store_true",
                    help="also bind what discovery finds by shape "
                         "(norm_fused, decoder_block, ...), calibrated "
                         "from the engine's own dummy forward just before "
                         "its first trace")
    ap.add_argument("--oneway", action="store_true",
                    help="free each original weight as its seat binds; "
                         "irreversible, and buys VRAM as it goes")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--kv-cache-bytes", type=int, default=None,
                    help="size the KV cache explicitly instead of deriving "
                         "it from --util. Needed on Jetson Thor, where "
                         "cudaMemGetInfo under-reports free memory badly "
                         "enough that the util-derived plan is wrong")
    ap.add_argument("--teacher-forced", default=None,
                    help="path to the other arm's tokens.json; scores "
                         "this arm's next-token choice against that "
                         "stream, one position at a time")
    ap.add_argument("--raw", action="store_true",
                    help="send the prompt as raw text; by default it is "
                         "rendered through the model's chat template with "
                         "thinking off, so a pane shows an answer rather "
                         "than a completion that runs on past it")
    ap.add_argument("--label", default=None)
    ap.add_argument("--sub", default=None)
    args = ap.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]

    report = {}
    if args.arm == "attach":
        if args.adapter:
            install_adapter(report)
        else:
            report = install_seats(
                tuple(s for s in args.seats.split(",") if s),
                args.oneway, args.scheme, args.expert_variant)
            if args.fusion:
                install_fusion(report)

    from vllm import LLM, SamplingParams

    engine_kwargs = ({"kv_cache_memory_bytes": args.kv_cache_bytes}
                     if args.kv_cache_bytes is not None else {})
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.util,
              max_num_seqs=max(levels), **engine_kwargs)
    engine = llm.llm_engine
    tok = llm.get_tokenizer()
    if not args.raw:
        global PROMPTS
        if args.prompt:
            PROMPTS = [args.prompt]
        PROMPTS = [tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
            for p in PROMPTS]
    out = pathlib.Path(args.out)
    colour = "stock" if args.arm == "base" else "ours"
    label = args.label or ("the engine, as shipped" if args.arm == "base"
                           else "+ FlashRT structures")
    sub = args.sub or ("vLLM default production form"
                       if args.arm == "base" else
                       "structure seats bound at load, inside the "
                       "engine's own graph")
    summary = {}

    if args.teacher_forced:
        ref = json.loads(pathlib.Path(args.teacher_forced).read_text())
        gate = teacher_forced(llm, ref, tok)
        print(f"teacher-forced same-token {gate['teacher_forced_same_token']}"
              f" over {gate['scored_positions']} positions", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"gate_{args.arm}.json").write_text(json.dumps(gate, indent=1))

    for n in levels:
        race(engine, SamplingParams, n, 8, tok)            # warm
        # the median repeat, not the fastest: min-of-N is a lower bound and
        # belongs in a footnote, never on the face of a film
        runs = []
        for _ in range(args.repeats):
            ev, ids, wall, pids = race(engine, SamplingParams, n,
                                       args.tokens, tok)
            runs.append((summarise(ev, ids, wall, n), ev, ids, pids))
        runs.sort(key=lambda r: r[0]["wall_s"])
        m, ev, ids, pids = runs[len(runs) // 2]
        m["wall_s_min"] = round(runs[0][0]["wall_s"], 4)
        m["wall_s_max"] = round(runs[-1][0]["wall_s"], 4)
        m["repeats"] = len(runs)
        # A one-chapter film uses the protocol's direct arm layout.
        # Keep the cN level for the existing multi-concurrency films.
        d = (out / args.arm if len(levels) == 1
             else out / f"c{n}" / args.arm)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"kind": "stream_batch", "label": label, "sub": sub,
                "color": colour, "concurrency": n, "arm": args.arm,
                "model": args.model, "max_tokens": args.tokens, **m}
        meta.update(report)
        meta["engine_kwargs"] = dict(engine_kwargs)
        if n == 1:
            # a single request reads better in the plain stream pane:
            # one big text buffer instead of a one-row dashboard
            meta["kind"] = "stream"
            meta["ttft_ms"] = m["ttft_ms_median"]
            meta["decode_tok_s"] = m["decode_tok_s_per_stream"]
            meta["n_tokens"] = m["n_tokens_total"]
            ev = [{"i": e["i"], "t": e["t"], "text": e["text"]}
                  for e in ev]
        (d / "events.json").write_text(json.dumps(
            {"meta": meta, "events": ev}, indent=1))
        (d / "tokens.json").write_text(json.dumps(
            {"prompt_ids": pids, "generated": ids}))
        summary[f"c{n}"] = m
        print(f"[c{n}] aggregate {m['aggregate_tok_s']} tok/s, "
              f"{m['decode_tok_s_per_stream']} per request, TTFT "
              f"{m['ttft_ms_median']} ms, all done {m['done_s']} s",
              flush=True)

    (out / f"metrics_{args.arm}.json").write_text(
        json.dumps({"arm": args.arm, "attach": report,
                    "levels": summary}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
