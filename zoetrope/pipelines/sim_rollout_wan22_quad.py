"""All four Wan2.2 arms in one process, against one eager baseline.

The ladder was assembled from separate runs, which is fine for a table
and wrong for a race: four panes on one wall clock have to share one
baseline. So this builds every form once and measures them back to back:

    eager      the diffusers WanPipeline as shipped
    compiled   the same host under torch.compile(reduce-overhead)
    band       W4A4 for every call whose own per-step score holds the
               band, FP8 for the tail — both whole-graph, the call
               choosing from its timestep
    sage       W4A4 + SageAttention2 for every call, whole-graph

The band arm reads its split off the **packaged** W4A4 scores rather
than the eager-form ones. Exporting shifts the arithmetic by around
5e-4, which is enough for a rule aimed exactly at the band to land just
under it; measuring the thing that will actually run removes the gap.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

import torch

#: point $FRT_WORKTREE at a FlashRT checkout to run against it in place.
if os.environ.get("FRT_WORKTREE"):
    sys.path.insert(0, os.environ["FRT_WORKTREE"])
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sim_rollout_wan22_hybrid as HY  # noqa: E402
from sim_rollout_wan22 import (  # noqa: E402
    MODEL, SEED, _CloneOut, cosine, decode, free, timed_call, write_arm)

#: the directory holding flashrt_wan_attn.py (the diffusers-wan2.2 demo)
SAGE_DIR = os.environ.get("WAN_SAGE_PROCESSOR", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="directory to write one sub-directory per arm")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--band", type=float, default=0.995)
    ap.add_argument("--prompt", default=None)
    cfg = ap.parse_args()
    if cfg.prompt:
        HY.PROMPT = cfg.prompt

    from diffusers import WanPipeline
    from flash_rt import structures
    from flash_rt.structures.aot import AotModule
    from flash_rt.structures.swap import attach as swap_attach

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16
                                       ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    host = pipe.transformer
    out = pathlib.Path(cfg.out)
    summary = {}

    with torch.no_grad():
        emb = pipe.encode_prompt(prompt=HY.PROMPT, negative_prompt="",
                                 do_classifier_free_guidance=True,
                                 max_sequence_length=512,
                                 device=torch.device("cuda"),
                                 dtype=torch.bfloat16)
    embeds = (emb[0], emb[1])
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    free()

    calls, sig = [], {}

    def hook(mod, a, kw, output):
        o = output[0] if isinstance(output, tuple) else output
        calls.append({"hidden": kw["hidden_states"].detach().cpu(),
                      "timestep": kw["timestep"].detach().cpu(),
                      "encoder": kw["encoder_hidden_states"].detach().cpu(),
                      "out": o.detach().cpu()})
        sig.setdefault("args", a)
        sig.setdefault("kwargs", kw)

    h = host.register_forward_hook(hook, with_kwargs=True)
    ref, _ = HY.run(pipe, cfg, embeds)
    h.remove()
    ts = [float(c["timestep"].reshape(-1)[0]) for c in calls]
    mid = calls[len(calls) // 2]
    print(f"captured {len(calls)} calls", flush=True)

    def take(name, label, sub, colour, mod, parity_ref=None):
        total, stamps = HY.timed(pipe, cfg, embeds, cfg.repeats)
        lat, _ = HY.run(pipe, cfg, embeds)
        cs = HY.stepwise(mod, calls)
        pc = timed_call(mod, mid)
        clip, dec = decode(pipe, lat)
        c = cosine(lat, ref)
        print(f"{name}: {total:.3f} s, {pc:.1f} ms/call, worst "
              f"{min(cs):.6f}, final {c:.6f}", flush=True)
        summary[name] = write_arm(
            out, name, label, sub, colour, cfg, total, stamps, clip, dec,
            round(c, 6), (min(cs), statistics.median(cs)), pc)
        return cs

    # ---- eager, then the host compiled --------------------------------
    take("eager", "Wan2.2, as shipped", "eager PyTorch", "stock", host)
    pipe.transformer = _CloneOut(
        torch.compile(host, mode="reduce-overhead"), host)
    HY.run(pipe, cfg, embeds)
    take("compiled", "the same host, compiled",
         "torch.compile + graph capture", "compiled", pipe.transformer)
    pipe.transformer = host
    torch._dynamo.reset()
    free()

    # ---- three packages ------------------------------------------------
    print("exporting three forms:", flush=True)
    w4_pkg, w4_plan, _ = HY.build_package(
        structures, swap_attach, host, calls, "nvfp4_balance",
        sig["args"], sig["kwargs"], "/tmp/frt_aot/quad_w4.pt2")
    fp8_pkg, fp8_plan, _ = HY.build_package(
        structures, swap_attach, host, calls, "fp8_static",
        sig["args"], sig["kwargs"], "/tmp/frt_aot/quad_fp8.pt2")
    sys.path.insert(0, SAGE_DIR)
    from flashrt_wan_attn import WanSageAttention2Processor
    orig_procs = dict(host.attn_processors)
    host.set_attn_processor(WanSageAttention2Processor())
    sage_pkg, sage_plan, _ = HY.build_package(
        structures, swap_attach, host, calls, "nvfp4_balance",
        sig["args"], sig["kwargs"], "/tmp/frt_aot/quad_w4sage.pt2")
    host.set_attn_processor(orig_procs)

    pipe.transformer = None
    host_cpu = host.to("cpu")
    free()
    kwsig = dict(sig["kwargs"])

    def wrap(pkg):
        loaded = structures.aot_load(pkg)

        def call(*a, **kw):
            for k, v in kwsig.items():
                if k not in kw and not torch.is_tensor(v):
                    kw[k] = v
            return loaded(*a, **{k: kw[k] for k in kwsig if k in kw})

        return AotModule(call, host_cpu)

    # two packages fit on this card and three do not, so the band arm
    # runs first and its pair is released before the Sage one is loaded
    w4, fp8 = wrap(w4_pkg), wrap(fp8_pkg)
    print(f"two packages resident "
          f"({torch.cuda.memory_allocated()/2**30:.1f} GiB)", flush=True)

    # the split comes off the packaged scores, which is what will run
    w4_cs = HY.stepwise(w4, calls)
    fp8_cs = HY.stepwise(fp8, calls)
    weak = sorted({t for t, c in zip(ts, w4_cs) if c < cfg.band})
    keep = len(ts) - sum(1 for t in ts if t in set(weak))
    print(f"packaged W4A4 holds {cfg.band} on {keep} of {len(ts)} calls; "
          f"the {len(weak)} it misses sit at t="
          f"{[int(t) for t in weak]}", flush=True)
    print("  W4 per-call profile: "
          + " ".join(f"{int(t)}:{c:.4f}" for t, c in zip(ts, w4_cs)),
          flush=True)
    print(f"  W4 alone worst {min(w4_cs):.6f}, FP8 alone worst "
          f"{min(fp8_cs):.6f}", flush=True)

    band_mod = HY._BySet(w4, fp8, weak, host_cpu)
    pipe.transformer = band_mod
    HY.run(pipe, cfg, embeds)
    band_mod._tally[:] = [0, 0]
    take("band", "+ FlashRT structures",
         f"W4A4 on {keep} of {len(ts)} calls, FP8 on the rest", "native",
         band_mod)
    summary["band"]["band"] = {
        "rule_band": cfg.band,
        "weak_timesteps": [int(t) for t in weak],
        "calls_w4": band_mod._tally[0], "calls_fp8": band_mod._tally[1],
        "paired_call_ms_w4": round(timed_call(w4, mid), 1),
        "paired_call_ms_fp8": round(timed_call(fp8, mid), 1),
        "w4_alone_worst": round(min(w4_cs), 6),
        "fp8_alone_worst": round(min(fp8_cs), 6),
        "split_read_from": "the packaged scores, not the eager-form ones",
    }

    pipe.transformer = None
    del band_mod, w4, fp8
    free()
    sage = HY._BySet(wrap(sage_pkg), None, (), host_cpu)
    print(f"band pair released, Sage package resident "
          f"({torch.cuda.memory_allocated()/2**30:.1f} GiB)", flush=True)
    pipe.transformer = sage
    HY.run(pipe, cfg, embeds)
    take("sage", "+ FlashRT structures, all 4-bit",
         "W4A4 + SageAttention2, whole-graph", "accent", sage)

    e = summary["eager"]["denoise_s"]
    summary["_ladder"] = {
        "one_process": True,
        "seams": {"w4": len(w4_plan.swaps), "fp8": len(fp8_plan.swaps),
                  "sage": len(sage_plan.swaps)},
        "vs_eager": {k: round(e / summary[k]["denoise_s"], 3)
                     for k in ("compiled", "band", "sage")},
        "vs_compiled": {k: round(summary["compiled"]["denoise_s"]
                                 / summary[k]["denoise_s"], 3)
                        for k in ("band", "sage")},
        "call_vs_eager": {
            k: round(summary["eager"]["paired_call_ms"]
                     / summary[k]["paired_call_ms"], 3)
            for k in ("compiled", "band", "sage")},
        "stepwise_worst": {k: summary[k]["stepwise_worst"]
                           for k in ("compiled", "band", "sage")},
        "fixture": {"steps": cfg.steps, "frames": cfg.frames,
                    "size": cfg.size, "seed": SEED, "prompt": HY.PROMPT},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["_ladder"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
