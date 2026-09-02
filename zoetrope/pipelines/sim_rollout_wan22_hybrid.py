"""Wan2.2, scheme by step: W4A4 while it holds, FP8 for the tail.

The per-step gate for W4A4 on a twenty-step schedule does not fail
everywhere. It falls monotonically with the timestep — 0.9992 at t=999,
0.9795 at t=211 — and crosses the 0.995 warn band only in the last five
or six steps. That is a schedule shape, not a defect, and it has an
obvious answer: run W4A4 while it passes and hand the tail to FP8.

Both forms are taken whole (`torch.export` + AOTInductor) and picked per
call by the timestep, which is a property of the request rather than of
a step counter, so the split survives any schedule length.

The text encoder runs once and then leaves the device: two baked
packages and an 11 GB encoder do not need to be resident together.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

#: point $FRT_WORKTREE at a FlashRT checkout to run against it in place.
if os.environ.get("FRT_WORKTREE"):
    sys.path.insert(0, os.environ["FRT_WORKTREE"])
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from sim_rollout_wan22 import (  # noqa: E402
    MODEL, PROMPT, SEED, STRUCTURES, _CloneOut, cosine, decode, free,
    timed_call, write_arm)

#: W4A4 keeps the band down to t=729 and loses it at t=682, so the split
#: sits between them. Expressed as a timestep, not a step index.
SPLIT_T = 700.0


class _BySet(torch.nn.Module):
    """Two whole-graph packages, chosen per call by the timestep.

    A threshold assumes the weak calls are a tail. They mostly are, but
    not always — one early call can score low and drag a threshold rule
    all the way up the schedule. Routing on the set of timesteps whose
    own score missed the band keeps every call the fast form can hold.
    """

    def __init__(self, fast, slow, weak, host):
        super().__init__()
        object.__setattr__(self, "_fast", fast)
        object.__setattr__(self, "_slow", slow)
        object.__setattr__(self, "_weak", {round(float(t), 3)
                                           for t in weak})
        object.__setattr__(self, "_host", host)
        object.__setattr__(self, "_tally", [0, 0])

    @property
    def device(self):
        return torch.device("cuda")

    def forward(self, *a, **kw):
        t = kw.get("timestep")
        v = round(float(t.reshape(-1)[0]) if torch.is_tensor(t)
                  else float(t), 3)
        if v in self._weak:
            self._tally[1] += 1
            return self._slow(*a, **kw)
        self._tally[0] += 1
        return self._fast(*a, **kw)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_host"), name)


class _ByTimestep(torch.nn.Module):
    """Two whole-graph packages, chosen per call by the timestep."""

    def __init__(self, coarse, fine, split, host):
        super().__init__()
        object.__setattr__(self, "_coarse", coarse)
        object.__setattr__(self, "_fine", fine)
        object.__setattr__(self, "_split", float(split))
        object.__setattr__(self, "_host", host)
        object.__setattr__(self, "_tally", [0, 0])

    @property
    def device(self):
        # the host module now lives on the CPU — only its weights, baked
        # into the packages, are on the card. The pipeline asks the
        # transformer where to put the latents, and the answer is here.
        return torch.device("cuda")

    def forward(self, *a, **kw):
        t = kw.get("timestep")
        v = float(t.reshape(-1)[0]) if torch.is_tensor(t) else float(t)
        if v >= self._split:
            self._tally[0] += 1
            return self._coarse(*a, **kw)
        self._tally[1] += 1
        return self._fine(*a, **kw)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_host"), name)


def run(pipe, cfg, embeds, stamps=None, output_type="latent", steps=None):
    gen = torch.Generator("cuda").manual_seed(SEED)
    cb = None
    if stamps is not None:
        def cb(p, i, t, kw):
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
            return kw
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pipe(prompt_embeds=embeds[0],
                   negative_prompt_embeds=embeds[1],
                   height=cfg.size, width=cfg.size, num_frames=cfg.frames,
                   num_inference_steps=steps or cfg.steps, generator=gen,
                   output_type=output_type, callback_on_step_end=cb)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    if stamps is not None:
        stamps[:] = [s - t0 for s in stamps]
    return (out.frames if hasattr(out, "frames") else out[0]), total


def timed(pipe, cfg, embeds, n=3):
    runs = []
    for _ in range(n):
        st = []
        _, total = run(pipe, cfg, embeds, stamps=st)
        runs.append((total, st))
    runs.sort(key=lambda r: r[0])
    return runs[len(runs) // 2]


def stepwise(mod, calls):
    out = []
    with torch.no_grad():
        for c in calls:
            y = mod(hidden_states=c["hidden"].to("cuda"),
                    timestep=c["timestep"].to("cuda"),
                    encoder_hidden_states=c["encoder"].to("cuda"),
                    return_dict=False)
            y = y[0] if isinstance(y, tuple) else y
            out.append(cosine(y.detach().cpu(), c["out"]))
    return out


def build_package(structures, swap_attach, host, calls, scheme, args, kw,
                  path):
    """Calibrate, bind, check the ledger, export, and hand the host back."""
    def fwd():
        with torch.no_grad():
            for c in calls:
                host(hidden_states=c["hidden"].to("cuda"),
                     timestep=c["timestep"].to("cuda"),
                     encoder_hidden_states=c["encoder"].to("cuda"),
                     return_dict=False)

    wanted = (STRUCTURES if scheme == "fp8_static"
              else tuple(x for x in STRUCTURES if x != "modnorm_qkv_chain"))
    plan = structures.auto_swaps(host, fwd, structures=wanted,
                                 scheme=scheme, verbose=False)
    handle = swap_attach(host, plan.swaps, observe=plan.observed,
                         revert=plan.revert)
    cs = stepwise(host, calls)
    pkg = structures.aot_package(host, args=args, kwargs=kw,
                                 package_path=path)
    rep = handle.report()
    fb = sum(r.get("fallbacks", 0) for r in rep.values())
    n = sum(r.get("calls", 0) for r in rep.values())
    print(f"  {scheme}: {len(plan.swaps)} seams, {n} calls, {fb} "
          f"fallbacks, worst {min(cs):.6f}", flush=True)
    assert fb == 0 and n > 0
    handle.detach()
    free()
    return pkg, plan, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="directory to write one sub-directory per arm")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt", default=None,
                    help="override the prompt every arm generates from")
    ap.add_argument("--split", type=float, default=SPLIT_T,
                    help="timestep at or above which W4A4 is used; 0 "
                         "puts the whole schedule on W4A4; -1 reads the "
                         "split off W4A4's own measured profile")
    ap.add_argument("--band", type=float, default=0.995,
                    help="the per-step score W4A4 has to hold to keep a "
                         "call, when --split is -1")
    ap.add_argument("--sage", action="store_true",
                    help="also swap the attention processor for "
                         "SageAttention2 inside the packaged band")
    cfg = ap.parse_args()
    global PROMPT
    if cfg.prompt:
        PROMPT = cfg.prompt

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

    # one text encode for the whole run; the encoder then leaves
    with torch.no_grad():
        embeds = pipe.encode_prompt(prompt=PROMPT, negative_prompt="",
                                    do_classifier_free_guidance=True,
                                    max_sequence_length=512,
                                    device=torch.device("cuda"),
                                    dtype=torch.bfloat16)
    embeds = (embeds[0], embeds[1])
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    free()
    print(f"prompt encoded, encoder released "
          f"({torch.cuda.memory_allocated()/2**30:.1f} GiB resident)",
          flush=True)

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
    ref, _ = run(pipe, cfg, embeds)
    h.remove()
    print(f"captured {len(calls)} calls", flush=True)
    mid = calls[len(calls) // 2]

    # ---- eager -------------------------------------------------------
    total, stamps = timed(pipe, cfg, embeds, cfg.repeats)
    pc = timed_call(host, mid)
    clip, dec = decode(pipe, ref)
    print(f"eager {total:.3f} s, {pc:.1f} ms/call", flush=True)
    summary["eager"] = write_arm(out, "eager", "Wan2.2, as shipped",
                                 "eager PyTorch", "stock", cfg, total,
                                 stamps, clip, dec, 1.0, (1.0, 1.0), pc)

    # ---- the same host, compiled --------------------------------------
    pipe.transformer = _CloneOut(
        torch.compile(host, mode="reduce-overhead"), host)
    run(pipe, cfg, embeds)
    total, stamps = timed(pipe, cfg, embeds, cfg.repeats)
    lat, _ = run(pipe, cfg, embeds)
    c = cosine(lat, ref)
    sw = stepwise(pipe.transformer, calls)
    pc = timed_call(pipe.transformer, mid)
    clip, dec = decode(pipe, lat)
    print(f"compiled {total:.3f} s, {pc:.1f} ms/call, worst {min(sw):.6f}",
          flush=True)
    summary["compiled"] = write_arm(
        out, "compiled", "the same host, compiled",
        "torch.compile + graph capture", "compiled", cfg, total, stamps,
        clip, dec, round(c, 6), (min(sw), statistics.median(sw)), pc)
    pipe.transformer = host
    torch._dynamo.reset()
    free()

    # ---- one package per band the schedule actually visits -------------
    if cfg.sage:
        # the attention swap belongs to our arm, not to the baselines, so
        # it goes on after they have been measured and before the export
        # $WAN_SAGE_PROCESSOR points at the directory holding
        # flashrt_wan_attn.py (the diffusers-wan2.2 demo).
        sys.path.insert(0, os.environ["WAN_SAGE_PROCESSOR"])
        from flashrt_wan_attn import WanSageAttention2Processor
        host.set_attn_processor(WanSageAttention2Processor())
        print("attention: SageAttention2", flush=True)

    ts = [float(c["timestep"].reshape(-1)[0]) for c in calls]
    tag = ("_sage" if cfg.sage else "") + f"_s{int(cfg.split)}"
    print("exporting the bands the schedule visits:", flush=True)
    w4_pkg, w4_plan, w4_cs = build_package(
        structures, swap_attach, host, calls, "nvfp4_balance",
        sig["args"], sig["kwargs"], f"/tmp/frt_aot/wan_w4{tag}.pt2")
    if cfg.split < 0:
        # the rule, rather than a constant: W4A4 keeps every call whose
        # own per-step score holds the band, and the tail it cannot hold
        # goes to FP8. The timestep this lands on is a property of the
        # content; the rule is not.
        bad = [t for t, c in zip(ts, w4_cs) if c < cfg.band]
        cfg.split = (max(bad) + 1.0) if bad else 0.0
        print(f"  split read off the profile: W4A4 holds {cfg.band} down "
              f"to t={cfg.split:.0f} ({sum(t >= cfg.split for t in ts)} "
              f"of {len(ts)} calls)", flush=True)
    if any(t < cfg.split for t in ts):
        fp8_pkg, fp8_plan, fp8_cs = build_package(
            structures, swap_attach, host, calls, "fp8_static",
            sig["args"], sig["kwargs"], f"/tmp/frt_aot/wan_fp8{tag}.pt2")
    else:
        fp8_pkg, fp8_plan, fp8_cs = w4_pkg, w4_plan, w4_cs
        print("  the whole schedule is above the split; no FP8 band",
              flush=True)

    blended = [w4_cs[i] if t >= cfg.split else fp8_cs[i]
               for i, t in enumerate(ts)]
    print(f"band split at t={cfg.split:.0f}: "
          f"{sum(t >= cfg.split for t in ts)} calls W4, "
          f"{sum(t < cfg.split for t in ts)} calls FP8", flush=True)
    print(f"predicted stepwise worst {min(blended):.6f} "
          f"(W4 alone {min(w4_cs):.6f}, FP8 alone {min(fp8_cs):.6f})",
          flush=True)

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

    coarse = wrap(w4_pkg)
    fine = coarse if fp8_pkg is w4_pkg else wrap(fp8_pkg)
    hybrid = _ByTimestep(coarse, fine, cfg.split, host_cpu)
    print(f"both packages resident "
          f"({torch.cuda.memory_allocated()/2**30:.1f} GiB)", flush=True)

    pipe.transformer = hybrid
    run(pipe, cfg, embeds)                                     # warm
    hybrid._tally[:] = [0, 0]
    total, stamps = timed(pipe, cfg, embeds, cfg.repeats)
    lat, _ = run(pipe, cfg, embeds)
    c = cosine(lat, ref)
    sw = stepwise(hybrid, calls)
    pc_w4 = timed_call(hybrid._coarse, mid)
    pc_fp8 = timed_call(hybrid._fine, mid)
    clip, dec = decode(pipe, lat)
    print(f"hybrid {total:.3f} s, worst {min(sw):.6f}, "
          f"W4 {pc_w4:.1f} / FP8 {pc_fp8:.1f} ms per call, "
          f"tally {hybrid._tally}", flush=True)
    summary["attach"] = write_arm(
        out, "attach", "+ FlashRT structures",
        (("W4A4" + (" + SageAttention2" if cfg.sage else "")
          + ", whole-graph AoT") if not any(t < cfg.split for t in ts)
         else f"W4A4 above t={cfg.split:.0f}, FP8 below, both whole-graph"),
        "accent", cfg, total, stamps, clip, dec, round(c, 6),
        (min(sw), statistics.median(sw)), pc_w4)

    summary["attach"]["band"] = {
        "split_timestep": cfg.split, "sage": bool(cfg.sage),
        "calls_w4": hybrid._tally[0], "calls_fp8": hybrid._tally[1],
        "paired_call_ms_w4": round(pc_w4, 1),
        "paired_call_ms_fp8": round(pc_fp8, 1),
        "seams_w4": len(w4_plan.swaps), "seams_fp8": len(fp8_plan.swaps),
        "stepwise_w4_alone_worst": round(min(w4_cs), 6),
        "stepwise_fp8_alone_worst": round(min(fp8_cs), 6),
        "stepwise_per_call": [round(x, 6) for x in sw],
        "timesteps": ts,
    }
    summary["_ladder"] = {
        "vs_eager": round(summary["eager"]["denoise_s"]
                          / summary["attach"]["denoise_s"], 3),
        "vs_compiled": round(summary["compiled"]["denoise_s"]
                             / summary["attach"]["denoise_s"], 3),
        "call_vs_eager_w4_band": round(summary["eager"]["paired_call_ms"]
                                       / pc_w4, 3),
        "stepwise_worst": round(min(sw), 6),
        "gate": "pass" if min(sw) >= 0.995 else "REFUSED",
        "fixture": {"steps": cfg.steps, "frames": cfg.frames,
                    "size": cfg.size, "seed": SEED},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["_ladder"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
