"""Record one Wan2.2 TI2V-5B clip per arm, with the wall time of every step.

Three arms, one process, one load, one prompt, one seed:

    eager      the diffusers WanPipeline as its authors ship it
    compiled   the same host under torch.compile(reduce-overhead)
               — compile plus CUDA-graph capture, the production form
    attach     the same host again, with FlashRT structures attached
               through auto_swaps: discover, calibrate, qualify, bind

Nothing about the host is edited.  The fixture is the one the Wan2.2
qualification receipts use — 480x480, 33 frames, seed 7 — so the numbers
here sit next to the recorded gate without a change of units.

Each arm writes `events.json` (a step-by-step wall clock) and
`frames.npy` (the clip it actually produced) for `race_compose.py`.
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

#: point $FRT_WORKTREE at a FlashRT checkout to run against it in place;
#: without it the installed package is used.
if os.environ.get("FRT_WORKTREE"):
    sys.path.insert(0, os.environ["FRT_WORKTREE"])

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PROMPT = ("A curious raccoon carefully opens a wooden box in a sunlit "
          "forest clearing, cinematic lighting, shallow depth of field")
SEED = 7
STRUCTURES = ("modnorm_qkv_chain", "vision_ffn", "qkv_pack", "linear_proj")


class _CloneOut(torch.nn.Module):
    """Under cudagraphs the two CFG calls share one graph output buffer,
    and the pipeline still reads the conditional prediction after the
    unconditional call has run.  Clone each call out of the graph."""

    def __init__(self, inner, host):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_host", host)

    def forward(self, *a, **kw):
        out = self._inner(*a, **kw)
        if isinstance(out, tuple):
            return (out[0].clone(),) + out[1:]
        return out.clone()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_host"), name)


def run_pipe(pipe, cfg, capture=None, stamps=None, output_type="latent",
             steps=None):
    """One generation.  `stamps` collects the wall time of each step."""
    gen = torch.Generator("cuda").manual_seed(SEED)
    hooks = []
    if capture is not None:
        def hook(module, args, kwargs, output):
            out = output[0] if isinstance(output, tuple) else output
            capture.append({
                "hidden": kwargs.get("hidden_states",
                                     args[0] if args else None
                                     ).detach().to("cpu"),
                "timestep": kwargs.get("timestep").detach().to("cpu"),
                "encoder": kwargs.get("encoder_hidden_states"
                                      ).detach().to("cpu"),
                "out": out.detach().to("cpu")})
        hooks.append(pipe.transformer.register_forward_hook(
            hook, with_kwargs=True))

    cb = None
    if stamps is not None:
        def cb(p, i, t, kw):
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
            return kw

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pipe(prompt=PROMPT, height=cfg.size, width=cfg.size,
                       num_frames=cfg.frames,
                       num_inference_steps=steps or cfg.steps,
                       generator=gen, output_type=output_type,
                       callback_on_step_end=cb)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        for h in hooks:
            h.remove()
    frames = out.frames if hasattr(out, "frames") else out[0]
    if stamps is not None:
        stamps[:] = [s - t0 for s in stamps]
    return frames, total


def timed(pipe, cfg, n=3):
    """Median of n generations, keeping the step clock of the median run."""
    runs = []
    for _ in range(n):
        stamps = []
        _, total = run_pipe(pipe, cfg, stamps=stamps)
        runs.append((total, stamps))
    runs.sort(key=lambda r: r[0])
    return runs[len(runs) // 2]


def timed_call(mod, c, n=20):
    """One transformer call, warm, median — the unit the per-call
    figures are quoted in.

    It is not the same thing as the wall time of a clip divided by the
    number of calls: that carries the pipeline's own fixed cost (text
    encoder, scheduler, CFG glue), which does not shrink when the
    transformer does. Both are reported, because at four steps the
    difference between them is most of the story."""
    kw = dict(hidden_states=c["hidden"].to("cuda"),
              timestep=c["timestep"].to("cuda"),
              encoder_hidden_states=c["encoder"].to("cuda"),
              return_dict=False)
    ts = []
    with torch.no_grad():
        mod(**kw)
        torch.cuda.synchronize()
        for _ in range(n):
            t0 = time.perf_counter()
            mod(**kw)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def stepwise(mod, calls):
    """Teacher-forced parity: feed each recorded eager input, compare the
    speed field one step at a time.  This is the reliable gate — a free
    run's final latent diverges chaotically even between eager and
    compile, so its cosine is a floor to sit above, not a target."""
    out = []
    with torch.no_grad():
        for c in calls:
            y = mod(hidden_states=c["hidden"].to("cuda"),
                    timestep=c["timestep"].to("cuda"),
                    encoder_hidden_states=c["encoder"].to("cuda"),
                    return_dict=False)
            y = y[0] if isinstance(y, tuple) else y
            out.append(cosine(y.detach().cpu(), c["out"]))
    return min(out), statistics.median(out)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def decode(pipe, latents):
    """The pipeline's own tail, on a latent this arm actually produced.

    Decoding the arm's own latent rather than re-running the pipeline
    keeps each pane showing the clip its own arm made — the whole point
    of putting the clips side by side at the end."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        lat = latents.to(pipe.vae.dtype)
        cfgv = pipe.vae.config
        mean = torch.tensor(cfgv.latents_mean).view(
            1, cfgv.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        std = 1.0 / torch.tensor(cfgv.latents_std).view(
            1, cfgv.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        video = pipe.vae.decode(lat / std + mean, return_dict=False)[0]
        arr = pipe.video_processor.postprocess_video(
            video, output_type="np")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    arr = np.asarray(arr[0] if isinstance(arr, (list, tuple)) else arr)
    if arr.ndim == 5:
        arr = arr[0]
    free()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8), dt


def write_arm(out_dir, name, label, sub, color, cfg, total, stamps,
              clip, decode_s, cos_vs_eager, step_parity=(1.0, 1.0),
              paired_call_ms=None):
    d = pathlib.Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "frames.npy", clip)
    events = [{"kind": "step", "i": i, "t": round(s, 4)}
              for i, s in enumerate(stamps)]
    meta = {
        "kind": "video", "label": label, "sub": sub, "color": color,
        "steps": cfg.steps, "frames": cfg.frames, "size": cfg.size,
        "seed": SEED, "prompt": PROMPT,
        "denoise_s": round(total, 4),
        "ms_per_step": round(total / cfg.steps * 1e3, 1),
        "ms_per_call": round(total / cfg.steps / 2 * 1e3, 1),
        "decode_s": round(total, 4),
        "vae_decode_s": round(decode_s, 4),
        "done_s": round(total, 4),
        "clip_fps": 16.0,
        "final_latent_vs_eager": cos_vs_eager,
        "stepwise_worst": round(step_parity[0], 6),
        "stepwise_median": round(step_parity[1], 6),
        "paired_call_ms": (round(paired_call_ms, 1)
                           if paired_call_ms else None),
    }
    (d / "events.json").write_text(json.dumps(
        {"meta": meta, "events": events}, indent=1))
    print(f"  wrote {d}  {total:.3f} s  ({meta['ms_per_call']} ms/call)",
          flush=True)
    return meta


def aot_arm(pipe, host, handle, cfg):
    """Take the swapped transformer whole: export it, AOT-compile it,
    hand the device back, and run the package instead.

    A graph break here is a defect rather than a fallback, so this is
    also the strictest check that the bound seams are compile-clean.
    The package bakes the weights, so the host module has to vacate the
    device before the package is loaded onto it."""
    from flash_rt.structures import aot_load, aot_package
    from flash_rt.structures.aot import AotModule

    seen = {}
    orig = host.forward

    def grab(*a, **kw):
        seen.setdefault("args", a)
        seen.setdefault("kwargs", kw)
        return orig(*a, **kw)

    host.forward = grab
    run_pipe(pipe, cfg, steps=1)
    host.forward = orig

    pkg = aot_package(host, args=seen["args"], kwargs=seen["kwargs"],
                      package_path=f"/tmp/frt_aot/wan_{cfg.scheme}_aot.pt2")
    rep = handle.report()
    fb = sum(r.get("fallbacks", 0) for r in rep.values())
    calls = sum(r.get("calls", 0) for r in rep.values())
    print(f"ledger before export: {calls} structure calls, "
          f"{fb} fallbacks", flush=True)
    assert fb == 0 and calls > 0, "the ledger has to be clean to export"

    handle.detach()
    pipe.transformer = None
    host_cpu = host.to("cpu")
    free()
    loaded = aot_load(pkg)
    sig = dict(seen["kwargs"])

    def call(*a, **kw):
        for k, v in sig.items():
            if k not in kw and not torch.is_tensor(v):
                kw[k] = v
        return loaded(*a, **{k: kw[k] for k in sig if k in kw})

    return AotModule(call, host_cpu)


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
    ap.add_argument("--scheme", default="fp8_static",
                    help="fp8_static (the 20-step ship form) or "
                         "nvfp4_balance (W4A4, a few-step schedule only)")
    ap.add_argument("--aot", action="store_true",
                    help="take the attached arm whole: torch.export plus "
                         "AOTInductor instead of torch.compile")
    cfg = ap.parse_args()
    global PROMPT
    if cfg.prompt:
        PROMPT = cfg.prompt

    from diffusers import WanPipeline
    from flash_rt import structures
    from flash_rt.structures.swap import attach as swap_attach

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16
                                       ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()      # the decode is not what is being raced
    host = pipe.transformer
    out = pathlib.Path(cfg.out)
    summary = {}

    # ---- eager: the host as shipped -----------------------------------
    calls = []
    ref, _ = run_pipe(pipe, cfg, capture=calls)
    print(f"captured {len(calls)} denoise calls", flush=True)
    total, stamps = timed(pipe, cfg, cfg.repeats)
    clip, dec = decode(pipe, ref)
    mid = calls[len(calls) // 2]
    pc = timed_call(host, mid)
    print(f"eager {total:.3f} s, {pc:.1f} ms per paired call, "
          f"clip {clip.shape}, vae decode {dec:.2f} s", flush=True)
    summary["eager"] = write_arm(
        out, "eager", "Wan2.2, as shipped", "eager PyTorch", "stock",
        cfg, total, stamps, clip, dec, 1.0, paired_call_ms=pc)

    # ---- compiled: the same host, the production form ------------------
    pipe.transformer = _CloneOut(
        torch.compile(host, mode="reduce-overhead"), host)
    run_pipe(pipe, cfg)                                   # warm / compile
    total, stamps = timed(pipe, cfg, cfg.repeats)
    lat, _ = run_pipe(pipe, cfg)
    c = cosine(lat, ref)
    sw = stepwise(pipe.transformer, calls)
    clip, dec = decode(pipe, lat)
    pc = timed_call(pipe.transformer, mid)
    print(f"compiled {total:.3f} s, {pc:.1f} ms per paired call, cos vs "
          f"eager {c:.6f}, stepwise worst {sw[0]:.6f}  <- the natural "
          f"floor", flush=True)
    summary["compiled"] = write_arm(
        out, "compiled", "the same host, compiled",
        "torch.compile + graph capture", "compiled",
        cfg, total, stamps, clip, dec, round(c, 6), sw,
        paired_call_ms=pc)
    pipe.transformer = host
    torch._dynamo.reset()
    free()

    # ---- attach: discover, calibrate, qualify, bind --------------------
    def fwd():
        with torch.no_grad():
            for c_ in calls:
                host(hidden_states=c_["hidden"].to("cuda"),
                     timestep=c_["timestep"].to("cuda"),
                     encoder_hidden_states=c_["encoder"].to("cuda"),
                     return_dict=False)

    wanted = (STRUCTURES if cfg.scheme == "fp8_static"
              else tuple(x for x in STRUCTURES if x != "modnorm_qkv_chain"))
    plan = structures.auto_swaps(host, fwd, structures=wanted,
                                 scheme=cfg.scheme, verbose=False)
    handle = swap_attach(host, plan.swaps, observe=plan.observed,
                         revert=plan.revert)
    print(f"seams bound: {len(plan.swaps)} (scheme {cfg.scheme})",
          flush=True)

    if cfg.aot:
        pipe.transformer = aot_arm(pipe, host, handle, cfg)
    else:
        pipe.transformer = _CloneOut(
            torch.compile(host, mode="reduce-overhead"), host)
    run_pipe(pipe, cfg)                                   # warm / compile
    total, stamps = timed(pipe, cfg, cfg.repeats)
    lat, _ = run_pipe(pipe, cfg)
    c = cosine(lat, ref)
    sw = stepwise(pipe.transformer, calls)
    clip, dec = decode(pipe, lat)
    pc = timed_call(pipe.transformer, mid)
    print(f"attach {total:.3f} s, {pc:.1f} ms per paired call, cos vs "
          f"eager {c:.6f}, stepwise worst {sw[0]:.6f}", flush=True)
    summary["attach"] = write_arm(
        out, "attach", "+ FlashRT structures",
        ("auto_swaps + whole-graph AoT" if cfg.aot
         else "auto_swaps + compile"),
        "accent", cfg, total, stamps, clip, dec, round(c, 6), sw,
        paired_call_ms=pc)

    summary["_ladder"] = {
        "seams": len(plan.swaps),
        "vs_eager": round(summary["eager"]["denoise_s"]
                          / summary["attach"]["denoise_s"], 3),
        "vs_compiled": round(summary["compiled"]["denoise_s"]
                             / summary["attach"]["denoise_s"], 3),
        "call_vs_eager": round(summary["eager"]["paired_call_ms"]
                               / summary["attach"]["paired_call_ms"], 3),
        "call_vs_compiled": round(summary["compiled"]["paired_call_ms"]
                                  / summary["attach"]["paired_call_ms"], 3),
        "fixture": {"steps": cfg.steps, "frames": cfg.frames,
                    "size": cfg.size, "seed": SEED},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["_ladder"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
