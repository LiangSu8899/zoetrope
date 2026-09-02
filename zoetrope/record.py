#!/usr/bin/env python3
"""Record one arm of one film. One arm per process — never two.

    python record.py --host openpi_pi05 --arm eager \
        --suite libero_spatial --task-id 0 --trial 0 \
        --out runs/pi05_race/openpi_eager
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def board_recipe() -> dict:
    """Power mode and clocks — a number without them is not comparable."""
    out = {}
    for label, cmd in (("nvpmodel", ["nvpmodel", "-q"]),
                       ("jetson_clocks", ["jetson_clocks", "--show"])):
        try:
            out[label] = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=20).stdout.strip()[:600]
        except Exception as exc:  # noqa: BLE001
            out[label] = f"unavailable: {type(exc).__name__}"
    for path, label in (
            ("/sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq", "gpu_cur_freq"),
            ("/sys/kernel/debug/bpmp/debug/clk/emc/rate", "emc_rate")):
        try:
            out[label] = pathlib.Path(path).read_text().strip()
        except Exception:  # noqa: BLE001
            pass
    return out


def build_host(args):
    if args.host == "openpi_pi05":
        from host_pi05_openpi import OpenPiPi05Host
        return OpenPiPi05Host(checkpoint=args.checkpoint, arm=args.arm,
                              config_name=args.config_name,
                              num_steps=args.num_steps,
                              fixed_noise=not args.fresh_noise,
                              seed=args.noise_seed,
                              compile_mode=args.compile_mode,
                              host_src=args.host_src)
    if args.host == "lerobot_pi05":
        from host_pi05_lerobot import LeRobotPi05Host
        return LeRobotPi05Host(checkpoint=args.checkpoint, arm=args.arm,
                               lerobot_src=args.lerobot_src,
                               tokenizer=args.tokenizer,
                               num_steps=args.num_steps,
                               fixed_noise=not args.fresh_noise,
                               seed=args.noise_seed,
                               compile_mode=args.compile_mode)
    if args.host == "explicit_pi05":
        from host_pi05_explicit import ExplicitPi05Host
        return ExplicitPi05Host(checkpoint=args.checkpoint,
                                lerobot_src=args.lerobot_src,
                                examples_src=args.host_src,
                                arm=args.arm,
                                tokenizer=args.tokenizer,
                                num_views=args.num_views)
    if args.host == "native_pi05":
        from host_pi05_native import NativePi05Host
        return NativePi05Host(checkpoint=args.checkpoint,
                              num_views=args.num_views,
                              use_fp4=args.use_fp4,
                              action_horizon=args.action_horizon,
                              fixed_noise=not args.fresh_noise,
                              seed=args.noise_seed)
    if args.host == "isaac_groot":
        from host_groot_isaac import IsaacGrootHost
        return IsaacGrootHost(checkpoint=args.checkpoint, arm=args.arm,
                              host_src=args.host_src,
                              embodiment_tag=args.embodiment_tag or
                              "libero_sim",
                              num_views=args.num_views,
                              denoising_steps=args.num_steps,
                              compile_mode=args.compile_mode)
    if args.host == "explicit_groot":
        from host_groot_explicit import ExplicitGrootHost
        return ExplicitGrootHost(checkpoint=args.checkpoint, arm=args.arm,
                                 host_src=args.host_src,
                                 examples_src=args.examples_src,
                                 embodiment_tag=args.embodiment_tag or
                                 "libero_sim",
                                 num_views=args.num_views,
                                 denoising_steps=args.num_steps)
    if args.host == "lerobot_groot":
        from host_groot_lerobot import LeRobotGrootHost
        return LeRobotGrootHost(checkpoint=args.checkpoint, arm=args.arm,
                                lerobot_src=args.lerobot_src,
                                embodiment_tag=args.embodiment_tag,
                                num_views=args.num_views,
                                compile_mode=args.compile_mode)
    if args.host == "native_groot":
        from host_groot_native import NativeGrootHost
        return NativeGrootHost(checkpoint=args.checkpoint,
                               host_src=args.host_src,
                               embodiment_tag=args.embodiment_tag or
                               "libero_sim",
                               use_fp4=args.use_fp4,
                               num_views=args.num_views,
                               denoising_steps=args.num_steps or 4,
                               action_horizon=args.action_horizon)
    raise SystemExit(f"unknown host {args.host}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--arm", default="eager")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--lerobot-src", default=None)
    ap.add_argument("--host-src", default=None)
    ap.add_argument("--examples-src", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--embodiment-tag", default=None)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--fresh-noise", action="store_true",
                    help="draw a new flow-matching sample every call")
    ap.add_argument("--replan", type=int, default=1)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--num-views", type=int, default=2)
    ap.add_argument("--action-horizon", type=int, default=None)
    ap.add_argument("--use-fp4", action="store_true")
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--warmup-rounds", type=int, default=20)
    ap.add_argument("--settle-rounds", type=int, default=60)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import simloop

    host = build_host(args)
    host.warmup_rounds = args.warmup_rounds
    host.settle_rounds = args.settle_rounds
    host.settle_seconds = args.settle_seconds
    host.build()

    out_dir = pathlib.Path(args.out)
    meta = simloop.rollout(
        host,
        suite=args.suite, task_id=args.task_id, trial=args.trial,
        seed=args.seed, replan=args.replan, out_dir=out_dir,
        resize_to=args.resize, max_steps=args.max_steps,
        extra_meta={"arm": args.arm,
                    "board": board_recipe(),
                    "argv": sys.argv[1:]})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
