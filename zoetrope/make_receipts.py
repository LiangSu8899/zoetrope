#!/usr/bin/env python3
"""Assemble a receipt from a directory of recorded arms.

Everything published about a film has to be traceable to the run that
produced it: per-arm latency, step counts, success, the swap/refusal
ledger, the board's power mode, the versions, and the checks that ran
before the arm was allowed to be recorded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def arm_record(run_dir: pathlib.Path) -> dict:
    blob = json.loads((run_dir / "events.json").read_text())
    meta, events = blob["meta"], blob["events"]
    latencies = [e["infer_ms"] for e in events if e["infer_ms"] is not None]
    report = meta.get("host_report", {})
    record = {
        "arm": run_dir.name,
        "host": report.get("host"),
        "median_infer_ms": round(meta["median_infer_ms"], 3),
        "policy_hz": round(meta["policy_hz"], 2) if meta["policy_hz"] else None,
        "p10_infer_ms": round(meta["p10_infer_ms"], 3) if latencies else None,
        "p90_infer_ms": round(meta["p90_infer_ms"], 3) if latencies else None,
        "iqr_ms": (round(meta["p90_infer_ms"] - meta["p10_infer_ms"], 3)
                   if latencies else None),
        "control_steps": meta["control_steps"],
        "policy_calls": meta["policy_calls"],
        "success": meta["success"],
        "task_time_s": round(episode_seconds(events, meta["control_hz"]), 2),
        "timed_region": report.get("timed_region", meta.get("timed_region")),
        "input_sensitivity": meta.get("input_sensitivity"),
    }
    for key in ("swaps", "observed", "refused", "parity", "ledger",
                "kernel_unavailable", "capture", "precision",
                "host_compile_peeled", "observation_rebuilds",
                # the explicit structure pipeline's own record
                "assembly", "book", "regions_refused", "seats",
                "seats_dropped_under_regions", "eager_parity_cosine",
                "captured_parity_cosine",
                # the native tier and the environment shims this board
                # needed, so a reader can tell what was not stock
                "load_model_kwargs", "frontend_infer_latency",
                "flash_attn", "scalar_tensor_cache",
                # GR00T: the explicit book, the frontend tier, and the
                # pinned flow-matching noise every arm integrates from
                "seats_bound", "refusals", "attention_variants",
                "cadence_statics", "graph_lowering", "noise_pinned",
                "parity_cosine_vs_host", "frontend", "prompt_tokens",
                "vision_tokens", "denoising_steps", "video_keys",
                "embodiment_tag", "frontend_latency",
                "frontend_embodiment_tag", "frontend_embodiment_id",
                "parity_cosine_raw_chunk"):
        if key in report:
            record[key] = report[key]
    return record


def first_decision_parity(root: pathlib.Path, arms: dict) -> dict:
    """Every arm's first action, against the eager arm's first action.

    Step 0 is the one decision every arm made from a byte-identical
    observation — same env, same seed, same initial state, same settle
    steps — so it is the only cross-arm comparison the rollouts
    themselves can support. After it the trajectories diverge, which is
    a property of closed-loop re-planning, not of an arm.
    """
    import math

    def step0(name):
        blob = json.loads((root / name / "events.json").read_text())
        return blob["events"][0]["action"]

    base_name = next((n for n in ("eager", "isaac_eager", "lerobot_eager")
                      if n in arms), None)
    if base_name is None:
        return {}
    base = step0(base_name)
    out = {"reference_arm": base_name, "reference_action": base, "arms": {}}
    for name in arms:
        if name == base_name:
            continue
        other = step0(name)
        if len(other) != len(base):
            out["arms"][name] = {"error": "action width differs"}
            continue
        num = sum(a * b for a, b in zip(base, other))
        den = (math.sqrt(sum(a * a for a in base))
               * math.sqrt(sum(b * b for b in other)))
        out["arms"][name] = {
            "cosine": round(num / den, 7) if den else None,
            "max_abs": round(max(abs(a - b)
                                 for a, b in zip(base, other)), 6),
            "per_dim_abs": [round(abs(a - b), 6)
                            for a, b in zip(base, other)],
        }
    return out


def episode_seconds(events, control_hz: float) -> float:
    """The film's own clock: the robot waits for its policy every step."""
    total, dt = 0.0, 1.0 / control_hz
    for event in events:
        if event["infer_ms"] is not None:
            total += event["infer_ms"] / 1000.0
        total += dt
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+",
                    help="one or more directories of arm sub-directories")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default=None)
    ap.add_argument("--annex", default=None,
                    help="a JSON file merged in at the top level — for "
                         "measurements taken at another boundary (a kernel "
                         "benchmark, say) that belong in the same receipt")
    args = ap.parse_args()

    films = {}
    board = versions = None
    for runs in args.runs:
        root = pathlib.Path(runs)
        arms = {}
        for run_dir in sorted(root.iterdir()):
            if not (run_dir / "events.json").exists():
                continue
            arms[run_dir.name] = arm_record(run_dir)
            meta = json.loads((run_dir / "events.json").read_text())["meta"]
            board = board or meta.get("board")
            versions = versions or {
                k: meta.get("host_report", {}).get(k)
                for k in ("torch", "transformers", "device_name")}
        if not arms:
            continue
        first = json.loads(
            (root / next(iter(arms)) / "events.json").read_text())["meta"]
        films[root.name] = {
            "suite": first["suite"], "task_id": first["task_id"],
            "trial": first["trial"], "seed": first["seed"],
            "task": first["task"], "replan": first["replan"],
            "control_hz": first["control_hz"],
            "first_decision_parity": first_decision_parity(root, arms),
            "arms": arms,
        }

    receipt = {
        "model": args.model,
        "device": (versions or {}).get("device_name"),
        "board": board,
        "versions": versions,
        "protocol": (
            "one arm per process; the robot waits for its own policy; "
            "re-plan every control step; each arm runs to its own "
            "completion; per-decision wall latency, median over the "
            "episode; host preprocessing and action decode outside the "
            "timed region for every arm alike"),
        "films": films,
    }
    if args.note:
        receipt["note"] = args.note
    if args.annex:
        receipt.update(json.loads(pathlib.Path(args.annex).read_text()))
    pathlib.Path(args.out).write_text(json.dumps(receipt, indent=2))

    for name, film in films.items():
        print(f"\n{name}: {film['task']}")
        for arm, rec in film["arms"].items():
            print(f"  {arm:16s} {rec['median_infer_ms']:8.2f} ms "
                  f"{rec['policy_hz'] or 0:6.1f} Hz  "
                  f"{rec['control_steps']:4d} steps  "
                  f"{rec['task_time_s']:6.2f} s  "
                  f"success={rec['success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
