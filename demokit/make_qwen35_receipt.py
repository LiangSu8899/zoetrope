#!/usr/bin/env python3
"""Assemble the receipt for the Qwen3.5 serving films.

Everything the handoff's checklist asks for, from the runs themselves:
the three rates kept separate, the token counts they were divided by,
both agreement measures, and the environment that produced them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def free_run_agreement(root: pathlib.Path, level: int) -> dict:
    """How long the two arms' streams stay identical, and how far they
    agree overall. This is a description, not a gate: greedy decoding
    follows its own equally valid continuation after one tie-break, so
    the number collapses as soon as the arms differ once."""
    base = json.loads(
        (root / f"c{level}" / "base" / "tokens.json").read_text())["generated"]
    att = json.loads(
        (root / f"c{level}" / "attach" / "tokens.json").read_text())["generated"]
    same = total = 0
    prefixes = []
    for xs, ys in zip(base, att):
        n = 0
        while n < min(len(xs), len(ys)) and xs[n] == ys[n]:
            n += 1
        prefixes.append(n)
        for x, y in zip(xs, ys):
            total += 1
            same += int(x == y)
    return {"rate": round(same / total, 4) if total else None,
            "identical_prefix_tokens": prefixes,
            "note": "free-run; not the parity gate (see teacher_forced)"}


def roofline(model_dir: pathlib.Path, bandwidth_gb_s: float,
             seat_bits: float, linear_attn_bound: bool,
             head_bits: float = 16.0) -> dict:
    """Bytes a single decode step must read, and the rate that caps.

    A rate nobody roofline-checked is a rate nobody has reason to
    believe. On a routed MoE the sum is not "the model": each token
    reads only the experts it routed to, plus everything dense. The
    terms are listed so a reader can disagree with one of them.
    """
    cfg = json.loads((model_dir / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    L = t["num_hidden_layers"]
    H, I = t["hidden_size"], t["moe_intermediate_size"]
    k, V = t["num_experts_per_tok"], t["vocab_size"]
    full = t.get("full_attention_interval")
    n_full = L // full if full else 10
    n_lin = L - n_full
    bf16 = 2.0
    seat = seat_bits / 8.0

    routed = k * (2 * I * H + H * I) * L               # params, per token
    shared = (2 * I * H + H * I) * L
    attn_full = n_full * (4.5 * H * H + H * 2 * H)     # qkv + o, approx
    attn_lin = n_lin * (6 * H * H + H * 2 * H)         # in_proj_qkvz + out
    gate = L * t["num_experts"] * H
    head = V * H

    def total(bits_routed, bits_seats, bits_linear_attn, bits_head):
        return (routed * bits_routed + shared * bits_seats
                + attn_full * bits_seats + attn_lin * bits_linear_attn
                + gate * bf16 + head * bits_head)

    # 30 of this model's 40 layers are gated-delta linear attention, and
    # they name their projections `in_proj_qkvz` / `out_proj` — which the
    # usual `qkv_proj,o_proj` seat list does not match. Whether they were
    # bound is the single biggest term in the attached arm's budget, so
    # the check asks rather than assumes.
    lin = seat if linear_attn_bound else bf16
    hd = head_bits / 8.0
    out = {"linear_attn_projections_bound": bool(linear_attn_bound),
           "lm_head_bits_attached": head_bits}
    for name, b in (("base", total(bf16, bf16, bf16, bf16)),
                    ("attach", total(seat, seat, lin, hd))):
        ms = b / (bandwidth_gb_s * 1e9) * 1e3
        out[name] = {"bytes_per_token_gb": round(b / 1e9, 2),
                     "ms_floor": round(ms, 2),
                     "tok_s_ceiling": round(1e3 / ms, 1)}
    out["bandwidth_gb_s"] = bandwidth_gb_s
    out["seat_bits"] = seat_bits
    out["terms"] = (
        "routed experts (top-k only) + shared expert + attention "
        "projections + router gate + lm_head. The router gate stays "
        "bf16 in both arms because no seat claims it. The "
        "linear-attention projections are "
        + ("bound at seat bits" if linear_attn_bound else "bf16 in both "
           "arms because no seat claims them")
        + f"; the lm_head reads at {head_bits:g} bits in the attached "
        "arm. This is a weight-traffic ceiling only: it is blind to "
        "launches and activations, so a measured rate above it means "
        "a term here is wrong, not that a kernel beat physics.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--floor-from", default=None,
                    help="gate_base.json from a base arm scored against "
                         "its own tokens — the gate's own floor")
    ap.add_argument("--control-from", default=None,
                    help="gate_attach.json from the smaller seat set, "
                         "scored against the same reference stream")
    ap.add_argument("--head-bits", type=float, default=16.0,
                    help="bits per lm_head weight in the attached arm: "
                         "16 when no seat claims the head, 8 for the "
                         "W8A16 slab path, 4.5 for an adopted NVFP4 "
                         "pack. Read it off the adapter's bind line.")
    ap.add_argument("--bind-line", default=None,
                    help="a log to lift the adapter's own bind report "
                         "line from, stored verbatim as the evidence "
                         "for --head-bits")
    ap.add_argument("--seat-bits", type=float, default=4.5,
                    help="effective bits per weight in a bound seat: "
                         "4 for the value plus the block scale")
    ap.add_argument("--bandwidth", type=float, default=273.0,
                    help="GB/s used for the roofline check")
    args = ap.parse_args()

    root = pathlib.Path(args.runs)
    levels = [int(x) for x in args.levels.split(",")]
    metrics = {arm: json.loads((root / f"metrics_{arm}.json").read_text())
               for arm in ("base", "attach")}

    per_level = {}
    for n in levels:
        row = {}
        for arm in ("base", "attach"):
            m = json.loads(
                (root / f"c{n}" / arm / "events.json").read_text())["meta"]
            row[arm] = {k: m[k] for k in (
                "ttft_ms_median", "ttft_ms_p90", "decode_tok_s_per_stream",
                "aggregate_tok_s", "n_tokens_total", "done_s", "wall_s")}
        row["aggregate_speedup"] = round(
            row["attach"]["aggregate_tok_s"] / row["base"]["aggregate_tok_s"], 3)
        row["per_request_speedup"] = round(
            row["attach"]["decode_tok_s_per_stream"]
            / row["base"]["decode_tok_s_per_stream"], 3)
        row["ttft_ratio"] = round(
            row["attach"]["ttft_ms_median"] / row["base"]["ttft_ms_median"], 3)
        row["free_run_agreement"] = free_run_agreement(root, n)
        per_level[f"c{n}"] = row

    att = metrics["attach"].get("attach", {})
    seats_asked = str(att.get("seats_asked", ""))
    # the adapter takes no seat list, so what it actually bound is read
    # back off its own per-seam report rather than off the request
    bound_names = " ".join(att.get("adapter_report", {}))
    linear_attn_bound = ("in_proj_qkvz" in seats_asked
                         or "in_proj_qkvz" in bound_names)

    def _gate(path, extra=None):
        f = pathlib.Path(path)
        if not f.exists():
            return None
        g = json.loads(f.read_text())
        g.update(extra or {})
        return g

    gate = _gate(root / "gate_attach.json")
    floor = _gate(args.floor_from, {
        "what": "the unmodified engine scored against its own tokens. It "
                "is not 1.0 and should not be: the free run comes off the "
                "engine's decode path and the gate scores through a "
                "prefill, and the two numeric paths part company at "
                "near-ties. This is the reference every arm is read "
                "against."}) if args.floor_from else None
    control = _gate(args.control_from, {
        "what": "the same gate, same reference stream, with only the "
                "expert banks and the ten full-attention layers bound — "
                "the thirty gated-delta layers left alone. The difference "
                "against the full seat set is what those thirty cost."}) \
        if args.control_from else None
    if gate is not None:
        gate["measured_on"] = (
            "the base arm's own streams, forced in position by position; "
            "the prompts are the recorder's eight, rendered through the "
            "model's chat template with thinking off")

    try:
        vllm_v = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:                                       # noqa: BLE001
        vllm_v = None

    receipt = {
        "model": args.model,
        "what": "a serving engine at four batch sizes, with and without "
                "structure seats bound into it",
        "engine": {"vllm": vllm_v, "arms_are_separate_processes": True,
                   "why": "a serving engine bakes its model tree into a "
                          "compiled, graph-captured decode path at load "
                          "time; attaching after that fails by "
                          "construction, so the two arms cannot share a "
                          "process and the streams are compared "
                          "afterwards from the ids each run wrote down"},
        "attach": dict(metrics["attach"]["attach"], **(
            {} if not args.bind_line else
            {"bind_report_lines": [
                ln.strip() for ln in pathlib.Path(args.bind_line)
                .read_text(errors="replace").splitlines()
                if "[structures.vllm]" in ln]})),
        "teacher_forced_parity": {
            "arm": gate,
            "floor_base_against_itself": floor,
            "control_without_gated_delta_seats": control,
            "reading": (
                "the arm's rate is read against the floor, not against "
                "1.0. Most of the distance is the expert bank: 64.4 GB "
                "at four bits, and this family has no eight-bit member "
                "to fall back to, so it is not a knob."
                + ("" if control is None else
                   " The thirty gated-delta projections account for the "
                   "difference between the two arm rows.")),
        },
        "levels": per_level,
        "roofline": roofline(pathlib.Path(args.model_dir), args.bandwidth,
                             args.seat_bits,
                             linear_attn_bound, args.head_bits),
        "boundary": "tokens are stamped as the engine emits them, from "
                    "LLMEngine.step(); every rate divides by tokens "
                    "actually produced, never by the cap asked for",
    }
    pathlib.Path(args.out).write_text(json.dumps(receipt, indent=1))

    print(f"{'c':>4} {'base':>8} {'attach':>8} {'x':>6} {'ttft b':>8} "
          f"{'ttft a':>8} {'free-run':>9}")
    for k, r in per_level.items():
        print(f"{k:>4} {r['base']['aggregate_tok_s']:>8} "
              f"{r['attach']['aggregate_tok_s']:>8} "
              f"{r['aggregate_speedup']:>6} "
              f"{r['base']['ttft_ms_median']:>8} "
              f"{r['attach']['ttft_ms_median']:>8} "
              f"{r['free_run_agreement']['rate']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
