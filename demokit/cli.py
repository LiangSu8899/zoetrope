"""`demokit` — draw recorded runs, and check that a run is well formed.

Recording happens inside whatever process runs the model, through
`demokit.hook`. This CLI covers the two things that happen afterwards.

    demokit check runs/myfilm/attach          # is this run directory valid?
    demokit draw  runs/myfilm --out film.webm # one chapter, arms in order
    demokit draw  --spec spec.json --out film.webm
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REQUIRED = ("kind", "label", "sub", "color", "done_s")
NEEDS = {
    "stream":       ("ttft_ms", "decode_tok_s", "n_tokens"),
    "video":        ("steps", "ms_per_step"),
    "stream_batch": ("concurrency", "aggregate_tok_s",
                     "decode_tok_s_per_stream", "ttft_ms_median"),
}
#: Robot rollouts predate the race contract and carry their own meta. They are
#: read by compose/sim.py, which derives the wall clock from `infer_ms` and the
#: control rate rather than reading a `t` off each event. Unifying the two is
#: item 1 in docs/PROTOCOL.md; until then, both shapes are valid and `draw`
#: picks the compositor that can read what it was given.
SIM_NEEDS = ("arm", "host", "median_infer_ms", "steps")


def shape_of(meta: dict) -> str:
    """Which contract this run was written against: 'race' or 'sim'."""
    return "race" if "kind" in meta else "sim"


def check(path: pathlib.Path) -> list[str]:
    """Every reason this run directory would not draw."""
    bad = []
    ev = path / "events.json"
    if not ev.exists():
        return [f"{path}: no events.json"]
    try:
        blob = json.loads(ev.read_text())
    except json.JSONDecodeError as e:
        return [f"{path}: events.json is not valid JSON ({e})"]
    meta, events = blob.get("meta", {}), blob.get("events", [])
    if not events:
        bad.append(f"{path}: no events")

    if shape_of(meta) == "sim":
        for k in SIM_NEEDS:
            if k not in meta:
                bad.append(f"{path}: a robot rollout needs meta[{k!r}]")
        if not (path / "frames.npy").exists():
            bad.append(f"{path}: a robot rollout needs frames.npy")
        if events and "infer_ms" not in events[0]:
            bad.append(f"{path}: a robot rollout event needs 'infer_ms'")
        return bad

    for k in REQUIRED:
        if k not in meta:
            bad.append(f"{path}: meta is missing {k!r}")
    kind = meta.get("kind")
    if kind not in NEEDS:
        bad.append(f"{path}: kind {kind!r} is not one of {tuple(NEEDS)}")
    else:
        for k in NEEDS[kind]:
            if k not in meta:
                bad.append(f"{path}: a {kind} pane needs meta[{k!r}]")
        if kind == "video" and not (path / "frames.npy").exists():
            bad.append(f"{path}: a video pane needs frames.npy")
    if events:
        ts = [e.get("t") for e in events]
        if any(t is None for t in ts):
            bad.append(f"{path}: an event has no 't'")
        elif ts != sorted(ts):
            bad.append(f"{path}: event timestamps are not sorted")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="demokit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate one or more run directories")
    c.add_parser = None
    c.add_argument("runs", nargs="+")

    d = sub.add_parser("draw", help="compose runs into a film")
    d.add_argument("runs", nargs="?", help="a directory of arms")
    d.add_argument("--arms", help="comma-separated order; default: sorted")
    d.add_argument("--spec", help="a multi-chapter spec JSON")
    d.add_argument("--out", required=True)
    d.add_argument("--title"); d.add_argument("--subtitle")
    d.add_argument("--note"); d.add_argument("--footer")
    d.add_argument("--pane", type=int, default=400)
    d.add_argument("--fps", type=int, default=30)
    d.add_argument("--speed", type=float, default=1.0)

    a = ap.parse_args(argv)

    if a.cmd == "check":
        bad = [b for r in a.runs for b in check(pathlib.Path(r))]
        for b in bad:
            print(b, file=sys.stderr)
        print(f"{len(a.runs) - len({b.split(':')[0] for b in bad})}"
              f"/{len(a.runs)} run(s) ready to draw")
        return 1 if bad else 0

    if a.runs and not a.spec:
        first = next((p for p in sorted(pathlib.Path(a.runs).iterdir())
                      if (p / "events.json").exists()), None)
        if first is not None:
            meta = json.loads((first / "events.json").read_text())["meta"]
            if shape_of(meta) == "sim":
                import subprocess
                cmd = [sys.executable,
                       str(pathlib.Path(__file__).parent / "compose"
                           / "sim_compose.py"),
                       "--runs", a.runs, "--out", a.out,
                       "--pane", str(a.pane), "--fps", str(a.fps),
                       "--speed", str(a.speed)]
                for flag in ("title", "note", "footer"):
                    if getattr(a, flag, None):
                        cmd += [f"--{flag}", getattr(a, flag)]
                return subprocess.run(cmd).returncode

    from demokit.compose import race_compose as R
    if a.spec:
        blob = json.loads(pathlib.Path(a.spec).read_text())
        chapters = [R.load_chapter(c) for c in blob["chapters"]]
        R.render(chapters, a.out, fps=blob.get("fps", a.fps),
                 pane=blob.get("pane", a.pane),
                 speed=blob.get("speed", a.speed))
        return 0
    if not a.runs:
        ap.error("give a runs directory, or --spec")
    root = pathlib.Path(a.runs)
    arms = (a.arms.split(",") if a.arms else
            sorted(p.name for p in root.iterdir()
                   if (p / "events.json").exists()))
    bad = [b for arm in arms for b in check(root / arm)]
    if bad:
        for b in bad:
            print(b, file=sys.stderr)
        return 1
    chapter = R.load_chapter({
        "runs": str(root), "arms": arms, "title": a.title,
        "subtitle": a.subtitle, "note": a.note, "footer": a.footer,
        "tail": 2.0, "seconds": None})
    R.render([chapter], a.out, fps=a.fps, pane=a.pane, speed=a.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
