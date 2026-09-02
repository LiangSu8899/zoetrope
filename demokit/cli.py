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
    "arch":         ("nodes", "depth", "stretch", "model_class"),
    "runtime":      ("launches", "distinct", "legend", "families", "origins"),
    "diagram":      ("diagram", "lit", "n_nodes", "providers"),
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
        if kind == "arch":
            nodes = meta.get("nodes") or []
            names = {n.get("node") for n in nodes}
            for n in nodes:
                if n.get("parent") not in names and n.get("parent") is not None:
                    bad.append(f"{path}: node {n.get('node')!r} hangs off "
                               f"{n.get('parent')!r}, which is not in the tree")
            fired = {e.get("node") for e in events}
            for n in fired - names:
                bad.append(f"{path}: an event names {n!r}, absent from nodes")
    if events:
        ts = [e.get("t") for e in events]
        if any(t is None for t in ts):
            bad.append(f"{path}: an event has no 't'")
        elif ts != sorted(ts):
            bad.append(f"{path}: event timestamps are not sorted")
    return bad


SKILL_SRC = pathlib.Path(__file__).resolve().parent.parent / ".ai" / "skills"

POINTER = """
## Recording a demo film

When asked to show what an optimization did — side-by-side arms of one model on
one wall clock — read `{path}` first. It covers the run-directory contract, the
three recording hooks, and how to choose arms so the comparison holds.
"""


def _skills(a) -> int:
    """Install the skill where each agent looks for one."""
    if not SKILL_SRC.exists():
        print(f"no skills at {SKILL_SRC}", file=sys.stderr)
        return 1
    names = sorted(p.name for p in SKILL_SRC.iterdir() if (p / "SKILL.md").exists())

    if a.action == "list":
        for n in names:
            first = (SKILL_SRC / n / "SKILL.md").read_text().splitlines()
            desc = next((l[len("description:"):].strip() for l in first[:8]
                         if l.startswith("description:")), "")
            print(f"{n}  {desc[:88]}")
        return 0

    roots = []
    if a.project:
        base = pathlib.Path(a.project)
        roots = [base / ".claude" / "skills", base / ".agents" / "skills"]
    else:
        home = pathlib.Path.home()
        if a.claude or not (a.claude or a.agents):
            roots.append(home / ".claude" / "skills")
        if a.agents or not (a.claude or a.agents):
            roots.append(home / ".agents" / "skills")

    if a.action == "where":
        for r in roots:
            print(r)
        return 0

    import shutil
    for root in roots:
        for n in names:
            dst = root / n
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(SKILL_SRC / n / "SKILL.md", dst / "SKILL.md")
            print(f"installed {n} -> {dst}")

    if a.agents_md:
        md = pathlib.Path(a.agents_md)
        target = roots[0] / names[0] / "SKILL.md"
        text = POINTER.format(path=target)
        existing = md.read_text() if md.exists() else ""
        if "Recording a demo film" in existing:
            print(f"{md}: pointer already present")
        else:
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text(existing.rstrip() + "\n" + text)
            print(f"pointed {md} at {target}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="demokit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate one or more run directories")
    c.add_parser = None
    c.add_argument("runs", nargs="+")

    k = sub.add_parser("skills", help="install the agent skill")
    k.add_argument("action", choices=["add", "list", "where"])
    k.add_argument("--claude", action="store_true", help="~/.claude/skills")
    k.add_argument("--agents", action="store_true", help="~/.agents/skills")
    k.add_argument("--project", metavar="DIR",
                   help="install into DIR/.claude/skills and DIR/.agents/skills")
    k.add_argument("--agents-md", metavar="PATH",
                   help="also append a pointer to this AGENTS.md, for agents "
                        "that read that rather than a skills directory")

    e = sub.add_parser("explain", help="draw a published mechanism: the "
                                      "explainer panels, or the whole menu")
    e.add_argument("spec", help="an explainer name, or a path to one")
    e.add_argument("--panel", type=int, default=0)
    e.add_argument("--palette", default="midnight")
    e.add_argument("--fps", type=int, default=30)
    e.add_argument("--frame"); e.add_argument("--out")
    e.add_argument("--menu", help="every panel of this spec, one PNG")
    e.add_argument("--sheet", help="one panel in every colour system")
    e.add_argument("--film", help="every panel in order, one film")

    L = sub.add_parser("looks", help="draw one stream recording in another "
                                     "visual language, or every one at once")
    L.add_argument("runs", help="a directory of arms")
    L.add_argument("--style", default="curve")
    L.add_argument("--palette", default="midnight")
    L.add_argument("--title"); L.add_argument("--sub")
    L.add_argument("--seconds", type=float, default=7.0)
    L.add_argument("--fps", type=int, default=30)
    L.add_argument("--at", type=float, default=0.62)
    L.add_argument("--out"); L.add_argument("--frame"); L.add_argument("--sheet")

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

    if a.cmd == "skills":
        return _skills(a)

    if a.cmd == "explain":
        from demokit.compose import explain_compose as E
        argv2 = [a.spec, "--panel", str(a.panel), "--palette", a.palette,
                 "--fps", str(a.fps)]
        for flag in ("frame", "out", "menu", "sheet", "film"):
            if getattr(a, flag, None):
                argv2 += [f"--{flag}", getattr(a, flag)]
        E.main(argv2)
        return 0

    if a.cmd == "looks":
        from demokit.compose import styles_compose as S
        argv2 = ["--run", a.runs, "--style", a.style, "--palette", a.palette,
                 "--seconds", str(a.seconds), "--fps", str(a.fps),
                 "--at", str(a.at)]
        for flag in ("title", "sub", "out", "frame", "sheet"):
            if getattr(a, flag, None):
                argv2 += [f"--{flag}", getattr(a, flag)]
        sys.argv = ["demokit looks"] + argv2
        S.main()
        return 0

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
