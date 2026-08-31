#!/usr/bin/env python3
"""Pin each FlashRT kernel package to a revision that has a build for
this board — and say which one, and why.

Two things make the obvious pin wrong here:

* the Hub API that resolves ``version=">=1"`` is unreachable from the
  board, so the revision has to be named explicitly; and
* **the newest release is not always the one built for aarch64.**
  `flashrt/fp4-fused-ops` publishes aarch64 at `v1` and x86-only at
  `v2`, so "highest tag" pins the package out of existence on this
  board and every FP4 rung of the pi0.5 region ladder refuses with
  "package unavailable" — a refusal the pin caused, not the board.

So the choice is: among the cached revisions, keep the ones that carry
the build variant this torch asks for, and prefer `main`, then the
highest tag. Prints shell `export` lines; `--json` prints the same
choice with the evidence for the receipt.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


def wanted_variant() -> str:
    from kernels.utils import build_variant

    return build_variant()


#: `kernels` accepts the exact build variant or either generic one; an
#: arch-agnostic package (the CuTe-DSL attention runtime is pure Python)
#: ships only as `torch-cuda`, and treating that as "no build for this
#: board" pins it out of existence exactly like a wrong tag does.
GENERIC_VARIANTS = ("torch-cuda", "torch-universal")


def candidates(repo_dir: pathlib.Path, variant: str) -> dict:
    """Every cached revision, and whether it can serve this board."""
    out = {}
    for snapshot in sorted((repo_dir / "snapshots").glob("*")):
        build = snapshot / "build"
        variants = sorted(p.name for p in build.glob("*")) if build.is_dir() \
            else []
        out[snapshot.name] = {
            "variants": variants,
            "usable": bool({variant, *GENERIC_VARIANTS}
                           .intersection(variants)),
            "aarch64": [v for v in variants if v.endswith("aarch64-linux")],
        }
    return out


def refs(repo_dir: pathlib.Path) -> dict:
    found = {}
    for ref in (repo_dir / "refs").glob("*"):
        if ref.is_file():
            found[ref.name] = ref.read_text().strip()
    return found


def tag_order(name: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", name)) or (0,)


def choose(repo_dir: pathlib.Path, variant: str) -> dict:
    seen = candidates(repo_dir, variant)
    pointers = refs(repo_dir)
    usable = {sha for sha, info in seen.items() if info["usable"]}
    record = {"repo": repo_dir.name.removeprefix("models--").replace("--", "/"),
              "wanted_variant": variant,
              "revisions": {sha[:8]: info["variants"] and
                            (info["usable"] and "usable" or "other-arch")
                            or "no build"
                            for sha, info in seen.items()},
              "refs": {k: v[:8] for k, v in pointers.items()}}

    if pointers.get("main") in usable:
        record.update(revision=pointers["main"], via="main")
        return record
    tags = sorted((t for t in pointers if t != "main"), key=tag_order,
                  reverse=True)
    for tag in tags:
        if pointers[tag] in usable:
            record.update(revision=pointers[tag], via=tag)
            if pointers.get("main") is not None:
                record["note"] = ("main has no build for this board; "
                                  f"pinned {tag}")
            return record
    if usable:                       # a revision no ref points at
        record.update(revision=sorted(usable)[0], via="cached")
        return record
    record["revision"] = None
    record["note"] = "no cached revision carries this board's variant"
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.cache or
                        pathlib.Path.home() / ".cache/huggingface/hub")
    variant = wanted_variant()
    records = [choose(d, variant) for d in
               sorted(root.glob("models--flashrt--*")) if (d / "refs").is_dir()]

    if args.json:
        print(json.dumps({"variant": variant, "packages": records}, indent=2))
        return 0
    for record in records:
        if not record.get("revision"):
            continue
        var = "FRT_KERNEL_REV_" + re.sub(r"[^A-Za-z0-9]", "_",
                                         record["repo"]).upper()
        print(f"export {var}={record['revision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
