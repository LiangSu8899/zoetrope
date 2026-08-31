#!/usr/bin/env python3
"""Alias the aarch64 build inside every cached *tagged* kernel revision.

`kernels.get_kernel(repo, version=">=1")` resolves a semver tag, so the
revision that has to carry a variant for this board is the tagged one,
not `main`. The published aarch64 artifact may sit under a neighbouring
torch minor; it is built against the torch stable ABI, and its cubin is
what decides whether it runs here — so this prints the embedded
architectures alongside every alias it makes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

PREFERRED = ["torch211-cxx11-cu130-aarch64-linux",
             "torch213-cxx11-cu130-aarch64-linux",
             "torch212-cxx11-cu130-aarch64-linux"]


def archs(so: pathlib.Path) -> list[str]:
    try:
        out = subprocess.run(["cuobjdump", "--list-elf", str(so)],
                             capture_output=True, text=True, timeout=180).stdout
    except Exception:  # noqa: BLE001
        return []
    found = set()
    for line in out.splitlines():
        for token in line.replace(".", " ").split():
            if token.startswith("sm_"):
                found.add(token)
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from kernels.utils import build_variant

    want = build_variant()
    root = pathlib.Path(args.cache or
                        pathlib.Path.home() / ".cache/huggingface/hub")
    print(f"[stage] board variant: {want}", flush=True)

    records = []
    for repo_dir in sorted(root.glob("models--flashrt--*")):
        for snapshot in sorted((repo_dir / "snapshots").glob("*")):
            build = snapshot / "build"
            if not build.is_dir():
                continue
            record = {"repo": repo_dir.name, "snapshot": snapshot.name}
            target = build / want
            if not target.exists():
                source = next((build / v for v in PREFERRED
                               if (build / v).is_dir()), None)
                if source is None:
                    record["note"] = "no aarch64 variant"
                    records.append(record)
                    continue
                target.symlink_to(source.name)
                record["aliased_from"] = source.name
            resolved = target.resolve()
            for so in resolved.glob("*.so"):
                record["archs"] = archs(so)
                break
            records.append(record)
            print(f"[stage] {repo_dir.name[18:]:42s} {snapshot.name[:8]} "
                  f"{record.get('aliased_from', want)[:34]:34s} "
                  f"{record.get('archs', record.get('note'))}", flush=True)

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
