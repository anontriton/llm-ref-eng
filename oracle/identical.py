#!/usr/bin/env python3
"""Check that two engine dumps are bit-identical, tensor by tensor.

    python oracle/identical.py web/build/dumps web/build/dumps-mt
    python oracle/identical.py engine/dumps engine/dumps-t8 --run long

compare.py asks whether an engine is within tolerance of the reference; this
asks a stricter question of two engine runs -- whether they produced exactly
the same bits. It is how "threads, SIMD and optimizations do not change a
number" is checked rather than argued: the wasm_simd128 build against wasm
scalar, the threaded build against the single-threaded one, an optimized
engine against the commit before it.

It compares the per-tensor sha256 each manifest records over the raw float32
bytes, and with --verify re-hashes the files too, so a stale dump cannot pass
on the strength of its manifest. The two dumps must describe the same weights,
the same runs and the same input ids, and hold the same set of tensors;
anything else is a usage error, not a difference.

Exit codes: 0 identical, 1 a tensor differs, 2 provenance or usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[dict, Path]:
    manifest = path / "manifest.json" if path.is_dir() else path
    if not manifest.is_file():
        raise SystemExit(f"no manifest at {manifest}")
    return json.loads(manifest.read_text()), manifest.parent


def file_hash(base: Path, run: dict, record: dict) -> str:
    array = np.load(base / run["dir"] / record["file"], allow_pickle=False)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="a dump directory or its manifest.json")
    parser.add_argument("b", type=Path, help="the dump to hold it to")
    parser.add_argument("--run", action="append", dest="runs",
                        help="only this run (repeatable)")
    parser.add_argument("--verify", action="store_true",
                        help="re-hash every file, not just trust the manifests")
    args = parser.parse_args()

    (ma, base_a), (mb, base_b) = load(args.a), load(args.b)
    problems = []
    if ma["model"]["weights_sha256"] != mb["model"]["weights_sha256"]:
        problems.append("different weights")
    runs_a = {r["name"]: r for r in ma["runs"]}
    runs_b = {r["name"]: r for r in mb["runs"]}
    names = args.runs or sorted(runs_a)
    for name in names:
        ra, rb = runs_a.get(name), runs_b.get(name)
        if ra is None or rb is None:
            problems.append(f"run {name} missing from {'a' if ra is None else 'b'}")
        elif ra["input_ids"] != rb["input_ids"]:
            problems.append(f"run {name}: input_ids differ")
        elif ra.get("kv_row") != rb.get("kv_row"):
            problems.append(f"run {name}: kv_row {ra.get('kv_row')} vs {rb.get('kv_row')}")
    if not args.runs and set(runs_a) != set(runs_b):
        problems.append(f"different runs: {sorted(set(runs_a) ^ set(runs_b))}")
    if problems:
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("NOT COMPARABLE", file=sys.stderr)
        return 2

    checked, differ = 0, []
    for name in names:
        ra, rb = runs_a[name], runs_b[name]
        ta = {t["name"]: t for t in ra["tensors"]}
        tb = {t["name"]: t for t in rb["tensors"]}
        if set(ta) != set(tb):
            print(f"  run {name}: tensors differ: {sorted(set(ta) ^ set(tb))[:5]}",
                  file=sys.stderr)
            return 2
        for tname in sorted(ta, key=lambda n: ta[n]["order"]):
            ha, hb = ta[tname]["sha256"], tb[tname]["sha256"]
            if args.verify:
                if file_hash(base_a, ra, ta[tname]) != ha or file_hash(base_b, rb, tb[tname]) != hb:
                    print(f"  run {name}: {tname} does not match its manifest (stale dump?)",
                          file=sys.stderr)
                    return 2
            checked += 1
            if ha != hb:
                differ.append((name, tname))

    if differ:
        print(f"  {len(differ)} of {checked} tensors differ; first, in forward order:")
        for name, tname in differ[:5]:
            print(f"    {name}: {tname}")
        print("NOT IDENTICAL", file=sys.stderr)
        return 1
    print(f"IDENTICAL  ({checked} tensors across {len(names)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
