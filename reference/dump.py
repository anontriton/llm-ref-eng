#!/usr/bin/env python3
"""Dump every intermediate activation of the reference model to .npy.

This produces the oracle: the fixed set of numbers the C++ engine is measured
against. It is deliberately boring and reproducible -- single-threaded, fp32,
fixed prompts, checksummed inputs and outputs.

    python reference/dump.py            # write oracle/activations + manifest
    python reference/dump.py --list     # show the runs without computing

Layout:
    oracle/manifest.json                one manifest covering every run
    oracle/activations/<run>/<name>.npy one file per tapped tensor

The engine writes the same layout under engine/dumps/ and oracle/compare.py
diffs the two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from weights import DEFAULT_WEIGHTS, load_gpt2

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "oracle"
ACTIVATIONS = OUT_DIR / "activations"
MANIFEST = OUT_DIR / "manifest.json"

# CLAUDE.md tolerance policy, fp32 phases. Recorded in the manifest so a dump
# carries the rules it was meant to be judged by.
TOLERANCE = {
    "max_abs": 1e-4,
    "max_rel": 1e-3,
    "rel_floor": 1e-2,
    "policy": "fp32",
}

# The fixed prompt set. Short, long, degenerate, and code, because they stress
# different things: T=1 exercises a 1x1 attention matrix, the long one exercises
# position embeddings well past the start of the table.
RUNS: dict[str, str] = {
    "capital": "The capital of France is",
    "unicorns": (
        "In a shocking finding, scientists discovered a herd of unicorns "
        "living in a remote valley"
    ),
    "code": "def fibonacci(n):",
    "newline": "\n",
    "long": (
        "The history of computing is often told as a story of hardware: "
        "vacuum tubes giving way to transistors, transistors to integrated "
        "circuits, and integrated circuits to the dense multicore processors "
        "of the present day. But the more interesting story is arguably the "
        "one about abstraction. Every generation of programmers has built a "
        "layer that let the next generation forget something the previous one "
        "had to know by heart, and the machine underneath has grown steadily "
        "less visible as a result."
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump_run(model, input_ids: torch.Tensor, run_dir: Path) -> list[dict]:
    """Run one forward pass, writing each tapped tensor to its own .npy."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in run_dir.glob("*.npy"):
        stale.unlink()

    records: list[dict] = []
    seen: set[str] = set()

    def tap(name: str, value: torch.Tensor) -> None:
        if name in seen:
            raise RuntimeError(f"tensor {name!r} tapped twice in one forward")
        seen.add(name)

        array = np.ascontiguousarray(value.detach().numpy(), dtype=np.float32)
        np.save(run_dir / f"{name}.npy", array, allow_pickle=False)
        record = {
            "order": len(records),
            "name": name,
            "file": f"{name}.npy",
            "shape": list(array.shape),
            "dtype": "float32",
            "sha256": sha256_array(array),
            # Cheap summary stats: enough to eyeball a dump without loading it,
            # and enough to spot an all-zero tensor at a glance.
            "min": float(array.min()),
            "max": float(array.max()),
            "absmax": float(np.abs(array).max()),
        }
        # Raw attention scores are only meaningful below the diagonal; the rest
        # gets masked away, and every engine is free to spell that differently.
        # The manifest carries the rule so compare.py needs no special cases.
        if name.endswith(".attn.scores"):
            record["region"] = "causal_lower_triangle"
        records.append(record)

    with torch.no_grad():
        model(input_ids, tap=tap)

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="subset of run names (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="list the runs and exit")
    args = parser.parse_args()

    names = args.runs or list(RUNS)
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        print(f"unknown run(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(RUNS)}", file=sys.stderr)
        return 2

    if args.list:
        for name in names:
            print(f"  {name:10s} {RUNS[name]!r:.70}")
        return 0

    # Single-threaded so the oracle is reproducible run to run: multithreaded
    # reductions can reassociate and shift the last couple of bits.
    torch.set_num_threads(1)
    torch.manual_seed(0)

    print("loading reference model...")
    model = load_gpt2(args.weights)

    from transformers import AutoTokenizer  # tokenization only
    tok = AutoTokenizer.from_pretrained(args.weights)

    runs = []
    for name in names:
        prompt = RUNS[name]
        input_ids = tok(prompt, return_tensors="pt").input_ids
        run_dir = ACTIVATIONS / name

        print(f"  {name:10s} T={input_ids.shape[1]:<4d} ", end="", flush=True)
        records = dump_run(model, input_ids, run_dir)
        total = sum(np.prod(r["shape"]) for r in records)
        print(f"{len(records):3d} tensors, {total * 4 / 1e6:.1f} MB")

        runs.append({
            "name": name,
            "prompt": prompt,
            "input_ids": input_ids[0].tolist(),
            "n_tokens": int(input_ids.shape[1]),
            "dir": f"activations/{name}",
            "tensors": records,
        })

    cfg = model.cfg
    manifest = {
        "schema": 1,
        "producer": "reference/dump.py",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "name": "gpt2-124m",
            "weights_sha256": sha256_file(args.weights / "model.safetensors"),
            "config": {
                "n_layer": cfg.n_layer, "n_head": cfg.n_head,
                "d_model": cfg.d_model, "d_ff": cfg.d_ff,
                "n_ctx": cfg.n_ctx, "vocab_size": cfg.vocab_size,
                "layer_norm_eps": cfg.layer_norm_eps,
            },
        },
        "dtype": "float32",
        "tolerance": TOLERANCE,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "threads": torch.get_num_threads(),
        },
        "runs": runs,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    n_tensors = sum(len(r["tensors"]) for r in runs)
    print(f"\nwrote {n_tensors} tensors across {len(runs)} runs")
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
