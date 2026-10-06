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

# Runs given as token ids rather than text. The prompts above stop at position
# 89, the eval's windows at 511, and the demo lets a sequence run to 1023, so
# without this most of the context window -- most of wpe, and attention over a
# long past -- was never checked layer by layer. "context" fills the window
# exactly: the first n_ctx ids of the pinned eval corpus, checked against the
# checksum eval/corpus.json records for the whole of it, so it adds no new
# pinned input. Its dump is large -- each attention tensor is 12 x 1024 x 1024
# -- about 2.0 GB of the oracle's 2.1.
ID_RUNS: dict[str, str] = {
    "context": "eval/corpus.tsv",
}

# Runs whose reference is computed in float64 rather than float32.
#
# At 1024 positions the fp32 rule stops being satisfiable by anything fp32.
# Measured against this same model run in float64, PyTorch's own fp32 forward
# breaks |x - ref| <= 1e-4 + 1e-3 * |ref| on 8 tensors, worst at 12x its
# budget, and the engine on 4, worst at 10x -- every miss an element near zero
# made by heavy cancellation, a score of -0.04 from terms summing to 148, a
# logit of -0.018 in a row that runs to 100. Holding the engine to PyTorch's
# fp32 numbers there measured PyTorch's rounding, not the engine: on every
# failing tensor the engine was the closer of the two to float64.
#
# So for these runs the oracle is the float64 forward, rounded to float32 for
# storage, and each tensor records `fp32_budget`: how much of the rule's
# budget PyTorch's fp32 forward uses against it, for comparison.
#
# What gates is the KV-cached dump at the last position, held to the rule
# unchanged: row 1023 attends over keys and values from every earlier
# position, so their errors reach it, and it passes with room to spare. The
# whole-sequence dump is compared and reported but does not gate. "No farther
# from float64 than PyTorch fp32, tensor by tensor" was tried as the gate and
# is not sound: it pits two independent roundings against each other 123
# times. The engine was the closer on 15 of the 16 tensors near the limit and
# lost ln_f.out by 20% -- with its own LayerNorm arithmetic at 1.5% of budget,
# the loss inherited entirely from where rounding happened to fall upstream.
# Passing that would have meant choosing a factor by hand. The other runs are
# untouched: there the rule holds for both, with room to spare.
FLOAT64_RUNS = {"context"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_ids(tsv: str, n: int) -> list[int]:
    """The first `n` ids of a pinned corpus, after checking the whole of it
    against the checksum its .json records."""
    path = ROOT / tsv
    meta = json.loads(path.with_suffix(".json").read_text())
    line = next(l for l in path.read_text().splitlines()
                if l and not l.startswith("#"))
    ids = [int(i) for i in line.split("\t")[2].split(",")]
    digest = hashlib.sha256(
        b"".join(i.to_bytes(4, "little") for i in ids)).hexdigest()
    if digest != meta["ids_sha256"]:
        raise SystemExit(f"{tsv}: ids do not match {path.with_suffix('.json').name}")
    if len(ids) < n:
        raise SystemExit(f"{tsv}: {len(ids)} ids, need {n}")
    return ids[:n]


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump_run(model, input_ids: torch.Tensor, run_dir: Path) -> list[dict]:
    """Run one forward pass, writing each tapped tensor to its own .npy.

    A float64 model is written rounded to float32: the files are float32
    whatever computed them, and that rounding is 6e-8 relative, far below the
    rule's 1e-3."""
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


def fp32_budgets(model, input_ids: torch.Tensor, run_dir: Path,
                 records: list[dict]) -> None:
    """Record, per tensor, how much of the fp32 rule PyTorch's own fp32
    forward uses against the float64 dump already in `run_dir`.

    The rule is applied by oracle/compare.py's own compare_tensor, so the
    allowance is measured with exactly the arithmetic that will be held to
    it."""
    sys.path.insert(0, str(ROOT / "oracle"))
    from compare import compare_tensor

    by_name = {r["name"]: r for r in records}

    def tap(name: str, value: torch.Tensor) -> None:
        record = by_name[name]
        truth = np.load(run_dir / record["file"]).astype(np.float64)
        verdict = compare_tensor(truth, value.detach().numpy().astype(np.float64),
                                 TOLERANCE, record.get("region"))
        record["fp32_budget"] = verdict["budget"]

    with torch.no_grad():
        model(input_ids, tap=tap)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="subset of run names (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="list the runs and exit")
    args = parser.parse_args()

    every = list(RUNS) + list(ID_RUNS)
    names = args.runs or every
    unknown = [n for n in names if n not in every]
    if unknown:
        print(f"unknown run(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(every)}", file=sys.stderr)
        return 2

    if args.list:
        for name in names:
            what = RUNS[name] if name in RUNS else f"<ids from {ID_RUNS[name]}>"
            print(f"  {name:10s} {what!r:.70}")
        return 0

    # Single-threaded so the oracle is reproducible run to run: multithreaded
    # reductions can reassociate and shift the last couple of bits.
    torch.set_num_threads(1)
    torch.manual_seed(0)

    print("loading reference model...")
    model = load_gpt2(args.weights)
    model64 = None  # built on first use: a float64 copy is 1 GB

    from transformers import AutoTokenizer  # tokenization only
    tok = AutoTokenizer.from_pretrained(args.weights)

    runs = []
    for name in names:
        if name in RUNS:
            prompt = RUNS[name]
            input_ids = tok(prompt, return_tensors="pt").input_ids
        else:
            ids = pinned_ids(ID_RUNS[name], model.cfg.n_ctx)
            input_ids = torch.tensor([ids], dtype=torch.long)
            # Carried for a human to read; nothing computes on it, and it need
            # not re-encode to the same ids.
            prompt = tok.decode(ids)
        run_dir = ACTIVATIONS / name

        print(f"  {name:10s} T={input_ids.shape[1]:<4d} ", end="", flush=True)
        float64 = name in FLOAT64_RUNS
        if float64:
            if model64 is None:
                model64 = load_gpt2(args.weights).double()
            records = dump_run(model64, input_ids, run_dir)
            fp32_budgets(model, input_ids, run_dir, records)
        else:
            records = dump_run(model, input_ids, run_dir)
        total = sum(np.prod(r["shape"]) for r in records)
        print(f"{len(records):3d} tensors, {total * 4 / 1e6:.1f} MB"
              + ("  (float64 reference)" if float64 else ""))

        run = {
            "name": name,
            "prompt": prompt,
            "input_ids": input_ids[0].tolist(),
            "n_tokens": int(input_ids.shape[1]),
            "dir": f"activations/{name}",
        }
        if float64:
            run["reference_dtype"] = "float64"
            # compare.py: a whole-sequence candidate is reported, a KV-cached
            # one gated. Said here, by the reference, so a candidate cannot
            # declare its own way out of the gate.
            run["whole_sequence"] = "report"
        run["tensors"] = records
        runs.append(run)

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
