#!/usr/bin/env python3
"""Validate the hand-written reference against HF's GPT2LMHeadModel.

This is the ONLY file in the project permitted to instantiate HF model code,
and it is a one-way check: HF validates the reference, it never becomes the
reference. Nothing downstream imports this module.

    python reference/validate_hf.py

Exits non-zero if the reference disagrees with HF beyond tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from weights import DEFAULT_WEIGHTS, load_gpt2

# CLAUDE.md tolerance policy, fp32 phases.
MAX_ABS = 1e-4
MAX_REL = 1e-3
REL_FLOOR = 1e-2      # ignore relative error on values this small
GREEDY_STEPS = 50

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley",
    "def fibonacci(n):",
    "\n",
]


def compare(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> tuple[bool, str]:
    """Max-abs and floored-relative comparison, as the tolerance policy says."""
    if ours.shape != theirs.shape:
        return False, f"{name}: shape {tuple(ours.shape)} vs {tuple(theirs.shape)}"

    diff = (ours - theirs).abs()
    max_abs = diff.max().item()

    # Relative error only where the reference value is big enough to mean
    # something; near zero, relative error is noise amplification.
    big = theirs.abs() > REL_FLOOR
    max_rel = (diff[big] / theirs[big].abs()).max().item() if big.any() else 0.0

    ok = max_abs < MAX_ABS and max_rel < MAX_REL
    return ok, f"{name}: max_abs {max_abs:.3e}  max_rel {max_rel:.3e}"


def greedy(step_fn, ids: torch.Tensor, steps: int) -> list[int]:
    """Argmax decoding, no cache, full forward each step. Slow and obvious."""
    ids = ids.clone()
    out = []
    for _ in range(steps):
        logits = step_fn(ids)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--steps", type=int, default=GREEDY_STEPS)
    args = parser.parse_args()

    torch.manual_seed(0)

    print("loading hand-written reference...")
    ours = load_gpt2(args.weights)

    print("loading HF GPT2LMHeadModel (validation only)...")
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tok = AutoTokenizer.from_pretrained(args.weights)
    theirs = GPT2LMHeadModel.from_pretrained(args.weights, dtype=torch.float32)
    theirs.eval()

    failures: list[str] = []

    print(f"\nlogits, {len(PROMPTS)} prompts (tolerance: abs < {MAX_ABS:g}, "
          f"rel < {MAX_REL:g} above |x| > {REL_FLOOR:g})")
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids
            a = ours(ids)
            b = theirs(ids).logits

            ok, line = compare(f"  T={ids.shape[1]:<3d} {prompt[:38]!r}", a, b)
            agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
            print(f"{line}  top1 {agree:.1%}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(line.strip())
            if agree < 1.0:
                failures.append(f"top-1 disagreement on {prompt!r}: {agree:.1%}")

    print(f"\ngreedy decode, {args.steps} steps from {PROMPTS[0]!r}")
    with torch.no_grad():
        ids = tok(PROMPTS[0], return_tensors="pt").input_ids
        a_ids = greedy(lambda x: ours(x), ids, args.steps)
        b_ids = greedy(lambda x: theirs(x).logits, ids, args.steps)

    if a_ids == b_ids:
        print(f"  {args.steps}/{args.steps} tokens identical")
        print(f"  -> {PROMPTS[0]}{tok.decode(a_ids)!r}")
    else:
        first = next(i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y)
        failures.append(
            f"greedy diverged at step {first}: ours {a_ids[first]} "
            f"({tok.decode([a_ids[first]])!r}) vs hf {b_ids[first]} "
            f"({tok.decode([b_ids[first]])!r})"
        )
        print(f"  DIVERGED at step {first}")

    print()
    if failures:
        print("VALIDATION FAILED", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print("VALIDATION PASSED -- the reference matches HF within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
