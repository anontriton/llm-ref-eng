#!/usr/bin/env python3
"""Check the engine's perplexity against the PyTorch reference.

    python eval/validate_ppl.py eval/runs/fp32

metrics.py comparing a run to itself proves the arithmetic is self-consistent
and nothing else. A harness can be perfectly self-consistent and still be
measuring the wrong thing -- an off-by-one in which token is the target, a
window boundary counted twice, a log taken of the wrong axis -- and every one
of those would survive a self-comparison and then quietly follow the int8
engine around, making a broken quantizer look fine or a working one look
broken.

So the number gets checked against something that did not come from the
engine: reference/model.py, the hand-written PyTorch model that Phase 0
validated against HuggingFace. It recomputes the same windows in torch, and the
two perplexities have to agree.

They should agree closely but not exactly. Both compute the same quantity from
the same weights, and the engine already tracks the reference to 52% of the
fp32 tolerance budget per tensor; a disagreement past the fourth decimal would
mean the harnesses differ in what they are counting, not that fp32 is
imprecise.

Exit codes: 0 agree, 1 disagree, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))


def corpus_ids(tsv: Path) -> list[int]:
    for line in tsv.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        return [int(v) for v in line.split("\t")[2].split(",")]
    raise SystemExit(f"no rows in {tsv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="an eval run directory")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "corpus.tsv")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="relative disagreement allowed in perplexity")
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["policy"] != "fp32":
        print(f"{args.run} has policy {manifest['policy']!r}; the reference "
              f"only speaks fp32, so only an fp32 run can be checked this way",
              file=sys.stderr)
        return 2

    window = manifest["eval"]["window"]
    windows = manifest["eval"]["windows"]
    ids = corpus_ids(args.corpus)[:window * windows]

    from weights import DEFAULT_WEIGHTS, load_gpt2
    weights = args.weights or DEFAULT_WEIGHTS
    print(f"reference  reference/model.py, weights {weights}")
    model = load_gpt2(str(weights))

    # Same windows, same targets, same exclusion of each window's first token.
    total_nll = 0.0
    count = 0
    per_position: list[float] = []
    with torch.no_grad():
        for w in range(windows):
            chunk = ids[w * window:(w + 1) * window]
            x = torch.tensor([chunk], dtype=torch.long)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[0].double()
            logprobs = torch.log_softmax(logits, dim=-1)
            targets = torch.tensor(chunk[1:], dtype=torch.long)
            picked = logprobs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
            nll = (-picked)
            per_position.extend(nll.tolist())
            total_nll += float(nll.sum())
            count += nll.numel()
            print(f"  window {w + 1}/{windows}  ppl so far "
                  f"{np.exp(total_nll / count):.4f}", flush=True)

    ref_ppl = float(np.exp(total_nll / count))

    engine_nll = np.load(args.run / "nll.npy").astype(np.float64)
    engine_ppl = float(np.exp(engine_nll.mean()))

    if engine_nll.size != count:
        print(f"\nposition count differs: engine {engine_nll.size}, "
              f"reference {count}", file=sys.stderr)
        return 1

    ref_nll = np.array(per_position)
    worst = float(np.abs(ref_nll - engine_nll).max())
    rel = abs(engine_ppl - ref_ppl) / ref_ppl

    print(f"\n  positions        {count}")
    print(f"  reference ppl    {ref_ppl:.6f}")
    print(f"  engine ppl       {engine_ppl:.6f}")
    print(f"  relative diff    {rel:.3e}   (allowed {args.rtol:.0e})")
    print(f"  worst per-token  {worst:.3e} nats")

    if rel > args.rtol:
        print("\nPERPLEXITY HARNESS DISAGREES WITH THE REFERENCE",
              file=sys.stderr)
        return 1
    print("\nPERPLEXITY HARNESS VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
