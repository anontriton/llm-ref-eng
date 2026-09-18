#!/usr/bin/env python3
"""Check the engine and the reference pick the same tokens, step after step.

The third fp32 criterion in CLAUDE.md: top-1 token identical for 50 consecutive
greedy steps. Layer-wise comparison (compare.py) proves the arithmetic tracks;
this proves the behaviour does, which is a different claim. It is also the
criterion that survives Phase 4, when layer-wise matching is void and only
agreement rates remain.

    python oracle/check_greedy.py                 # 50 steps from the first run
    python oracle/check_greedy.py --steps 20 --run code

Greedy decoding is unforgiving by design: one token of divergence changes every
token after it, so a single disagreement is reported with the step it happened
at rather than as a percentage.

Exit codes: 0 pass, 1 divergence, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))

from weights import DEFAULT_WEIGHTS, load_gpt2  # noqa: E402

ENGINE = ROOT / "engine" / "build" / "tools" / "gpt2_generate"


def reference_greedy(model, ids: list[int], steps: int) -> list[int]:
    out: list[int] = []
    cur = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(steps):
            logits = model(cur)
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--run", default=None, help="run name (default: the first)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    args = parser.parse_args()

    if not args.engine.exists():
        print(f"no engine binary at {args.engine} -- build it first", file=sys.stderr)
        return 2

    cmd = [str(args.engine), "--steps", str(args.steps)]
    if args.run:
        cmd += ["--run", args.run]
    print(f"engine    {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return 2
    got = json.loads(proc.stdout)

    print(f"reference reference/model.py, {args.steps} steps")
    torch.set_num_threads(1)
    model = load_gpt2(args.weights)
    want = reference_greedy(model, got["prompt_ids"], args.steps)
    mine = got["generated_ids"]

    first = next((i for i, (a, b) in enumerate(zip(want, mine)) if a != b), None)
    print(f"\nrun {got['run']}  prompt {got['prompt_ids']}")

    if first is None:
        print(f"  reference {want}")
        print(f"  engine    {mine}")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.weights)
            print(f"\n  -> {tok.decode(got['prompt_ids'])!r}"
                  f" + {tok.decode(mine)!r}")
        except Exception:
            pass
        print(f"\nGREEDY CHECK PASSED  ({args.steps}/{args.steps} tokens identical)")
        return 0

    print(f"  FAIL   diverged at step {first}: reference {want[first]}, "
          f"engine {mine[first]}")
    print(f"         reference {want[:first + 1]}")
    print(f"         engine    {mine[:first + 1]}")
    print("\nGREEDY CHECK FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
