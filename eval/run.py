#!/usr/bin/env python3
"""Run the quantized-phase eval and record it as commit-tagged JSON.

    python eval/run.py --out eval/runs/fp32
    python eval/run.py --out eval/runs/int8 \\
                       --weights weights/gpt2-124m-int8.bin \\
                       --reference eval/runs/fp32

engine/tools/gpt2_eval measures and dumps; eval/metrics.py judges; this adds
the provenance that makes the number traceable and writes the summary to
eval/results/, which is committed. Same three-way split as bench/, and for the
same reason: a metric without a commit behind it cannot be compared to
anything later.

The dumps themselves stay in eval/runs/ and are gitignored -- the sampled
logits alone are 50 MB, and they are reproducible from the committed corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOL = ROOT / "engine" / "build-avx2" / "tools" / "gpt2_eval"
RESULTS = ROOT / "eval" / "results"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def git_commit() -> tuple[str, bool]:
    """Dirty ignores generated eval output, which cannot change what was
    measured -- the same carve-out bench/run.py makes for its results."""
    sha = git("rev-parse", "HEAD")
    lines = [l for l in git("status", "--porcelain").splitlines()
             if not re.match(r"^..\s+\"?eval/(runs|results)/", l)]
    return sha, bool(lines)


def cpu_name() -> str:
    try:
        m = re.search(r"^model name\s*:\s*(.+)$",
                      Path("/proc/cpuinfo").read_text(), re.MULTILINE)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--weights", type=Path, default=None,
                        help="weight file; its header decides the policy")
    parser.add_argument("--out", type=Path, required=True,
                        help="dump directory, under eval/runs/")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--kl-stride", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reference", type=Path, default=None,
                        help="an fp32 run to judge this one against")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if not args.tool.is_file():
        print(f"{args.tool} not found -- build it first", file=sys.stderr)
        return 1

    sha, dirty = git_commit()
    if dirty:
        print("WARNING: working tree is dirty; this run is recorded as dirty "
              "and does not count as a result.", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    cmd = [str(args.tool), "--out", str(args.out),
           "--window", str(args.window), "--windows", str(args.windows),
           "--kl-stride", str(args.kl_stride), "--threads", str(args.threads)]
    if args.weights is not None:
        cmd += ["--weights", str(args.weights)]
    print(f"running {' '.join(cmd)}", file=sys.stderr)
    started = datetime.now(timezone.utc)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    measured = json.loads(proc.stdout)

    verdict = None
    if args.reference is not None:
        judge = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "metrics.py"), str(args.out),
             "--reference", str(args.reference), "--json"],
            capture_output=True, text=True, cwd=ROOT)
        # metrics.py prints a human table, then the JSON, then its verdict
        # line. raw_decode stops at the end of the object instead of guessing
        # where it ends.
        start = judge.stdout.find("\n{")
        if start == -1:
            sys.stdout.write(judge.stdout)
            sys.stderr.write(judge.stderr)
            print("metrics.py did not produce a verdict", file=sys.stderr)
            return judge.returncode or 1
        sys.stdout.write(judge.stdout[:start])
        verdict, _ = json.JSONDecoder().raw_decode(judge.stdout[start + 1:])

    manifest = json.loads((args.out / "manifest.json").read_text())
    result = {
        "schema": 1,
        "commit": sha,
        "dirty": dirty,
        "timestamp": started.isoformat().replace("+00:00", "Z"),
        "policy": measured["policy"],
        "backend": measured["backend"],
        "threads": measured["threads"],
        "corpus": manifest["corpus"],
        "eval": manifest["eval"],
        "metrics": {
            "perplexity": measured["perplexity"],
            "mean_nll": measured["mean_nll"],
        },
        # Absent for a run with no reference, which is what the fp32 baseline
        # is: there is nothing above it to be judged against.
        "vs_fp32": verdict,
        "machine": {"cpu": cpu_name(), "cores": os.cpu_count()},
    }

    text = json.dumps(result, indent=2) + "\n"
    print(text)
    if args.no_write:
        return 0

    parts = [started.strftime("%Y%m%dT%H%M%SZ"), sha[:7], measured["policy"]]
    if dirty:
        parts.append("dirty")
    out = RESULTS / ("-".join(parts) + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0 if (verdict is None or verdict["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
