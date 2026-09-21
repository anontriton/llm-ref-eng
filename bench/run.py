#!/usr/bin/env python3
"""Run the benchmark and record the result as commit-tagged JSON.

    python bench/run.py                      # -> bench/results/<stamp>-<sha>-scalar.json
    python bench/run.py --generate 32        # a cheaper run
    python bench/run.py --no-write           # print, record nothing
    python bench/run.py --tool engine/build-avx2/tools/gpt2_bench

The backend and build type are read from the binary and from the CMake cache
beside it, not passed in, so a result cannot claim a backend it was not built
with.

engine/tools/gpt2_bench measures; this adds the provenance that makes a number
mean something -- which commit produced it, on which machine, with which build.
Same split as the oracle: the C++ side produces, the Python side records and
judges. Timing lives in C++ because a subprocess boundary would fold process
startup and weight loading into the measurement.

A speedup claim is only as good as the pair of results it compares, so the
fields that would invalidate such a comparison -- commit, build type, backend,
thread count, prompt length, generate count -- are all recorded, and a run on a
a dirty working tree is marked dirty and does not count as a result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOL = ROOT / "engine" / "build" / "tools" / "gpt2_bench"
DEFAULT_PROMPTS = ROOT / "bench" / "prompts.tsv"
DEFAULT_WEIGHTS = ROOT / "weights" / "gpt2-124m.bin"
RESULTS = ROOT / "bench" / "results"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def git_commit() -> tuple[str, bool]:
    """Returns (sha, dirty). Untracked files count as dirty: an untracked
    source file can change what gets built.

    Results are the exception. They are this script's own output, they cannot
    change what was built or measured, and counting them would mark every run
    after the first of a session dirty -- including the second half of a
    before/after pair, which is exactly when it matters most.
    """
    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    lines = [line for line in status.splitlines()
             if not line[3:].lstrip('"').startswith("bench/results/")]
    return sha, bool(lines)


def cmake_build_type(tool: Path) -> str:
    """Read the build type out of the CMake cache beside the tool, so pointing
    --tool at a second build directory reports that build rather than the
    default one."""
    cache = tool.parent.parent / "CMakeCache.txt"
    if not cache.is_file():
        return "unknown"
    m = re.search(r"^CMAKE_BUILD_TYPE:STRING=(.*)$", cache.read_text(),
                  re.MULTILINE)
    return (m.group(1).strip() or "unknown") if m else "unknown"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--run", default="", help="prompt name (default: first)")
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--prefill-repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1,
                        help="worker threads, including the calling one; "
                             "results must not depend on this")
    parser.add_argument("--kv-cache", action="store_true",
                        help="decode through the engine's KV cache")
    parser.add_argument("--label", default="",
                        help="optional suffix for the result filename")
    parser.add_argument("--no-write", action="store_true",
                        help="print the JSON but do not write to results/")
    args = parser.parse_args()

    if not args.tool.is_file():
        print(f"{args.tool} not found -- build it first:\n"
              f"    cmake --build engine/build", file=sys.stderr)
        return 1

    sha, dirty = git_commit()
    if dirty:
        print("WARNING: working tree is dirty. Per bench/README.md this run "
              "is recorded as dirty and does not count as a result.",
              file=sys.stderr)

    cmd = [str(args.tool),
           "--weights", str(args.weights),
           "--prompts", str(args.prompts),
           "--generate", str(args.generate),
           "--prefill-repeat", str(args.prefill_repeat),
           "--threads", str(args.threads)]
    if args.kv_cache:
        cmd += ["--kv-cache"]
    if args.run:
        cmd += ["--run", args.run]

    print(f"running {' '.join(cmd)}", file=sys.stderr)
    started = datetime.now(timezone.utc)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print(f"gpt2_bench exited {proc.returncode}", file=sys.stderr)
        return proc.returncode
    measured = json.loads(proc.stdout)

    result = {
        "schema": 1,
        "commit": sha,
        "dirty": dirty,
        "timestamp": started.isoformat().replace("+00:00", "Z"),
        "config": {
            "build": cmake_build_type(args.tool),
            # Both reported by the binary itself. A benchmark that had to be
            # told which backend it was running would eventually be told wrong.
            "backend": measured["backend"],
            "threads": measured["threads"],
            "dtype": "fp32",
            # Taken from the tool's own report rather than from the flag, so
            # the record says what actually ran.
            "kv_cache": measured["kv_cache"],
        },
        "prompt": {
            "name": measured["run"],
            "tokens": measured["prompt_tokens"],
            "generate": measured["generate"],
        },
        "metrics": {
            "tokens_per_sec": measured["tokens_per_sec"],
            "prefill_ms": measured["prefill_ms"],
            "decode_ms_per_token": measured["decode_ms_per_token"],
        },
        "detail": {
            "prefill_repeat": measured["prefill_repeat"],
            "decode_steps": measured["decode_steps"],
            "decode_total_ms": measured["decode_total_ms"],
            "weights_load_ms": measured["weights_load_ms"],
        },
        # Not a metric. If an optimization changes these ids, it changed the
        # engine's output, and the number above is measuring a different
        # program than the one it is being compared to.
        "output": {"generated_ids": measured["generated_ids"]},
        "machine": {
            "cpu": cpu_name(),
            "cores": os.cpu_count(),
            "platform": platform.platform(),
            "compiler": compiler_version(),
        },
    }

    text = json.dumps(result, indent=2) + "\n"
    print(text)

    if args.no_write:
        return 0

    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    parts = [stamp, sha[:7], measured["backend"]]
    if measured["threads"] > 1:
        parts.append(f"t{measured['threads']}")
    # In the name, because a cached and an uncached run at the same commit are
    # not the same measurement and should not look alike in a directory
    # listing.
    parts.append("kv" if measured["kv_cache"] else "nokv")
    if args.label:
        parts.append(args.label)
    if dirty:
        parts.append("dirty")
    out = RESULTS / ("-".join(parts) + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out.relative_to(ROOT)}", file=sys.stderr)
    return 0


def compiler_version() -> str:
    cxx = shutil.which("g++")
    if not cxx:
        return "unknown"
    try:
        first = subprocess.run([cxx, "--version"], capture_output=True,
                               text=True, check=True).stdout.splitlines()[0]
        return first.strip()
    except (subprocess.CalledProcessError, OSError, IndexError):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
