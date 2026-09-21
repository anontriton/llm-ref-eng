#!/usr/bin/env python3
"""Prove that compare.py catches what it is supposed to catch.

A comparison tool that always passes is worse than no comparison tool, because
it converts an unknown into false confidence. These tests build deliberately
broken candidate dumps out of the reference and assert that compare.py rejects
each one, at the right tensor, for the right reason.

    python oracle/test_compare.py

Requires the reference oracle to exist (run reference/dump.py first).
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "oracle"
COMPARE = ORACLE / "compare.py"
MANIFEST = ORACLE / "manifest.json"
RUN = "capital"          # smallest run; these tests are about logic, not size


def build_candidate(dest: Path, mutate: Callable[[dict, Path], None] | None) -> Path:
    """Copy one run of the reference into dest, then let `mutate` break it."""
    manifest = json.loads(MANIFEST.read_text())
    manifest = copy.deepcopy(manifest)
    manifest["runs"] = [r for r in manifest["runs"] if r["name"] == RUN]
    manifest["producer"] = "test_compare.py"

    shutil.copytree(ORACLE / "activations" / RUN, dest / "activations" / RUN)
    if mutate is not None:
        mutate(manifest, dest)

    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest / "manifest.json"


def rewrite(manifest: dict, dest: Path, tensor: str,
            fn: Callable[[np.ndarray], np.ndarray], *, refresh_hash: bool = True):
    """Replace one tensor's contents, optionally keeping the manifest honest."""
    run = manifest["runs"][0]
    record = next(r for r in run["tensors"] if r["name"] == tensor)
    path = dest / run["dir"] / record["file"]

    array = fn(np.load(path))
    array = np.ascontiguousarray(array, dtype=np.float32)
    np.save(path, array, allow_pickle=False)

    record["shape"] = list(array.shape)
    if refresh_hash:
        record["sha256"] = hashlib.sha256(array.tobytes()).hexdigest()


def run_compare(candidate: Path, *extra: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(COMPARE), str(candidate), "--run", RUN, *extra],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# cases: (name, mutation, expected exit code, text that must appear)
# --------------------------------------------------------------------------

def _nudge(delta: float):
    return lambda a: a + delta


def _perturb_upper_triangle(a):
    a = a.copy()
    t = a.shape[-1]
    upper = np.triu(np.ones((t, t), dtype=bool), k=1)
    a[..., upper] += 100.0
    return a


def _perturb_lower_triangle(a):
    a = a.copy()
    t = a.shape[-1]
    lower = np.tril(np.ones((t, t), dtype=bool))
    a[..., lower] += 0.5
    return a


def _scale(factor: float):
    """Relative error of exactly (factor - 1) on every element."""
    return lambda a: a * factor


def _nudge_smallest(delta: float):
    """Perturb only the element closest to zero."""
    def fn(a):
        a = a.copy()
        a.reshape(-1)[np.argmin(np.abs(a))] += delta
        return a
    return fn


def _inject_nan(a):
    a = a.copy()
    a.reshape(-1)[0] = np.nan
    return a


def _wrong_weights(manifest: dict, dest: Path) -> None:
    manifest["model"]["weights_sha256"] = "0" * 64


def _wrong_tokens(manifest: dict, dest: Path) -> None:
    manifest["runs"][0]["input_ids"][0] += 1


def _as_kv_row(row: int, *, hold: int | None = None):
    """Rewrite the candidate as a KV-cached dump: one position, not T.

    `hold` lets the dump declare row `row` while actually holding a different
    one, which is how the reference-side slice gets proved to be anchored to
    the declared position rather than trivially agreeing with whatever it is
    handed.
    """
    def mutate(manifest: dict, dest: Path) -> None:
        run = manifest["runs"][0]
        run["kv_row"] = row
        held = row if hold is None else hold
        for record in run["tensors"]:
            path = dest / run["dir"] / record["file"]
            array = np.load(path)
            # Positions live on axis 1 for activations, axis 2 (queries) for
            # attention tensors -- the same rule compare.py slices by.
            axis = 1 if array.ndim == 3 else 2
            array = np.ascontiguousarray(np.take(array, [held], axis=axis),
                                         dtype=np.float32)
            np.save(path, array, allow_pickle=False)
            record["shape"] = list(array.shape)
            record["sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
    return mutate


def _quantized_policy(manifest: dict, dest: Path) -> None:
    manifest["tolerance"]["policy"] = "int8"


CASES = [
    ("clean copy passes",
     None, 0, "ORACLE COMPARISON PASSED"),

    ("error above tolerance is caught",
     lambda m, d: rewrite(m, d, "block.5.attn.probs", _nudge(1e-2)),
     1, "first divergence at block.5.attn.probs"),

    ("error below tolerance passes",
     lambda m, d: rewrite(m, d, "block.5.attn.probs", _nudge(1e-6)),
     0, "ORACLE COMPARISON PASSED"),

    ("earliest divergence is reported, not the loudest",
     lambda m, d: (rewrite(m, d, "block.9.mlp.out", _nudge(10.0)),
                   rewrite(m, d, "block.2.ln_1.out", _nudge(1e-2))),
     1, "first divergence at block.2.ln_1.out"),

    ("masked upper triangle of attn.scores is ignored",
     lambda m, d: rewrite(m, d, "block.0.attn.scores", _perturb_upper_triangle),
     0, "ORACLE COMPARISON PASSED"),

    ("causal lower triangle of attn.scores is not ignored",
     lambda m, d: rewrite(m, d, "block.0.attn.scores", _perturb_lower_triangle),
     1, "first divergence at block.0.attn.scores"),

    ("NaN is caught as a pattern difference, not swallowed",
     lambda m, d: rewrite(m, d, "block.3.mlp.fc.out", _inject_nan),
     1, "nan pattern differs"),

    ("shape mismatch is caught",
     lambda m, d: rewrite(m, d, "block.1.attn.qkv", lambda a: a[:, :-1, :]),
     1, "shape"),

    ("stale dump (file does not match its own manifest hash) is caught",
     lambda m, d: rewrite(m, d, "block.7.out", _nudge(1e-9), refresh_hash=False),
     1, "stale dump"),

    ("different weights are a provenance failure",
     _wrong_weights, 2, "different weights"),

    ("different prompt tokens are a provenance failure",
     _wrong_tokens, 2, "input_ids differ"),

    ("quantized policy voids layer-wise comparison",
     _quantized_policy, 2, "layer-wise matching is void"),

    # The next three pin the combined rule, |a - b| <= max_abs + max_rel * |a|.
    # block.2.mlp.out holds one of GPT-2's outlier dimensions (~2317 in this
    # run), which is exactly where a bare absolute limit stops being meaningful.

    ("relative error within max_rel passes, even when absolute error is large",
     lambda m, d: rewrite(m, d, "block.2.mlp.out", _scale(1 + 5e-4)),
     0, "ORACLE COMPARISON PASSED"),

    ("relative error above max_rel is caught, even at large magnitude",
     lambda m, d: rewrite(m, d, "block.2.mlp.out", _scale(1 + 5e-3)),
     1, "first divergence at block.2.mlp.out"),

    # A masked attention position is an exact zero; leaking 5e-4 of probability
    # into it is 5x max_abs, and the relative term offers no slack at zero.
    ("near-zero values are still held to max_abs",
     lambda m, d: rewrite(m, d, "block.5.attn.probs", _nudge_smallest(5e-4)),
     1, "first divergence at block.5.attn.probs"),

    # A KV-cached dump holds one position where the reference holds T. The
    # comparison stays real: it slices the reference to the declared row, and
    # the next two cases pin both directions of that.
    ("kv_row dump compares against the reference's matching row",
     _as_kv_row(4), 0, "ORACLE COMPARISON PASSED"),

    ("kv_row dump holding the wrong row is caught",
     _as_kv_row(4, hold=0), 1, "first divergence at embed.out"),

    # With one query row, the causal triangle cannot be read off the tensor's
    # own shape: np.tril of a 1-by-T grid marks a single column and would
    # excuse every other key the row actually attended to.
    ("kv_row still checks the whole causal row of attn.scores",
     lambda m, d: (_as_kv_row(4)(m, d),
                   rewrite(m, d, "block.0.attn.scores",
                           lambda a: a + np.array([0, 0, 0, 1e-2, 0],
                                                  dtype=np.float32))),
     1, "first divergence at block.0.attn.scores"),
]


def main() -> int:
    if not MANIFEST.exists():
        print(f"no oracle at {MANIFEST} -- run reference/dump.py first",
              file=sys.stderr)
        return 2

    passed = 0
    for i, (name, mutate, want_code, want_text) in enumerate(CASES, 1):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = build_candidate(Path(tmp) / "dump", mutate)
            code, output = run_compare(candidate)

        problems = []
        if code != want_code:
            problems.append(f"exit {code}, wanted {want_code}")
        if want_text not in output:
            problems.append(f"output missing {want_text!r}")

        if problems:
            print(f"  FAIL  {i:2d}. {name}")
            for p in problems:
                print(f"          {p}")
            print("        ---- compare.py said ----")
            for line in output.strip().splitlines():
                print(f"        {line}")
            return 1

        print(f"  ok    {i:2d}. {name}")
        passed += 1

    print(f"\n{passed}/{len(CASES)} oracle self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
