#!/usr/bin/env python3
"""Compare a candidate activation dump against the reference oracle.

This is the sole authority on whether the engine is correct. It reads two
manifests, checks they describe the same run, then walks every tensor in
forward order and reports the FIRST divergence -- because once block 3 is
wrong, every later mismatch is an echo, not a finding.

    python oracle/compare.py engine/dumps/manifest.json
    python oracle/compare.py engine/dumps/manifest.json --run capital -v
    python oracle/compare.py oracle/manifest.json          # reference vs itself

The fp32 rule is the numpy.allclose form, applied per element:

    |candidate - reference| <= max_abs + max_rel * |reference|

Both numbers come from the reference manifest. Combining them this way, rather
than demanding each independently, is what makes the rule satisfiable in fp32:
GPT-2's residual stream has outlier dimensions near 2650, where one fp32 ulp is
2.4e-4 -- so a bare 1e-4 absolute limit would demand bit-exact agreement with
PyTorch's GEMM summation order, which is not a correctness property. Near zero
the max_abs term governs, so values produced by cancellation are still held to
an absolute bound rather than an unbounded relative one.

Exit codes: 0 pass, 1 numeric divergence, 2 provenance or usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_MANIFEST = ROOT / "oracle" / "manifest.json"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> tuple[dict, Path]:
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise SystemExit(f"no manifest at {path}")
    return json.loads(path.read_text()), path.parent


def load_tensor(base: Path, run: dict, record: dict, verify_hash: bool) -> np.ndarray:
    path = base / run["dir"] / record["file"]
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=False)

    if list(array.shape) != record["shape"]:
        raise ValueError(
            f"{record['name']}: manifest says {record['shape']}, "
            f"file holds {list(array.shape)}"
        )
    if verify_hash:
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(
                f"{record['name']}: file does not match its manifest hash "
                f"(stale dump?)"
            )
    return array.astype(np.float64)


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def region_mask(shape: tuple[int, ...], region: str | None,
                q_offset: int = 0) -> np.ndarray | None:
    """Which elements are in scope. None means all of them.

    'causal_lower_triangle' restricts attention scores to the part every
    implementation must actually compute; what an engine leaves in the masked
    upper triangle is its own business, and shows up in .probs regardless.

    `q_offset` is the absolute position of the first query row. It is 0 for a
    whole-sequence tensor, where this is exactly np.tril. It is not 0 for a
    KV-cached dump, which holds one query row from partway along the sequence:
    that row legitimately attends to every key up to its own position, so
    deriving the triangle from the tensor's own shape would mark a single
    column in scope and quietly excuse the other T-1.
    """
    if region is None:
        return None
    if region == "causal_lower_triangle":
        t_q, t_k = shape[-2], shape[-1]
        rows = np.arange(q_offset, q_offset + t_q)[:, None]
        cols = np.arange(t_k)[None, :]
        return cols <= rows
    raise ValueError(f"unknown region {region!r}")


def slice_row(ref: np.ndarray, name: str, row: int) -> np.ndarray:
    """Reduce a whole-sequence reference tensor to the single position `row`.

    A KV-cached decode step computes one position, so it dumps [1, 1, C] where
    the reference holds [1, T, C]. Comparing them is legitimate rather than a
    convenience: attention is causal, so position `row` attends only to
    positions <= row and its activations do not depend on whether later tokens
    were present. The reference's row `row` is therefore exactly what the
    cached step must produce, and the oracle already contains it.

    Axis 1 carries positions for activations ([1, T, C]); attention tensors are
    [1, H, T_q, T_k] and carry queries on axis 2. The key axis is left whole --
    the cached step really does attend across all T keys.
    """
    if ref.ndim == 3:
        axis = 1
    elif ref.ndim == 4:
        axis = 2
    else:
        raise ValueError(f"{name}: cannot take a position row from shape "
                         f"{ref.shape}")
    if not 0 <= row < ref.shape[axis]:
        raise ValueError(f"{name}: kv_row {row} outside axis {axis} of "
                         f"length {ref.shape[axis]}")
    return np.take(ref, [row], axis=axis)


def compare_tensor(ref: np.ndarray, cand: np.ndarray, tol: dict,
                   region: str | None, q_offset: int = 0) -> dict:
    """Return a verdict dict for one tensor pair."""
    result: dict = {"ok": True, "reason": None}

    if ref.shape != cand.shape:
        return {"ok": False, "reason":
                f"shape {cand.shape} != reference {ref.shape}"}

    mask = region_mask(ref.shape, region, q_offset)
    if mask is not None:
        mask = np.broadcast_to(mask, ref.shape)
    else:
        mask = np.ones(ref.shape, dtype=bool)

    # Non-finite values must line up exactly before any arithmetic: inf - inf
    # is nan, which would otherwise quietly poison every statistic below.
    for label, test in (("nan", np.isnan), ("+inf", lambda x: np.isposinf(x)),
                        ("-inf", lambda x: np.isneginf(x))):
        a, b = test(ref) & mask, test(cand) & mask
        if not np.array_equal(a, b):
            n = int((a ^ b).sum())
            where = np.argwhere(a ^ b)[0]
            return {"ok": False, "reason":
                    f"{label} pattern differs at {n} position(s), "
                    f"first at {tuple(int(i) for i in where)}"}

    finite = np.isfinite(ref) & np.isfinite(cand) & mask
    if not finite.any():
        return result | {"budget": 0.0, "max_abs": 0.0, "max_rel": 0.0, "n": 0}

    diff = np.zeros(ref.shape)
    diff[finite] = np.abs(ref[finite] - cand[finite])

    # The verdict: each element against its own bound. `budget` is the fraction
    # of that bound the worst element uses, so 1.0 is exactly at the limit and
    # anything above fails. Reporting it shows how much headroom a passing
    # engine has, which is what tells you whether an optimization is safe.
    bound = tol["max_abs"] + tol["max_rel"] * np.abs(ref)
    used = np.zeros(ref.shape)
    used[finite] = diff[finite] / bound[finite]
    worst_idx = np.unravel_index(np.argmax(used), used.shape)
    budget = float(used[worst_idx])

    # Diagnostics only -- they no longer decide anything. Relative error is
    # reported only above rel_floor, where it carries meaning; below it, it is
    # noise amplification.
    abs_idx = np.unravel_index(np.argmax(diff), diff.shape)
    max_abs = float(diff[abs_idx])
    rel = np.zeros(ref.shape)
    big = finite & (np.abs(ref) > tol["rel_floor"])
    if big.any():
        rel[big] = diff[big] / np.abs(ref[big])
    max_rel = float(rel.max())

    ok = budget <= 1.0
    return {
        "ok": ok,
        "reason": None if ok else "tolerance exceeded",
        "budget": budget,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "worst_at": tuple(int(i) for i in worst_idx),
        "ref_at_worst": float(ref[worst_idx]),
        "cand_at_worst": float(cand[worst_idx]),
        "bound_at_worst": float(bound[worst_idx]),
        "n": int(finite.sum()),
    }


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def check_provenance(ref: dict, cand: dict) -> list[str]:
    """A dump whose manifest disagrees with the run being compared is a
    failure, not a warning."""
    problems = []

    if ref.get("schema") != cand.get("schema"):
        problems.append(f"schema {cand.get('schema')} != {ref.get('schema')}")

    if ref["model"]["weights_sha256"] != cand["model"]["weights_sha256"]:
        problems.append(
            f"different weights: reference "
            f"{ref['model']['weights_sha256'][:12]}..., candidate "
            f"{cand['model']['weights_sha256'][:12]}..."
        )

    for key, want in ref["model"]["config"].items():
        got = cand["model"]["config"].get(key)
        if got != want:
            problems.append(f"config.{key}: {got} != {want}")

    if ref["dtype"] != cand["dtype"]:
        problems.append(f"dtype {cand['dtype']} != {ref['dtype']}")

    # Either side declaring a non-fp32 policy voids the whole comparison: an
    # int8 engine must not be held to a layer-wise tolerance, and an int8
    # oracle is not a tolerance to hold anything to.
    for label, manifest in (("reference", ref), ("candidate", cand)):
        policy = manifest["tolerance"].get("policy")
        if policy != "fp32":
            problems.append(
                f"{label} tolerance policy is {policy!r}: layer-wise matching "
                f"is void for quantized runs -- use perplexity / top-1 "
                f"agreement / logit KL instead"
            )

    cand_runs = {r["name"]: r for r in cand["runs"]}
    for run in ref["runs"]:
        other = cand_runs.get(run["name"])
        if other is None:
            continue  # absent runs are handled by the caller
        if other["input_ids"] != run["input_ids"]:
            problems.append(f"run {run['name']}: input_ids differ")

    return problems


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path,
                        help="candidate manifest.json (or its directory)")
    parser.add_argument("--reference", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--run", action="append", dest="runs",
                        help="only this run (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every tensor, not just the worst")
    parser.add_argument("--allow-subset", action="store_true",
                        help="permit the candidate to omit tensors or runs "
                             "(for incremental bring-up)")
    parser.add_argument("--no-verify-hash", action="store_true",
                        help="skip the per-file checksum of the dumps")
    args = parser.parse_args()

    ref_manifest, ref_base = load_manifest(args.reference)
    cand_manifest, cand_base = load_manifest(args.candidate)
    verify = not args.no_verify_hash

    print(f"reference  {args.reference}")
    print(f"           {ref_manifest['producer']}, {ref_manifest['created']}")
    print(f"candidate  {args.candidate}")
    print(f"           {cand_manifest['producer']}, {cand_manifest['created']}")

    problems = check_provenance(ref_manifest, cand_manifest)
    if problems:
        print("\nPROVENANCE MISMATCH", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    tol = ref_manifest["tolerance"]
    print(f"\nprovenance ok  |  tolerance: |cand - ref| <= "
          f"{tol['max_abs']:g} + {tol['max_rel']:g} * |ref|")

    cand_runs = {r["name"]: r for r in cand_manifest["runs"]}
    wanted = args.runs or [r["name"] for r in ref_manifest["runs"]]
    failed = False
    checked = 0

    for ref_run in ref_manifest["runs"]:
        name = ref_run["name"]
        if name not in wanted:
            continue

        cand_run = cand_runs.get(name)
        if cand_run is None:
            if args.allow_subset:
                print(f"\nrun {name}: skipped (absent from candidate)")
                continue
            print(f"\nrun {name}: MISSING from candidate", file=sys.stderr)
            failed = True
            continue

        cand_records = {r["name"]: r for r in cand_run["tensors"]}
        kv_row = cand_run.get("kv_row")
        if kv_row is None:
            print(f"\nrun {name}  (T={ref_run['n_tokens']}, "
                  f"{len(ref_run['tensors'])} tensors)")
        else:
            print(f"\nrun {name}  (T={ref_run['n_tokens']}, "
                  f"{len(ref_run['tensors'])} tensors, KV-cached row {kv_row})")

        worst = None
        divergence = None

        for record in sorted(ref_run["tensors"], key=lambda r: r["order"]):
            tname = record["name"]
            other = cand_records.get(tname)
            if other is None:
                if args.allow_subset:
                    continue
                divergence = (tname, {"ok": False, "reason":
                                      "missing from candidate"})
                break

            try:
                a = load_tensor(ref_base, ref_run, record, verify)
                b = load_tensor(cand_base, cand_run, other, verify)
                if kv_row is not None:
                    a = slice_row(a, tname, kv_row)
            except (FileNotFoundError, ValueError) as exc:
                divergence = (tname, {"ok": False, "reason": str(exc)})
                break

            verdict = compare_tensor(a, b, tol, record.get("region"),
                                     q_offset=kv_row or 0)
            checked += 1

            if args.verbose:
                flag = "ok  " if verdict["ok"] else "FAIL"
                print(f"  {flag} {tname:32s} "
                      f"budget {verdict.get('budget', 0) * 100:6.1f}%  "
                      f"abs {verdict.get('max_abs', 0):.3e}  "
                      f"rel {verdict.get('max_rel', 0):.3e}")

            if not verdict["ok"]:
                divergence = (tname, verdict)
                break

            if worst is None or verdict["budget"] > worst[1]["budget"]:
                worst = (tname, verdict)

        if divergence is None:
            tname, v = worst
            print(f"  PASS   worst tensor {tname} at {v['budget'] * 100:.1f}% of "
                  f"budget  (abs {v['max_abs']:.3e}, rel {v['max_rel']:.3e})")
        else:
            failed = True
            tname, v = divergence
            order = next((r["order"] for r in ref_run["tensors"]
                          if r["name"] == tname), -1)
            print(f"  FAIL   first divergence at {tname} (tensor #{order})")
            if v["reason"] == "tolerance exceeded":
                err = abs(v['cand_at_worst'] - v['ref_at_worst'])
                print(f"         {v['budget'] * 100:.1f}% of budget at index "
                      f"{v['worst_at']}  (ref {v['ref_at_worst']:+.6f}, "
                      f"candidate {v['cand_at_worst']:+.6f})")
                print(f"         |diff| {err:.6e} > bound {v['bound_at_worst']:.6e}"
                      f"  (max_abs {v['max_abs']:.3e}, max_rel {v['max_rel']:.3e})")
            else:
                print(f"         {v['reason']}")

    print()
    if failed:
        print("ORACLE COMPARISON FAILED", file=sys.stderr)
        return 1
    print(f"ORACLE COMPARISON PASSED  ({checked} tensors within tolerance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
