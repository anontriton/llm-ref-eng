#!/usr/bin/env python3
"""Judge a quantized engine against the fp32 engine.

    python eval/metrics.py eval/runs/int8 --reference eval/runs/fp32
    python eval/metrics.py eval/runs/fp32 --reference eval/runs/fp32   # self

This is to Phase 4 what oracle/compare.py is to Phase 2: the sole authority on
whether a run passes. It exists because compare.py cannot do this job -- it
refuses a non-fp32 policy outright, on the grounds that layer-wise tolerance is
meaningless once the arithmetic changes representation. An int8 engine is not a
worse fp32 engine; it is a different function that has to behave like the same
model.

CLAUDE.md names the three replacements, and they answer different questions:

  perplexity        did the model get worse at the language? An absolute
                    number, comparable across engines and to the outside world.
  top-1 agreement   does it pick the same words? The thing a user sees, and
                    the criterion that survives into Phase 5.
  KL divergence     how far did the whole distribution move? Catches damage
                    that has not yet changed an argmax -- a model one nudge
                    away from picking differently everywhere still looks
                    perfect to top-1.

All three, because each is blind to something the others catch. KL is computed
over the sampled positions the engine dumped whole logits for; perplexity and
agreement use every predicted position.

Exit codes: 0 pass, 1 a metric outside its limit, 2 provenance or usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Proposed, not measured -- there is no int8 engine yet to calibrate against.
# The fp32 tolerance in CLAUDE.md was revised once on evidence, and these
# should expect the same treatment: if int8 lands outside them, the first
# question is whether the limit or the quantizer is wrong, and the answer has
# to come from a measurement rather than from taste.
DEFAULTS = {
    "max_ppl_ratio": 1.02,      # perplexity may rise 2%
    "min_top1_agreement": 0.98,  # 98% of argmaxes unchanged
    "max_mean_kl": 0.01,        # nats, averaged over sampled positions
    "max_p99_kl": 0.05,         # nats, worst positions still bounded
}


def load_run(path: Path) -> tuple[dict, Path]:
    base = path if path.is_dir() else path.parent
    manifest = base / "manifest.json" if path.is_dir() else path
    if not manifest.is_file():
        raise SystemExit(f"no manifest at {manifest}")
    return json.loads(manifest.read_text()), base


def check_provenance(ref: dict, cand: dict) -> list[str]:
    """Two runs are comparable only if they measured the same thing. A metric
    computed across different corpora or window sizes is not a comparison, it
    is a coincidence."""
    problems = []

    if ref.get("schema") != cand.get("schema"):
        problems.append(f"schema {cand.get('schema')} != {ref.get('schema')}")

    if ref["model"]["weights_sha256"] != cand["model"]["weights_sha256"]:
        problems.append(
            f"different weights: {cand['model']['weights_sha256'][:12]}... vs "
            f"{ref['model']['weights_sha256'][:12]}...")

    if ref["model"]["vocab_size"] != cand["model"]["vocab_size"]:
        problems.append("vocab_size differs")

    if ref["corpus"]["ids_sha256"] != cand["corpus"]["ids_sha256"]:
        problems.append(
            f"different corpus: {cand['corpus']['ids_sha256'][:12]}... vs "
            f"{ref['corpus']['ids_sha256'][:12]}...")

    for key in ("window", "windows", "predictions", "kl_stride", "samples"):
        if ref["eval"][key] != cand["eval"][key]:
            problems.append(
                f"eval.{key}: {cand['eval'][key]} != {ref['eval'][key]}")

    # The reference has to be the fp32 engine. Comparing int4 against int8 and
    # calling the result agreement would measure how alike two approximations
    # are, not how good either one is.
    if ref.get("policy") != "fp32":
        problems.append(
            f"reference policy is {ref.get('policy')!r}: the reference must be "
            f"the fp32 engine, since every metric here is defined relative to "
            f"it")

    return problems


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--max-ppl-ratio", type=float,
                        default=DEFAULTS["max_ppl_ratio"])
    parser.add_argument("--min-top1-agreement", type=float,
                        default=DEFAULTS["min_top1_agreement"])
    parser.add_argument("--max-mean-kl", type=float,
                        default=DEFAULTS["max_mean_kl"])
    parser.add_argument("--max-p99-kl", type=float,
                        default=DEFAULTS["max_p99_kl"])
    parser.add_argument("--json", action="store_true",
                        help="emit the metrics as JSON on stdout")
    args = parser.parse_args()

    ref_manifest, ref_base = load_run(args.reference)
    cand_manifest, cand_base = load_run(args.candidate)

    print(f"reference  {args.reference}")
    print(f"           policy {ref_manifest['policy']}, backend "
          f"{ref_manifest['backend']}")
    print(f"candidate  {args.candidate}")
    print(f"           policy {cand_manifest['policy']}, backend "
          f"{cand_manifest['backend']}")

    problems = check_provenance(ref_manifest, cand_manifest)
    if problems:
        print("\nPROVENANCE MISMATCH", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    ref_nll = np.load(ref_base / "nll.npy").astype(np.float64)
    cand_nll = np.load(cand_base / "nll.npy").astype(np.float64)
    ref_top1 = np.load(ref_base / "top1.npy")
    cand_top1 = np.load(cand_base / "top1.npy")
    ref_logits = np.load(ref_base / "logits_sample.npy").astype(np.float64)
    cand_logits = np.load(cand_base / "logits_sample.npy").astype(np.float64)

    # Perplexity from the dumped per-position values, not from the summary the
    # engine printed: the judge should not take the defendant's word for it.
    ref_ppl = float(np.exp(ref_nll.mean()))
    cand_ppl = float(np.exp(cand_nll.mean()))
    ratio = cand_ppl / ref_ppl

    agreement = float((ref_top1 == cand_top1).mean())

    # KL(P_fp32 || P_cand): the expectation is taken under the fp32
    # distribution, which is the one being treated as correct. It is not
    # symmetric, and this direction is the one that asks "how much probability
    # mass did the reference put where the candidate does not follow".
    ref_lp = log_softmax(ref_logits)
    cand_lp = log_softmax(cand_logits)
    kl = (np.exp(ref_lp) * (ref_lp - cand_lp)).sum(axis=-1)
    mean_kl = float(kl.mean())
    p99_kl = float(np.percentile(kl, 99))
    max_kl = float(kl.max())

    tol = {
        "max_ppl_ratio": args.max_ppl_ratio,
        "min_top1_agreement": args.min_top1_agreement,
        "max_mean_kl": args.max_mean_kl,
        "max_p99_kl": args.max_p99_kl,
    }
    checks = [
        ("perplexity", f"{cand_ppl:.4f} vs {ref_ppl:.4f}  (x{ratio:.5f})",
         ratio <= tol["max_ppl_ratio"], f"<= x{tol['max_ppl_ratio']}"),
        ("top-1 agreement", f"{agreement * 100:.3f}%  "
         f"({int((ref_top1 == cand_top1).sum())}/{ref_top1.size})",
         agreement >= tol["min_top1_agreement"],
         f">= {tol['min_top1_agreement'] * 100:.1f}%"),
        ("mean KL", f"{mean_kl:.6f} nats",
         mean_kl <= tol["max_mean_kl"], f"<= {tol['max_mean_kl']}"),
        ("p99 KL", f"{p99_kl:.6f} nats  (max {max_kl:.6f})",
         p99_kl <= tol["max_p99_kl"], f"<= {tol['max_p99_kl']}"),
    ]

    print(f"\nprovenance ok  |  {ref_manifest['eval']['predictions']} "
          f"predictions, {ref_manifest['eval']['samples']} sampled for KL\n")
    failed = False
    for name, value, ok, limit in checks:
        mark = "ok  " if ok else "FAIL"
        print(f"  {mark} {name:17s} {value:44s} {limit}")
        if not ok:
            failed = True

    if args.json:
        print("\n" + json.dumps({
            "policy": cand_manifest["policy"],
            "backend": cand_manifest["backend"],
            "perplexity": cand_ppl,
            "reference_perplexity": ref_ppl,
            "perplexity_ratio": ratio,
            "top1_agreement": agreement,
            "mean_kl": mean_kl,
            "p99_kl": p99_kl,
            "max_kl": max_kl,
            "tolerance": tol,
            "pass": not failed,
        }, indent=2))

    if failed:
        print("\nQUANTIZED EVAL FAILED", file=sys.stderr)
        return 1
    print("\nQUANTIZED EVAL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
