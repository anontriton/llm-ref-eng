#!/usr/bin/env python3
"""Which use of wte takes the quantization damage, and what could ship instead.

    python reference/wte_sim.py

Phase 5's download is the weight file, and 154 MB of the shipping 243 MB is
wte, kept fp32 because Phase 4 measured int8 wte changing 17% of argmaxes.
wte does two jobs -- the embedding lookup and the tied lm_head -- and Phase 4
never separated them. This does, and then tries representations of the table
that a browser could be asked to download.

Every variant keeps the twelve layers at int8 per-channel, as the engine ships
them, and is scored against the all-fp32 model with eval/metrics.py's own
metrics and thresholds, over every predicted position of the 8x512 eval slice.
The split variants give the embedding and lm_head separate tables; they are a
diagnosis, not a format, since shipping both would be larger than either.

Measured:

    variant                    download  ppl ratio   top-1   mean KL  decisive
    wte fp32 (ships today)       243.3   x0.99844   97.505%  0.001134  0.024%  pass
    int8 row, both uses          127.7   x1.01536   83.023%  0.041824  4.183%  FAIL
    int8 row, lm_head only           --  x1.01540   83.121%  0.041851  4.159%  FAIL
    int8 row, embedding only         --  x0.99843   97.578%  0.001134  0.024%  pass
    int8 row g64, both           129.9   x1.00490   89.408%  0.014891  1.590%  FAIL
    fp16, both                   166.1   x0.99847   97.529%  0.001150  0.024%  pass
    bf16, both                   166.1   x0.99731   93.713%  0.007755  0.269%  FAIL
    int8 row + 4 cols fp32       128.5   x0.99912   96.747%  0.001620  0.049%  FAIL
    int8 row + 8 cols fp32       129.3   x0.99805   97.407%  0.001327  0.024%  pass

The damage is all lm_head. The embedding absorbs int8 for free; the head does
not, and finer groups along d_model only get it to 89%. The reason is the
input: ln_f's output has a few enormous hidden dimensions -- dim 496 averages
|x| = 201 against a median of 0.35 -- so an int8 error in those columns of wte
is multiplied by two hundred, and those same columns set every row's scale.
Take the 8 largest out of the int8 table and keep them in fp32, and the head
behaves like fp32 again at 129 MB -- 8 of 768 columns, 1.6 MB, held back.

The margin is honest but not wide: top-1 clears its 97% bar by 0.4 points,
while decisive disagreement is exactly the shipping engine's. Four columns
miss top-1 by a quarter point. Keeping them in fp16 instead of fp32 scored
97.456% at 8 and 97.285% at 16 in an earlier run, so the count is not
monotone at this resolution; 8 is where it clears, not an optimum.

The outlier columns are chosen on eval/calib.tsv, not on the slice they are
scored on. fp16 also passes, at 166 MB, with no new format beyond a dtype.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import DEFAULTS  # noqa: E402
from quant_sim import PROJECTION, corpus_ids, quant_dequant  # noqa: E402
from weights import DEFAULT_WEIGHTS, load_gpt2  # noqa: E402

WINDOW, WINDOWS = 512, 8
MB = 1e6
# The shipping int8 file, less the fp32 wte inside it: layers, norms, biases, wpe.
SHIPPING_INT8_BYTES = 243_312_256


def forward(model, ids: torch.Tensor, emb: torch.Tensor,
            head: torch.Tensor) -> torch.Tensor:
    """reference/model.py's forward, with the embedding and the tied head fed
    from separate tables so each use can be quantized alone."""
    T = ids.shape[1]
    x = emb[ids] + model.wpe[torch.arange(T)]
    for i, block in enumerate(model.h):
        x = block(x, f"block.{i}", lambda *a: None)
    return model.ln_f(x) @ head.T


def outlier_columns(model, calib: list[int]) -> torch.Tensor:
    """Hidden dims of ln_f's output, largest mean |x| first. Measured on the
    calibration slice, never on the one being scored."""
    with torch.no_grad():
        ids = torch.tensor([calib])
        x = model.wte[ids] + model.wpe[torch.arange(len(calib))]
        for i, block in enumerate(model.h):
            x = block(x, f"block.{i}", lambda *a: None)
        act = model.ln_f(x)[0].abs().mean(0)
    order = act.argsort(descending=True)
    print("ln_f |x| by dim, largest:",
          ", ".join(f"{int(d)}={float(act[d]):.1f}" for d in order[:8]),
          f"(median {float(act.median()):.3f})")
    return order


def main() -> int:
    ids = corpus_ids(ROOT / "eval" / "corpus.tsv")[:WINDOW * WINDOWS]
    calib = corpus_ids(ROOT / "eval" / "calib.tsv")[:WINDOW]

    ref = load_gpt2(str(DEFAULT_WEIGHTS))
    q = load_gpt2(str(DEFAULT_WEIGHTS))
    with torch.no_grad():
        for name, p in q.named_parameters():
            if PROJECTION.search(name):
                p.data = quant_dequant(p.data, None, 8, False, 0)

    W = ref.wte.data.clone()
    V, D = W.shape
    rest = SHIPPING_INT8_BYTES / MB - V * D * 4 / MB
    order = outlier_columns(ref, calib)

    def int8_rows(group=None):
        # One scale per row -- lm_head's output channel -- as the engine does.
        return quant_dequant(W, group, 8, False, 1)

    def int8_plus(k):
        keep = order[:k]
        Wm = W.clone()
        Wm[:, keep] = 0          # outliers no longer set the row's scale
        out = quant_dequant(Wm, None, 8, False, 1)
        out[:, keep] = W[:, keep]
        return out

    int8_bytes = V * D + V * 4
    tables = {  # name -> (table, download MB for wte in that form)
        "fp32": (W, V * D * 4),
        "int8 row": (int8_rows(), int8_bytes),
        "int8 row g64": (int8_rows(64), V * D + V * (D // 64) * 4),
        "fp16": (W.half().float(), V * D * 2),
        "bf16": (W.bfloat16().float(), V * D * 2),
        "int8+4": (int8_plus(4), int8_bytes + V * 4 * 4),
        "int8+8": (int8_plus(8), int8_bytes + V * 8 * 4),
    }
    variants = [  # label, embedding table, lm_head table
        ("wte fp32 (ships today)", "fp32", "fp32"),
        ("int8 row, both uses", "int8 row", "int8 row"),
        ("int8 row, lm_head only", "fp32", "int8 row"),
        ("int8 row, embedding only", "int8 row", "fp32"),
        ("int8 row g64, both", "int8 row g64", "int8 row g64"),
        ("fp16, both", "fp16", "fp16"),
        ("bf16, both", "bf16", "bf16"),
        ("int8 row + 4 cols fp32", "int8+4", "int8+4"),
        ("int8 row + 8 cols fp32", "int8+8", "int8+8"),
    ]

    acc = {v[0]: {"nll": [], "agree": [], "kl": [], "decisive": []}
           for v in variants}
    ref_nll = []
    with torch.no_grad():
        for w in range(WINDOWS):
            chunk = torch.tensor([ids[w * WINDOW:(w + 1) * WINDOW]])
            tgt = chunk[0, 1:]
            rows = torch.arange(len(tgt))
            r_lp = torch.log_softmax(forward(ref, chunk, W, W)[0, :-1].double(), -1)
            r_p, r_top = r_lp.exp(), r_lp.argmax(-1)
            ref_nll.append(-r_lp.gather(1, tgt[:, None])[:, 0])
            for label, e, h in variants:
                c_lp = torch.log_softmax(
                    forward(q, chunk, tables[e][0], tables[h][0])[0, :-1].double(), -1)
                c_top = c_lp.argmax(-1)
                # Same definitions as eval/metrics.py: KL(P_fp32 || P_cand),
                # and a flip is decisive when fp32 preferred its own pick by
                # more than the margin.
                margin = r_p[rows, r_top] - r_p[rows, c_top]
                a = acc[label]
                a["nll"].append(-c_lp.gather(1, tgt[:, None])[:, 0])
                a["agree"].append(r_top == c_top)
                a["kl"].append((r_p * (r_lp - c_lp)).sum(-1))
                a["decisive"].append((r_top != c_top)
                                     & (margin > DEFAULTS["decisive_margin"]))
            print(f"  window {w + 1}/{WINDOWS}", file=sys.stderr)

    ref_ppl = float(torch.cat(ref_nll).mean().exp())
    print(f"fp32 perplexity {ref_ppl:.4f} over {len(torch.cat(ref_nll))} positions\n")
    print(f"{'variant':26} {'download':>9} {'ppl ratio':>10} {'top-1':>8} "
          f"{'mean KL':>9} {'p99 KL':>8} {'decisive':>9}")
    for label, e, h in variants:
        a = {k: torch.cat(v) for k, v in acc[label].items()}
        ratio = float(a["nll"].mean().exp()) / ref_ppl
        agree = float(a["agree"].double().mean())
        kl = a["kl"].numpy()
        p99 = float(np.percentile(kl, 99))
        dec = float(a["decisive"].double().mean())
        ok = (ratio <= DEFAULTS["max_ppl_ratio"]
              and agree >= DEFAULTS["min_top1_agreement"]
              and kl.mean() <= DEFAULTS["max_mean_kl"]
              and p99 <= DEFAULTS["max_p99_kl"]
              and dec <= DEFAULTS["max_decisive_rate"])
        dl = f"{rest + tables[e][1] / MB:7.1f}MB" if e == h else "       --"
        print(f"{label:26} {dl} {ratio:10.5f} {agree * 100:7.3f}% "
              f"{kl.mean():9.6f} {p99:8.5f} {dec * 100:8.3f}%  "
              f"{'pass' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
