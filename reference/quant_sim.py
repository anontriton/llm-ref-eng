#!/usr/bin/env python3
"""Try a quantization scheme in PyTorch before writing it in C++.

    python reference/quant_sim.py --bits 8 --group none
    python reference/quant_sim.py --bits 4 --group 64 --asym
    python reference/quant_sim.py --bits 4 --group 64 --include wte

Quantizes the reference model's weights in place and measures perplexity on the
same WikiText-2 slice, in the same windows, that eval/ uses -- so a number here
is directly comparable to one from the engine.

This exists because the int8 scheme was chosen here, in an afternoon, rather
than by writing four C++ backends and measuring them. Deciding representation,
group size and axis is cheap in torch and expensive in a kernel, and the
simulation predicted the engine to five decimal places when it was finally
built: 36.8946 simulated, 36.894570 measured.

It answers one question only -- how much damage does this scheme do. Speed,
memory traffic and whether the arithmetic is reproducible across backends are
questions for the engine, and the int8 work found the engine disagreeing with
the simulation about which trade was worth taking.

Measured on the 8x512 slice, quantizing the twelve layers' projections and
leaving wte, layer norms and biases in fp32:

    scheme                      eff. bits   perplexity   vs fp32
    fp32                             32.0      36.3366         --
    int8  sym per-channel             8.00     36.2798      -0.2%
    int6  sym group 64                6.50     36.4479      +0.3%
    int5  sym group 64                5.50     37.4885      +3.2%
    int4  asym group 32               6.00     37.7196      +3.8%
    int4  asym group 64               5.00     39.2197      +7.9%
    int4  sym group 32                5.00     39.9095      +9.8%
    int4  sym group 64                4.50     42.3685     +16.6%
    int4  sym group 128               4.25     42.9408     +18.2%
    int4  sym per-channel             4.00     55.1786     +51.9%

Two things fall out of that table. Round-to-nearest int4 does not reach the
Phase 4 thresholds at any group size -- the best of them is eight times over
the 2% perplexity budget, and getting there needs a quantizer that compensates
for its own error rather than one that rounds. And at equal cost, spending bits
on the codebook beats spending them on the scales: int5 at group 64 costs 5.50
bits and lands at +3.2%, while int4 asymmetric at group 32 costs 6.00 and lands
at +3.8%.

Effective bits counts the scales, which are the part that is easy to forget:
a symmetric group of G carries one fp32 scale, an asymmetric group carries two,
so 32/G bits or 64/G bits per weight on top of the codes. Per-channel is the
group-equals-the-whole-axis case and its overhead is 32/n_in, under 0.05 bits
here, which is why it is reported as the bare width.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))

PROJECTION = re.compile(r"(c_attn|c_proj|c_fc)\.weight$")


def corpus_ids(tsv: Path) -> list[int]:
    for line in tsv.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        return [int(v) for v in line.split("\t")[2].split(",")]
    raise SystemExit(f"no rows in {tsv}")


def quant_dequant(w: torch.Tensor, group: int | None, bits: int,
                  asym: bool, axis: int) -> torch.Tensor:
    """Round-trip through the scheme, staying in fp32.

    `axis` is the one the matmul reduces over, so groups are contiguous runs of
    the terms that get summed -- which is the only grouping a kernel can apply
    cheaply, since it can scale a partial sum once per group instead of once
    per weight.
    """
    if axis == 1:
        return quant_dequant(w.t().contiguous(), group, bits, asym, 0).t().contiguous()

    n_in, n_out = w.shape
    g = n_in if group is None else group
    if n_in % g:
        raise SystemExit(f"reduction axis {n_in} is not divisible by group {g}")
    wg = w.reshape(n_in // g, g, n_out)

    if asym:
        lo = wg.amin(dim=1, keepdim=True)
        hi = wg.amax(dim=1, keepdim=True)
        levels = 2 ** bits - 1
        scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
        zero = torch.round(-lo / scale)
        q = torch.clamp(torch.round(wg / scale) + zero, 0, levels)
        out = (q - zero) * scale
    else:
        amax = wg.abs().amax(dim=1, keepdim=True)
        qmax = 2 ** (bits - 1) - 1
        scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
        q = torch.clamp(torch.round(wg / scale), -qmax, qmax)
        out = q * scale

    return out.reshape(n_in, n_out)


def perplexity(model, ids: list[int], window: int, windows: int) -> float:
    total, count = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            chunk = ids[w * window:(w + 1) * window]
            logits = model(torch.tensor([chunk], dtype=torch.long))
            if isinstance(logits, tuple):
                logits = logits[0]
            lp = torch.log_softmax(logits[0].double(), dim=-1)
            tgt = torch.tensor(chunk[1:], dtype=torch.long)
            nll = -lp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)
            total += float(nll.sum())
            count += nll.numel()
    return float(np.exp(total / count))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group", default="none",
                        help="'none' for one scale per output channel, else a "
                             "group size along the reduction axis")
    parser.add_argument("--asym", action="store_true",
                        help="carry a zero point per group as well as a scale")
    parser.add_argument("--include", nargs="*", default=[],
                        help="extra tensors to quantize, e.g. wte")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "eval" / "corpus.tsv")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--fp32", action="store_true",
                        help="quantize nothing; prints the baseline")
    args = parser.parse_args()

    group = None if args.group == "none" else int(args.group)
    ids = corpus_ids(args.corpus)[:args.window * args.windows]

    from weights import DEFAULT_WEIGHTS, load_gpt2
    model = load_gpt2(str(DEFAULT_WEIGHTS))

    touched = 0
    if not args.fp32:
        with torch.no_grad():
            for name, p in model.named_parameters():
                is_proj = PROJECTION.search(name)
                is_extra = any(name.endswith(x) or name == x for x in args.include)
                if not (is_proj or is_extra):
                    continue
                # wte is [vocab, d_model] and its output channel as the tied
                # lm_head is the vocabulary, so it reduces over dim 1; the
                # projections are [in, out] and reduce over dim 0.
                axis = 1 if (is_extra and not is_proj) else 0
                p.data = quant_dequant(p.data, group, args.bits, args.asym, axis)
                touched += 1

    overhead = 0.0 if group is None else (32.0 / group) * (2 if args.asym else 1)
    eff = 32.0 if args.fp32 else args.bits + overhead
    ppl = perplexity(model, ids, args.window, args.windows)

    scheme = "fp32" if args.fp32 else (
        f"int{args.bits} {'asym' if args.asym else 'sym'} "
        f"group {args.group}")
    print(f"scheme        {scheme}")
    print(f"tensors       {touched}")
    print(f"effective     {eff:.2f} bits/weight")
    print(f"perplexity    {ppl:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
