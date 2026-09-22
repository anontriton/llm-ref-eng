#!/usr/bin/env python3
"""GPTQ: quantize each weight so the layer's *output* stays put.

    python reference/gptq.py --bits 4 --group 64            # measure only
    python reference/gptq.py --bits 4 --group 64 --save-npz out.npz

Round-to-nearest quantization minimizes the error in each weight, which is not
the thing anyone cares about. reference/quant_sim.py measured what that costs
at 4 bits: the best grouping lands 16% over fp32 perplexity, eight times the
budget in eval/metrics.py. The weights are individually close and the layer's
output is not.

GPTQ (Frantar et al., 2022) minimizes the error in the layer's output instead.
It walks the input dimension one column at a time, and after rounding a column
it pushes the resulting output error into the columns not yet quantized, so
they can absorb it. Which columns can absorb what is decided by the Hessian of
the layer's output error with respect to its weights, H = 2 X^T X, estimated
from calibration activations -- an input direction the data never exercises
gets no say.

    for each column i, in order:
        q      <- quantize(w[:, i])
        err    <- (w[:, i] - dequant(q)) / Hinv[i, i]
        w[:, i+1:] -= err * Hinv[i, i+1:]

The division by Hinv[i, i] is what makes it more than bookkeeping: it converts
a weight error into the weight update elsewhere that cancels its effect on the
output, to second order.

Calibration comes from the WikiText-2 *train* split. The eval slice is the test
split, and quantizing against the data you then report perplexity on measures
nothing. eval/calib.tsv and eval/corpus.tsv are checksummed separately and this
refuses to run if they are the same text.

Sequential over blocks: block i is calibrated on activations produced by blocks
0..i-1 as already quantized, so each block compensates for the drift the ones
before it introduced rather than for a fp32 input it will never see.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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


def quant_group(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-row scale for one group of columns.

    w is [n_out, group]; the scale is one number per output row, which is the
    axis the engine can apply once after a reduction.
    """
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    return scale, torch.tensor(float(qmax))


def gptq_layer(W: torch.Tensor, H: torch.Tensor, bits: int, group: int,
               percdamp: float = 0.01, blocksize: int = 128,
               act_order: bool = False):
    """Quantize one [n_in, n_out] weight against Hessian H [n_in, n_in].

    Returns (codes [n_in, n_out] int8, scales [n_groups, n_out] float32).
    """
    Wt = W.t().contiguous().double()          # [n_out, n_in]; columns are inputs
    n_out, n_in = Wt.shape
    H = H.clone().double()

    # An input the calibration set never moves has no opinion about its column.
    # Zeroing the weight would change the layer; leaving H singular would break
    # the Cholesky, so the diagonal is propped up instead.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0

    # Activation ordering: quantize the input directions the calibration data
    # exercises most, first, while the columns still to come have the most room
    # to absorb their error. Columns are permuted, so groups are formed over
    # the permuted order -- which is why this costs the engine a per-column
    # group index and breaks the group == kLeaf correspondence.
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        Wt = Wt[:, perm]
        H = H[perm][:, perm]

    damp = percdamp * torch.mean(torch.diag(H))
    H[range(n_in), range(n_in)] += damp

    # Hinv, then its upper Cholesky factor: row i of this is exactly the update
    # direction for the columns after i.
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    codes = torch.zeros(n_out, n_in, dtype=torch.int8)
    n_groups = (n_in + group - 1) // group
    scales = torch.zeros(n_groups, n_out, dtype=torch.float32)

    for b0 in range(0, n_in, blocksize):
        b1 = min(b0 + blocksize, n_in)
        Wb = Wt[:, b0:b1].clone()
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[b0:b1, b0:b1]

        scale = None
        for j in range(b1 - b0):
            i = b0 + j
            if i % group == 0:
                # From the current, already-compensated weights: the scale has
                # to describe the numbers actually being rounded, not the ones
                # this group held before earlier columns pushed error into it.
                gj = min(group, b1 - b0 - j)
                scale, qmax = quant_group(Wb[:, j:j + gj], bits)
                scales[i // group] = scale.float()

            w = Wb[:, j]
            q = torch.clamp(torch.round(w / scale), -qmax, qmax)
            codes[:, i] = q.to(torch.int8)

            err = (w - q * scale) / Hb[j, j]
            if j + 1 < b1 - b0:
                Wb[:, j + 1:] -= err.unsqueeze(1) * Hb[j, j + 1:].unsqueeze(0)
            Eb[:, j] = err

        # Carry the block's accumulated error into everything still to come.
        if b1 < n_in:
            Wt[:, b1:] -= Eb @ Hinv[b0:b1, b1:]

    if perm is not None:
        # Back to the original column order. The scales stay in permuted-group
        # order, which is the part an engine would have to carry an index for;
        # here the simulation just dequantizes before returning.
        inv = torch.argsort(perm)
        deq = torch.zeros_like(Wt)
        for gi in range(n_groups):
            lo, hi = gi * group, min((gi + 1) * group, n_in)
            deq[:, lo:hi] = codes[:, lo:hi].double() * scales[gi].double().unsqueeze(1)
        return ("dequantized", deq[:, inv].t().contiguous().float())

    return codes.t().contiguous(), scales


def dequantize(codes: torch.Tensor, scales: torch.Tensor,
               group: int) -> torch.Tensor:
    """codes [n_in, n_out] int8, scales [n_groups, n_out] -> fp32 [n_in, n_out]."""
    n_in, n_out = codes.shape
    out = codes.float().reshape(-1, group, n_out) * scales.unsqueeze(1)
    return out.reshape(n_in, n_out)


def collect(model, ids: list[int], window: int, windows: int):
    """Per-position nll, argmax and log-probs over the eval windows.

    The same quantities eval/ computes in the engine, so a simulated scheme can
    be held to the same four metrics before anyone writes a kernel for it.
    """
    nll, top1, lps = [], [], []
    with torch.no_grad():
        for w in range(windows):
            chunk = ids[w * window:(w + 1) * window]
            logits = model(torch.tensor([chunk], dtype=torch.long))
            if isinstance(logits, tuple):
                logits = logits[0]
            lp = torch.log_softmax(logits[0].double(), dim=-1)[:-1]
            tgt = torch.tensor(chunk[1:], dtype=torch.long)
            nll.append(-lp.gather(1, tgt.unsqueeze(1)).squeeze(1))
            top1.append(lp.argmax(dim=-1))
            lps.append(lp[::16])          # strided, as the engine samples for KL
    return (torch.cat(nll), torch.cat(top1), torch.cat(lps))


def perplexity(model, ids: list[int], window: int, windows: int) -> float:
    return float(np.exp(collect(model, ids, window, windows)[0].mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group", type=int, default=64,
                        help="columns per scale, along the reduction axis; 64 "
                             "matches backend::kAccumLanes' leaf so the engine "
                             "can scale a partial sum instead of a weight")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--act-order", action="store_true",
                        help="quantize high-Hessian columns first; simulation "
                             "only, since it permutes group membership")
    parser.add_argument("--calib", type=Path, default=ROOT / "eval" / "calib.tsv")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "corpus.tsv")
    parser.add_argument("--calib-seqs", type=int, default=24)
    parser.add_argument("--calib-len", type=int, default=512)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--save-npz", type=Path, default=None,
                        help="write codes and scales for scripts/ to pack")
    parser.add_argument("--compare", action="store_true",
                        help="also report top-1 agreement and KL against the "
                             "unquantized reference, as eval/metrics.py would")
    args = parser.parse_args()

    calib_ids = corpus_ids(args.calib)
    eval_ids = corpus_ids(args.corpus)
    if hashlib.sha256(str(calib_ids).encode()).hexdigest() == \
       hashlib.sha256(str(eval_ids).encode()).hexdigest():
        raise SystemExit("calibration and eval corpora are the same text; "
                         "GPTQ would be tuned on its own test set")

    need = args.calib_seqs * args.calib_len
    if len(calib_ids) < need:
        raise SystemExit(f"calibration set has {len(calib_ids)} ids, need {need}")

    from weights import DEFAULT_WEIGHTS, load_gpt2
    model = load_gpt2(str(DEFAULT_WEIGHTS))
    model.eval()

    baseline = None
    if args.compare:
        print("collecting the fp32 reference first...", flush=True)
        baseline = collect(model, eval_ids, args.window, args.windows)
    cfg = model.cfg if hasattr(model, "cfg") else model.config

    print(f"calibration  {args.calib.name}, {args.calib_seqs} x {args.calib_len} tokens")
    print(f"scheme       int{args.bits}, group {args.group}, percdamp {args.percdamp}")

    # Block.forward calls its tap unconditionally; only GPT2.forward defaults
    # it. Driving blocks directly means supplying one.
    no_tap = lambda *a, **k: None

    # Hidden states entering block 0, for every calibration sequence.
    with torch.no_grad():
        hidden = []
        for s in range(args.calib_seqs):
            x = torch.tensor([calib_ids[s * args.calib_len:(s + 1) * args.calib_len]],
                             dtype=torch.long)
            positions = torch.arange(x.shape[1])
            hidden.append(model.wte[x] + model.wpe[positions])

    saved: dict[str, np.ndarray] = {}
    t0 = time.time()

    for bi, block in enumerate(model.h):
        targets = {
            "attn.c_attn": block.attn.c_attn,
            "attn.c_proj": block.attn.c_proj,
            "mlp.c_fc": block.mlp.c_fc,
            "mlp.c_proj": block.mlp.c_proj,
        }
        hess = {k: torch.zeros(m.weight.shape[0], m.weight.shape[0], dtype=torch.double)
                for k, m in targets.items()}
        counts = {k: 0 for k in targets}

        handles = []
        def make_hook(key):
            def hook(_mod, inputs, _out):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
                hess[key] += 2.0 * (x.t() @ x)
                counts[key] += x.shape[0]
            return hook
        for k, m in targets.items():
            handles.append(m.register_forward_hook(make_hook(k)))

        with torch.no_grad():
            for h in hidden:
                block(h, f"block.{bi}", no_tap)
        for hd in handles:
            hd.remove()

        for k, m in targets.items():
            codes, scales = gptq_layer(m.weight.data, hess[k], args.bits,
                                       args.group, args.percdamp,
                                       act_order=args.act_order)
            with torch.no_grad():
                if isinstance(codes, str):
                    m.weight.data = scales
                else:
                    m.weight.data = dequantize(codes, scales, args.group)
            if args.save_npz is not None and not isinstance(codes, str):
                saved[f"h.{bi}.{k}.weight.codes"] = codes.numpy()
                saved[f"h.{bi}.{k}.weight.scales"] = scales.numpy()

        # Re-run with the quantized block so the next one calibrates on what it
        # will actually be fed.
        with torch.no_grad():
            hidden = [block(h, f"block.{bi}", no_tap) for h in hidden]

        print(f"  block {bi:2d}/{len(model.h)}  {time.time() - t0:6.1f}s", flush=True)

    overhead = 32.0 / args.group
    nll, top1, lp = collect(model, eval_ids, args.window, args.windows)
    ppl = float(np.exp(nll.mean()))
    print()
    print(f"effective    {args.bits + overhead:.2f} bits/weight")
    print(f"perplexity   {ppl:.4f}   (fp32 36.3366, RTN int4 g64 42.3685)")

    if baseline is not None:
        b_nll, b_top1, b_lp = baseline
        ratio = ppl / float(np.exp(b_nll.mean()))
        agree = float((b_top1 == top1).double().mean())
        kl = (b_lp.exp() * (b_lp - lp)).sum(dim=-1)
        # Decisive: fp32 preferred its own pick by more than 0.05 probability.
        p = b_lp.exp()
        idx = torch.arange(b_lp.shape[0])
        stride_top1_b = b_top1[::16][:b_lp.shape[0]]
        stride_top1_c = top1[::16][:b_lp.shape[0]]
        margin = p[idx, stride_top1_b] - p[idx, stride_top1_c]
        decisive = float(((stride_top1_b != stride_top1_c) & (margin > 0.05))
                         .double().mean())
        print(f"ppl ratio    x{ratio:.5f}")
        print(f"top-1        {agree * 100:.3f}%")
        print(f"mean KL      {float(kl.mean()):.6f} nats")
        print(f"p99 KL       {float(np.percentile(kl.numpy(), 99)):.6f} nats")
        print(f"decisive     {decisive * 100:.3f}%")

    if args.save_npz is not None:
        np.savez(args.save_npz, **saved)
        print(f"wrote        {args.save_npz}  ({len(saved)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
