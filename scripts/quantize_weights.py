#!/usr/bin/env python3
"""Quantize the flat weight file to int8, symmetric, per output channel.

    python scripts/quantize_weights.py         # -> weights/gpt2-124m-int8.bin

Reads the fp32 file and rewrites the large matrices as int8 plus a float32
scale per output channel. Everything else -- layer norms, biases, the position
table -- stays fp32: together they are 3.6 MB of a 498 MB file, and they are
the parameters least able to absorb the error.

    w[i, j] ~= q[i, j] * scale[j],  q in [-127, 127]

Symmetric, so there is no zero point: weight distributions here are centred
near zero, and an asymmetric scheme would spend a subtraction per element to
recover a fraction of a percent.

Per output channel rather than per tensor, which is not a refinement but the
whole thing working or not. Measured on the Phase 4 eval slice, simulating the
scheme in PyTorch before any of it was written in C++:

    fp32                                36.3366
    per-tensor scales                   42.3039   +16.4%
    per-channel, the 48 layer matrices  36.2798   free
    per-channel, wte alone              36.9502   +1.69%
    per-channel, everything             36.8946   +1.54%

One scale for a whole matrix is dominated by that matrix's largest outlier and
throws away resolution everywhere else. Per channel, the layers quantize for
nothing at all -- the small improvement is noise on a 4088-position slice, not
a gain -- and effectively all of the cost is wte.

wte is left in fp32 by default, and that decision came from the engine rather
than the simulation. Quantizing it too compresses 3.90x instead of 2.05x, and
perplexity still passes at +1.54% -- but measured against the fp32 engine on
the same slice:

                        perplexity   top-1     mean KL
    int8, wte fp32       x0.99844   97.48%    0.00117
    int8, wte int8       x1.01536   83.02%    0.04139

Seventeen percent of argmaxes change and the distribution moves 35x further,
for 1.9x more compression. Perplexity alone would have waved that through,
which is the argument for measuring more than one thing. --quantize-wte takes
the trade if the memory matters more.

The output channel is the axis the matmul reduces *to*:

    [in, out] projections   scale per column, shape [out]
    wte [vocab, d_model]    scale per row, shape [vocab] -- as lm_head its
                            output channel is the vocabulary, and the same
                            per-row scale is what an embedding lookup wants

Scales ride along as ordinary f32 tensors named "<tensor>.scale", so the file
format needs nothing new beyond a dtype value it already had a field for.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from convert_weights import (  # noqa: E402
    QUANT_INT8_PER_CHANNEL, config_from_json, read_bin, write_bin,
)

DEFAULT_IN = ROOT / "weights" / "gpt2-124m.bin"
DEFAULT_OUT = ROOT / "weights" / "gpt2-124m-int8.bin"
DEFAULT_CONFIG = ROOT / "weights" / "gpt2-124m" / "config.json"

# The matrices worth quantizing: every one that a matmul streams. Together they
# are 99.3% of the file.
PROJECTION = re.compile(r"h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight$")


def quantize(w: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 along `axis`, which is the axis reduced over -- so the
    scale ends up with one entry per output channel."""
    amax = np.abs(w).max(axis=axis, keepdims=True)
    # A channel of exact zeros would divide by zero; its quantized values are
    # zero regardless, so any positive scale reconstructs it exactly.
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, np.ascontiguousarray(scale.reshape(-1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quantize-wte", action="store_true",
                        help="also quantize the embedding table; measured to "
                             "cost far more than it saves, see above")
    parser.add_argument("--keep-fp32", nargs="*", default=[],
                        help="further tensor names to leave alone")
    args = parser.parse_args()

    meta, tensors = read_bin(args.src)
    if meta["quant"] != 0:
        raise SystemExit(f"{args.src} is already quantized (quant={meta['quant']})")
    cfg = config_from_json(args.config)

    out: dict[str, np.ndarray] = {}
    n_quantized = 0
    src_bytes = sum(a.nbytes for a in tensors.values())

    for name, a in tensors.items():
        if name in args.keep_fp32 or (name == "wte" and not args.quantize_wte):
            out[name] = a
            continue

        if name == "wte":
            axis = 1           # reduce over d_model -> one scale per vocab row
        elif PROJECTION.fullmatch(name):
            axis = 0           # reduce over n_in -> one scale per output column
        else:
            out[name] = a
            continue

        q, scale = quantize(a.astype(np.float32), axis)
        out[name] = q
        out[f"{name}.scale"] = scale
        n_quantized += 1

        rel = float(np.abs(q.astype(np.float32) * scale.reshape(
            (-1, 1) if axis == 1 else (1, -1)) - a).max() / np.abs(a).max())
        print(f"  {name:28s} {str(a.shape):14s} -> int8 + {scale.size:5d} "
              f"scales   worst rel {rel:.5f}")

    dst_bytes = sum(a.nbytes for a in out.values())
    source_sha = bytes.fromhex(meta["source_sha256"])
    write_bin(args.out, cfg, out, source_sha, quant=QUANT_INT8_PER_CHANNEL)

    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")
    print(f"  {n_quantized} tensors quantized, {len(out)} total")
    print(f"  {src_bytes / 1e6:.1f} MB -> {dst_bytes / 1e6:.1f} MB "
          f"({src_bytes / dst_bytes:.2f}x smaller)")
    print(f"  source sha256 {meta['source_sha256'][:16]}... (carried through)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
