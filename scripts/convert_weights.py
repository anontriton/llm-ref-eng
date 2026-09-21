#!/usr/bin/env python3
"""Convert the GPT-2 safetensors checkpoint into the flat binary the C++ engine loads.

Deferred until Phase 2 on purpose: the on-disk layout should follow the engine's
loader, not be guessed at in Phase 0. What the engine wants is a file it can read
in one pass with no JSON parser, no external dependency, and no per-tensor
allocation -- so that is what this writes.

    python scripts/convert_weights.py            # -> weights/gpt2-124m.bin
    python scripts/convert_weights.py --verify    # re-read and check against source

Layout (little-endian throughout):

    magic   "GPT2WTS1"                       8 bytes
    u32     version = 1
    u32     n_tensors
    u32     data_start                       byte offset of the data block
    u32     reserved
    config  u32 n_layer, n_head, d_model,
            d_ff, n_ctx, vocab_size
            f64 layer_norm_eps
    u8      source_sha256[32]                sha256 of model.safetensors
    entry[] n_tensors x 128 bytes            fixed stride, see ENTRY_FORMAT
    pad     to a 64-byte boundary
    data    contiguous float32, row-major

The source checksum travels inside the file so the engine can stamp it into its
own dump manifest without ever seeing the original checkpoint. That is what lets
oracle/compare.py verify the engine and the reference ran on the same weights.

Weights are stored in the checkpoint's own [in, out] orientation -- the one the
reference's Linear already adopts -- so neither this script nor the engine ever
transposes. `h.{i}.attn.bias` is dropped: it is HF's precomputed causal mask, a
buffer, not a learned weight.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))

from weights import DEFAULT_WEIGHTS, config_from_json, read_safetensors  # noqa: E402

MAGIC = b"GPT2WTS1"
VERSION = 1
NAME_FIELD = 64
MAX_DIMS = 4
ENTRY_SIZE = 128
DTYPE_F32 = 0
DTYPE_I8 = 1

# Header word 4, describing the file as a whole rather than one tensor.
QUANT_NONE = 0
QUANT_INT8_PER_CHANNEL = 1
DTYPE_SIZE = {DTYPE_F32: 4, DTYPE_I8: 1}
HEADER_FIXED = 8 + 16 + 32 + 32          # magic + counts + config + sha256
DATA_ALIGN = 64

IGNORED_SUFFIXES = (".attn.bias", ".attn.masked_bias")


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def expected_names(cfg) -> list[str]:
    """Every tensor the engine needs, in the order it wants them.

    Listing them explicitly rather than taking whatever the checkpoint happens
    to hold means a renamed or missing key is a loud failure here, not a subtly
    wrong number twelve layers later.
    """
    names = ["wte", "wpe"]
    for i in range(cfg.n_layer):
        names += [
            f"h.{i}.ln_1.weight", f"h.{i}.ln_1.bias",
            f"h.{i}.attn.c_attn.weight", f"h.{i}.attn.c_attn.bias",
            f"h.{i}.attn.c_proj.weight", f"h.{i}.attn.c_proj.bias",
            f"h.{i}.ln_2.weight", f"h.{i}.ln_2.bias",
            f"h.{i}.mlp.c_fc.weight", f"h.{i}.mlp.c_fc.bias",
            f"h.{i}.mlp.c_proj.weight", f"h.{i}.mlp.c_proj.bias",
        ]
    names += ["ln_f.weight", "ln_f.bias"]
    return names


def expected_shape(name: str, cfg) -> tuple[int, ...]:
    """The shape each tensor must have. A shape check here is the cheapest
    possible place to catch the Conv1D transpose gotcha."""
    d, f, v, c = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.n_ctx
    if name == "wte":
        return (v, d)
    if name == "wpe":
        return (c, d)
    leaf = name.split(".", 2)[-1] if name.startswith("h.") else name
    table = {
        "ln_1.weight": (d,), "ln_1.bias": (d,),
        "ln_2.weight": (d,), "ln_2.bias": (d,),
        "ln_f.weight": (d,), "ln_f.bias": (d,),
        "attn.c_attn.weight": (d, 3 * d), "attn.c_attn.bias": (3 * d,),
        "attn.c_proj.weight": (d, d), "attn.c_proj.bias": (d,),
        "mlp.c_fc.weight": (d, f), "mlp.c_fc.bias": (f,),
        "mlp.c_proj.weight": (f, d), "mlp.c_proj.bias": (d,),
    }
    if leaf not in table:
        raise KeyError(f"no expected shape for {name!r}")
    return table[leaf]


def gather(weights_dir: Path) -> tuple[object, dict[str, np.ndarray]]:
    cfg = config_from_json(weights_dir / "config.json")
    raw = read_safetensors(weights_dir / "model.safetensors")

    tensors: dict[str, np.ndarray] = {}
    for key, tensor in raw.items():
        if key.endswith(IGNORED_SUFFIXES):
            continue
        name = {"wte.weight": "wte", "wpe.weight": "wpe"}.get(key, key)
        tensors[name] = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)

    wanted = expected_names(cfg)
    missing = [n for n in wanted if n not in tensors]
    extra = [n for n in tensors if n not in wanted]
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensor(s): {missing[:5]}")
    if extra:
        raise SystemExit(f"checkpoint holds {len(extra)} unexpected tensor(s): {extra[:5]}")

    for name in wanted:
        want = expected_shape(name, cfg)
        got = tensors[name].shape
        if got != want:
            raise SystemExit(f"{name}: shape {got}, expected {want}")

    return cfg, {n: tensors[n] for n in wanted}


def write_bin(out: Path, cfg, tensors: dict[str, np.ndarray], source_sha: bytes,
              quant: int = QUANT_NONE) -> None:
    names = list(tensors)
    data_start = align_up(HEADER_FIXED + len(names) * ENTRY_SIZE, DATA_ALIGN)

    header = bytearray()
    header += MAGIC
    header += struct.pack("<IIII", VERSION, len(names), data_start, quant)
    # eps is f64, not f32: the engine stamps it into its dump manifest and
    # oracle/compare.py compares config values exactly. Narrowing 1e-5 to f32
    # and widening it back yields 1.0000000116860974e-05, which reads as a
    # provenance failure rather than a rounding detail. Kernels still use f32.
    header += struct.pack(
        "<IIIIIId",
        cfg.n_layer, cfg.n_head, cfg.d_model,
        cfg.d_ff, cfg.n_ctx, cfg.vocab_size,
        cfg.layer_norm_eps,
    )
    header += source_sha
    assert len(header) == HEADER_FIXED, len(header)

    offset = 0
    for name in names:
        a = tensors[name]
        if len(a.shape) > MAX_DIMS:
            raise SystemExit(f"{name}: {len(a.shape)} dims exceeds {MAX_DIMS}")
        raw_name = name.encode("ascii")
        if len(raw_name) >= NAME_FIELD:
            raise SystemExit(f"{name}: name too long for a {NAME_FIELD}-byte field")

        entry = bytearray()
        entry += raw_name.ljust(NAME_FIELD, b"\0")
        dtype = DTYPE_I8 if a.dtype == np.int8 else DTYPE_F32
        entry += struct.pack("<II", a.ndim, dtype)
        dims = list(a.shape) + [0] * (MAX_DIMS - a.ndim)
        entry += struct.pack("<4Q", *dims)
        entry += struct.pack("<QQQ", offset, a.nbytes, 0)
        assert len(entry) == ENTRY_SIZE, len(entry)
        header += entry
        offset += a.nbytes

    header += b"\0" * (data_start - len(header))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(header)
        for name in names:
            f.write(tensors[name].tobytes(order="C"))
    tmp.replace(out)


def read_bin(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read the format back. Used by --verify and by the engine's unit test
    fixture; keeping a Python reader honest keeps the C++ reader honest."""
    blob = path.read_bytes()
    if blob[:8] != MAGIC:
        raise SystemExit(f"{path}: bad magic {blob[:8]!r}")
    version, n_tensors, data_start, quant = struct.unpack_from("<IIII", blob, 8)
    if version != VERSION:
        raise SystemExit(f"{path}: version {version}, expected {VERSION}")
    n_layer, n_head, d_model, d_ff, n_ctx, vocab_size, eps = struct.unpack_from(
        "<IIIIIId", blob, 24)
    source_sha = blob[56:88]

    meta = {
        "version": version, "n_tensors": n_tensors, "data_start": data_start,
        "config": {"n_layer": n_layer, "n_head": n_head, "d_model": d_model,
                   "d_ff": d_ff, "n_ctx": n_ctx, "vocab_size": vocab_size,
                   "layer_norm_eps": eps},
        "quant": quant,
        "source_sha256": source_sha.hex(),
    }

    tensors: dict[str, np.ndarray] = {}
    for i in range(n_tensors):
        base = HEADER_FIXED + i * ENTRY_SIZE
        name = blob[base:base + NAME_FIELD].split(b"\0", 1)[0].decode("ascii")
        ndim, dtype = struct.unpack_from("<II", blob, base + NAME_FIELD)
        dims = struct.unpack_from("<4Q", blob, base + NAME_FIELD + 8)[:ndim]
        offset, nbytes, _ = struct.unpack_from("<QQQ", blob, base + NAME_FIELD + 40)
        if dtype not in DTYPE_SIZE:
            raise SystemExit(f"{name}: dtype {dtype}")
        start = data_start + offset
        np_dtype = np.int8 if dtype == DTYPE_I8 else np.float32
        a = np.frombuffer(blob, dtype=np_dtype,
                          count=nbytes // DTYPE_SIZE[dtype], offset=start)
        tensors[name] = a.reshape(dims)
    return meta, tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <weights>/../gpt2-124m.bin")
    parser.add_argument("--verify", action="store_true",
                        help="read the result back and compare against the source")
    args = parser.parse_args()

    out = args.out or args.weights.parent / "gpt2-124m.bin"

    print(f"reading {args.weights}")
    cfg, tensors = gather(args.weights)
    total = sum(a.size for a in tensors.values())
    print(f"  {len(tensors)} tensors, {total:,} parameters ({total / 1e6:.1f}M)")

    source_sha = sha256_file(args.weights / "model.safetensors")
    print(f"  source sha256 {source_sha.hex()[:16]}...")

    write_bin(out, cfg, tensors, source_sha)
    size = out.stat().st_size
    print(f"wrote {out}  ({size / 1e6:.1f} MB)")

    if args.verify:
        meta, back = read_bin(out)
        if meta["source_sha256"] != source_sha.hex():
            raise SystemExit("verify: source checksum did not round-trip")
        if len(back) != len(tensors):
            raise SystemExit("verify: tensor count changed")
        for name, want in tensors.items():
            got = back[name]
            if got.shape != want.shape:
                raise SystemExit(f"verify: {name} shape {got.shape} != {want.shape}")
            if not np.array_equal(got, want):
                raise SystemExit(f"verify: {name} bytes differ")
        print(f"verify: {len(back)} tensors round-tripped bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
