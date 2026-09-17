"""Load the GPT-2 124M checkpoint into the hand-written reference model.

The checkpoint stores linear weights as [in, out] and computes `x @ W + b`
(HF calls this Conv1D). `model.Linear` adopts that same layout deliberately, so
loading is a straight copy with no transposes -- which is the point: a transpose
bug here would be invisible in shapes for the square 768x768 projections and
would corrupt every number downstream.

Only two keys need renaming (`wte.weight` -> `wte`, `wpe.weight` -> `wpe`)
because the embeddings are bare Parameters rather than modules. One key is
deliberately dropped: `h.{i}.attn.bias` is HF's precomputed 1024x1024 causal
mask, a buffer rather than a learned weight -- we build the mask on the fly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model import GPT2, GPT2Config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "gpt2-124m"

# HF buffers that are not parameters of our model.
IGNORED_SUFFIXES = (".attn.bias", ".attn.masked_bias")


def config_from_json(path: Path) -> GPT2Config:
    """Build our config from the checkpoint's own config.json, and refuse
    anything that is not the architecture this reference implements."""
    raw = json.loads(path.read_text())

    if raw.get("activation_function") != "gelu_new":
        raise ValueError(
            f"expected activation gelu_new (tanh approximation), got "
            f"{raw.get('activation_function')!r}"
        )
    if raw.get("n_embd") % raw.get("n_head") != 0:
        raise ValueError("n_embd must be divisible by n_head")

    return GPT2Config(
        n_layer=raw["n_layer"],
        n_head=raw["n_head"],
        d_model=raw["n_embd"],
        d_ff=4 * raw["n_embd"],          # GPT-2 has no separate n_inner here
        n_ctx=raw["n_positions"],
        vocab_size=raw["vocab_size"],
        layer_norm_eps=raw["layer_norm_epsilon"],
    )


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Read a safetensors file into a dict of tensors.

    Hand-rolled rather than importing the safetensors package: the format is a
    u64 header length, a JSON header of {name: {dtype, shape, data_offsets}},
    then a contiguous data block. Phase 2's C++ loader will have to do this too,
    so it is worth having it written out once in a language we can read.
    """
    import struct

    blob = path.read_bytes()
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8 : 8 + header_len])
    header.pop("__metadata__", None)
    data_start = 8 + header_len

    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    out: dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        if meta["dtype"] not in dtypes:
            raise ValueError(f"{name}: unsupported dtype {meta['dtype']}")
        begin, end = meta["data_offsets"]
        buf = bytearray(blob[data_start + begin : data_start + end])
        tensor = torch.frombuffer(buf, dtype=dtypes[meta["dtype"]])
        out[name] = tensor.reshape(meta["shape"])
    return out


def load_gpt2(weights_dir: Path | str = DEFAULT_WEIGHTS) -> GPT2:
    """Build a GPT2 and fill it from the checkpoint. Strict about everything."""
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"{weights_dir} not found -- run scripts/download_weights.py first"
        )

    cfg = config_from_json(weights_dir / "config.json")
    model = GPT2(cfg)

    raw = read_safetensors(weights_dir / "model.safetensors")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key.endswith(IGNORED_SUFFIXES):
            continue
        if key == "wte.weight":
            state["wte"] = tensor
        elif key == "wpe.weight":
            state["wpe"] = tensor
        else:
            state[key] = tensor

    # strict=True catches a missing, extra, or misshapen tensor -- which is
    # exactly the class of bug that otherwise shows up as "the logits are
    # slightly wrong" fifty steps later.
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _summary(model: GPT2) -> str:
    total = sum(p.numel() for p in model.parameters())
    # wte is counted once but used twice (embedding + tied output head).
    return f"{total:,} parameters ({total / 1e6:.1f}M), tied lm_head"


if __name__ == "__main__":
    m = load_gpt2()
    print(_summary(m))
