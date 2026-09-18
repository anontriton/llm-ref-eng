"""Hand-written GPT-2 124M forward pass.

This is the source of truth for numerics. Every number the C++ engine produces
is ultimately checked against what this file computes, so it is written to be
read: explicit shapes, no fused shortcuts, no HF model code.

Two deliberate choices that differ from idiomatic PyTorch:

1. `Linear` stores its weight as [in, out] and computes `x @ W + b`, which is
   the layout the GPT-2 checkpoint actually holds (HF calls it Conv1D).
   `nn.Linear` stores [out, in] and computes `x @ W.T + b`. Adopting the
   checkpoint's layout means weight loading needs no transpose and the
   transpose gotcha cannot silently bite us.

2. `LayerNorm` is spelled out rather than calling `nn.LayerNorm`, because the
   C++ port has to reproduce exactly these operations in this order.

Module and parameter names mirror the checkpoint (`wte`, `wpe`, `h.{i}`,
`ln_f`) so a state dict maps across with no renaming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

# A tap receives (name, tensor) for every intermediate activation. Phase 1's
# dumper passes one in; by default it does nothing.
Tap = Callable[[str, torch.Tensor], None]


def _no_tap(name: str, value: torch.Tensor) -> None:
    pass


@dataclass(frozen=True)
class GPT2Config:
    """GPT-2 124M ("gpt2" small)."""

    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 3072
    n_ctx: int = 1024
    vocab_size: int = 50257
    layer_norm_eps: float = 1e-5

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """The tanh approximation of GELU, which is what GPT-2 was trained with.

    NOT the erf formulation. The two peak 4.7e-4 apart near |x| ~ 2.7 --
    mid-range, not the tails, where both converge -- which is ~5x our 1e-4
    absolute tolerance. Using the wrong one is a silent accuracy bug, not a
    crash.
    """
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """Normalize over the last dimension, then scale and shift.

    Uses the biased (population) variance, i.e. divide by N, not N-1.
    """

    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias


class Linear(nn.Module):
    """y = x @ W + b with W stored [in, out] -- the checkpoint's own layout."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a fused QKV projection."""

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.cfg = cfg
        # 768 -> 2304, split into three 768-wide chunks: query, key, value.
        self.c_attn = Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, prefix: str, tap: Tap) -> torch.Tensor:
        cfg = self.cfg
        B, T, C = x.shape

        qkv = self.c_attn(x)                       # [B, T, 3C]
        tap(f"{prefix}.qkv", qkv)
        q, k, v = qkv.split(cfg.d_model, dim=-1)   # three [B, T, C]

        # [B, T, C] -> [B, n_head, T, d_head]
        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, cfg.n_head, cfg.d_head).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)

        # Scale by 1/sqrt(d_head) = 1/sqrt(64), NOT 1/sqrt(d_model).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.d_head)
        # Tapped BEFORE masking, so the oracle holds finite numbers. The
        # comparison only trusts the causal (lower-triangular) region, which is
        # the part any implementation has to compute; how an implementation
        # spells "masked" (-inf, -1e9, or simply not computing it) is its own
        # business and shows up in .probs, where it actually matters.
        tap(f"{prefix}.scores", scores)

        # Causal mask: position t may not attend to anything after t.
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        tap(f"{prefix}.probs", probs)

        out = probs @ v                            # [B, n_head, T, d_head]
        # Merge heads back: [B, n_head, T, d_head] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        tap(f"{prefix}.out", out)
        return out


class MLP(nn.Module):
    """Two-layer feed-forward net, 768 -> 3072 -> 768, with gelu_new between."""

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.c_fc = Linear(cfg.d_model, cfg.d_ff)
        self.c_proj = Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor, prefix: str, tap: Tap) -> torch.Tensor:
        h = self.c_fc(x)
        tap(f"{prefix}.fc.out", h)
        h = gelu_new(h)
        tap(f"{prefix}.act.out", h)
        out = self.c_proj(h)
        tap(f"{prefix}.out", out)
        return out


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln_1(x)), then x + mlp(ln_2(x))."""

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.d_model, cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, prefix: str, tap: Tap) -> torch.Tensor:
        normed = self.ln_1(x)
        tap(f"{prefix}.ln_1.out", normed)
        x = x + self.attn(normed, f"{prefix}.attn", tap)

        normed = self.ln_2(x)
        tap(f"{prefix}.ln_2.out", normed)
        x = x + self.mlp(normed, f"{prefix}.mlp", tap)

        tap(f"{prefix}.out", x)
        return x


class GPT2(nn.Module):
    """GPT-2 language model. Output head is tied to the token embedding."""

    def __init__(self, cfg: GPT2Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GPT2Config()
        c = self.cfg
        self.wte = nn.Parameter(torch.zeros(c.vocab_size, c.d_model))
        self.wpe = nn.Parameter(torch.zeros(c.n_ctx, c.d_model))
        self.h = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = LayerNorm(c.d_model, c.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        tap: Optional[Tap] = None,
    ) -> torch.Tensor:
        """input_ids: [B, T] int64. Returns logits [B, T, vocab_size]."""
        tap = tap or _no_tap
        B, T = input_ids.shape
        if T > self.cfg.n_ctx:
            raise ValueError(f"sequence length {T} exceeds context {self.cfg.n_ctx}")

        # Learned absolute positions, added to the token embedding.
        positions = torch.arange(T, device=input_ids.device)
        x = self.wte[input_ids] + self.wpe[positions]
        tap("embed.out", x)

        for i, block in enumerate(self.h):
            x = block(x, f"block.{i}", tap)

        x = self.ln_f(x)
        tap("ln_f.out", x)

        # lm_head is tied to wte: there is no separate output matrix.
        logits = x @ self.wte.T
        tap("logits", logits)
        return logits
