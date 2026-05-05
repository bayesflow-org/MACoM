import math

import torch
import torch.nn as nn
import numpy as np


# ── RoPE helpers ─────────────────────────────────────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by splitting in half: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, num_heads, T, head_dim).

    cos/sin have shape (T, head_dim) and are broadcast over batch and heads.
    """
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Precomputes cos/sin tables for positions 0..max_seq_len and returns slices
    for the actual sequence length at forward time. If the sequence exceeds the
    cached length the table is extended automatically.

    Args:
        head_dim    : dimension of each attention head (must be even)
        max_seq_len : initial cache size (extended lazily if exceeded)
        base        : frequency base (default 10000, as in the original paper)

    Reference: Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary
    Position Embedding." https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
        self._max_cached = max_seq_len

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)           # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
        self.register_buffer("_cos", emb.cos(), persistent=False)
        self.register_buffer("_sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._max_cached:
            self._build_cache(seq_len)
            self._max_cached = seq_len
        return self._cos[:seq_len], self._sin[:seq_len]


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal embedding for a scalar input.

    Maps (..., 1) → (..., dim) using sin/cos at geometrically spaced frequencies.
    No learned parameters; negligible compute overhead.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalEmbedding dim must be even"
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=x.device, dtype=x.dtype)
            / max(half - 1, 1)
        )
        args = x * freqs  # (..., half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., dim)


class FourierEmbedding(nn.Module):
    """Fourier projection embedding with normally distributed frequencies.

    Based on Tancik et al. (2020), Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains.

    Maps (..., 1) → (..., embed_dim) via sin/cos projections with scaled random frequencies.
    Frequencies are trainable by default, allowing the model to learn which frequency
    range is most informative.

    Args:
        embed_dim   : output dimensionality (must be even; embed_dim // 2 frequencies)
        scale       : std of the initial frequency distribution; controls the range of
                      frequencies at initialisation. Default 1.0 is appropriate for
                      flow-matching time t ∈ [0, 1].
        trainable   : if True (default), frequencies are learned during training.
                      If False, they are fixed after initialisation.
    """
    def __init__(self, embed_dim: int, scale: float = 1.0, trainable: bool = True):
        super().__init__()
        assert embed_dim % 2 == 0, "FourierEmbedding embed_dim must be even"
        self.embed_dim = embed_dim
        self.scale = scale

        frequencies = torch.randn(embed_dim // 2) * scale
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        proj = x * self.frequencies * 2 * math.pi   # (..., embed_dim // 2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., embed_dim)





