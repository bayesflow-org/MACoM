import torch
import torch.nn as nn
import torch.nn.functional as F


_GLU_ACTIVATIONS = {
    "swiglu": F.silu,
    "geglu": F.gelu,
    "reglu": F.relu,
    "liglu": None,
}


class FFN(nn.Module):
    """Gated feedforward network (FFN) for transformer blocks.

    Implements a GLU-style feedforward layer: gate and value branches are
    projected separately, the gate is passed through an activation, and the
    result is multiplied element-wise before the output projection.

    Four variants are supported:
      - SwiGLU (default): gate = SiLU(x W_gate)  — LLaMA, PaLM-2
      - GEGLU:            gate = GELU(x W_gate)   — T5 v1.1, PaLM
      - ReGLU:            gate = ReLU(x W_gate)
      - LiGLU:            gate = x W_gate         (no activation)

    The intermediate dimension is set to ``2/3 * expansion_factor * embed_dim``
    (rounded up to the nearest multiple of 64) so that parameter count stays
    comparable to a plain FFN with the same ``expansion_factor``.

    Parameters
    ----------
    embed_dim : int
        Input and output dimensionality.
    expansion_factor : float
        Multiplier controlling intermediate width before the 2/3 GLU scaling.
    glu_variant : str
        One of "swiglu", "geglu", "reglu", "liglu".
    dropout : float
        Dropout applied after gating and before the output projection.

    References
    ----------
    Noam Shazeer (2020). "GLU Variants Improve Transformer."
    https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        embed_dim: int = 64,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        dropout: float = 0.0,
    ):
        super().__init__()

        glu_variant = glu_variant.lower()
        if glu_variant not in _GLU_ACTIVATIONS:
            raise ValueError(
                f"Unknown GLU variant '{glu_variant}'. "
                f"Choose from: {list(_GLU_ACTIVATIONS.keys())}."
            )

        self.glu_variant = glu_variant
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor

        # Scale so total params ≈ standard FFN; round up to nearest multiple of 64
        intermediate_dim = int(embed_dim * expansion_factor * 2 / 3)
        intermediate_dim = ((intermediate_dim + 63) // 64) * 64
        self.intermediate_dim = intermediate_dim

        self.gate_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.gate_fn = _GLU_ACTIVATIONS[glu_variant]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.gate_fn is not None:
            gate = self.gate_fn(gate)
        x = gate * self.up_proj(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.down_proj(x)
