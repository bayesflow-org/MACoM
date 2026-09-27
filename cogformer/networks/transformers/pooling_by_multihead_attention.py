import torch
import torch.nn as nn
from .multihead_attention_block import MultiheadAttentionBlock, AdaptiveRMSNorm
from ..mlp.ffn import FFN


class PoolingByMultiheadAttention(nn.Module):
    """PMA (Lee et al., 2019): learned seeds cross-attend to the set.

    pre_pool_ffn adds the row-wise feed-forward the Set Transformer paper defines
    PMA over, PMA(Z) = MAB(S, rFF(Z)), as BayesFlow >= 2.0.13 does: a pre-RMSNorm
    SwiGLU FFN on each set element with a zero-initialised residual scale, so the
    block starts out pooling Z itself. gated_residual gates the MAB's FFN branch the
    same way; its attention branch is left ungated, also as in BayesFlow, since the
    pooled output must depend on the set from the first step.
    """
    def __init__(
        self,
        input_dim: int,
        seed_dim: int,
        num_heads: int = 4,
        num_seeds: int = 1,
        layer_norm: bool = False,
        dropout: float = 0.0,
        expansion_factor: float = 4.0,
        pre_pool_ffn: bool = False,
        gated_residual: bool = False,
    ):
        super().__init__()
        self.seed = nn.Parameter(torch.Tensor(1, num_seeds, seed_dim))
        nn.init.xavier_uniform_(self.seed)

        if pre_pool_ffn:
            self.pre_norm = AdaptiveRMSNorm(input_dim) if layer_norm else None
            self.pre_ffn = FFN(
                embed_dim=input_dim,
                expansion_factor=expansion_factor,
                dropout=dropout,
            )
            self.pre_scale = nn.Parameter(torch.zeros(()))
        else:
            self.pre_ffn = None

        self.mab = MultiheadAttentionBlock(
            query_dim=seed_dim,
            key_dim=input_dim,
            num_heads=num_heads,
            layer_norm=layer_norm,
            dropout=dropout,
            expansion_factor=expansion_factor,
            gate_attention=False,
            gate_ffn=gated_residual,
        )

    def forward(self, x):
        if self.pre_ffn is not None:
            h = self.pre_norm(x) if self.pre_norm is not None else x
            x = x + self.pre_scale * self.pre_ffn(h)
        return self.mab(self.seed.repeat(x.size(0), 1, 1), x)
