from torch import nn
from .multihead_attention_block import MultiheadAttentionBlock
from .self_attention_block import SelfAttentionBlock

# gated_residual gates SELF-attention and FFN branches only. Cross-attention (MAB)
# branches stay ungated: in the decoder they are the only path from the encoder to
# the output, and a zero gate there gives the whole encoder exactly zero gradient on
# the first step (measured). BayesFlow applies the same rule inside its PMA.


def cross_attention_layers(
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
    expansion_factor: float = 4.0,
    gated_residual: bool = False,
):
    layers = []
    for i in range(num_layers):
        layers.append(MultiheadAttentionBlock(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
            expansion_factor=expansion_factor,
            gate_attention=False,
            gate_ffn=gated_residual,
        ))
    return nn.ModuleList(layers)


def self_attention_layers(
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
    query_dim: int = 128,
    skip_first: bool = False,
    skip_last: bool = False,
    expansion_factor: float = 4.0,
    gated_residual: bool = False,
):
    layers = []

    if skip_first:
        l = MultiheadAttentionBlock(
            query_dim=query_dim, key_dim=key_dim, num_heads=num_heads,
            dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
            gate_attention=False, gate_ffn=gated_residual,
        )
    else:
        l = SelfAttentionBlock(
            input_dim=query_dim, num_heads=num_heads,
            dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
            gated_residual=gated_residual,
        )
    layers.append(l)

    for _ in range(1, num_layers - 1):
        layers.append(SelfAttentionBlock(
            input_dim=query_dim, num_heads=num_heads,
            dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
            gated_residual=gated_residual,
        ))

    if skip_last:
        l = MultiheadAttentionBlock(
            query_dim=query_dim, key_dim=key_dim, num_heads=num_heads,
            dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
            gate_attention=False, gate_ffn=gated_residual,
        )
    else:
        l = SelfAttentionBlock(
            input_dim=query_dim, num_heads=num_heads,
            dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
            gated_residual=gated_residual,
        )
    layers.append(l)

    return nn.ModuleList(layers)


def mixed_attention_layers(
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
    mab_first: bool = True,
    expansion_factor: float = 4.0,
    gated_residual: bool = False,
):
    layers = []
    for i in range(num_layers):
        use_mab = (i % 2 == 0) if mab_first else (i % 2 == 1)
        if use_mab:
            layers.append(MultiheadAttentionBlock(
                query_dim=query_dim, key_dim=key_dim, num_heads=num_heads,
                dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
                gate_attention=False, gate_ffn=gated_residual,
            ))
        else:
            layers.append(SelfAttentionBlock(
                input_dim=query_dim, num_heads=num_heads,
                dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
                gated_residual=gated_residual,
            ))
    return nn.ModuleList(layers)


def custom_attention_layers(
    pattern: tuple = (1, 0, 0),
    query_dim: int = 128,
    key_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    layer_norm: bool = True,
    dropout: float = 0.1,
    expansion_factor: float = 4.0,
    gated_residual: bool = False,
):
    """Patterned attention layers: 1 = MAB (cross-attention), 0 = SAB (self-attention)."""
    design = [pattern[i % len(pattern)] for i in range(num_layers)]
    layers = []
    for i in range(num_layers):
        if design[i] == 1:
            layers.append(MultiheadAttentionBlock(
                query_dim=query_dim, key_dim=key_dim, num_heads=num_heads,
                dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
                gate_attention=False, gate_ffn=gated_residual,
            ))
        else:
            layers.append(SelfAttentionBlock(
                input_dim=query_dim, num_heads=num_heads,
                dropout=dropout, layer_norm=layer_norm, expansion_factor=expansion_factor,
                gated_residual=gated_residual,
            ))
    return nn.ModuleList(layers)
