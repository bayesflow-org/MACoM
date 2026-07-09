import torch
import torch.nn as nn

from .attention_layers import (
    self_attention_layers,
    mixed_attention_layers,
    custom_attention_layers,
    cross_attention_layers,
)
from .self_attention_block import SelfAttentionBlock
from cogformer.utils.tensor_utils import broadcast_right
from .embeddings import SinusoidalEmbedding, FourierEmbedding
from cogformer.networks.mlp.time_mlp import TimeMLP


class Decoder(nn.Module):

    def __init__(
            self,
            input_dim: int = 2,
            seed_dim: int = 128,
            proj_dim: int = 64,
            num_layers: int = 3,
            num_heads: int = 4,
            layer_norm: bool = False,
            dropout: float = 0.05,
            layer_design: str = None,
            layer_kwargs: dict = None,
            time_embedding_dim: int = 32,
            pos_embedding_dim: int = 32,
            model_embedding_dim: int = 0,
            num_param_indices: int = 16,
            num_regressor_indices: int = 8,
            num_category_indices: int = 8,
            time_freq_trainable: bool = True,
            block_film: bool = False,
            expansion_factor: float = 4.0,
            self_attn_padding_mask: bool = True,
            output_conditioning: str = "concat",
            block_conditioning: str = "concat",
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        # When False, self-attention does not receive a key_padding_mask. This lets
        # the fused FlashAttention kernel fire (a boolean mask forces the slower
        # mem-efficient path), at the cost of active tokens attending to padded
        # cells. Inactive token *outputs* are still zeroed after each layer.
        self.self_attn_padding_mask = self_attn_padding_mask

        if layer_kwargs is None:
            layer_kwargs = {}

        self.time_embedding = SinusoidalEmbedding(time_embedding_dim)

        # Factorized coefficient-identity embeddings. Each token identifies its
        # coefficient as (param, reg_a, reg_b, slot_a, slot_b) integer indices; 0 = "none".
        # The regressor table is shared across both positions (pairs are canonically
        # ordered a < b), while the two slot positions get separate tables because
        # interaction rows (i, j) and (j, i) are distinct coefficients.
        self.param_embedding = nn.Embedding(num_param_indices, pos_embedding_dim)
        self.regressor_embedding = nn.Embedding(num_regressor_indices, pos_embedding_dim)
        self.category_embedding = nn.Embedding(num_category_indices, pos_embedding_dim)
        self.category_embedding_b = nn.Embedding(num_category_indices, pos_embedding_dim)
        # Fused positional term over the raw (unnormalized) design-row index: gives
        # tokens an unambiguous pointer to their design column even when the same
        # factorized code sits at different padded positions across batch elements
        # (possible with interaction blocks under heterogeneous regressor counts).
        self.row_embedding = SinusoidalEmbedding(pos_embedding_dim)

        # input: theta_t (1) + param_embedding (pos_embedding_dim) + reg_embedding (pos_embedding_dim)
        #        + t_embedding (time_embedding_dim) [+ model_embedding (model_embed_dim)]
        self.input_proj = nn.Linear(1 + 2 * pos_embedding_dim + time_embedding_dim + model_embedding_dim, proj_dim)

        if layer_design is not None:
            match layer_design:
                case "self_attention":
                    self.layers = self_attention_layers(
                        query_dim=proj_dim,
                        key_dim=seed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        layer_norm=layer_norm,
                        dropout=dropout,
                        expansion_factor=expansion_factor,
                        **layer_kwargs
                    )
                case "mixed_attention":
                    self.layers = mixed_attention_layers(
                        query_dim=proj_dim,
                        key_dim=seed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        layer_norm=layer_norm,
                        dropout=dropout,
                        expansion_factor=expansion_factor,
                        **layer_kwargs
                    )
                case "cross_attention":
                    self.layers = cross_attention_layers(
                        query_dim=proj_dim,
                        key_dim=seed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        layer_norm=layer_norm,
                        dropout=dropout,
                        expansion_factor=expansion_factor,
                    )
                case "custom_attention":
                    self.layers = custom_attention_layers(
                        query_dim=proj_dim,
                        key_dim=seed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        layer_norm=layer_norm,
                        dropout=dropout,
                        expansion_factor=expansion_factor,
                        **layer_kwargs
                    )
        else:
            self.layers = mixed_attention_layers(
                query_dim=proj_dim,
                key_dim=seed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                layer_norm=layer_norm,
                dropout=dropout,
                expansion_factor=expansion_factor,
                **layer_kwargs
            )

        self.output_proj = TimeMLP(
            input_dim=proj_dim,
            time_embedding_dim=time_embedding_dim,
            conditioning=output_conditioning,
        )
        self.num_heads = num_heads

        # Per-block time conditioning applied after each attention layer.
        # block_film toggles it on; block_conditioning selects the mechanism:
        #   "film"   : FiLM affine modulation, out = (1 + gamma) * out + beta
        #   "concat" : residual small MLP over [out ; time_embedding]
        self.block_film = block_film
        self.block_conditioning = block_conditioning
        if block_film:
            assert block_conditioning in ("film", "concat"), \
                f"block_conditioning must be 'film' or 'concat', got {block_conditioning!r}"
            if block_conditioning == "film":
                self.film_layers = nn.ModuleList([
                    nn.Linear(time_embedding_dim, 2 * proj_dim)
                    for _ in range(num_layers)
                ])
            else:
                self.block_mlp_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(proj_dim + time_embedding_dim, proj_dim),
                        nn.Mish(),
                        nn.Linear(proj_dim, proj_dim),
                    )
                    for _ in range(num_layers)
                ])

    def embed_positions(self, query):
        """Embed factorized coefficient identities.

        query : (B, T, 6) int64 — (param, reg_a, reg_b, slot_a, slot_b, row_pos) per token.
        Returns (param_embedding, regressor_embedding), each (B, T, pos_embedding_dim).
        """
        param_embedding = self.param_embedding(query[..., 0])
        regressor_embedding = (
            self.regressor_embedding(query[..., 1])
            + self.regressor_embedding(query[..., 2])
            + self.category_embedding(query[..., 3])
            + self.category_embedding_b(query[..., 4])
            + self.row_embedding(query[..., 5:6].float())
        )
        return param_embedding, regressor_embedding

    def velocity(self, theta_t, query, key, t, query_mask=None, model_embedding=None,
                 param_embedding=None, regressor_embedding=None):
        """
        param_embedding, regressor_embedding : optional pre-computed positional embeddings.
            If provided (e.g. pre-computed once before an ODE loop), the embedding
            lookups are skipped.
        """
        # key_padding_mask for self-attention: True = inactive token, should be ignored.
        # Disabled when self_attn_padding_mask is False (FlashAttention A/B).
        key_padding_mask = (
            (query_mask <= 0)
            if (query_mask is not None and self.self_attn_padding_mask)
            else None
        )

        # Factorized identity embeddings (skip if pre-computed)
        if param_embedding is None or regressor_embedding is None:
            param_embedding, regressor_embedding = self.embed_positions(query)
        time_embedding = self.time_embedding(t)                      # (B, T, time_embedding_dim)

        components = [theta_t, param_embedding, regressor_embedding, time_embedding]
        if model_embedding is not None:
            components.append(model_embedding)                       # (B, T, model_embed_dim)
        out = torch.cat(components, dim=-1)
        out = self.input_proj(out)

        # Run through attention layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SelfAttentionBlock):
                out = layer(query=out, key_padding_mask=key_padding_mask)
            else:
                out = layer(query=out, key=key, attn_mask=None)

            if query_mask is not None:
                out = out * query_mask[..., None]

            if self.block_film:
                if self.block_conditioning == "film":
                    gamma, beta = self.film_layers[i](time_embedding).chunk(2, dim=-1)
                    out = (1 + gamma) * out + beta
                else:
                    out = out + self.block_mlp_layers[i](torch.cat([out, time_embedding], dim=-1))
                if query_mask is not None:
                    out = out * query_mask[..., None]

        return self.output_proj(out, time_embedding)

    def forward(self, theta, query, key, query_mask=None, model_embedding=None):
        """Returns predicted and target velocity (potentially masked)."""

        # Generate time
        t = torch.rand(theta.shape[0], device=theta.device)
        t = broadcast_right(t, theta.shape)

        # Generate z = theta0
        z = torch.randn_like(theta)

        # Interpolation
        theta_t = t * theta + (1 - t) * z

        # Predict velocity v at theta_t
        pred_velocity = self.velocity(theta_t, query, key, t, query_mask, model_embedding)
        target_velocity = theta - z

        if query_mask is not None:
            pred_velocity = pred_velocity * query_mask[..., None]
            target_velocity = target_velocity * query_mask[..., None]

        return pred_velocity, target_velocity
