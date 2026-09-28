import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from cogformer.networks.transformers.encoder import Encoder
from cogformer.networks.transformers.decoder import Decoder
from cogformer.networks.transformers.multihead_attention_block import MultiheadAttentionBlock
from cogformer.networks.transformers.pooling_by_multihead_attention import PoolingByMultiheadAttention
from cogformer.networks.mlp.ffn import FFN
from cogformer.utils.ode_solver import ode_step


class CogFormer(nn.Module):

    def __init__(
            self,
            encoder_input_dim: int = 18,
            decoder_input_dim: int = 2,
            proj_dim: int = 64,
            encoder_num_layers: int = 3,
            decoder_num_layers: int = 3,
            encoder_num_heads: int = 4,
            decoder_num_heads: int = 4,
            num_seeds: int = 10,
            seed_dim: int = 128,
            encoder_embed_dim: int = 64,
            ffn_expansion_factor: float = 4.0,
            layer_norm: bool = True,
            dropout: float = 0.0,
            layer_dropout: float = 0.0,
            encoder_layer_design: str = None,
            decoder_layer_design: str = None,
            encoder_layer_kwargs: dict = None,
            decoder_layer_kwargs: dict = None,
            time_embedding_dim: int = 32,
            pos_embedding_dim: int = 32,
            num_param_indices: int = 16,
            num_regressor_indices: int = 8,
            num_category_indices: int = 8,
            num_models: int | None = None,
            model_embed_dim: int = 8,
            time_freq_trainable: bool = True,
            block_film: bool = True,
            self_attn_padding_mask: bool = True,
            output_conditioning: str = "linear",
            block_conditioning: str = "film",
            gated_residual: bool = True,
            pre_pool_ffn: bool = True,
            weight_init: str = "orthogonal",
    ):
        super().__init__()
        # gated_residual / pre_pool_ffn / weight_init="orthogonal" port BayesFlow's
        # 2.0.13 attention defaults (zero-init residual scales, rFF before PMA pooling,
        # orthogonal kernels). All default ON since 2026-09-28: the A/B at the final
        # recipe (r4c4, b128, 500 ep, 18 validation cases) found recovery r 0.64 -> 0.75,
        # NRMSE 0.179 -> 0.153 and contraction 0.51 -> 0.61 at unchanged calibration.
        # Checkpoints trained before that need all three passed as False/"default".
        # The first two add parameters, so loading a checkpoint with the wrong setting
        # fails in load_state_dict rather than silently; weight_init only affects a fresh
        # model and is invisible at load.
        assert weight_init in ("default", "orthogonal"), \
            f"weight_init must be 'default' or 'orthogonal', got {weight_init!r}"

        self.model_embedding = (
            nn.Embedding(num_models, model_embed_dim) if num_models is not None else None
        )
        self._model_embed_dim = model_embed_dim if num_models is not None else 0

        self.encoder = Encoder(
            input_dim=encoder_input_dim,
            embed_dim=encoder_embed_dim,
            num_layers=encoder_num_layers,
            num_seeds=num_seeds,
            seed_dim=seed_dim,
            num_heads=encoder_num_heads,
            layer_norm=layer_norm,
            dropout=dropout,
            layer_dropout=layer_dropout,
            layer_design=encoder_layer_design,
            layer_kwargs=encoder_layer_kwargs,
            expansion_factor=ffn_expansion_factor,
            gated_residual=gated_residual,
            pre_pool_ffn=pre_pool_ffn,
        )

        self.decoder = Decoder(
            input_dim=decoder_input_dim,
            seed_dim=seed_dim,
            proj_dim=proj_dim,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
            layer_norm=layer_norm,
            dropout=dropout,
            layer_design=decoder_layer_design,
            layer_kwargs=decoder_layer_kwargs,
            time_embedding_dim=time_embedding_dim,
            pos_embedding_dim=pos_embedding_dim,
            num_param_indices=num_param_indices,
            num_regressor_indices=num_regressor_indices,
            num_category_indices=num_category_indices,
            model_embedding_dim=self._model_embed_dim,
            time_freq_trainable=time_freq_trainable,
            block_film=block_film,
            expansion_factor=ffn_expansion_factor,
            self_attn_padding_mask=self_attn_padding_mask,
            output_conditioning=output_conditioning,
            block_conditioning=block_conditioning,
            gated_residual=gated_residual,
        )

        if weight_init == "orthogonal":
            self._orthogonal_init()

    def _orthogonal_init(self):
        """Orthogonal kernels in every attention and FFN projection, and the PMA seeds.

        Matches BayesFlow's kernel_initializer="orthogonal" scope: attention and FFN
        weights only. Embeddings, input/output projections and FiLM layers keep their
        defaults. nn.MultiheadAttention packs Q/K/V into one (3E, E) in_proj_weight
        when kdim == embed_dim; each E x E slice is initialised separately so Q, K and
        V are each orthogonal, as BayesFlow's three separate Dense layers are.
        """
        for m in self.modules():
            if isinstance(m, MultiheadAttentionBlock):
                a = m.attn
                if a.in_proj_weight is not None:
                    for w in a.in_proj_weight.data.chunk(3, dim=0):
                        nn.init.orthogonal_(w)
                else:
                    for w in (a.q_proj_weight, a.k_proj_weight, a.v_proj_weight):
                        nn.init.orthogonal_(w)
                nn.init.orthogonal_(a.out_proj.weight)
            elif isinstance(m, FFN):
                for lin in (m.gate_proj, m.up_proj, m.down_proj):
                    nn.init.orthogonal_(lin.weight)
            elif isinstance(m, PoolingByMultiheadAttention):
                nn.init.orthogonal_(m.seed.data[0])

    def forward(self, params, input_data, param_indices, regressor_indices, params_mask=None, model_ids=None):

        encoder_tokens = self.encoder(input_data)

        pos_embeddings = torch.cat([param_indices, regressor_indices], dim=-1)

        model_embedding = None
        if self.model_embedding is not None and model_ids is not None:
            model_embedding = self.model_embedding(model_ids)                                        # (B, model_embed_dim)
            model_embedding = model_embedding.unsqueeze(1).expand(-1, pos_embeddings.shape[1], -1)   # (B, T, model_embed_dim)

        pred_velocity, target_velocity = self.decoder(
            theta=params,
            query=pos_embeddings,
            key=encoder_tokens,
            query_mask=params_mask,
            model_embedding=model_embedding,
        )

        return pred_velocity, target_velocity

    def _velocity(self, theta, t, enc_k, mask_q, emb_q, param_emb, reg_emb, amp_dtype, use_amp):
        with torch.autocast(device_type=theta.device.type, dtype=amp_dtype, enabled=use_amp):
            out = self._compiled_vel(
                theta_t=theta, query=None, key=enc_k, t=t,
                query_mask=mask_q, model_embedding=emb_q,
                param_embedding=param_emb, regressor_embedding=reg_emb,
            )
        return out.float()

    @torch.no_grad()
    def sample(self, input_data, param_indices, regressor_indices, params_mask,
               steps=1000, num_samples=100, model_ids=None, solver="euler"):
        """
        solver : "euler" | "midpoint" | "rk4"
            NFE per step: euler=1, midpoint=2, rk4=4.
        """
        batch_size = input_data.shape[0]
        num_tokens = param_indices.shape[1]
        device = input_data.device

        use_amp = device.type == "cuda"
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp else torch.float32

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            encoder_tokens = self.encoder(input_data)                               # (B, num_seeds, seed_dim)
            pos_embeddings = torch.cat([param_indices, regressor_indices], dim=-1)  # (B, num_tokens, 6)

            model_emb = None
            if self.model_embedding is not None and model_ids is not None:
                model_emb = self.model_embedding(model_ids)
                model_emb = model_emb.unsqueeze(1).expand(-1, num_tokens, -1)       # (B, num_tokens, embed_dim)

            enc_k  = encoder_tokens.repeat_interleave(num_samples, dim=0)           # (B*S, num_seeds, seed_dim)
            pos_q  = pos_embeddings.repeat_interleave(num_samples, dim=0)           # (B*S, num_tokens, 6)
            mask_q = params_mask.repeat_interleave(num_samples, dim=0) if params_mask is not None else None
            emb_q  = model_emb.repeat_interleave(num_samples, dim=0) if model_emb is not None else None

            # Pre-compute identity embeddings — constant across all ODE steps
            param_emb, reg_emb = self.decoder.embed_positions(pos_q)  # each (B*S, num_tokens, pos_embedding_dim)

        # Lazily compile decoder.velocity (torch.compile(cogformer) only wraps forward())
        if not hasattr(self, '_compiled_vel'):
            self._compiled_vel = torch.compile(self.decoder.velocity, dynamic=True)

        theta_t = torch.randn((batch_size * num_samples, num_tokens, 1), device=device)
        t       = torch.zeros((batch_size * num_samples, num_tokens, 1), device=device)
        dt      = 1.0 / steps

        v_fn = functools.partial(self._velocity, enc_k=enc_k, mask_q=mask_q,
                                 emb_q=emb_q, param_emb=param_emb, reg_emb=reg_emb,
                                 amp_dtype=amp_dtype, use_amp=use_amp)

        for _ in tqdm(range(steps), desc="Sampling", unit="step", leave=True):
            theta_t = ode_step(v_fn, theta_t, t, dt, solver)
            t.add_(dt)

        return theta_t.reshape(batch_size, num_samples, num_tokens, 1).cpu().numpy()  # (B, S, T, 1)

    def compute_loss(self, pred_velocity, target_velocity, param_masks) -> torch.Tensor:

        sq_err = F.mse_loss(pred_velocity, target_velocity, reduction="none").squeeze(-1)
        sq_err_masked = sq_err * param_masks
        num_active_params = param_masks.sum(dim=-1)
        return torch.mean(sq_err_masked.sum(dim=-1) / num_active_params)
