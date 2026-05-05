import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from cogformer.networks.transformers.encoder import Encoder
from cogformer.networks.transformers.decoder import Decoder
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
            layer_norm: bool = True,
            dropout: float = 0.0,
            layer_dropout: float = 0.0,
            encoder_layer_design: str = None,
            decoder_layer_design: str = None,
            encoder_layer_kwargs: dict = None,
            decoder_layer_kwargs: dict = None,
            time_embedding_dim: int = 32,
            pos_embedding_dim: int = 32,
            num_models: int | None = None,
            model_embed_dim: int = 8,
            time_freq_trainable: bool = True,
            block_film: bool = False,
            use_rope: bool = True,
    ):
        super().__init__()

        self.model_embedding = (
            nn.Embedding(num_models, model_embed_dim) if num_models is not None else None
        )
        self._model_embed_dim = model_embed_dim if num_models is not None else 0

        self.encoder = Encoder(
            input_dim=encoder_input_dim,
            num_layers=encoder_num_layers,
            num_seeds=num_seeds,
            seed_dim=seed_dim,
            num_heads=encoder_num_heads,
            layer_norm=layer_norm,
            dropout=dropout,
            layer_dropout=layer_dropout,
            layer_design=encoder_layer_design,
            layer_kwargs=encoder_layer_kwargs,
            use_rope=use_rope,
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
            model_embedding_dim=self._model_embed_dim,
            time_freq_trainable=time_freq_trainable,
            block_film=block_film,
        )

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

    def _velocity(self, theta, t, enc_k, pos_q, mask_q, emb_q, amp_dtype, use_amp):
        with torch.autocast(device_type=theta.device.type, dtype=amp_dtype, enabled=use_amp):
            out = self.decoder.velocity(
                theta_t=theta, query=pos_q, key=enc_k, t=t,
                query_mask=mask_q, model_embedding=emb_q,
            )
        return out.float()

    @torch.no_grad()
    def sample(self, input_data, param_indices, regressor_indices, params_mask,
               steps=1000, num_samples=100, model_ids=None, solver="euler",
               samples_chunk_size=None, encoder_tokens=None):
        """
        solver : "euler" | "midpoint" | "rk4"
            NFE per step: euler=1, midpoint=2, rk4=4.
        samples_chunk_size : int or None
            Process num_samples in chunks of this size. Peak GPU memory scales with
            batch_size * samples_chunk_size instead of batch_size * num_samples.
            The encoder runs once; only the ODE loop is repeated per chunk.
        encoder_tokens : Tensor or None
            Pre-computed encoder output (B, num_seeds, seed_dim). If provided the
            encoder forward pass is skipped — useful when sampling multiple chunks
            from one large batch that was encoded once externally.
        """
        batch_size = input_data.shape[0]
        num_tokens = param_indices.shape[1]
        device = input_data.device

        use_amp = device.type == "cuda"
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp else torch.float32

        # Encode once — shared across all sample chunks
        if encoder_tokens is None:
            encoder_tokens = self.encoder(input_data)                           # (B, num_seeds, seed_dim)
        pos_embeddings = torch.cat([param_indices, regressor_indices], dim=-1)  # (B, num_tokens, 2)

        model_emb = None
        if self.model_embedding is not None and model_ids is not None:
            model_emb = self.model_embedding(model_ids)
            model_emb = model_emb.unsqueeze(1).expand(-1, num_tokens, -1)       # (B, num_tokens, embed_dim)

        s_chunk = samples_chunk_size if samples_chunk_size is not None else num_samples
        dt = 1.0 / steps
        all_chunks = []

        for s_start in range(0, num_samples, s_chunk):
            s_count = min(s_chunk, num_samples - s_start)

            enc_k  = encoder_tokens.repeat_interleave(s_count, dim=0)           # (B*s, num_seeds, seed_dim)
            pos_q  = pos_embeddings.repeat_interleave(s_count, dim=0)           # (B*s, num_tokens, 2)
            mask_q = params_mask.repeat_interleave(s_count, dim=0) if params_mask is not None else None
            emb_q  = model_emb.repeat_interleave(s_count, dim=0) if model_emb is not None else None

            theta_t = torch.randn((batch_size * s_count, num_tokens, 1), device=device)
            t       = torch.zeros((batch_size * s_count, num_tokens, 1), device=device)

            v_fn = functools.partial(self._velocity, enc_k=enc_k, pos_q=pos_q, mask_q=mask_q,
                                     emb_q=emb_q, amp_dtype=amp_dtype, use_amp=use_amp)

            for _ in tqdm(range(steps), desc="Sampling", unit="step", leave=(s_start == 0)):
                theta_t = ode_step(v_fn, theta_t, t, dt, solver)
                t = t + dt

            all_chunks.append(theta_t.reshape(batch_size, s_count, num_tokens, 1).cpu())
            del enc_k, pos_q, mask_q, emb_q, theta_t, t

        return torch.cat(all_chunks, dim=1).numpy()  # (B, num_samples, num_tokens, 1)

    def compute_loss(self, pred_velocity, target_velocity, param_masks) -> torch.Tensor:

        sq_err = F.mse_loss(pred_velocity, target_velocity, reduction="none").squeeze(-1)
        sq_err_masked = sq_err * param_masks
        num_active_params = param_masks.sum(dim=-1)
        return torch.mean(sq_err_masked.sum(dim=-1) / num_active_params)
