import torch
import numpy as np
from collections.abc import Mapping, Sequence

from cogformer.simulator.design_utils import build_position_codes


class Adapter:

    @staticmethod
    def _resolve_position_codes(samples: dict, n_tokens: int) -> np.ndarray:
        """Return per-design-row factorized codes, shape (batch_size, n_tokens, 4) int64.

        Prefers the simulator-emitted "position_codes"; falls back to rebuilding them
        from per-element design configs (e.g. batches saved to disk before the key
        existed). Padding rows keep the all-zero "none" code.
        """
        codes = samples.get("position_codes")
        if codes is not None:
            return np.asarray(codes, dtype=np.int64)

        design_configs = samples.get("design_configs")
        max_num_categories = samples.get("max_num_categories")
        if design_configs is None or max_num_categories is None:
            raise KeyError(
                "Adapter needs 'position_codes' (or 'design_configs' + 'max_num_categories' "
                "to rebuild them) in the samples dict."
            )
        batch_size = len(design_configs)
        codes = np.zeros((batch_size, n_tokens, 4), dtype=np.int64)
        for i, dc in enumerate(design_configs):
            element_codes = build_position_codes(dc, max_num_categories=max_num_categories)
            codes[i, :element_codes.shape[0]] = element_codes
        return codes

    @staticmethod
    def to_torch(x: np.ndarray, copy: bool = False) -> torch.Tensor:
        return torch.tensor(x) if copy else torch.from_numpy(x)

    @staticmethod
    def to_device(batch, device):
        """Recursively move tensors in mappings/sequences to the given device."""
        if isinstance(batch, Mapping):
            return {k: Adapter.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            return [Adapter.to_device(v, device) for v in batch]
        elif hasattr(batch, "to"):
            return batch.to(device)
        else:
            return batch

    @staticmethod
    def adapt(
        samples: dict,
        device: str | torch.device = torch.device("cuda"),
        max_alternatives: int = 4,
    ) -> dict:
        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]

        batch_size, num_obs = design_matrices.shape[0], design_matrices.shape[1]
        n_params = samples["n_params"]

        num_alternatives = np.asarray(
            samples.get("num_alternatives", np.full(batch_size, 2)), dtype=np.float32
        ).reshape(batch_size)
        denom = np.maximum(num_alternatives - 1.0, 1.0).reshape(batch_size, 1, 1)
        choices_normalized = choices / denom
        alt_col = np.repeat(
            ((num_alternatives - 1.0) / max(max_alternatives - 1, 1)).reshape(batch_size, 1, 1),
            num_obs, axis=1,
        ).astype(np.float32)

        _ptc = samples.get("per_trial_context", {})
        _zeros = np.zeros((batch_size, num_obs), dtype=np.float32)

        # favored_idx: which accumulator is boosted per trial (RDM); 0 for other models.
        # Normalized to [0, 1] using the same max_alternatives denominator as choices.
        fav_raw = np.asarray(_ptc.get("favored_idx", _zeros), dtype=np.float32)
        fav_col = (fav_raw / max(max_alternatives - 1, 1))[..., None]  # (B, T, 1)

        # theta: per-trial heading direction (CDM); 0 for other models.
        # Normalized to [-1, 1] via division by π.
        theta_raw = np.asarray(_ptc.get("theta", _zeros), dtype=np.float32)
        theta_col = (theta_raw / np.pi)[..., None]  # (B, T, 1)

        input_data = Adapter.to_torch(
            np.concatenate([design_matrices, rts, choices_normalized, alt_col, fav_col, theta_col], axis=-1)
        ).to(torch.float32)

        n_tokens = samples["param_masks"].shape[-1] // n_params
        row_codes = Adapter._resolve_position_codes(samples, n_tokens)  # (B, n_tokens, 4)

        # Token order matches the flattened param outputs: design rows outermost,
        # intrinsic params innermost. Each token carries its factorized identity
        # (reg_a, reg_b, slot_a, slot_b) plus the raw padded design-row index as a
        # fifth channel — an unnormalized positional signal for token-to-column
        # binding that, unlike the factors, is allowed to vary with layout.
        param_idx = np.tile(np.arange(n_params, dtype=np.int64), n_tokens)[..., None]
        parameter_indices = Adapter.to_torch(
            np.tile(param_idx, (batch_size, 1, 1)), copy=True
        )

        row_pos = np.repeat(np.arange(n_tokens, dtype=np.int64), n_params)[None, :, None]
        regressor_indices = Adapter.to_torch(
            np.concatenate(
                [
                    np.repeat(row_codes, n_params, axis=1),
                    np.broadcast_to(row_pos, (batch_size, n_tokens * n_params, 1)),
                ],
                axis=-1,
            ),
            copy=True,
        )

        param_masks = Adapter.to_torch(samples["param_masks"]).to(torch.float32)
        param_matrices = Adapter.to_torch(samples["param_matrices"]).to(torch.float32)

        raw_ids = samples.get("model_ids", None)
        model_ids = (
            Adapter.to_torch(np.asarray(raw_ids, dtype=np.int64))
            if raw_ids is not None else None
        )

        out = {
            "input_data": input_data,
            "param_indices": parameter_indices,
            "regressor_indices": regressor_indices,
            "param_masks": param_masks,
            "param_matrices": param_matrices,
            "model_ids": model_ids,
        }

        return Adapter.to_device(out, device)
