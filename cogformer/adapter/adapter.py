import torch
import numpy as np
from collections.abc import Mapping, Sequence


class Adapter:

    @staticmethod
    def _build_position_indices(
        n_params: int,
        max_num_regressors: int,
        max_num_categories: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (parameter_indices, regressor_indices), each shape (num_tokens, 1)."""
        n_total = max_num_regressors * (max_num_regressors + 1) // 2
        n_tokens = n_total * (max_num_categories - 1) + 1
        param_idx = np.tile(np.linspace(0.0, 1.0, n_params), n_tokens)[..., None]
        reg_idx = np.repeat(np.linspace(0.0, 1.0, n_tokens), n_params)[..., None]
        return param_idx, reg_idx

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

        input_data = Adapter.to_torch(
            np.concatenate([design_matrices, rts, choices_normalized, alt_col], axis=-1)
        ).to(torch.float32)

        param_idx, reg_idx = Adapter._build_position_indices(
            n_params=n_params,
            max_num_regressors=samples["max_num_regressors"],
            max_num_categories=samples["max_num_categories"],
        )

        parameter_indices = Adapter.to_torch(
            np.tile(param_idx, (batch_size, 1, 1))
        ).to(torch.float32)

        regressor_indices = Adapter.to_torch(
            np.tile(reg_idx, (batch_size, 1, 1))
        ).to(torch.float32)

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
