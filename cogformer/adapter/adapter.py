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
    def scatter_to_design_rows(
        pred: np.ndarray,
        row_gather_indices: np.ndarray,
        num_active_rows: np.ndarray,
        num_design_rows: int,
        n_params: int,
    ) -> np.ndarray:
        """Scatter predictions on compacted tokens back to the canonical design-row layout.

        pred : (batch, num_samples, compact_rows * n_params, ...) — output of
            CogFormer.sample on a batch adapted with compact_tokens=True.
        Returns (batch, num_samples, num_design_rows * n_params, ...) with zeros at
        rows that were inactive (their coefficients are masked everywhere downstream).
        """
        pred = np.asarray(pred)
        gather = np.asarray(row_gather_indices)
        active = np.asarray(num_active_rows).astype(np.int64)
        batch_size, num_samples = pred.shape[0], pred.shape[1]
        compact_rows = pred.shape[2] // n_params
        tail = pred.shape[3:]

        src = pred.reshape(batch_size, num_samples, compact_rows, n_params, *tail)
        out = np.zeros((batch_size, num_samples, num_design_rows, n_params, *tail), dtype=pred.dtype)
        for i in range(batch_size):
            k = int(active[i])
            out[i, :, gather[i, :k]] = src[i, :, :k].swapaxes(0, 1)
        return out.reshape(batch_size, num_samples, num_design_rows * n_params, *tail)

    @staticmethod
    def adapt(
        samples: dict,
        device: str | torch.device = torch.device("cuda"),
        max_alternatives: int = 4,
        compact_tokens: bool = False,
    ) -> dict:
        """Convert a simulator batch into network-ready tensors.

        compact_tokens : if True, drop design rows whose coefficients are fully masked
            (padding from unused regressor slots, category levels, and interaction
            blocks) and keep only active rows, padded to the batch maximum (rounded up
            to a multiple of 8). Decoder cost scales ~quadratically with the token
            count, so this is a large training speedup; the factorized identity codes
            plus the canonical row index in the fifth channel keep dropped-row layouts
            unambiguous. Use scatter_to_design_rows to map sampled predictions back to
            the canonical layout.
        """
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

        n_rows = samples["param_masks"].shape[-1] // n_params
        row_codes = Adapter._resolve_position_codes(samples, n_rows)  # (B, n_rows, 4)
        masks_np = np.asarray(samples["param_masks"], dtype=np.float32).reshape(batch_size, n_rows, n_params)
        matrices_np = np.asarray(samples["param_matrices"], dtype=np.float32).reshape(batch_size, n_rows, n_params)
        # Canonical design-row index per row; stays the design-column pointer even
        # after compaction (the fifth identity channel must keep pointing at the
        # encoder-side column the coefficient belongs to).
        row_index = np.broadcast_to(np.arange(n_rows, dtype=np.int64)[None, :], (batch_size, n_rows))

        extra: dict = {}
        if compact_tokens:
            row_active = masks_np.any(axis=-1)                       # (B, n_rows)
            num_active = row_active.sum(axis=1).astype(np.int64)
            compact_rows = max(int(num_active.max()), 1)
            compact_rows = min(-(-compact_rows // 8) * 8, n_rows)    # multiple of 8 caps recompile variety
            gather = np.zeros((batch_size, compact_rows), dtype=np.int64)
            for i in range(batch_size):
                idx = np.flatnonzero(row_active[i])[:compact_rows]
                gather[i, :idx.shape[0]] = idx
            pad = np.arange(compact_rows)[None, :] >= num_active[:, None]

            row_codes = np.take_along_axis(row_codes, gather[..., None], axis=1)
            masks_np = np.take_along_axis(masks_np, gather[..., None], axis=1)
            matrices_np = np.take_along_axis(matrices_np, gather[..., None], axis=1)
            row_index = gather.copy()
            row_codes[pad] = 0
            masks_np[pad] = 0.0
            matrices_np[pad] = 0.0
            row_index[pad] = 0

            extra = {
                "row_gather_indices": Adapter.to_torch(gather),
                "num_active_rows": Adapter.to_torch(num_active),
                "num_design_rows": n_rows,
            }
            n_rows = compact_rows

        # Token order matches the flattened param outputs: design rows outermost,
        # intrinsic params innermost. Each token carries its factorized identity
        # (reg_a, reg_b, slot_a, slot_b) plus the canonical design-row index as a
        # fifth channel — an unnormalized positional signal for token-to-column
        # binding that, unlike the factors, is allowed to vary with layout.
        param_idx = np.tile(np.arange(n_params, dtype=np.int64), n_rows)[..., None]
        parameter_indices = Adapter.to_torch(
            np.tile(param_idx, (batch_size, 1, 1)), copy=True
        )

        regressor_indices = Adapter.to_torch(
            np.concatenate(
                [
                    np.repeat(row_codes, n_params, axis=1),
                    np.repeat(row_index, n_params, axis=1)[..., None],
                ],
                axis=-1,
            ),
            copy=True,
        )

        param_masks = torch.from_numpy(
            np.ascontiguousarray(masks_np.reshape(batch_size, n_rows * n_params))
        )
        param_matrices = torch.from_numpy(
            np.ascontiguousarray(matrices_np.reshape(batch_size, n_rows * n_params))
        )

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
            "n_params": n_params,
            **extra,
        }

        return Adapter.to_device(out, device)
