"""
Utilities for packaging real behavioral data into the CogFormer batch format.

Real data enters as raw arrays (rts, choices, condition columns). This module
converts them to the padded design-matrix / param-mask format expected by
Adapter.adapt, processing one subject at a time (batch_size=1).

Design matrix column layout (keep_intercept=True, max_num_categories=C,
max_num_regressors=R, block_width = C-1, n_total = R*(R+1)//2):

  col 0        : intercept (1.0)
  cols 1..bw   : u_1 block  (first col = regressor value, rest zero)
  cols bw+1..2bw: u_2 block
  ...
  cols n_main*bw+1..(n_main+1)*bw : u_1:u_2 block
  ... (interactions follow all main effects in the same order as build_design_config)

Continuous (real-valued) regressors always occupy the FIRST column of their block.
"""
import numpy as np
from cogformer.simulator.design_utils import build_parameter_mask


def _regressor_keys(design_config: dict) -> list[str]:
    """Return non-intercept keys in design_config order."""
    return [k for k in design_config if k != "1"]


def build_real_design_matrix(
    regressors: dict[str, np.ndarray],
    design_config: dict,
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool = True,
) -> np.ndarray:
    """
    Build a zero-padded design matrix from per-trial regressor arrays.

    Parameters
    ----------
    regressors : dict mapping design_config key → 1-D float array (num_obs,)
        e.g. {"u_1": stim_type, "u_2": truth, "u_1:u_2": interaction}
    design_config : dict
        Same format as CogFormer design configs.
    max_num_regressors, max_num_categories : ints
        Must match the trained CogFormer.
    """
    block_width = max_num_categories - 1
    n_total = max_num_regressors * (max_num_regressors + 1) // 2
    max_num_cols = n_total * block_width + (1 if keep_intercept else 0)

    num_obs = next(iter(regressors.values())).shape[0]
    dm = np.zeros((num_obs, max_num_cols), dtype=np.float32)

    col_offset = 0
    if keep_intercept:
        dm[:, 0] = 1.0
        col_offset = 1

    keys = _regressor_keys(design_config)
    for j, key in enumerate(keys):
        if key in regressors:
            col = col_offset + j * block_width
            dm[:, col] = np.asarray(regressors[key], dtype=np.float32)

    return dm


def build_real_param_mask(
    design_config: dict,
    intrinsic_params: list[str],
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool = True,
) -> np.ndarray:
    """
    Build a padded param_mask (max_num_cols, n_params) for real data.

    Returns shape (max_num_cols * n_params,) — flat, matching Adapter expectations.
    """
    block_width = max_num_categories - 1
    n_total = max_num_regressors * (max_num_regressors + 1) // 2
    max_num_cols = n_total * block_width + (1 if keep_intercept else 0)
    n_params = len(intrinsic_params)

    local_mask = build_parameter_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        fixed_params=None,
        max_num_categories=max_num_categories,
        keep_intercept=keep_intercept,
    )  # shape: (n_local_rows, n_params)

    padded = np.zeros((max_num_cols, n_params), dtype=np.float32)
    padded[:local_mask.shape[0], :] = local_mask
    return padded.flatten()


def build_subject_batch(
    rts: np.ndarray,
    choices: np.ndarray,
    design_matrix: np.ndarray,
    param_mask: np.ndarray,
    n_params: int,
    max_num_regressors: int,
    max_num_categories: int,
    num_alternatives: int = 2,
    model_id: int | None = None,
) -> dict:
    """
    Pack a single subject's data into the batch dict expected by Adapter.adapt.

    Parameters
    ----------
    rts, choices : (num_obs,) arrays — real observations
    design_matrix : (num_obs, max_num_cols)
    param_mask : flat (max_num_cols * n_params,) array
    """
    rts = np.asarray(rts, dtype=np.float32).reshape(-1, 1)
    choices = np.asarray(choices, dtype=np.float32).reshape(-1, 1)
    num_obs = rts.shape[0]

    batch = {
        "design_matrices":  design_matrix[np.newaxis],            # (1, num_obs, max_num_cols)
        "sim_data": {
            "rts":     rts[np.newaxis],                           # (1, num_obs, 1)
            "choices": choices[np.newaxis],                       # (1, num_obs, 1)
        },
        "param_matrices":   np.zeros((1, len(param_mask)), dtype=np.float32),
        "param_masks":      param_mask[np.newaxis],               # (1, max_num_cols * n_params)
        "n_params":         n_params,
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "num_alternatives": np.array([num_alternatives], dtype=np.int32),
    }
    if model_id is not None:
        batch["model_ids"] = np.array([model_id], dtype=np.int64)
    return batch


def load_ddm_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load DDM CSV data.

    Returns
    -------
    data : dict mapping subject_id → dict with rts, choices, stim_type, truth
    subject_ids : list of int
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Drop subjects with fewer than 100 trials (same filter as notebook)
    counts = df.groupby("id")["id"].transform("size")
    df = df[counts >= 100].copy()

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)
        stim_type   = sdf["stim_type"].values * 1.0 - 0.5   # effect coding
        truth       = sdf["factual_truth"].values * 1.0 - 0.5
        interaction = stim_type * truth
        data[int(sid)] = {
            "rts":         sdf["rt"].values.astype(np.float32),
            "choices":     sdf["resp"].values.astype(np.float32),
            "u_1":         stim_type.astype(np.float32),
            "u_2":         truth.astype(np.float32),
            "u_1:u_2":     interaction.astype(np.float32),
        }
    return data, sorted(data.keys())


def load_cdm_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load CDM CSV data (Kvam 2019).

    choices = deviation = response - targetOrientation (in radians, ∈ (-π, π))
    Regressors: isSpeed (u_1) and isCued (u_2), effect-coded as ±0.5.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    data = {}
    for sid in sorted(df["Participant"].unique()):
        sdf = df[df["Participant"] == sid].reset_index(drop=True)
        is_speed    = sdf["isSpeed"].values * 1.0 - 0.5   # effect coding
        is_cued     = sdf["isCued"].values * 1.0 - 0.5
        interaction = is_speed * is_cued
        deviation   = (sdf["response"].values - sdf["targetOrientation"].values).astype(np.float32)
        data[int(sid)] = {
            "rts":     sdf["RT"].values.astype(np.float32),
            "choices": deviation,
            "u_1":     is_speed.astype(np.float32),
            "u_2":     is_cued.astype(np.float32),
            "u_1:u_2": interaction.astype(np.float32),
        }
    return data, sorted(data.keys())
