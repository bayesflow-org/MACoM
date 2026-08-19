"""
Utilities for packaging real behavioral data into the CogFormer batch format.

Real data enters as raw arrays (rts, choices, condition columns). This module
converts them to the padded design-matrix / param-mask format expected by
Adapter.adapt, processing one subject at a time (batch_size=1).

Design matrix column layout (keep_intercept=True, max_num_categories=C,
max_num_regressors=R, block_width = C-1, inter_block_width = (C-1)²):

  col 0              : intercept (1.0)
  cols 1..bw         : u_1 block  (first col = regressor value, rest zero for continuous)
  cols bw+1..2bw     : u_2 block
  ...
  cols R*bw+1..R*bw+ibw : u_1:u_2 block  (ibw = inter_block_width)
  ... (interactions follow all main effects)

Continuous regressors occupy the FIRST column of their block.
Categorical regressors (dummy-coded) occupy columns 0..(n_dummies-1) of their block.
Interaction blocks hold the outer product of each pair of parent blocks.
"""
import numpy as np
from cogformer.simulator.design_utils import build_parameter_mask, build_position_codes


def _regressor_keys(design_config: dict) -> list[str]:
    """Return non-intercept keys in design_config order."""
    return [k for k in design_config if k != "1"]


def _compute_max_num_cols(
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool,
) -> int:
    block_width = max_num_categories - 1
    n_main = max_num_regressors
    n_inter = max_num_regressors * (max_num_regressors - 1) // 2
    return n_main * block_width + n_inter * block_width ** 2 + (1 if keep_intercept else 0)


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
    regressors : dict mapping design_config key → array of shape (num_obs,) or (num_obs, n_cols)
        Continuous regressors: 1-D array → placed in first column of block.
        Categorical / interaction regressors: 2-D array → placed in first n_cols columns of block.
        e.g. {"u_1": speed_cond, "u_2": difficulty_dummies, "u_1:u_2": interaction_dummies}
    design_config : dict
        Same format as CogFormer design configs.
    max_num_regressors, max_num_categories : ints
        Must match the trained CogFormer.
    """
    block_width = max_num_categories - 1
    inter_block_width = block_width ** 2
    max_num_cols = _compute_max_num_cols(max_num_regressors, max_num_categories, keep_intercept)

    num_obs = next(iter(regressors.values())).shape[0]
    dm = np.zeros((num_obs, max_num_cols), dtype=np.float32)

    col_offset = 0
    if keep_intercept:
        dm[:, 0] = 1.0
        col_offset = 1

    keys = _regressor_keys(design_config)
    cur = col_offset
    start_pos = {}
    for key in keys:
        start_pos[key] = cur
        cur += inter_block_width if ":" in key else block_width

    for key in keys:
        if key not in regressors:
            continue
        arr = np.asarray(regressors[key], dtype=np.float32)
        start = start_pos[key]
        width = inter_block_width if ":" in key else block_width
        if arr.ndim == 1:
            dm[:, start] = arr
        else:
            n_cols = min(arr.shape[1], width)
            dm[:, start:start + n_cols] = arr[:, :n_cols]

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
    max_num_cols = _compute_max_num_cols(max_num_regressors, max_num_categories, keep_intercept)
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
    design_config: dict,
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
    design_config : dict — used to build factorized position codes; must be the same
        config the design matrix and param mask were built from
    """
    rts = np.asarray(rts, dtype=np.float32).reshape(-1, 1)
    choices = np.asarray(choices, dtype=np.float32).reshape(-1, 1)
    num_obs = rts.shape[0]

    max_num_cols = len(param_mask) // n_params
    codes = build_position_codes(design_config, max_num_categories=max_num_categories)
    position_codes = np.zeros((max_num_cols, 4), dtype=np.int64)
    position_codes[:codes.shape[0]] = codes

    batch = {
        "position_codes":   position_codes[np.newaxis],           # (1, max_num_cols, 4)
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
    Load DDM CSV data (Stump et al., 2026).

    Regressors:
      u_1  : stim_type,  effect coded ±0.5  (continuous)
      u_2  : truth,      effect coded ±0.5  (continuous)
      u_1:u_2 : stim_type × truth           (continuous × continuous → 1 column)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    counts = df.groupby("id")["id"].transform("size")
    df = df[counts >= 100].copy()

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)
        stim_type   = (sdf["stim_type"].values * 1.0 - 0.5).astype(np.float32)
        truth       = (sdf["factual_truth"].values * 1.0 - 0.5).astype(np.float32)
        interaction = (stim_type * truth).astype(np.float32)
        data[int(sid)] = {
            "rts":         sdf["rt"].values.astype(np.float32),
            "choices":     sdf["resp"].values.astype(np.float32),
            "u_1":         stim_type,
            "u_2":         truth,
            "u_1:u_2":     interaction,
        }
    return data, sorted(data.keys())


def load_rdm_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load RDM CSV data (3-alternative Stroop, 149 subjects).

    Regressors:
      u_1     : congruent, effect coded ±0.5  (congruent=+0.5, incongruent=-0.5)
      u_2     : session,   effect coded ±0.5  (session 0 → -0.5, session 1 → +0.5)
      u_1:u_2 : congruent × session           (continuous × continuous → 1 column)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    counts = df.groupby("id")["id"].transform("size")
    df = df[counts >= 100].copy()

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)
        congruent   = (sdf["congruent"].values * 1.0 - 0.5).astype(np.float32)
        session     = (sdf["session"].values * 1.0 - 0.5).astype(np.float32)
        interaction = (congruent * session).astype(np.float32)
        data[int(sid)] = {
            "rts":     sdf["rt"].values.astype(np.float32),
            "choices": sdf["resp"].values.astype(np.float32),
            "u_1":     congruent,
            "u_2":     session,
            "u_1:u_2": interaction,
        }
    return data, sorted(data.keys())


def load_cdm_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load CDM CSV data (Kvam, 2019).

    Regressors:
      u_1     : speed_condition, effect coded ±0.5      (continuous, 1 column)
      u_2     : difficulty (3 levels: 1,2,3), dummy coded to (d2, d3)
                reference category = 1                  (categorical, 2 columns)
      u_1:u_2 : speed_condition × (d2, d3)             (cat×cont → 2 columns)

    RESPONSE SCALE: raw ``resp_deviation`` spans [-pi/2, pi/2], but CDM emits
    ``arctan2(x1, x0)`` on [-pi, pi] — the column is a HALF-ANGLE. It is doubled
    here so data and simulator share a support. Evidence (2026-08-19, mc r4c4,
    500 draws, choice-dimension PPC): KS 0.067->0.026 (Speeded) and 0.099->0.034
    (Unspeeded); PPC/observed angular sd mismatch +29% -> +3%; quantiles inside
    the 95% band 1/5 -> 5/5 for Unspeeded. Confirm against Kvam (2019)'s response
    apparatus before treating as settled.
    """
    RESP_DEVIATION_SCALE = 2.0
    import pandas as pd
    df = pd.read_csv(csv_path)

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)
        speed = (sdf["speed_condition"].values * 1.0 - 0.5).astype(np.float32)
        difficulty = sdf["difficulty"].values

        # Dummy coding: reference category = 1
        d2 = (difficulty == 2).astype(np.float32)
        d3 = (difficulty == 3).astype(np.float32)

        # cat × cont interaction: each dummy × speed_condition
        inter_d2 = d2 * speed
        inter_d3 = d3 * speed

        data[int(sid)] = {
            "rts":     sdf["rt"].values.astype(np.float32),
            "choices": (sdf["resp_deviation"].values * RESP_DEVIATION_SCALE).astype(np.float32),
            "u_1":     speed,
            "u_2":     np.column_stack([d2, d3]),          # (num_obs, 2)
            "u_1:u_2": np.column_stack([inter_d2, inter_d3]),  # (num_obs, 2)
        }
    return data, sorted(data.keys())
