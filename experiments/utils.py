"""Shared utility functions for CogFormer experiment scripts."""

import numpy as np


def infer_free_fixed_intrinsics(
    design_config: dict,
    all_intrinsics: list[str],
    default_fixed_values: dict[str, float],
) -> tuple[list[str], list[str], dict[str, float]]:
    """Partition intrinsics into free/fixed based on which appear in design_config.

    Returns (free_intrinsics, fixed_intrinsics, fixed_values).
    """
    used = {p for params in design_config.values() for p in params}
    free = [p for p in all_intrinsics if p in used]
    fixed = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed if p in default_fixed_values}
    return free, fixed, fixed_values


def build_encoder_input_dim(
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool,
) -> int:
    """Compute the encoder input dimension for a CogFormer model."""
    block_width = max_num_categories - 1
    n_main = max_num_regressors
    n_inter = max_num_regressors * (max_num_regressors - 1) // 2
    design_cols = n_main * block_width + n_inter * block_width ** 2 + (1 if keep_intercept else 0)
    return design_cols + 3  # +3 for sim_data features (rt, choice, and one auxiliary)


def add_global_params(
    cfg: dict,
    global_params: list[str],
    global_variable_names: list[str],
) -> dict:
    """Return a copy of cfg with global_params appended to intrinsic_params,
    variable_names, and the intercept ("1") column of benchmark_configs and
    val_scenarios.
    """
    cfg = dict(cfg)
    cfg["intrinsic_params"] = cfg["intrinsic_params"] + global_params
    cfg["variable_names"] = cfg["variable_names"] + global_variable_names
    cfg["global_params"] = list(global_params)

    def _add_to_intercept(dc: dict) -> dict:
        return {
            k: (ps + [p for p in global_params if p not in ps] if k == "1" else list(ps))
            for k, ps in dc.items()
        }

    if "benchmark_configs" in cfg:
        cfg["benchmark_configs"] = {
            name: _add_to_intercept(dc)
            for name, dc in cfg["benchmark_configs"].items()
        }

    if "val_scenarios" in cfg:
        cfg["val_scenarios"] = [
            (tag, _add_to_intercept(dc)) for tag, dc in cfg["val_scenarios"]
        ]

    return cfg


def strip_global_params(cfg: dict) -> dict:
    """Return a copy of cfg with global_params removed from intrinsic_params,
    variable_names, free_params, benchmark_configs, and val_scenarios.
    """
    global_p = set(cfg.get("global_params", []))
    if not global_p:
        return cfg

    cfg = dict(cfg)
    keep = [i for i, p in enumerate(cfg["intrinsic_params"]) if p not in global_p]
    cfg["intrinsic_params"] = [cfg["intrinsic_params"][i] for i in keep]
    cfg["variable_names"] = [cfg["variable_names"][i] for i in keep]
    cfg["free_params"] = [p for p in cfg.get("free_params", []) if p not in global_p]
    cfg["global_params"] = []

    def _strip_from(dc: dict) -> dict:
        return {k: [p for p in ps if p not in global_p] for k, ps in dc.items()}

    if "benchmark_configs" in cfg:
        cfg["benchmark_configs"] = {
            name: _strip_from(dc) for name, dc in cfg["benchmark_configs"].items()
        }

    if "val_scenarios" in cfg:
        cfg["val_scenarios"] = [
            (tag, _strip_from(dc)) for tag, dc in cfg["val_scenarios"]
        ]

    return cfg


def reshape_bf_to_cf(
    bf_samples: np.ndarray,
    design_config: dict,
    intrinsic_params: list[str],
) -> np.ndarray:
    """Reshape flat BayesFlow posterior samples into (*, num_rows, num_cols) layout.

    bf_samples shape: (*leading, num_active_params)
    Returns shape:    (*leading, num_rows, num_cols)
    """
    *leading, num_active = bf_samples.shape
    num_rows = len(design_config)
    num_cols = len(intrinsic_params)
    col_idx = {p: j for j, p in enumerate(intrinsic_params)}

    result = np.zeros((*leading, num_rows, num_cols))
    flat_pos = 0
    for row_i, active_params in enumerate(design_config.values()):
        ordered = [p for p in intrinsic_params if p in active_params]
        for p in ordered:
            result[..., row_i, col_idx[p]] = bf_samples[..., flat_pos]
            flat_pos += 1

    assert flat_pos == num_active, f"Expected {num_active} active params but mapped {flat_pos}."
    return result
