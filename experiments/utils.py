"""Shared utility functions for CogFormer experiment scripts."""

import math
import os
import numpy as np
import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def recommend_batch_size(base: int = 64) -> int:
    """Suggest a batch size based on available GPU VRAM.

    CogFormer's O(N²) decoder self-attention (N≈1267 tokens) saturates GPU memory
    bandwidth even at small batch sizes, so throughput per step is roughly flat
    regardless of batch size.  Larger batches are a *gradient quality* lever, not
    a wall-clock speed lever — only use them if you also reduce epochs proportionally.

    Formula: scale by sqrt(vram / 8 GB) so the increase is conservative.
    Cap at 4× base to avoid OOM on attention-heavy architectures.
    Falls back to base when no CUDA device is present.
    """
    if not torch.cuda.is_available():
        return base
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    scale = max(1.0, math.sqrt(vram_gb / 8.0))
    return int(min(base * 4, 2 ** round(math.log2(base * scale))))


def recommend_sim_workers() -> int:
    """Number of outer PrefetchQueue workers scaled to available CPU cores.

    Each outer worker spawns (cpu_count // n_outer) inner numba threads,
    so the total thread count stays proportional to the machine.
    """
    cpu = os.cpu_count() or 4
    return max(3, min(8, cpu // 6))


class PrefetchQueue:
    """Overlap CPU simulation with GPU training by keeping n_workers batches in flight.

    While the GPU processes batch i, batches i+1..i+n_workers simulate in parallel
    threads, so the GPU never stalls waiting for data after the initial warm-up.
    Each worker thread should use its own simulator instance to avoid contention;
    pass a callable that handles this (e.g. via threading.local).
    """

    def __init__(self, sample_fn, n_workers: int = 3):
        self._fn = sample_fn
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._queue = deque(self._executor.submit(self._fn) for _ in range(n_workers))

    def next(self):
        batch = self._queue.popleft().result()
        self._queue.append(self._executor.submit(self._fn))
        return batch

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
        return False


def train_step(net, adapted, config, optimizer, scheduler):
    """One flow-matching gradient step; returns (loss_scalar, current_lr)."""
    optimizer.zero_grad(set_to_none=True)
    device_type = config["device"].type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
        pred_velocity, target_velocity = net(
            adapted["param_matrices"][..., None],
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            model_ids=adapted.get("model_ids"),
        )
        _module = getattr(net, 'module', net)  # handles DDP and DataParallel
        L = _module.compute_loss(pred_velocity, target_velocity, adapted["param_masks"])
    L.backward()
    if config["gradient_clip_norm"] is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip_norm"])
    optimizer.step()
    return L.detach().item(), scheduler.get_last_lr()[0]


def add_architecture_args(parser):
    """Register shared CogFormer architecture arguments."""
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)
    parser.add_argument("--compile_mode", type=str, default="max-autotune-no-cudagraphs",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])


def add_training_args(parser):
    """Register shared training schedule and design-space arguments."""
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=recommend_batch_size(64))
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--dry_run_epochs", type=int, default=None,
                        help="If set, train for this many epochs only and record extrapolated time.")
    parser.add_argument("--max_num_regressors", type=int, default=5)
    parser.add_argument("--max_num_categories", type=int, default=5)
    parser.add_argument("--keep_intercept", action="store_true", default=True)
    parser.add_argument("--add_interaction", action="store_true", default=True)


def chunked_sample(model, adapted, num_samples, steps, solver="euler",
                   samples_chunk_size=None):
    """Call model.sample() in chunks of samples_chunk_size to limit peak GPU memory.

    Peak memory scales with batch_size * samples_chunk_size * num_tokens.
    If samples_chunk_size is None the full num_samples are drawn in one pass.
    """
    input_data      = adapted["input_data"]
    param_indices   = adapted["param_indices"]
    regressor_idx   = adapted["regressor_indices"]
    params_mask     = adapted["param_masks"]
    model_ids       = adapted.get("model_ids")

    if samples_chunk_size is None or samples_chunk_size >= num_samples:
        return model.sample(
            input_data, param_indices, regressor_idx, params_mask,
            num_samples=num_samples, steps=steps, solver=solver,
            model_ids=model_ids,
        )

    chunks = []
    for s_start in range(0, num_samples, samples_chunk_size):
        s_count = min(samples_chunk_size, num_samples - s_start)
        chunks.append(model.sample(
            input_data, param_indices, regressor_idx, params_mask,
            num_samples=s_count, steps=steps, solver=solver,
            model_ids=model_ids,
        ))
    return np.concatenate(chunks, axis=1)


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
    return design_cols + 5  # +5: rt, choices_normalized, alt_col, favored_idx, theta


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
        def _process_scenario(s):
            tag, dc, opts = s if len(s) == 3 else (*s, {})
            new_dc = dc if opts.get("skip_global", False) else _add_to_intercept(dc)
            return (tag, new_dc, {k: v for k, v in opts.items() if k != "skip_global"})
        cfg["val_scenarios"] = [_process_scenario(s) for s in cfg["val_scenarios"]]

    if "holdout_configs" in cfg:
        cfg["holdout_configs"] = [
            {**hc, "design_config": _add_to_intercept(hc["design_config"])}
            for hc in cfg["holdout_configs"]
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
            (tag, _strip_from(dc), opts) for tag, dc, opts in cfg["val_scenarios"]
        ]

    if "holdout_configs" in cfg:
        cfg["holdout_configs"] = [
            {**hc, "design_config": _strip_from(hc["design_config"])}
            for hc in cfg["holdout_configs"]
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
