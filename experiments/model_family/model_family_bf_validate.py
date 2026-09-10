import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import keras
import logging
import numpy as np
import pandas as pd
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from cogformer.simulator.model_family import ModelFamily
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import bf_colors

from ..configs import MODELS as _BASE_MODELS
from ..utils import add_global_params, reshape_bf_to_cf, seed_case


_BF_DIRS = {
    "ddm": ("ddm_families_bf", "./outputs/figures/model_family/bf/ddm"),
    "cdm": ("cdm_families_bf", "./outputs/figures/model_family/bf/cdm"),
    "rdm": ("rdm_families_bf", "./outputs/figures/model_family/bf/rdm"),
    "lba": ("lba_families_bf", "./outputs/figures/model_family/bf/lba"),
}

# The BF baselines estimate the contamination parameter (_bf_cases_with_globals puts it
# in every case), so intrinsic_params, variable_names and benchmark_configs must all
# carry it here too. Without this, intrinsic_params is one short and reshape_bf_to_cf
# maps 6 of the 7 active params and trips its own assertion.
GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

MODELS = {}
for _key, _raw in {k: v for k, v in _BASE_MODELS.items() if k in _BF_DIRS}.items():
    _base = add_global_params(_raw, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    _cfg = dict(_base)
    _cfg["case_configs"] = _cfg["bf_case_configs"]
    _cfg["ckpt_prefix"], _cfg["figures_dir"] = _BF_DIRS[_key]
    _cfg["data_dir"] = "./outputs/data/model_family"
    _cfg["evals_dir"] = "./outputs/evaluations/model_family/bf"
    MODELS[_key] = _cfg


class ModelFamilyBF(bf.simulators.Simulator):
    def __init__(self, model_key: str, case: str):
        model_cfg = MODELS[model_key]
        case_configs = model_cfg["case_configs"]
        if case not in case_configs:
            raise ValueError(f"Unknown case: {case}. Must be one of {list(case_configs.keys())}")
        self.case = case
        self.config = case_configs[case]
        self.link_fn = model_cfg["link_fn"]
        self.model_family = ModelFamily(
            # model_kwargs carries per-family simulator settings that differ from the
            # class defaults -- today only LBA(fixed_A=0.5), which every other LBA arm
            # pins via --lba_fixed_A. Constructing bare here trained the baseline on a
            # different generative model than the arm it is compared against.
            model=model_cfg["model_cls"](**model_cfg.get("model_kwargs", {})),
            prior=model_cfg["prior_fn"](),
            regressed_params=self.config["regressed_params"],
            mask_randomizer_kwargs=dict(
                free_intrinsics=self.config["free_intrinsics"],
                fixed_intrinsics=self.config["fixed_intrinsics"],
                fixed_values=self.config["fixed_values"],
            ),
        )

    def sample(self, batch_size, num_obs=500, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]
        sample_kwargs = {
            "min_num_regressors": self.config["min_num_regressors"],
            "max_num_regressors": self.config["max_num_regressors"],
            "max_num_categories": self.config["max_num_categories"],
        }
        if self.config["fixed_config"]:
            sample_kwargs["fixed_config"] = True
        if self.config["design_config"] is not None:
            sample_kwargs["design_config"] = self.config["design_config"]
        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=self.config["flatten_param_outputs"],
            link_fun=self.link_fn(),
            **sample_kwargs,
            **kwargs,
        )
        param_matrices = samples["param_matrices"]
        param_masks = samples["param_masks"]
        if self.config["squeeze_outputs"]:
            param_matrices = param_matrices.squeeze(axis=1)
            param_masks = param_masks.squeeze(axis=1)
        return {
            "design_matrices": samples["design_matrices"],
            "rts": samples["sim_data"]["rts"],
            "choices": samples["sim_data"]["choices"],
            "params": param_matrices,
            "masks": param_masks,
        }


def test(model_key: str, case: str):
    model_cfg = MODELS[model_key]
    config = model_cfg["case_configs"][case]
    simulator = ModelFamilyBF(model_key=model_key, case=case)
    samples = simulator.sample(4)
    masks = samples["masks"]
    params = samples["params"]

    num_active = int(masks[0].astype(bool).sum())
    expected = config["expected_num_active"]
    if expected is not None:
        status = "OK" if num_active == expected else f"FAIL (expected {expected})"
        print(f"  [Check] Active params: {num_active}  →  {status}")
        assert num_active == expected
    else:
        print(f"  [Check] Active params: {num_active}  →  (variable, no assertion)")

    if config["fixed_config"] or config["design_config"] is not None:
        consistent = np.all(masks == masks[0])
        print(f"  [Check] Mask consistent across batch: {'OK' if consistent else 'FAIL'}")
        assert consistent

    zero_where_masked = np.allclose(params * (1 - masks), 0.0)
    print(f"  [Check] Params zero where masked: {'OK' if zero_where_masked else 'FAIL'}")
    assert zero_where_masked

    checkpoint_path = f"./outputs/checkpoints/model_family/bf/{model_key}/{MODELS[model_key]['ckpt_prefix']}_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)
    val_sims = simulator.sample(100)
    post_draws = approximator.sample(
        conditions={"rts": val_sims["rts"], "choices": val_sims["choices"], "design_matrices": val_sims["design_matrices"]},
        num_samples=100,
    )
    print(f"Targets: {val_sims['params'].shape}, Estimates: {post_draws['params'].shape}")


def main(model_key: str, case: str, batch_size: int = 200, num_samples: int = 200,
         skip_posteriors: bool = False, skip_log_gamma: bool = True, seed: int = 0):
    model_cfg = MODELS[model_key]
    model_name = model_key
    case_config = model_cfg["case_configs"][case]
    param_names = case_config["param_names"]
    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]

    simulator = ModelFamilyBF(model_key=model_key, case=case)
    ckpt_path = f"./outputs/checkpoints/model_family/bf/{model_key}/{model_cfg['ckpt_prefix']}_{case}/model.keras"
    approximator = keras.saving.load_model(ckpt_path)

    data_dir = Path(model_cfg["data_dir"])
    figures_dir = Path(model_cfg["figures_dir"]) / case
    evals_dir = Path(model_cfg["evals_dir"])
    for d in (data_dir, figures_dir, evals_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seeded per (model, case), like model_class_cf_validate. Without this the
    # validation datasets were redrawn on every invocation, so re-running this
    # script SILENTLY BROKE the pairing with any CogFormer predictions already
    # generated against the previous npz -- and C2ST's only guard is a shape check
    # that a re-pairing still passes. Seeding by identity (not by position) also
    # makes a single case reproducible without running the whole sweep.
    seed_case(seed, model_key, case)
    val_sims = simulator.sample(batch_size)
    post_draws = approximator.sample(
        conditions={"rts": val_sims["rts"], "choices": val_sims["choices"], "design_matrices": val_sims["design_matrices"]},
        num_samples=num_samples,
    )
    targets = val_sims["params"]
    estimates = post_draws["params"]

    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)
    true_params = targets[:, active_idx]

    # Checkpoints trained after 2026-09-10 emit ONLY the active cells: the trainer
    # slices the inference target because the full grid is up to 60% structurally
    # zero (see model_family_bf_train.ModelFamilyBF). Older checkpoints emit the
    # whole grid. Tell them apart by width rather than by a flag, so both load.
    #
    # The simulator here is NOT sliced, so true_set/param_masks stay full-grid, and
    # pred_set is scattered back to full-grid before saving. Every downstream
    # consumer -- model_family_c2st, which does pred_set[:, :, active_idx] -- is
    # therefore unaffected by which era produced the checkpoint.
    n_grid = int(masks.shape[-1])
    n_active = int(active_idx.sum())
    if estimates.shape[-1] == n_active:
        pred_params = estimates
        pred_full = np.zeros(estimates.shape[:2] + (n_grid,), dtype=estimates.dtype)
        pred_full[:, :, active_idx] = estimates
        logging.info(f"[{case}] active-only checkpoint ({n_active} of {n_grid} cells); "
                     f"scattered to full grid for the npz")
    elif estimates.shape[-1] == n_grid:
        pred_params = estimates[:, :, active_idx]
        pred_full = estimates
        logging.info(f"[{case}] legacy full-grid checkpoint ({n_grid} cells)")
    else:
        raise ValueError(
            f"[{case}] checkpoint emits {estimates.shape[-1]} params, which matches "
            f"neither the active count ({n_active}) nor the full grid ({n_grid})."
        )
    active_param_names = [name for name, active in zip(param_names, active_idx) if active] \
        if len(param_names) == len(active_idx) else param_names

    np.savez(
        data_dir / f"{model_name}_{case}_data.npz",
        rts=val_sims["rts"],
        choices=val_sims["choices"],
        true_set=val_sims["params"],
        pred_set=pred_full,
        design_matrices=val_sims["design_matrices"],
        param_masks=val_sims["masks"],
    )
    logging.info(f"Saved data to {data_dir}")

    rmse = bf.diagnostics.metrics.root_mean_squared_error(
        estimates=pred_params, targets=true_params, variable_names=active_param_names
    )
    calibration_errors = bf.diagnostics.metrics.calibration_error(
        estimates=pred_params, targets=true_params, variable_names=active_param_names
    )
    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=pred_params, targets=true_params, variable_names=active_param_names
    )
    metrics_dict = {
        rmse["metric_name"]: rmse["values"],
        calibration_errors["metric_name"]: calibration_errors["values"],
        contraction["metric_name"]: contraction["values"],
    }
    if not skip_log_gamma:
        log_gamma = bf.diagnostics.metrics.calibration_log_gamma(
            estimates=pred_params, targets=true_params, variable_names=active_param_names
        )
        metrics_dict[log_gamma["metric_name"]] = log_gamma["values"]
    metrics = pd.DataFrame(metrics_dict)
    metrics.to_csv(evals_dir / f"{model_name}_families_bf_{case}_evaluations.csv", sep=";")
    logging.info("Metric evaluation is now finished.")

    design_config = model_cfg["benchmark_configs"][case]
    adaptive_colors = bf_colors()

    true_grid = reshape_bf_to_cf(true_params, design_config, intrinsic_params)
    pred_grid = reshape_bf_to_cf(pred_params, design_config, intrinsic_params)
    params_mask = reshape_bf_to_cf(np.ones((1, true_params.shape[-1])), design_config, intrinsic_params)[0]

    # Split the grid description from the palette: the plotting helpers take the colors,
    # compute_adaptive_metrics does not and raises TypeError on intercept_color.
    common = dict(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=variable_names,
    )
    plot_colors = dict(
        intercept_color=adaptive_colors["intercept"],
        main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
    )

    recovery_fig = adaptive_recovery(true=true_grid, pred=pred_grid, **common, **plot_colors)
    recovery_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_recovery.pdf", bbox_inches="tight")
    plt.close(recovery_fig)

    coverage_fig = adaptive_coverage(true=true_grid, pred=pred_grid, **common, **plot_colors)
    coverage_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_coverage.pdf", bbox_inches="tight")
    plt.close(coverage_fig)

    ecdf_fig = adaptive_ecdf(true=true_grid, pred=pred_grid, difference=True, **common, **plot_colors)
    ecdf_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_ecdf.pdf", bbox_inches="tight")
    plt.close(ecdf_fig)

    metrics_fig = plot_adaptive_metrics(true=true_grid, pred=pred_grid, skip_log_gamma=skip_log_gamma, **common, **plot_colors)
    metrics_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_metrics.pdf", bbox_inches="tight")
    plt.close(metrics_fig)

    metrics_df = compute_adaptive_metrics(true=true_grid, pred=pred_grid, skip_log_gamma=skip_log_gamma, **common)
    metrics_df.to_csv(figures_dir / f"{model_name}_family_{case}_bf_metrics.csv")
    logging.info(f"Saved diagnostics to {figures_dir}")

    if not skip_posteriors:
        for i in range(10):
            posterior_fig = adaptive_posterior(
                samples=pred_grid[i],
                targets=true_grid[i],
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=2,
                intercept_color=adaptive_colors["intercept"],
                main_effect_color=adaptive_colors["main_effect"],
                interaction_color=adaptive_colors["interaction"],
            )
            posterior_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_posterior_{i}.pdf", bbox_inches="tight")
            plt.close(posterior_fig)


if __name__ == "__main__":
    # DDM_CASE_CONFIGS was never defined or imported here, so this raised NameError on
    # every invocation -- model_family_bf_validate has never actually run, which is why
    # outputs/data/model_family did not exist. Mirror model_family_bf_train, which reads
    # the case list off the already-built registry.
    all_cases = list(MODELS["ddm"]["case_configs"].keys())

    parser = argparse.ArgumentParser(description="Model Family BayesFlow Validation")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--case", type=str, default="intercept_only", choices=all_cases)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; the actual seed is derived from (model, case).")
    parser.add_argument("--skip_posteriors", action="store_true")
    parser.add_argument("--skip_log_gamma", action="store_true", default=True)
    parser.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        test(model_key=args.model, case=args.case)
    else:
        main(
            model_key=args.model,
            case=args.case,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            skip_posteriors=args.skip_posteriors,
            skip_log_gamma=args.skip_log_gamma,
            seed=args.seed,
        )
