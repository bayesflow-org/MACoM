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
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import bf_colors

from ..configs import DDM_BF_CASE_CONFIGS, DDM_BENCHMARK_CONFIGS
from ..utils import reshape_bf_to_cf


CASE = "interaction"
CASE_CONFIG = DDM_BF_CASE_CONFIGS[CASE]
DESIGN_CONFIG = DDM_BENCHMARK_CONFIGS[CASE]
INTRINSIC_PARAMS = ["v", "a", "z", "tau", "s_v", "s_tau"]
VARIABLE_NAMES = [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

INFERENCE_NETWORKS = {
    "coupling_flow": bf.networks.CouplingFlow,
    "flow_matching": bf.networks.FlowMatching,
    "diffusion_model": bf.networks.DiffusionModel,
    "stable_consistency": bf.networks.StableConsistencyModel,
}

CKPT_DIR = Path("./outputs/checkpoints/baselines")
DATA_DIR = Path("./outputs/data/baselines")
FIGURES_DIR = Path("./outputs/figures/baselines")


class DDMInteractionSimulator(bf.simulators.Simulator):
    def __init__(self):
        self.config = CASE_CONFIG
        self.model_family = ModelFamily(
            model=DDM(),
            prior=ddm_priors(),
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
            "design_config": self.config["design_config"],
        }

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=self.config["flatten_param_outputs"],
            link_fun=ddm_link_fun(),
            **sample_kwargs,
            **kwargs,
        )

        return {
            "design_matrices": samples["design_matrices"],
            "rts": samples["sim_data"]["rts"],
            "choices": samples["sim_data"]["choices"],
            "params": samples["param_matrices"],
            "masks": samples["param_masks"],
        }


def build_adapter():
    return (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )


def build_summary_network():
    return bf.networks.SetTransformer(
        summary_dim=32, seed_dim=64,
        num_heads=(4, 4, 4, 4), mlp_depths=(1, 1, 1, 1),
        embed_dims=(128, 128, 128, 128), mlp_widths=(128, 128, 128, 128),
        num_seeds=4,
    )


def ckpt_path(net_name: str) -> str:
    return str(CKPT_DIR / net_name / f"ddm_interaction_{net_name}")


def train(net_name: str, epochs: int = 1000, steps_per_epoch: int = 100, batch_size: int = 64):
    simulator = DDMInteractionSimulator()
    inference_net = INFERENCE_NETWORKS[net_name]()

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=build_adapter(),
        summary_network=build_summary_network(),
        inference_network=inference_net,
        checkpoint_filepath=ckpt_path(net_name),
    )

    logging.info(f"[{net_name}] Starting training: {epochs} epochs × {steps_per_epoch} steps")
    history = workflow.fit_online(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
    )
    logging.info(f"[{net_name}] Training complete.")
    return history


def validate(net_name: str, batch_size: int = 200, num_samples: int = 200, skip_log_gamma: bool = True):
    simulator = DDMInteractionSimulator()
    approximator = keras.saving.load_model(ckpt_path(net_name) + "/model.keras")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = FIGURES_DIR / net_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    val_sims = simulator.sample(batch_size)
    post_draws = approximator.sample(
        conditions={
            "rts": val_sims["rts"],
            "choices": val_sims["choices"],
            "design_matrices": val_sims["design_matrices"],
        },
        num_samples=num_samples,
    )

    targets = val_sims["params"]
    estimates = post_draws["params"]
    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)

    true_params = targets[:, active_idx]
    pred_params = estimates[:, :, active_idx]
    active_param_names = CASE_CONFIG["param_names"]

    np.savez(
        DATA_DIR / f"ddm_interaction_{net_name}_data.npz",
        rts=val_sims["rts"],
        choices=val_sims["choices"],
        true_set=targets,
        pred_set=estimates,
        design_matrices=val_sims["design_matrices"],
        param_masks=masks,
    )
    logging.info(f"[{net_name}] Saved data to {DATA_DIR}")

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
    metrics.index = active_param_names
    metrics.to_csv(DATA_DIR / f"ddm_interaction_{net_name}_metrics.csv", sep=";")
    logging.info(f"[{net_name}] Saved metrics.")

    true_grid = reshape_bf_to_cf(true_params, DESIGN_CONFIG, INTRINSIC_PARAMS)
    pred_grid = reshape_bf_to_cf(pred_params, DESIGN_CONFIG, INTRINSIC_PARAMS)
    params_mask = reshape_bf_to_cf(np.ones((1, true_params.shape[-1])), DESIGN_CONFIG, INTRINSIC_PARAMS)[0]
    adaptive_colors = bf_colors()

    common = dict(
        design_config=DESIGN_CONFIG,
        intrinsic_params=INTRINSIC_PARAMS,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=VARIABLE_NAMES,
        intercept_color=adaptive_colors["intercept"],
        main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
    )

    plot_calls = [
        ("recovery", lambda: adaptive_recovery(true=true_grid, pred=pred_grid, **common)),
        ("coverage", lambda: adaptive_coverage(true=true_grid, pred=pred_grid, **common)),
        ("ecdf", lambda: adaptive_ecdf(true=true_grid, pred=pred_grid, difference=True, **common)),
        ("metrics", lambda: plot_adaptive_metrics(true=true_grid, pred=pred_grid, skip_log_gamma=skip_log_gamma, **common)),
    ]
    for suffix, call in plot_calls:
        fig = call()
        fig.savefig(fig_dir / f"ddm_interaction_{net_name}_{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)

    metrics_df = compute_adaptive_metrics(true=true_grid, pred=pred_grid, skip_log_gamma=skip_log_gamma, **common)
    metrics_df.to_csv(fig_dir / f"ddm_interaction_{net_name}_adaptive_metrics.csv")
    logging.info(f"[{net_name}] Saved diagnostic figures to {fig_dir}")


def compare(skip_log_gamma: bool = True):
    """Load per-network metric CSVs and produce a grouped comparison figure."""
    records = {}
    for net_name in INFERENCE_NETWORKS:
        csv_path = DATA_DIR / f"ddm_interaction_{net_name}_metrics.csv"
        if not csv_path.exists():
            logging.warning(f"Missing metrics for {net_name}, skipping.")
            continue
        df = pd.read_csv(csv_path, sep=";", index_col=0)
        records[net_name] = df

    if not records:
        logging.error("No metric files found. Run validation first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_cols = list(next(iter(records.values())).columns)
    net_names = list(records.keys())
    net_labels = [n.replace("_", "\n") for n in net_names]

    summary = {}
    for net_name, df in records.items():
        summary[net_name] = df.mean()
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(FIGURES_DIR / "ddm_interaction_comparison_summary.csv", sep=";")

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(net_names))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(net_names)))

    for ax, metric in zip(axes, metric_cols):
        means = [records[n][metric].mean() for n in net_names]
        sems = [records[n][metric].sem() for n in net_names]
        bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(net_labels, fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.set_ylabel("Mean ± SEM", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("DDM Interaction — Inference Network Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ddm_interaction_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved comparison figure to {FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Baseline Inference Network Comparison (DDM Interaction)")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "validate", "compare"],
        help="Operation to perform"
    )
    parser.add_argument(
        "--net", type=str, default="all",
        choices=list(INFERENCE_NETWORKS.keys()) + ["all"],
        help="Which inference network to use (ignored for compare mode)"
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--include_log_gamma", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    skip_log_gamma = not args.include_log_gamma

    nets = list(INFERENCE_NETWORKS.keys()) if args.net == "all" else [args.net]

    if args.mode == "train":
        for net_name in nets:
            train(net_name, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size)
    elif args.mode == "validate":
        for net_name in nets:
            validate(net_name, batch_size=args.val_batch_size, num_samples=args.num_samples, skip_log_gamma=skip_log_gamma)
    elif args.mode == "compare":
        compare(skip_log_gamma=skip_log_gamma)


if __name__ == "__main__":
    main()
