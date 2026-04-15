import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, cdm_priors, rdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, cdm_link_fun, rdm_link_fun
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors


MODELS = {
    "ddm": {
        "model_cls": DDM,
        "prior_fn": ddm_priors,
        "link_fn": ddm_link_fun,
        "name": "DDM",
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$\pi$"],
        "free_params": ["v", "a", "z", "tau"],
        "fixed_params": ["s_v", "s_tau"],
        "global_params": ["pi"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir": "./outputs/figures/model_family/cf/ddm/",
        # DDM uses singular "family", CDM/RDM use plural "families"
        "pred_filename_template": "{model}_family_{case}_cf_pred.npz",
        "benchmark_configs": {
            "intercept_only": {
                "1": ["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
                "u_1": [], "u_2": [], "u_1:u_2": [],
            },
            "regressed": {
                "1": ["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": [],
            },
            "fixed": {
                "1": ["v", "a", "z", "tau", "pi"],
                "u_1": [], "u_2": [], "u_1:u_2": [],
            },
            "fixed_regressed": {
                "1": ["v", "a", "z", "tau", "pi"],
                "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": [],
            },
            "interaction": {
                "1": ["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "a", "z", "tau", "s_v"],
                "u_2": ["v", "a", "z", "tau"],
                "u_1:u_2": ["v", "a", "z"],
            },
            "full": {
                "1": ["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
                "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
                "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
            },
        },
    },
    "cdm": {
        "model_cls": CDM,
        "prior_fn": cdm_priors,
        "link_fn": cdm_link_fun,
        "name": "CDM",
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$\pi$"],
        "free_params": ["v", "v_theta", "a", "tau"],
        "fixed_params": ["s_v", "s_tau"],
        "global_params": ["pi"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir": "./outputs/figures/model_family/cf/cdm/",
        "pred_filename_template": "{model}_families_{case}_cf_pred.npz",
        "benchmark_configs": {
            "intercept_only": {
                "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": [], "u_2": [], "u_1:u_2": [],
            },
            "regressed": {
                "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": [],
            },
            "fixed": {
                "1": ["v", "v_theta", "a", "tau", "pi"],
                "u_1": [], "u_2": [], "u_1:u_2": [],
            },
            "fixed_regressed": {
                "1": ["v", "v_theta", "a", "tau", "pi"],
                "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": [],
            },
            "interaction": {
                "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "v_theta", "a", "tau", "s_v"],
                "u_2": ["v", "v_theta", "a", "tau"],
                "u_1:u_2": ["v", "v_theta", "a"],
            },
            "full": {
                "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
                "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            },
        },
    },
    "rdm": {
        "model_cls": RDM,
        "prior_fn": rdm_priors,
        "link_fn": rdm_link_fun,
        "name": "RDM",
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$", r"$\pi$"],
        "free_params": ["v", "v_diff", "a", "tau"],
        "fixed_params": ["s_v", "s_tau"],
        "global_params": ["pi"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir": "./outputs/figures/model_family/cf/rdm/",
        "pred_filename_template": "{model}_families_{case}_cf_pred.npz",
        "benchmark_configs": {
            "intercept_only": {
                "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": [], "u_2": [], "u_1:u_2": [],
            },
            "regressed": {
                "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": [],
            },
            "fixed": {
                "1": ["v", "v_diff", "a", "tau", "pi"],
                "u_1": [], "u_2": [], "u_1:u_2": [],
            },
            "fixed_regressed": {
                "1": ["v", "v_diff", "a", "tau", "pi"],
                "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": [],
            },
            "interaction": {
                "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "v_diff", "a", "tau", "s_v"],
                "u_2": ["v", "v_diff", "a", "tau"],
                "u_1:u_2": ["v", "v_diff", "a"],
            },
            "full": {
                "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"],
                "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
                "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            },
        },
    },
}


def strip_global_params(cfg: dict) -> dict:
    """Return a shallow copy of cfg with all global_params removed from
    intrinsic_params, variable_names, free_params, and benchmark_configs."""
    global_p = set(cfg.get("global_params", []))
    if not global_p:
        return cfg
    cfg = dict(cfg)
    keep = [i for i, p in enumerate(cfg["intrinsic_params"]) if p not in global_p]
    cfg["intrinsic_params"] = [cfg["intrinsic_params"][i] for i in keep]
    cfg["variable_names"]   = [cfg["variable_names"][i]   for i in keep]
    cfg["free_params"]      = [p for p in cfg["free_params"]  if p not in global_p]
    cfg["global_params"]    = []
    cfg["benchmark_configs"] = {
        name: {k: [p for p in ps if p not in global_p] for k, ps in dc.items()}
        for name, dc in cfg["benchmark_configs"].items()
    }
    return cfg


def load_validation_data(data_path: Path):
    dataset = np.load(data_path, allow_pickle=True)
    if "true_set" not in dataset.files:
        raise ValueError("BF npz is missing 'true_set'. Re-save from bf pipeline with true_set included.")
    return {
        "design_matrices": dataset["design_matrices"],
        "sim_data": {"rts": dataset["rts"], "choices": dataset["choices"]},
        "param_matrices": dataset["true_set"],
        "param_masks": dataset["param_masks"],
    }


def infer_free_fixed_intrinsics(design_config, all_intrinsics, default_fixed_values):
    used = {p for plist in design_config.values() for p in plist}
    free_intrinsics = [p for p in all_intrinsics if p in used]
    fixed_intrinsics = [p for p in all_intrinsics if p not in used]
    fixed_values = {p: default_fixed_values[p] for p in fixed_intrinsics if p in default_fixed_values}
    return free_intrinsics, fixed_intrinsics, fixed_values


def build_encoder_input_dim(max_num_regressors, max_num_categories, keep_intercept):
    max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
    return max_total_regressors * (max_num_categories - 1) + (4 if keep_intercept else 3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                   help="Benchmark model (ddm, cdm, rdm)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (defaults to model-specific path)")
    p.add_argument("--pred_dir", type=str, default="./outputs/data/model_family/")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--solver", type=str, default="rk4",
                   choices=["euler", "midpoint", "rk4"])
    p.add_argument("--include_full", action="store_true", default=False)
    p.add_argument("--skip_posteriors", action="store_true", default=False)
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--layer_dropout", type=float, default=0.05)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    p.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include global parameters (e.g. pi) in estimation. "
                        "Use --no-estimate-global-params to disable.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = MODELS[args.model]
    if not args.estimate_global_params:
        model_cfg = strip_global_params(model_cfg)
    model_name = args.model
    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]
    default_fixed_values = model_cfg["default_fixed_values"]
    link_fn = model_cfg["link_fn"]
    outdir_root = Path(args.outdir) if args.outdir else Path(model_cfg["outdir"])

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = ModelFamily(
        model=model_cfg["model_cls"](),
        name=model_cfg["name"],
        prior_fun=model_cfg["prior_fn"](),
        mask_randomizer_kwargs={
            "free_intrinsics": intrinsic_params,
            "fixed_intrinsics": [],
            "fixed_values": {},
            "global_params": model_cfg.get("global_params"),
        },
    )
    adapter = Adapter()

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, args.keep_intercept
    )
    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()

    colors = cogformer_fm_colors()
    benchmark = dict(model_cfg["benchmark_configs"])
    if not args.include_full:
        benchmark.pop("full", None)

    pred_filename_template = model_cfg["pred_filename_template"]

    for cfg_name, design_config in benchmark.items():
        print(f"Validating case {cfg_name}")
        outdir = outdir_root / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)

        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config, intrinsic_params, default_fixed_values
        )

        val_sample_config = {
            "mask_randomizer_kwargs": {
                "free_intrinsics": free_intr,
                "fixed_intrinsics": fixed_intr,
                "fixed_values": fixed_vals,
            },
            "min_num_regressors": 0,
            "fixed_config": True,
        }

        if args.data_dir is not None:
            data_path = Path(args.data_dir) / f"{model_name}_{cfg_name}_data.npz"
            test_samples = load_validation_data(data_path)

            num_rows = len(design_config)
            num_params = len(intrinsic_params)
            design_matrices = test_samples["design_matrices"]

            if design_matrices.shape[-1] == 1:
                batch_size, num_obs = design_matrices.shape[:2]
                dm = np.zeros((batch_size, num_obs, num_rows))
                dm[:, :, 0] = design_matrices.squeeze(axis=-1)
                test_samples["design_matrices"] = dm

            if test_samples["param_matrices"].shape[-1] < num_rows * num_params:
                batch_size = test_samples["param_matrices"].shape[0]
                pmat = np.zeros((batch_size, num_rows * num_params))
                pmask = np.zeros((batch_size, num_rows * num_params))
                pmat[:, :test_samples["param_matrices"].shape[-1]] = test_samples["param_matrices"]
                pmask[:, :test_samples["param_masks"].shape[-1]] = test_samples["param_masks"]
                test_samples["param_matrices"] = pmat
                test_samples["param_masks"] = pmask

            file_batch = test_samples["sim_data"]["rts"].shape[0]
            assert file_batch == args.batch_size, (
                f"Batch size mismatch: file={file_batch}, args={args.batch_size}"
            )
            test_samples = test_samples | model_family_config
        else:
            test_samples = model_family.batch_sample(
                **model_family_config,
                **val_sample_config,
                batch_size=args.batch_size,
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=link_fn(),
            )

        adapted = adapter.adapt(test_samples, intrinsic_params=model_family.intrinsic_params)
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_cols = len(intrinsic_params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        pred_set = cogformer.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=args.num_sample_steps,
            num_samples=args.num_samples,
            solver=args.solver,
        )
        pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_cols)

        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]

        pred_dir = Path(args.pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_filename = pred_filename_template.format(model=model_name, case=cfg_name)
        pred_path = pred_dir / pred_filename
        np.savez(pred_path, pred_set=pred_set, true_set=true_set, params_mask=params_mask)
        logging.info(f"[saved] {pred_path}")

        fig = adaptive_recovery(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        figpath = outdir / f"{model_name}_family_{cfg_name}_fm_mixed_recovery.pdf"
        fig.savefig(figpath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {figpath}")

        coverage = adaptive_coverage(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        coverage_path = outdir / f"{model_name}_family_{cfg_name}_fm_mixed_coverage.pdf"
        coverage.savefig(coverage_path, bbox_inches="tight")
        plt.close(coverage)
        logging.info(f"[saved] {coverage_path}")

        ecdf = adaptive_ecdf(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            difference=True,
        )
        ecdf_path = outdir / f"{model_name}_family_{cfg_name}_fm_mixed_ecdf.pdf"
        ecdf.savefig(ecdf_path, bbox_inches="tight")
        plt.close(ecdf)
        logging.info(f"[saved] {ecdf_path}")

        metrics_fig = plot_adaptive_metrics(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            skip_log_gamma=args.skip_log_gamma,
        )
        metrics_fig_path = outdir / f"{model_name}_family_{cfg_name}_fm_mixed_metrics.pdf"
        metrics_fig.savefig(metrics_fig_path, bbox_inches="tight")
        plt.close(metrics_fig)
        logging.info(f"[saved] {metrics_fig_path}")

        metrics_df = compute_adaptive_metrics(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            skip_log_gamma=args.skip_log_gamma,
        )
        metrics_csv_path = outdir / f"{model_name}_family_{cfg_name}_fm_mixed_metrics.csv"
        metrics_df.to_csv(metrics_csv_path)
        logging.info(f"[saved] {metrics_csv_path}")

        if not args.skip_posteriors:
            for i in range(min(10, args.batch_size)):
                posterior = adaptive_posterior(
                    samples=pred_set[i],
                    design_config=design_config,
                    intrinsic_params=intrinsic_params,
                    max_num_categories=args.max_num_categories,
                    parameter_mask=params_mask,
                    unfold=False,
                    intercept_color=colors["intercept"],
                    main_effect_color=colors["main_effect"],
                    interaction_color=colors["interaction"],
                )
                posterior_path = outdir / f"{model_name}_family_{cfg_name}_fm_mixed_posterior_{i}.pdf"
                posterior.savefig(posterior_path, bbox_inches="tight")
                plt.close(posterior.fig)

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
