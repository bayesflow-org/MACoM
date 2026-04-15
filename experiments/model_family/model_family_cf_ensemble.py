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
from cogformer.simulator.context_manager import ContextManager
from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery
from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf
from cogformer.utils.plot_utils import cogformer_fm_colors


MODELS = {
    "ddm": {
        "model_cls": DDM, "prior_fn": ddm_priors, "link_fn": ddm_link_fun,
        "name": "DDM",
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "free_params": ["v", "a", "z", "tau"],
        "fixed_params": ["s_v", "s_tau"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir": "./outputs/figures/model_family/cf/ddm/ensemble/",
        # DDM uses "Intercept" (short) while CDM/RDM use "Intercept Only"
        "case_labels": {
            "intercept_only": "Intercept",
            "regressed": "Regressed",
            "fixed": "Fixed",
            "fixed_regressed": "Fixed Reg.",
            "interaction": "Interaction",
        },
        "benchmark_configs": {
            "intercept_only": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "fixed": {"1": ["v", "a", "z", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "a", "z", "tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "interaction": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
        },
    },
    "cdm": {
        "model_cls": CDM, "prior_fn": cdm_priors, "link_fn": cdm_link_fun,
        "name": "CDM",
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "free_params": ["v", "v_theta", "a", "tau"],
        "fixed_params": ["s_v", "s_tau"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir": "./outputs/figures/model_family/cf/cdm/ensemble/",
        "case_labels": {
            "intercept_only": "Intercept Only",
            "regressed": "Regressed",
            "fixed": "Fixed Variability",
            "fixed_regressed": "Fixed + Regressed",
            "interaction": "With Interaction",
        },
        "benchmark_configs": {
            "intercept_only": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "fixed": {"1": ["v", "v_theta", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "v_theta", "a", "tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "interaction": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v"], "u_2": ["v", "v_theta", "a", "tau"], "u_1:u_2": ["v", "v_theta", "a"]},
        },
    },
    "rdm": {
        "model_cls": RDM, "prior_fn": rdm_priors, "link_fn": rdm_link_fun,
        "name": "RDM",
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "free_params": ["v", "v_diff", "a", "tau"],
        "fixed_params": ["s_v", "s_tau"],
        "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
        "outdir": "./outputs/figures/model_family/cf/rdm/ensemble/",
        "case_labels": {
            "intercept_only": "Intercept Only",
            "regressed": "Regressed",
            "fixed": "Fixed Variability",
            "fixed_regressed": "Fixed + Regressed",
            "interaction": "With Interaction",
        },
        "benchmark_configs": {
            "intercept_only": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "fixed": {"1": ["v", "v_diff", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "interaction": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
        },
    },
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
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (defaults to model-specific path)")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--num_random_configs", type=int, default=8)
    p.add_argument("--benchmark_only", action="store_true", default=False)
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
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = MODELS[args.model]
    model_name = args.model
    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]
    free_params = model_cfg["free_params"]
    fixed_params = model_cfg["fixed_params"]
    default_fixed_values = model_cfg["default_fixed_values"]
    link_fn = model_cfg["link_fn"]
    n_cols = len(intrinsic_params)

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "num_obs": args.num_obs,
        "add_interaction": args.add_interaction,
    }

    model_family = ModelFamily(
        model=model_cfg["model_cls"](), name=model_cfg["name"], prior_fun=model_cfg["prior_fn"](),
        mask_randomizer_kwargs={"free_intrinsics": intrinsic_params, "fixed_intrinsics": [], "fixed_values": {}},
    )
    adapter = Adapter()
    cm = ContextManager()

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

    outdir = Path(args.outdir) if args.outdir else Path(model_cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    benchmark = model_cfg["benchmark_configs"]
    case_labels = model_cfg["case_labels"]

    def run_inference(design_config):
        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config, intrinsic_params, default_fixed_values
        )
        samples = model_family.batch_sample(
            **model_family_config,
            mask_randomizer_kwargs={"free_intrinsics": free_intr, "fixed_intrinsics": fixed_intr, "fixed_values": fixed_vals},
            min_num_regressors=0, fixed_config=True,
            batch_size=args.batch_size, flatten_param_outputs=True,
            design_config=design_config, link_fun=link_fn(),
        )
        adapted = adapter.adapt(samples, intrinsic_params=model_family.intrinsic_params)
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        pred_set = cogformer.sample(
            adapted["input_data"], adapted["param_indices"],
            adapted["regressor_indices"], adapted["param_masks"],
            steps=args.num_sample_steps, num_samples=args.num_samples,
        )
        pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_cols)

        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]
        return true_set, pred_set, params_mask

    # ── Benchmark ensemble ─────────────────────────────────────────────────────
    true_list, pred_list, masks, design_configs, labels = [], [], [], [], []

    for cfg_name, design_config in benchmark.items():
        logging.info(f"Sampling benchmark case: {cfg_name}")
        true_set, pred_set, params_mask = run_inference(design_config)
        true_list.append(true_set)
        pred_list.append(pred_set)
        masks.append(params_mask)
        design_configs.append(design_config)
        labels.append(case_labels[cfg_name])

    for plot_fn, plot_name in [
        (ensemble_recovery, "recovery"),
        (ensemble_coverage, "coverage"),
        (ensemble_ecdf, "ecdf"),
    ]:
        fig = plot_fn(
            true_list=true_list, pred_list=pred_list,
            design_configs=design_configs, intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_masks=masks, variable_names=variable_names,
            labels=labels, thumb_scale=2.0,
        )
        fig_path = outdir / f"{model_name}_ensemble_benchmark_{plot_name}.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {fig_path}")

    if args.benchmark_only:
        logging.info("Done.")
        return

    # ── Random configs ensemble ────────────────────────────────────────────────
    random_design_configs = [
        cm.build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=args.max_num_regressors,
            free_intrinsics=free_params,
            fixed_intrinsics=fixed_params,
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
        )
        for _ in range(args.num_random_configs)
    ]
    random_labels = [f"Random {i + 1}" for i in range(args.num_random_configs)]
    random_true_list, random_pred_list, random_masks = [], [], []

    for rand_dc in random_design_configs:
        rand_true, rand_pred, rand_mask = run_inference(rand_dc)
        random_true_list.append(rand_true)
        random_pred_list.append(rand_pred)
        random_masks.append(rand_mask)

    for plot_fn, plot_name in [
        (ensemble_recovery, "recovery"),
        (ensemble_coverage, "coverage"),
        (ensemble_ecdf, "ecdf"),
    ]:
        fig = plot_fn(
            true_list=random_true_list, pred_list=random_pred_list,
            design_configs=random_design_configs, intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_masks=random_masks, variable_names=variable_names,
            labels=random_labels, n_colors=args.num_random_configs,
        )
        fig_path = outdir / f"{model_name}_ensemble_random_{plot_name}.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {fig_path}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
