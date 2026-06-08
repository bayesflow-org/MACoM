import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.simulator.design_utils import build_parameter_mask, build_design_matrix, build_random_design_config
from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery
from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, strip_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim, chunked_sample


_CASE_LABELS = {
    "ddm": {
        "intercept_only": "Intercept", "regressed": "Regressed",
        "fixed": "Fixed", "fixed_regressed": "Fixed Reg.", "interaction": "Interaction",
    },
    "cdm": {
        "intercept_only": "Intercept Only", "regressed": "Regressed",
        "fixed": "Fixed Variability", "fixed_regressed": "Fixed + Regressed", "interaction": "With Interaction",
    },
    "rdm": {
        "intercept_only": "Intercept Only", "regressed": "Regressed",
        "fixed": "Fixed Variability", "fixed_regressed": "Fixed + Regressed", "interaction": "With Interaction",
    },
}
_OUTDIRS = {
    "ddm": "./outputs/figures/model_family/cf/ddm/ensemble/",
    "cdm": "./outputs/figures/model_family/cf/cdm/ensemble/",
    "rdm": "./outputs/figures/model_family/cf/rdm/ensemble/",
}

MODELS = {}
for _key, _base in {k: v for k, v in _BASE_MODELS.items() if k in _OUTDIRS}.items():
    _cfg = add_global_params(_base, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    # Ensemble only uses a subset of benchmark cases (no "full")
    _cfg["benchmark_configs"] = {
        k: v for k, v in _cfg["benchmark_configs"].items() if k != "full"
    }
    _cfg["outdir"] = _OUTDIRS[_key]
    _cfg["case_labels"] = _CASE_LABELS[_key]
    MODELS[_key] = _cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (defaults to model-specific path)")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help="Samples per ODE loop; peak GPU memory scales with batch_size * samples_chunk_size.")
    p.add_argument("--solver", type=str, default="midpoint",
                   choices=["euler", "midpoint", "rk4"])
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
        model=model_cfg["model_cls"](), name=model_cfg["name"], prior=model_cfg["prior_fn"](),
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
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    cogformer = torch.compile(cogformer, dynamic=True)

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
        adapted = adapter.adapt(samples)
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        pred_set = chunked_sample(
            cogformer, adapted, num_samples=args.num_samples,
            steps=args.num_sample_steps, solver=args.solver,
            samples_chunk_size=args.samples_chunk_size,
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
        build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=args.max_num_regressors,
            free_intrinsics=free_params,
            fixed_intrinsics=fixed_params,
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
            global_params=model_cfg.get("global_params"),
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
