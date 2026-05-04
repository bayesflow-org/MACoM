import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily
from cogformer.simulator.design_utils import split_design_space, build_random_design_config
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, strip_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim

HOLDOUT_TYPES = ["random", "variability", "interaction", "sparsity"]

MODELS = {
    key: add_global_params(base, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for key, base in _BASE_MODELS.items()
}


def _holdout_configs(holdout_type: str, model_cfg: dict, args) -> list[tuple[str, dict]]:
    """Return (name, design_config) pairs for the held-out validation set."""
    intrinsic_params = model_cfg["intrinsic_params"]
    free_params = model_cfg["free_params"]
    fixed_params = model_cfg["fixed_params"]
    default_fixed = model_cfg["default_fixed_values"]
    global_params = model_cfg.get("global_params", [])
    benchmark = model_cfg["benchmark_configs"]

    if holdout_type == "random":
        # Set B: slope patterns not seen during training.
        _, set_b = split_design_space(
            free_intrinsics=free_params,
            split_ratio=args.split_ratio,
            seed=args.split_seed,
        )
        logging.info(f"[random holdout] Validating on {len(set_b)} held-out slope patterns.")
        configs = []
        rng = np.random.default_rng(args.val_seed)
        for i in range(args.num_holdout_configs):
            dc = build_random_design_config(
                intrinsic_params=intrinsic_params,
                num_regressors=args.max_num_regressors,
                free_intrinsics=free_params,
                fixed_intrinsics=fixed_params,
                keep_intercept=args.keep_intercept,
                add_interaction=True,
                allowed_slope_patterns=set_b,
                global_params=global_params,
                free_prob=1.0,
            )
            configs.append((f"random_holdout_{i}", dc))
        return configs

    elif holdout_type == "variability":
        # Configs where variability params (s_v, s_tau) appear as slopes —
        # three scenarios of increasing exposure.
        all_params = intrinsic_params  # includes global (pi) if present
        var_params = fixed_params      # ["s_v", "s_tau"]
        main_params = free_params      # ["v", "a", "z", "tau"]

        configs = []

        # Scenario 1: variability-only slopes — only s_v / s_tau regressed.
        dc_var_only = {"1": list(all_params)}
        for r in range(1, args.max_num_regressors + 1):
            dc_var_only[f"u_{r}"] = list(var_params)
        if args.max_num_regressors >= 2:
            dc_var_only["u_1:u_2"] = list(var_params)
        configs.append(("variability_only", dc_var_only))

        # Scenario 2: mixed — main params AND variability params all regressed.
        dc_mixed = {"1": list(all_params)}
        for r in range(1, args.max_num_regressors + 1):
            dc_mixed[f"u_{r}"] = list(main_params) + list(var_params)
        if args.max_num_regressors >= 2:
            dc_mixed["u_1:u_2"] = list(main_params) + list(var_params)
        configs.append(("variability_mixed", dc_mixed))

        # Scenario 3: full benchmark config (already has s_v as a slope in u_1).
        configs.append(("variability_full", benchmark["full"]))

        return configs

    elif holdout_type == "interaction":
        # Configs that contain interaction terms — never seen during training.
        return [
            ("interaction", benchmark["interaction"]),
            ("full", benchmark["full"]),
        ]

    elif holdout_type == "sparsity":
        # Dense configs — high free_prob, all params regressed across many regressors.
        # Use the densest benchmark configs.
        return [
            ("interaction", benchmark["interaction"]),
            ("full", benchmark["full"]),
        ]

    raise ValueError(f"Unknown holdout_type: {holdout_type}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    p.add_argument("--holdout_type", type=str, required=True, choices=HOLDOUT_TYPES)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--pred_dir", type=str, default="./outputs/data/generalization/")

    # Validation settings
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])
    p.add_argument("--sample_chunk_size", type=int, default=50,
                   help="Batch items per cogformer.sample() call; reduces peak GPU memory.")
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help="Samples per cogformer.sample() call; reduces peak GPU memory. "
                        "Peak scales with batch_chunk * samples_chunk instead of batch_chunk * num_samples.")
    p.add_argument("--skip_posteriors", action="store_true", default=False)
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")

    # Architecture (must match training)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_dropout", type=float, default=0.0)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)

    # Holdout-specific
    p.add_argument("--split_seed", type=int, default=0,
                   help="[random] Must match the seed used during training.")
    p.add_argument("--split_ratio", type=float, default=0.5,
                   help="[random] Must match the ratio used during training.")
    p.add_argument("--num_holdout_configs", type=int, default=10,
                   help="[random] Number of random held-out configs to evaluate.")
    p.add_argument("--val_seed", type=int, default=42,
                   help="[random] RNG seed for sampling held-out design configs.")

    p.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = MODELS[args.model]
    if not args.estimate_global_params:
        model_cfg = strip_global_params(model_cfg)

    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]
    link_fn = model_cfg["link_fn"]

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
        prior=model_cfg["prior_fn"](),
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
    cogformer = torch.compile(cogformer, dynamic=True)

    colors = cogformer_fm_colors()
    model_name = args.model

    outdir_root = (
        Path(args.outdir) if args.outdir
        else Path(f"./outputs/figures/generalization/cf/{model_name}/{args.holdout_type}")
    )

    pred_dir = Path(args.pred_dir) / model_name / args.holdout_type
    pred_dir.mkdir(parents=True, exist_ok=True)

    holdout_cases = _holdout_configs(args.holdout_type, model_cfg, args)

    for cfg_name, design_config in holdout_cases:
        print(f"Validating holdout case: {cfg_name}")
        outdir = outdir_root / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)

        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config, intrinsic_params, model_cfg["default_fixed_values"]
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

        test_samples = model_family.batch_sample(
            **model_family_config,
            **val_sample_config,
            batch_size=args.batch_size,
            flatten_param_outputs=True,
            design_config=design_config,
            link_fun=link_fn(),
        )
        adapted = adapter.adapt(test_samples, device=device)

        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_cols = len(intrinsic_params)
        n_rows = true_set.shape[1] // n_cols
        true_set = true_set.reshape(args.batch_size, n_rows, n_cols)

        pred_chunks = []
        for c_start in range(0, args.batch_size, args.sample_chunk_size):
            c_end = min(c_start + args.sample_chunk_size, args.batch_size)
            pred_chunks.append(cogformer.sample(
                adapted["input_data"][c_start:c_end],
                adapted["param_indices"][c_start:c_end],
                adapted["regressor_indices"][c_start:c_end],
                adapted["param_masks"][c_start:c_end],
                steps=args.num_sample_steps,
                num_samples=args.num_samples,
                solver=args.solver,
                samples_chunk_size=args.samples_chunk_size,
            ))
        pred_set = np.concatenate(pred_chunks, axis=0)
        pred_set = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_cols)

        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]

        tag = f"{model_name}_{args.holdout_type}_{cfg_name}"
        np.savez(pred_dir / f"{tag}_cf_pred.npz",
                 pred_set=pred_set, true_set=true_set, params_mask=params_mask)
        logging.info(f"[saved] {pred_dir / f'{tag}_cf_pred.npz'}")

        fig = adaptive_recovery(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        fig.savefig(outdir / f"{tag}_recovery.pdf", bbox_inches="tight")
        plt.close(fig)

        coverage = adaptive_coverage(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )
        coverage.savefig(outdir / f"{tag}_coverage.pdf", bbox_inches="tight")
        plt.close(coverage)

        ecdf = adaptive_ecdf(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            difference=True,
        )
        ecdf.savefig(outdir / f"{tag}_ecdf.pdf", bbox_inches="tight")
        plt.close(ecdf)

        metrics_fig = plot_adaptive_metrics(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            skip_log_gamma=args.skip_log_gamma,
        )
        metrics_fig.savefig(outdir / f"{tag}_metrics.pdf", bbox_inches="tight")
        plt.close(metrics_fig)

        metrics_df = compute_adaptive_metrics(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
            parameter_mask=params_mask,
            variable_names=variable_names,
            skip_log_gamma=args.skip_log_gamma,
        )
        metrics_df.to_csv(outdir / f"{tag}_metrics.csv")
        logging.info(f"[saved] {outdir / f'{tag}_metrics.csv'}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
