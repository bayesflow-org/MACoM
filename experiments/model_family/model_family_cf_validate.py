import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, strip_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim, chunked_sample


# CF validate uses global params (pi) and adds script-specific output keys
_OUTDIR_MAP = {
    "ddm": ("./outputs/figures/model_family/cf/ddm/", "{model}_family_{case}_cf_pred.npz"),
    "cdm": ("./outputs/figures/model_family/cf/cdm/", "{model}_families_{case}_cf_pred.npz"),
    "rdm": ("./outputs/figures/model_family/cf/rdm/", "{model}_families_{case}_cf_pred.npz"),
}

MODELS = {}
for _key, _base in {k: v for k, v in _BASE_MODELS.items() if k in _OUTDIR_MAP}.items():
    _cfg = add_global_params(_base, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    _cfg["outdir"], _cfg["pred_filename_template"] = _OUTDIR_MAP[_key]
    MODELS[_key] = _cfg


def load_validation_data(data_path: Path):
    dataset = np.load(data_path, allow_pickle=True)
    if "true_set" not in dataset.files:
        raise ValueError("BF npz is missing 'true_set'. Re-save from bf pipeline with true_set included.")
    return {
        "design_matrices": dataset["design_matrices"],
        "sim_data": {"rts": dataset["rts"], "choices": dataset["choices"]},
        "param_matrices": dataset["true_set"],
        "param_masks": dataset["param_masks"],
        "n_params": dataset["true_set"].shape[-1],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                   help="Benchmark model (ddm, cdm, rdm)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (defaults to model-specific path)")
    p.add_argument("--pred_dir", type=str, default="./outputs/data/model_family/")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--solver", type=str, default="midpoint",
                   choices=["euler", "midpoint", "rk4"])
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help="Samples per ODE loop; peak GPU memory scales with batch_size * samples_chunk_size.")
    p.add_argument("--include_full", action="store_true", default=False)
    p.add_argument("--configs", nargs="+", default=None,
                   help="Subset of benchmark configs to run (e.g. --configs intercept_only regressed)."
                        " Runs all configs by default.")
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
    p.add_argument("--self_attn_padding_mask", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Must match the flag the checkpoint was trained with. Use "
                        "--no-self_attn_padding_mask for checkpoints trained mask-off.")
    p.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include global parameters (e.g. pi) in estimation. "
                        "Use --no-estimate-global-params to disable.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
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
        self_attn_padding_mask=args.self_attn_padding_mask,  # must match training run
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    cogformer = torch.compile(cogformer, dynamic=True)

    colors = cogformer_fm_colors()
    benchmark = dict(model_cfg["benchmark_configs"])
    if not args.include_full:
        benchmark.pop("full", None)
    if args.configs is not None:
        benchmark = {k: v for k, v in benchmark.items() if k in args.configs}

    pred_filename_template = model_cfg["pred_filename_template"]

    link_fun = link_fn()
    n_cols = len(intrinsic_params)

    # Phase 1: prepare all data
    cases = []
    for cfg_name, design_config in benchmark.items():
        print(f"Preparing data: {cfg_name}")
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
                link_fun=link_fun,
            )

        adapted = adapter.adapt(test_samples, device=device)
        true_flat = adapted["param_matrices"].detach().cpu().numpy()
        n_rows = true_flat.shape[1] // n_cols
        true_set = true_flat.reshape(args.batch_size, n_rows, n_cols)
        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape((args.batch_size, n_rows, n_cols))[0]
        cases.append({
            "cfg_name": cfg_name, "design_config": design_config,
            "adapted": adapted, "true_set": true_set,
            "params_mask": params_mask, "n_rows": n_rows, "outdir": outdir,
        })

    # Phase 2: run all ODE integrations
    for case in cases:
        print(f"Sampling: {case['cfg_name']}")
        case["pred_set"] = chunked_sample(
            cogformer, case["adapted"],
            num_samples=args.num_samples,
            steps=args.num_sample_steps,
            solver=args.solver,
            samples_chunk_size=args.samples_chunk_size,
        ).reshape(args.batch_size, args.num_samples, case["n_rows"], n_cols)

    # Phase 3: save predictions and generate figures
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cfg_name     = case["cfg_name"]
        design_config = case["design_config"]
        true_set     = case["true_set"]
        pred_set     = case["pred_set"]
        params_mask  = case["params_mask"]
        outdir       = case["outdir"]
        n_rows       = case["n_rows"]

        pred_path = pred_dir / pred_filename_template.format(model=model_name, case=cfg_name)
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
