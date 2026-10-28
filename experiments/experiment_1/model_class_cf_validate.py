import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.design_utils import DESIGN_GENERATORS
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery, global_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage, global_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf, global_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import (
    adaptive_metrics as plot_adaptive_metrics,
    global_metrics as plot_global_metrics,
)
from cogformer.diagnostics.metric.adaptive_metrics import (
    adaptive_metrics as compute_adaptive_metrics,
    global_metrics as compute_global_metrics,
)
from cogformer.utils.plot_utils import cogformer_mc_colors

from ..configs import MODELS as _BASE_MODELS
from ..utils import (load_bf_validation_data, add_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim,
                     chunked_sample, resolve_val_scenario, seed_case, add_seed_arg,
                     bench_to_training_layout, add_structure_conditioning_arg)

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}


def get_benchmark_design_configs():
    return {
        m.upper(): {
            "intrinsic_params": cfg["intrinsic_params"],
            "variable_names": cfg["variable_names"],
            "default_fixed_values": cfg["default_fixed_values"],
            "val_scenarios": cfg["val_scenarios"],
        }
        for m, cfg in _MODELS_WITH_PI.items()
    }


def infer_mask_randomizer_kwargs(design_config, all_intrinsics, default_fixed_values):
    free, fixed, values = infer_free_fixed_intrinsics(design_config, all_intrinsics, default_fixed_values)
    return {"free_intrinsics": free, "fixed_intrinsics": fixed, "fixed_values": values}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/figures/model_class/cf/")
    p.add_argument("--pred_dir", type=str, default=None,
                   help="Where prediction .npz files go. Defaults to <outdir>/predictions so a single "
                        "--outdir keeps figures and predictions together; pass explicitly to "
                        "separate them.")
    p.add_argument("--data_dir", type=str, default=None, help="Directory with BayesFlow validation data")

    # Validation settings
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--design_generator", choices=DESIGN_GENERATORS, default="mixture",
                   help="Generator for newly simulated evaluation data; saved --data_dir datasets are unchanged.")

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help="Samples per ODE loop; peak GPU memory scales with batch_size * samples_chunk_size.")
    p.add_argument("--include_full", action="store_true", default=False,
                   help="Deprecated / no effect: validation now runs the fixed val_scenarios set.")
    p.add_argument("--skip_posteriors", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip posterior pairplots (DEFAULT). They are not deliverables -- the finalize allowlists drop them -- and cost time per case; pass --no-skip_posteriors to draw them.")
    p.add_argument("--skip_log_gamma", action="store_true", default=True, help="Skip log gamma metric (slow, skipped by default)")
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false", help="Include log gamma metric")

    # Architecture (must match training)
    add_structure_conditioning_arg(p)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=128)
    p.add_argument("--encoder_embed_dim", type=int, default=64,
                   help="Per-trial embedding width; must match the training run (load is strict).")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_dropout", type=float, default=0.0)
    p.add_argument("--model_embed_dim", type=int, default=8)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    p.add_argument("--self_attn_padding_mask", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Must match the flag the checkpoint was trained with. Use "
                        "--no-self_attn_padding_mask for checkpoints trained mask-off.")

    # Must match the training run; see model_class_cf_train. Default ON = the CogFormer
    # class default. The first two add parameters, so a mismatch fails in load_state_dict.
    p.add_argument("--gated_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pre_pool_ffn", action=argparse.BooleanOptionalAction, default=True)
    add_seed_arg(p)
    return p.parse_args()


@torch.no_grad()
def plot_case(case, colors, max_num_categories, max_num_regressors,
              skip_posteriors=True, skip_log_gamma=True):
    """Every figure for one (family, case), from arrays that are already in hand.

    Split out of main's phase 3 so ``replot_per_case`` can redraw these from the
    saved ``*_cf_pred.npz`` without a checkpoint or a GPU. It touches nothing but
    ``case``'s arrays and metadata, which is why it can be driven either by a live
    validation run or by what that run persisted.

    The defaults MATCH parse_args': --skip_log_gamma and --skip_posteriors are both
    on there, and a different default here would make a replot silently produce a
    different set of columns from the run whose numbers it is redrawing.
    """
    model_name    = case["model_name"]
    cfg_name      = case["cfg_name"]
    design_config = case["design_config"]
    intrinsic_params = case["intrinsic_params"]
    variable_names   = case["variable_names"]
    true_set      = case["true_set"]
    pred_set      = case["pred_set"]
    params_mask   = case["params_mask"]
    outdir        = case["outdir"]
    categorical_regressors = case["categorical_regressors"]
    tag = f"{model_name.lower()}_{cfg_name}"
    # Shared row-labeling kwargs: the mask layout is padded to the training
    # regressor budget, and mixed cases carry contrast-coded regressors.
    label_kwargs = {
        "max_num_regressors": max_num_regressors,
        "categorical_regressors": categorical_regressors,
    }

    fig = adaptive_recovery(
        true=true_set, pred=pred_set,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=params_mask,
        variable_names=variable_names,
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
        global_params=GLOBAL_PARAMS,
        **label_kwargs,
    )
    fig_path = outdir / f"{tag}_fm_recovery.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {fig_path}")

    # Global parameters (e.g. pi) are estimated jointly but plotted separately.
    glob_fig = global_recovery(
        true=true_set, pred=pred_set,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        global_params=GLOBAL_PARAMS,
        parameter_mask=params_mask,
        variable_names=variable_names,
        global_color=colors["intercept"],
        **label_kwargs,
    )
    if glob_fig is not None:
        glob_path = outdir / f"{tag}_fm_recovery_global.pdf"
        glob_fig.savefig(glob_path, bbox_inches="tight")
        plt.close(glob_fig)
        logging.info(f"[saved] {glob_path}")

    coverage_fig = adaptive_coverage(
        true=true_set, pred=pred_set,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=params_mask,
        variable_names=variable_names,
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
        global_params=GLOBAL_PARAMS,
        **label_kwargs,
    )
    coverage_path = outdir / f"{tag}_fm_coverage.pdf"
    coverage_fig.savefig(coverage_path, bbox_inches="tight")
    plt.close(coverage_fig)
    logging.info(f"[saved] {coverage_path}")

    # Global parameters (e.g. pi): separate coverage panel.
    cov_glob = global_coverage(
        true=true_set, pred=pred_set, design_config=design_config,
        variable_names=variable_names, global_params=GLOBAL_PARAMS,
        intrinsic_params=intrinsic_params, parameter_mask=params_mask,
        max_num_categories=max_num_categories,
        **label_kwargs,
    )
    if cov_glob is not None:
        cov_glob.savefig(outdir / f"{tag}_fm_coverage_global.pdf", bbox_inches="tight")
        plt.close(cov_glob)

    ecdf_fig = adaptive_ecdf(
        true=true_set, pred=pred_set,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=params_mask,
        variable_names=variable_names,
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
        difference=True,
        global_params=GLOBAL_PARAMS,
        **label_kwargs,
    )
    ecdf_path = outdir / f"{tag}_fm_ecdf.pdf"
    ecdf_fig.savefig(ecdf_path, bbox_inches="tight")
    plt.close(ecdf_fig)
    logging.info(f"[saved] {ecdf_path}")

    # Global parameters (e.g. pi): separate ECDF panel.
    ecdf_glob = global_ecdf(
        true=true_set, pred=pred_set, design_config=design_config,
        variable_names=variable_names, global_params=GLOBAL_PARAMS,
        intrinsic_params=intrinsic_params, parameter_mask=params_mask,
        max_num_categories=max_num_categories, difference=True,
        **label_kwargs,
    )
    if ecdf_glob is not None:
        ecdf_glob.savefig(outdir / f"{tag}_fm_ecdf_global.pdf", bbox_inches="tight")
        plt.close(ecdf_glob)

    metrics_fig = plot_adaptive_metrics(
        true=true_set, pred=pred_set,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=params_mask,
        variable_names=variable_names,
        intercept_color=colors["intercept"],
        main_effect_color=colors["main_effect"],
        interaction_color=colors["interaction"],
        skip_log_gamma=skip_log_gamma,
        global_params=GLOBAL_PARAMS,
        **label_kwargs,
    )
    metrics_fig_path = outdir / f"{tag}_fm_metrics.pdf"
    metrics_fig.savefig(metrics_fig_path, bbox_inches="tight")
    plt.close(metrics_fig)
    logging.info(f"[saved] {metrics_fig_path}")

    # Global parameters (e.g. pi): separate metric heatmap.
    metrics_glob = plot_global_metrics(
        true=true_set, pred=pred_set, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        global_params=GLOBAL_PARAMS, parameter_mask=params_mask,
        variable_names=variable_names, skip_log_gamma=skip_log_gamma,
        **label_kwargs,
    )
    if metrics_glob is not None:
        metrics_glob.savefig(outdir / f"{tag}_fm_metrics_global.pdf", bbox_inches="tight")
        plt.close(metrics_glob)

    metrics_df = compute_adaptive_metrics(
        true=true_set, pred=pred_set,
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        parameter_mask=params_mask,
        variable_names=variable_names,
        skip_log_gamma=skip_log_gamma,
        global_params=GLOBAL_PARAMS,
        **label_kwargs,
    )
    metrics_csv_path = outdir / f"{tag}_fm_metrics.csv"
    metrics_df.to_csv(metrics_csv_path)
    logging.info(f"[saved] {metrics_csv_path}")

    # Global parameters (e.g. pi): separate metrics CSV.
    metrics_glob_df = compute_global_metrics(
        true=true_set, pred=pred_set, design_config=design_config,
        intrinsic_params=intrinsic_params, max_num_categories=max_num_categories,
        global_params=GLOBAL_PARAMS, parameter_mask=params_mask,
        variable_names=variable_names, skip_log_gamma=skip_log_gamma,
        **label_kwargs,
    )
    if not metrics_glob_df.empty:
        metrics_glob_df.to_csv(outdir / f"{tag}_fm_metrics_global.csv")

    if not skip_posteriors:
        for i in range(10):
            posterior_fig = adaptive_posterior(
                samples=pred_set[i],
                targets=true_set[i],
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=max_num_categories,
                parameter_mask=params_mask,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
                **label_kwargs,
            )
            posterior_path = outdir / f"{tag}_fm_posterior_{i}.pdf"
            posterior_fig.savefig(posterior_path, bbox_inches="tight")
            plt.close(posterior_fig)

def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild ModelClass to access model_index, n_global_params, and local_to_global maps.
    # Seeded runs must simulate serially; the worker pool has its own RNG state.
    _sim_workers = 1 if args.seed is not None else None
    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM",
                           prior=ddm_priors(), num_workers=_sim_workers,
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=RDM(), name="RDM",
                           prior=rdm_priors(), num_workers=_sim_workers,
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=CDM(), name="CDM",
                           prior=cdm_priors(), num_workers=_sim_workers,
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"]),
    }
    link_funs = {
        "DDM": ddm_link_fun(),
        "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(),
    }
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()

    encoder_input_dim = build_encoder_input_dim(
        max_num_regressors=args.max_num_regressors,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )

    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_structure_conditioning=args.encoder_structure_conditioning,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        encoder_embed_dim=args.encoder_embed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=model_class.num_models,
        model_embed_dim=args.model_embed_dim,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        self_attn_padding_mask=args.self_attn_padding_mask,  # must match training run
        gated_residual=args.gated_residual,
        pre_pool_ffn=args.pre_pool_ffn,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    cogformer = torch.compile(cogformer, dynamic=True)

    colors = cogformer_mc_colors()
    all_model_configs = get_benchmark_design_configs()
    training_max_cats = args.max_num_categories
    n_global_params = model_class.n_global_params

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
    }

    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(args.outdir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Predictions -> {pred_dir}")

    # Phase 1: prepare all data
    cases = []
    for model_name, model_cfg in all_model_configs.items():
        mf = model_class.model_families[model_name]
        model_id = model_class.model_index[model_name]
        link_fun = model_class.link_funs[model_name]
        intrinsic_params = model_cfg["intrinsic_params"]
        variable_names = model_cfg["variable_names"]
        default_fixed_values = model_cfg["default_fixed_values"]
        global_indices = model_class.local_to_global[model_name]

        for cfg_name, design_config, scenario_opts in model_cfg["val_scenarios"]:
            max_cats, min_cats, mix_opts, skip = resolve_val_scenario(scenario_opts, training_max_cats)
            n_main_effects = sum(1 for k in design_config if k != "1" and ":" not in k)
            if n_main_effects > args.max_num_regressors:
                print(f"Skipping {model_name} / {cfg_name}: needs {n_main_effects} regressors, "
                      f"model trained with max_num_regressors={args.max_num_regressors}")
                continue
            if skip:
                print(f"Skipping {model_name} / {cfg_name}: needs max_num_categories={max_cats} "
                      f"> training {training_max_cats}")
                continue
            print(f"Preparing data: {model_name} / {cfg_name}")
            seed_case(args.seed, model_name, cfg_name)
            outdir = Path(args.outdir) / model_name.lower() / cfg_name
            outdir.mkdir(parents=True, exist_ok=True)

            mask_kwargs = infer_mask_randomizer_kwargs(
                design_config=design_config,
                all_intrinsics=intrinsic_params,
                default_fixed_values=default_fixed_values,
            )
            # Shared benchmark data (the e1_data stage) covers the bench5 cases of
            # DDM/RDM/CDM. Cases it does not cover -- mixed_predictor, every LBA case --
            # fall through to seeded simulation rather than being skipped, so a run with
            # --data_dir still produces every scenario.
            data_path = (Path(args.data_dir) / f"{model_name.lower()}_{cfg_name}_data.npz"
                         if args.data_dir is not None else None)
            if data_path is not None and not data_path.exists():
                print(f"{model_name} / {cfg_name}: not in {args.data_dir}; simulating (seeded)")
                data_path = None

            def self_simulate():
                samples = mf.batch_sample(
                    design_config=design_config,
                    batch_size=args.batch_size,
                    num_obs=args.num_obs,
                    mask_randomizer_kwargs=mask_kwargs,
                    max_num_regressors=args.max_num_regressors,
                    max_num_categories=max_cats,
                    min_num_categories=min_cats,
                    keep_intercept=args.keep_intercept,
                    add_interaction=args.add_interaction,
                    flatten_param_outputs=True,
                    fixed_config=True,
                    link_fun=link_fun,
                    pad_to_max_num_categories=training_max_cats,
                    design_generator=args.design_generator,
                    **mix_opts,
                )
                samples["model_ids"] = np.full(args.batch_size, model_id, dtype=np.int64)
                return samples

            keep_rows = None
            if data_path is not None:
                # Must equal build_encoder_input_dim's design_cols, or the encoder gets
                # the wrong width (the old formula multiplied interaction blocks by
                # block_width instead of block_width**2: 31 cols instead of 67, and the
                # forward pass died with "mat1 and mat2 shapes cannot be multiplied
                # (2000x36 and 72x64)"). Derive it from the same function so the two
                # cannot drift; the -5 drops the rt/choice/alt/favored/theta channels.
                max_dm_cols = build_encoder_input_dim(
                    max_num_regressors=args.max_num_regressors,
                    max_num_categories=args.max_num_categories,
                    keep_intercept=args.keep_intercept,
                ) - 5
                test_samples = load_bf_validation_data(
                    data_path=data_path,
                    design_config=design_config,
                    intrinsic_params=intrinsic_params,
                    model_id=model_id,
                    batch_size=args.batch_size,
                    max_dm_cols=max_dm_cols,
                )
                test_samples["max_num_regressors"] = args.max_num_regressors
                # The scenario's resolved max_cats, NOT args.max_num_categories: the BF
                # parameter grid has len(design_config) rows, and build_position_codes
                # expands by block_width = max_num_categories - 1. These benchmark
                # scenarios pin max_cats=2 (block_width 1), giving the 4 rows the BF grid
                # actually has; passing the training max of 4 yields 16 and fails to
                # broadcast. It also keeps the grid equal to the BF side, which is what
                # model_family_c2st's guard requires.
                test_samples["max_num_categories"] = max_cats
                # batch_sample emits per-element position_codes; the BF loader cannot,
                # so hand the adapter the design config and let it rebuild them via its
                # documented fallback. Without this the paired branch dies in
                # Adapter._resolve_position_codes.
                test_samples["design_configs"] = [design_config] * args.batch_size
                # The network sees the canonical training layout, not the compact file
                # grid (see bench_to_training_layout); keep_rows maps back for saving.
                if max_cats < training_max_cats:
                    test_samples, keep_rows = bench_to_training_layout(
                        test_samples, self_simulate(), design_config,
                        bench_max_num_categories=max_cats)
            else:
                test_samples = self_simulate()

            test_samples["param_matrices"], test_samples["param_masks"] = \
                model_class.lift_to_global_space(
                    model_name,
                    test_samples["param_matrices"],
                    test_samples["param_masks"],
                )
            test_samples["n_params"] = n_global_params

            # Adapter.adapt defaults to torch.device('cuda'); pass the resolved device
            # so this runs on a CPU-only host too (the finetune validator already does).
            adapted = adapter.adapt(test_samples, device=device)
            true_flat = adapted["param_matrices"].detach().cpu().numpy()
            n_rows = true_flat.shape[1] // n_global_params
            true_set = true_flat.reshape(args.batch_size, n_rows, n_global_params)[:, :, global_indices]
            params_mask = adapted["param_masks"].detach().cpu().numpy()
            params_mask = params_mask.reshape(args.batch_size, n_rows, n_global_params)[0][:, global_indices]
            if keep_rows is not None:
                true_set, params_mask = true_set[:, keep_rows], params_mask[keep_rows]
            cases.append({
                "keep_rows": keep_rows,
                "model_name": model_name, "cfg_name": cfg_name,
                "design_config": design_config, "intrinsic_params": intrinsic_params,
                "variable_names": variable_names, "global_indices": global_indices,
                "adapted": adapted, "true_set": true_set,
                "params_mask": params_mask, "n_rows": n_rows, "outdir": outdir,
                # Contrast-coded regressors for row labeling (e.g. mixed_predictor's u_1)
                "categorical_regressors": mix_opts.get("categorical_regressors"),
                # The grid's OWN row layout, for labelling. Shared benchmark files load
                # as the compact grid (the scenario's category count, 2 for bench5);
                # self-simulated cases are padded to the training count. Labelling a
                # compact grid at the training count names u_2 / u_1:u_2 as extra
                # categories of u_1.
                "plot_cats": max_cats if data_path is not None else args.max_num_categories,
            })

    # Phase 2: run all ODE integrations
    for case in cases:
        print(f"Sampling: {case['model_name']} / {case['cfg_name']}")
        seed_case(args.seed, case["model_name"], case["cfg_name"], salt="ode")
        raw = chunked_sample(
            cogformer, case["adapted"],
            num_samples=args.num_samples,
            steps=args.num_sample_steps,
            solver=args.solver,
            samples_chunk_size=args.samples_chunk_size,
        )
        case["pred_set"] = raw.reshape(
            args.batch_size, args.num_samples, case["n_rows"], n_global_params
        )[:, :, :, case["global_indices"]]
        if case["keep_rows"] is not None:
            case["pred_set"] = case["pred_set"][:, :, case["keep_rows"]]

        # Persist immediately rather than at the top of phase 3. Sampling is the
        # expensive part (~20 min/case) and lives only in memory until written, so a
        # failure anywhere in phase-3 plotting used to discard every case that had not
        # yet been plotted -- one AttributeError in a posterior plot cost a full 7-hour
        # run. Each case is now durable the moment its ODE integration finishes.
        tag = f"{case['model_name'].lower()}_{case['cfg_name']}"
        np.savez(
            pred_dir / f"{tag}_cf_pred.npz",
            pred_set=case["pred_set"], true_set=case["true_set"],
            params_mask=case["params_mask"],
            # Layout the grid is in, so replot_per_case labels rows the same way.
            max_num_categories=case["plot_cats"],
        )
        logging.info(f"[saved] {pred_dir / f'{tag}_cf_pred.npz'}")

    # Phase 3: generate figures (predictions are already written in phase 2)
    for case in cases:
        plot_case(case, colors, case["plot_cats"], args.max_num_regressors,
                  skip_posteriors=args.skip_posteriors,
                  skip_log_gamma=args.skip_log_gamma)

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
