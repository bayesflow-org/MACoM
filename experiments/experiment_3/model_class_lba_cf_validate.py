"""
Evaluate a ModelClass CogFormer (trained on DDM + RDM + CDM) on held-out LBA data.

LBA shares the same parameter structure as RDM {v, v_diff, a, tau, s_v, s_tau} plus
the contamination parameter pi (now estimated by the model class). Its local parameters
map cleanly into the 9-dim global parameter space:

    Global:  v   a   z   tau  s_v  s_tau  pi  v_diff  v_theta
    Index:   0   1   2   3    4    5      6   7       8
    LBA:     v   a   -   tau  s_v  s_tau  pi  v_diff  -

The CogFormer is run with a chosen model_id (default: RDM = 1, since the param
spaces overlap exactly). Use --all_model_ids to evaluate under all three known
model embeddings and compare recovery.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.design_utils import DESIGN_GENERATORS
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors, lba_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun, lba_link_fun
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
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

from ..configs import MODELS as _BASE_MODELS, LBA_CONFIG
from ..utils import (add_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim,
                     chunked_sample, resolve_val_scenario, seed_case, add_seed_arg,
                     load_bf_validation_data, bench_to_training_layout)


GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}

# LBA extended with pi — same as what add_global_params would produce
_LBA_WITH_PI = add_global_params(LBA_CONFIG, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)

# LBA local param order (with pi): ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"]
# Global param order (DDM→RDM→CDM with pi):
#   ["v", "a", "z", "tau", "s_v", "s_tau", "pi", "v_diff", "v_theta"]
#   Index: 0   1   2   3    4    5      6    7       8
LBA_LOCAL_PARAMS = _LBA_WITH_PI["intrinsic_params"]
LBA_TO_GLOBAL = [0, 7, 1, 3, 4, 5, 6]  # LBA local index i → global index LBA_TO_GLOBAL[i]
N_GLOBAL_PARAMS = 9


def lift_lba_to_global(
    param_matrices: np.ndarray,
    param_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift LBA's 7-dim local param layout to the 9-dim global space."""
    n_lba = len(LBA_LOCAL_PARAMS)
    batch_size = param_matrices.shape[0]
    n_rows = param_matrices.shape[1] // n_lba

    pm_local = param_matrices.reshape(batch_size, n_rows, n_lba)
    mask_local = param_masks.reshape(batch_size, n_rows, n_lba)

    pm_global = np.zeros((batch_size, n_rows, N_GLOBAL_PARAMS), dtype=param_matrices.dtype)
    mask_global = np.zeros((batch_size, n_rows, N_GLOBAL_PARAMS), dtype=param_masks.dtype)

    for local_idx, global_idx in enumerate(LBA_TO_GLOBAL):
        pm_global[:, :, global_idx] = pm_local[:, :, local_idx]
        mask_global[:, :, global_idx] = mask_local[:, :, local_idx]

    return pm_global.reshape(batch_size, -1), mask_global.reshape(batch_size, -1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/figures/generalization/model_class_lba/cf/")
    p.add_argument("--pred_dir", type=str, default=None,
                   help="Where prediction .npz files go. Defaults to <outdir>/predictions so a "
                        "single --outdir keeps figures and predictions together; pass explicitly "
                        "to separate them.")

    # Which model embedding to use during inference (default: RDM=1)
    p.add_argument("--eval_model_id", type=int, default=1,
                   help="Model ID to condition on (0=DDM, 1=RDM, 2=CDM). "
                        "Ignored when --all_model_ids is set.")
    p.add_argument("--all_model_ids", action="store_true", default=False,
                   help="Evaluate LBA under each of the 3 known model IDs.")

    # Benchmark cases
    p.add_argument("--lba_fixed_A", type=float, default=0.5,
                   help="Pin LBA's start-point range A so the intrinsic `a` is the real "
                        "response threshold (A + b == a). MUST match the value the "
                        "checkpoint was finetuned at (0.5) -- fixed_A changes what `a` "
                        "MEANS, so a mismatch compares two different quantities under one "
                        "name. Pass None only to reproduce the old parameterization where "
                        "`a` IS A; zero-shot transfer collapses there (0.86 -> 0.24) "
                        "because the pretrained families all treat `a` as a threshold.")
    p.add_argument("--scenarios", type=str, default=None,
                   help="Comma-separated val_scenario tags to run (default: all six).")
    p.add_argument("--include_full", action="store_true", default=False,
                   help="Deprecated / no effect: LBA zero-shot now runs the same six "
                        "val_scenarios as every other family, so there is no 'full' "
                        "benchmark config to opt into.")

    # Validation settings
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--design_generator", choices=DESIGN_GENERATORS, default="mixture",
                   help="Generator for self-simulated scenarios (mixed_predictor); --data_dir datasets are loaded as stored.")

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help="Samples per ODE loop; peak GPU memory scales with batch_size * samples_chunk_size.")
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")

    # Architecture (must match the checkpoint)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--encoder_embed_dim", type=int, default=64)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_dropout", type=float, default=0.0)
    p.add_argument("--model_embed_dim", type=int, default=8)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    p.add_argument("--self_attn_padding_mask", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Must match the flag the --checkpoint (MC base) was trained with. Use "
                        "--no-self_attn_padding_mask for a mask-off checkpoint.")

    p.add_argument("--data_dir", type=str, default=None,
                   help="Shared benchmark datasets (<family>_<case>_data.npz, the e1_data "
                        "stage). LBA cases found there are LOADED, so the zero-shot, "
                        "finetuned and BayesFlow arms score identical datasets; cases not "
                        "there fall back to seeded simulation.")
    add_seed_arg(p)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild ModelClass (same setup as training) to get model_index, n_global_params, etc.
    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM",
                           prior=ddm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=RDM(), name="RDM",
                           prior=rdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=CDM(), name="CDM",
                           prior=cdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"]),
    }
    link_funs = {
        "DDM": ddm_link_fun(),
        "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(),
    }
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    assert model_class.n_global_params == N_GLOBAL_PARAMS, (
        f"Global param space mismatch: expected {N_GLOBAL_PARAMS}, "
        f"got {model_class.n_global_params} ({model_class.global_param_names})"
    )

    # Standalone LBA family for simulation only.
    # num_workers=1 under --seed: the simulation pool carries its own RNG state, so a
    # parallel batch_sample is not reproducible (see seed_case).
    lba_family = ModelFamily(
        model=LBA(fixed_A=args.lba_fixed_A),
        name="LBA",
        prior=lba_priors(),
        num_workers=1 if args.seed is not None else None,
        intrinsic_params=LBA_LOCAL_PARAMS,
        mask_randomizer_kwargs={
            "free_intrinsics": _LBA_WITH_PI["free_params"],
            "fixed_intrinsics": _LBA_WITH_PI["fixed_params"],
            "fixed_values": _LBA_WITH_PI["default_fixed_values"],
            "global_params": GLOBAL_PARAMS,
        },
    )
    lba_link_fn = lba_link_fun()

    adapter = Adapter()

    encoder_input_dim = build_encoder_input_dim(
        max_num_regressors=args.max_num_regressors,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )

    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        encoder_embed_dim=args.encoder_embed_dim,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=model_class.num_models,
        model_embed_dim=args.model_embed_dim,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        self_attn_padding_mask=args.self_attn_padding_mask,  # must match the MC checkpoint's training run
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    cogformer = torch.compile(cogformer, dynamic=True)

    colors = cogformer_mc_colors()

    # Zero-shot LBA is validated on the SAME six scenarios as every other family
    # (intercept_only, fixed, regressed, fixed_regressed, interaction, mixed_predictor)
    # so generalization numbers are comparable with the in-distribution validation.
    # The older `benchmark_configs` set diverged: it ran up to 5 regressors and 3-way
    # pairwise interactions, so its numbers were not comparable with anything.
    benchmarks = _LBA_WITH_PI["val_scenarios"]
    if args.scenarios:
        want = {t.strip() for t in args.scenarios.split(",") if t.strip()}
        benchmarks = [b for b in benchmarks if b[0] in want]
        missing = want - {b[0] for b in benchmarks}
        if missing:
            raise ValueError(f"--scenarios unknown tags {sorted(missing)}; "
                             f"available: {[b[0] for b in _LBA_WITH_PI['val_scenarios']]}")
        logging.info(f"Running scenarios: {[b[0] for b in benchmarks]}")

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
        "design_generator": args.design_generator,
    }

    eval_ids = (
        list(model_class.model_index.values())
        if args.all_model_ids
        else [args.eval_model_id]
    )
    id_to_name = {v: k for k, v in model_class.model_index.items()}

    # fixed_A changes what `a` means, so runs at different values are not
    # interchangeable and must not share an output directory -- the same reason
    # model_class_cf_finetune_lba.py puts it in the checkpoint filename. Without
    # this, an A-sweep silently overwrites its own earlier figures.
    outdir_root = Path(args.outdir) / (
        f"A{'none' if args.lba_fixed_A is None else format(args.lba_fixed_A, 'g')}")
    logging.info(f"lba_fixed_A={args.lba_fixed_A} -> outputs under {outdir_root}")

    pred_dir = Path(args.pred_dir) if args.pred_dir else outdir_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Predictions -> {pred_dir}")

    variable_names   = _LBA_WITH_PI["variable_names"]
    intrinsic_params = LBA_LOCAL_PARAMS

    # Phase 1: prepare all data (one dataset per config, shared across eval_ids)
    config_data = []
    training_max_cats = args.max_num_categories
    for cfg_name, design_config, scenario_opts in benchmarks:
        max_cats, min_cats, mix_opts, skip = resolve_val_scenario(
            scenario_opts, training_max_cats
        )
        n_main_effects = sum(1 for k in design_config if k != "1" and ":" not in k)
        if n_main_effects > args.max_num_regressors:
            print(f"Skipping LBA / {cfg_name}: needs {n_main_effects} regressors, "
                  f"model trained with max_num_regressors={args.max_num_regressors}")
            continue
        if skip:
            print(f"Skipping LBA / {cfg_name}: needs max_num_categories={max_cats} "
                  f"> training {training_max_cats}")
            continue
        print(f"Preparing LBA data: {cfg_name}")
        # Keyed on "LBA" so the zero-shot and finetuned runs draw identical test sets.
        seed_case(args.seed, "LBA", cfg_name)
        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config, LBA_LOCAL_PARAMS, _LBA_WITH_PI["default_fixed_values"]
        )
        # Shared benchmark data first (see --data_dir); the seed pins parameters but not
        # the simulated RTs, so only LOADING pairs this arm with the finetuned and BF ones.
        data_path = (Path(args.data_dir) / f"lba_{cfg_name}_data.npz"
                     if args.data_dir is not None else None)
        if data_path is not None and not data_path.exists():
            print(f"LBA / {cfg_name}: not in {args.data_dir}; simulating (seeded)")
            data_path = None

        def self_simulate():
            return lba_family.batch_sample(
                design_config=design_config,
                batch_size=args.batch_size,
                num_obs=args.num_obs,
                mask_randomizer_kwargs={
                    "free_intrinsics": free_intr,
                    "fixed_intrinsics": fixed_intr,
                    "fixed_values": fixed_vals,
                    "global_params": GLOBAL_PARAMS,
                },
                flatten_param_outputs=True,
                fixed_config=True,
                link_fun=lba_link_fn,
                # Scenarios pin their own (usually narrower) category count, so the design
                # matrix must be padded back to the training width or the encoder input dim
                # will not match the checkpoint.
                pad_to_max_num_categories=training_max_cats,
                **{**model_family_config,
                   "max_num_categories": max_cats,
                   "min_num_categories": min_cats,
                   **mix_opts},
            )

        keep_rows = None
        if data_path is not None:
            test_samples = load_bf_validation_data(
                data_path=data_path, design_config=design_config,
                intrinsic_params=LBA_LOCAL_PARAMS, model_id=0,   # overwritten per eval id
                batch_size=args.batch_size,
                max_dm_cols=build_encoder_input_dim(
                    max_num_regressors=args.max_num_regressors,
                    max_num_categories=args.max_num_categories,
                    keep_intercept=args.keep_intercept) - 5,
            )
            # Same three fields, same reasons, as model_class_lba_finetune_validate's
            # paired branch: training regressor ceiling, the scenario's category count
            # (the shared grid has len(design_config) rows), and the design config so
            # the adapter can rebuild position codes.
            test_samples["max_num_regressors"] = args.max_num_regressors
            test_samples["max_num_categories"] = max_cats
            test_samples["design_configs"] = [design_config] * args.batch_size
            # The network sees the canonical training layout, not the compact file grid
            # (see utils.bench_to_training_layout); keep_rows maps back for saving.
            if max_cats < training_max_cats:
                test_samples, keep_rows = bench_to_training_layout(
                    test_samples, self_simulate(), design_config,
                    bench_max_num_categories=max_cats)
        else:
            test_samples = self_simulate()
        pm_global, mask_global = lift_lba_to_global(
            test_samples["param_matrices"],
            test_samples["param_masks"],
        )
        test_samples["param_matrices"] = pm_global
        test_samples["param_masks"]    = mask_global
        test_samples["n_params"]       = N_GLOBAL_PARAMS

        n_rows = pm_global.shape[1] // N_GLOBAL_PARAMS
        true_set   = pm_global.reshape(args.batch_size, n_rows, N_GLOBAL_PARAMS)[:, :, LBA_TO_GLOBAL]
        params_mask = mask_global.reshape(args.batch_size, n_rows, N_GLOBAL_PARAMS)[0][:, LBA_TO_GLOBAL]
        if keep_rows is not None:
            true_set, params_mask = true_set[:, keep_rows], params_mask[keep_rows]
        config_data.append({
            "cfg_name": cfg_name, "design_config": design_config,
            "test_samples": test_samples, "true_set": true_set,
            "params_mask": params_mask, "n_rows": n_rows, "keep_rows": keep_rows,
            # Row layout for LABELLING: compact (the scenario's count) when loaded from
            # the shared benchmark files, padded to training otherwise.
            "plot_cats": max_cats if data_path is not None else args.max_num_categories,
        })

    # Phase 2: run all ODE integrations (config × eval_id)
    cases = []
    for cd in config_data:
        for model_id in eval_ids:
            id_label = id_to_name[model_id]
            samples_copy = dict(cd["test_samples"])
            samples_copy["model_ids"] = np.full(args.batch_size, model_id, dtype=np.int64)
            adapted = adapter.adapt(samples_copy, device=device)
            print(f"Sampling: {cd['cfg_name']} / model_id={model_id} ({id_label})")
            seed_case(args.seed, "LBA", cd["cfg_name"], salt="ode")
            raw = chunked_sample(
                cogformer, adapted,
                num_samples=args.num_samples,
                steps=args.num_sample_steps,
                solver=args.solver,
                samples_chunk_size=args.samples_chunk_size,
            )
            pred_set = raw.reshape(
                args.batch_size, args.num_samples, cd["n_rows"], N_GLOBAL_PARAMS
            )[:, :, :, LBA_TO_GLOBAL]
            if cd["keep_rows"] is not None:
                pred_set = pred_set[:, :, cd["keep_rows"]]
            outdir = outdir_root / cd["cfg_name"] / f"mid_{id_label.lower()}"
            outdir.mkdir(parents=True, exist_ok=True)
            cases.append({
                "cfg_name": cd["cfg_name"], "design_config": cd["design_config"],
                "model_id": model_id, "id_label": id_label,
                "true_set": cd["true_set"], "pred_set": pred_set,
                "params_mask": cd["params_mask"], "outdir": outdir,
                "plot_cats": cd["plot_cats"],
            })

    # Phase 3: save predictions and generate figures
    for case in cases:
        cfg_name      = case["cfg_name"]
        design_config = case["design_config"]
        model_id      = case["model_id"]
        true_set      = case["true_set"]
        pred_set      = case["pred_set"]
        params_mask   = case["params_mask"]
        outdir        = case["outdir"]
        tag = f"lba_{cfg_name}_mid{model_id}"

        np.savez(
            pred_dir / f"{tag}_cf_pred.npz",
            pred_set=pred_set, true_set=true_set, params_mask=params_mask,
            max_num_categories=case["plot_cats"],   # row layout, for replots
        )
        logging.info(f"[saved] {pred_dir / f'{tag}_cf_pred.npz'}")

        # max_num_regressors is forwarded to every plot and metric call below, not only
        # to the sampler. Without it expand_design_config_to_mask cannot account for the
        # rows an interaction block occupies and invents phantom main effects (u_3, u_4,
        # u_5 ...) to cover them, so `u_1:u_2` is labelled and COLOURED as a main effect.
        # The cell values are unaffected -- this is purely which name sits beside a row --
        # but every interaction/mixed_predictor figure and metrics.csv this script wrote
        # before 2026-09-15 carries the wrong row names.
        fig = adaptive_recovery(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            global_params=GLOBAL_PARAMS,
        )
        fig.savefig(outdir / f"{tag}_recovery.pdf", bbox_inches="tight")
        plt.close(fig)

        # Global parameters (e.g. pi) are estimated jointly but plotted separately.
        glob_fig = global_recovery(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            global_params=GLOBAL_PARAMS,
            parameter_mask=params_mask,
            variable_names=variable_names,
            global_color=colors["intercept"],
        )
        if glob_fig is not None:
            glob_fig.savefig(outdir / f"{tag}_recovery_global.pdf", bbox_inches="tight")
            plt.close(glob_fig)

        coverage_fig = adaptive_coverage(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            global_params=GLOBAL_PARAMS,
        )
        coverage_fig.savefig(outdir / f"{tag}_coverage.pdf", bbox_inches="tight")
        plt.close(coverage_fig)

        # Global parameters (e.g. pi): separate coverage panel.
        cov_glob = global_coverage(
            true=true_set, pred=pred_set, design_config=design_config,
            variable_names=variable_names, global_params=GLOBAL_PARAMS,
            intrinsic_params=intrinsic_params, parameter_mask=params_mask,
            max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
        )
        if cov_glob is not None:
            cov_glob.savefig(outdir / f"{tag}_coverage_global.pdf", bbox_inches="tight")
            plt.close(cov_glob)

        ecdf_fig = adaptive_ecdf(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            difference=True,
            global_params=GLOBAL_PARAMS,
        )
        ecdf_fig.savefig(outdir / f"{tag}_ecdf.pdf", bbox_inches="tight")
        plt.close(ecdf_fig)

        # Global parameters (e.g. pi): separate ECDF panel.
        ecdf_glob = global_ecdf(
            true=true_set, pred=pred_set, design_config=design_config,
            variable_names=variable_names, global_params=GLOBAL_PARAMS,
            intrinsic_params=intrinsic_params, parameter_mask=params_mask,
            max_num_categories=case["plot_cats"], difference=True,
            max_num_regressors=args.max_num_regressors,
        )
        if ecdf_glob is not None:
            ecdf_glob.savefig(outdir / f"{tag}_ecdf_global.pdf", bbox_inches="tight")
            plt.close(ecdf_glob)

        metrics_fig = plot_adaptive_metrics(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
            skip_log_gamma=args.skip_log_gamma,
            global_params=GLOBAL_PARAMS,
        )
        metrics_fig.savefig(outdir / f"{tag}_metrics.pdf", bbox_inches="tight")
        plt.close(metrics_fig)

        # Global parameters (e.g. pi): separate metric heatmap.
        metrics_glob = plot_global_metrics(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params, max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            global_params=GLOBAL_PARAMS, parameter_mask=params_mask,
            variable_names=variable_names, skip_log_gamma=args.skip_log_gamma,
        )
        if metrics_glob is not None:
            metrics_glob.savefig(outdir / f"{tag}_metrics_global.pdf", bbox_inches="tight")
            plt.close(metrics_glob)

        metrics_df = compute_adaptive_metrics(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            parameter_mask=params_mask,
            variable_names=variable_names,
            skip_log_gamma=args.skip_log_gamma,
            global_params=GLOBAL_PARAMS,
        )
        metrics_df.to_csv(outdir / f"{tag}_metrics.csv")
        logging.info(f"[saved] {outdir / f'{tag}_metrics.csv'}")

        # Global parameters (e.g. pi): separate metrics CSV.
        metrics_glob_df = compute_global_metrics(
            true=true_set, pred=pred_set, design_config=design_config,
            intrinsic_params=intrinsic_params, max_num_categories=case["plot_cats"],
            max_num_regressors=args.max_num_regressors,
            global_params=GLOBAL_PARAMS, parameter_mask=params_mask,
            variable_names=variable_names, skip_log_gamma=args.skip_log_gamma,
        )
        if not metrics_glob_df.empty:
            metrics_glob_df.to_csv(outdir / f"{tag}_metrics_global.csv")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
