"""Validate a fine-tuned (DDM+RDM+CDM+LBA) ModelClass checkpoint.

Evaluates:
  1. LBA recovery under the dedicated LBA embedding (model_id=3).
  2. DDM / RDM / CDM recovery under their original embeddings (0/1/2)
     to quantify catastrophic forgetting.

Outputs (per model × benchmark condition):
  - {tag}_pred.npz        (pred_set, true_set, params_mask)
  - {tag}_metrics.csv
  - {tag}_recovery.pdf
  - {tag}_coverage.pdf
  - {tag}_ecdf.pdf
  - {tag}_metrics.pdf
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors, lba_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun, lba_link_fun
from .model_class_cf_validate import load_bf_validation_data
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
from ..utils import (
    add_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim, chunked_sample,
    resolve_val_scenario, seed_case, add_seed_arg
)

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}
_LBA_WITH_PI = add_global_params(LBA_CONFIG, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)

LBA_LOCAL_PARAMS = _LBA_WITH_PI["intrinsic_params"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lba_fixed_A", type=float, default=0.5,
                   help="MUST match the value the checkpoint was finetuned with. Pin LBA's start-point range A so the intrinsic `a` is the real response "
                             "threshold (A + b == a), matching what `a` means for DDM/RDM/CDM. "
                             "Without this the shared `a` column carries two incompatible "
                             "meanings across families. Pass 0 to also remove start-point "
                             "variability entirely; None restores the original parameterization.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/figures/model_class/cf/lba_finetune/")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory with BayesFlow validation data. When set, every arm "
                        "scores the SAME simulated datasets the BF baseline was scored "
                        "on, which is what makes the C2ST comparisons paired.")
    p.add_argument("--pred_dir", type=str, default=None,
                   help="Where prediction .npz files go. Defaults to <outdir>/predictions so a single "
                        "--outdir keeps figures and predictions together; pass explicitly to "
                        "separate them.")

    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)

    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])
    p.add_argument("--samples_chunk_size", type=int, default=1)
    p.add_argument("--include_full", action="store_true", default=False)
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")

    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
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
                   help="Must match the flag the finetuned checkpoint was trained with. Use "
                        "--no-self_attn_padding_mask for checkpoints finetuned mask-off.")

    add_seed_arg(p)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask_kwargs = {
        "DDM": {"free_intrinsics": _MODELS_WITH_PI["ddm"]["free_params"],
                "fixed_intrinsics": _MODELS_WITH_PI["ddm"]["fixed_params"],
                "fixed_values": _MODELS_WITH_PI["ddm"]["default_fixed_values"],
                "global_params": GLOBAL_PARAMS},
        "RDM": {"free_intrinsics": _MODELS_WITH_PI["rdm"]["free_params"],
                "fixed_intrinsics": _MODELS_WITH_PI["rdm"]["fixed_params"],
                "fixed_values": _MODELS_WITH_PI["rdm"]["default_fixed_values"],
                "global_params": GLOBAL_PARAMS},
        "CDM": {"free_intrinsics": _MODELS_WITH_PI["cdm"]["free_params"],
                "fixed_intrinsics": _MODELS_WITH_PI["cdm"]["fixed_params"],
                "fixed_values": _MODELS_WITH_PI["cdm"]["default_fixed_values"],
                "global_params": GLOBAL_PARAMS},
        "LBA": {"free_intrinsics": _LBA_WITH_PI["free_params"],
                "fixed_intrinsics": _LBA_WITH_PI["fixed_params"],
                "fixed_values": _LBA_WITH_PI["default_fixed_values"],
                "global_params": GLOBAL_PARAMS},
    }

    # Seeded runs must simulate serially; the worker pool has its own RNG state.
    _sim_workers = 1 if args.seed is not None else None
    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors(),
                           num_workers=_sim_workers,
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"],
                           mask_randomizer_kwargs=mask_kwargs["DDM"]),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors(),
                           num_workers=_sim_workers,
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"],
                           mask_randomizer_kwargs=mask_kwargs["RDM"]),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors(),
                           num_workers=_sim_workers,
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"],
                           mask_randomizer_kwargs=mask_kwargs["CDM"]),
        "LBA": ModelFamily(model=LBA(fixed_A=args.lba_fixed_A), name="LBA", prior=lba_priors(),
                           num_workers=_sim_workers,
                           intrinsic_params=LBA_LOCAL_PARAMS,
                           mask_randomizer_kwargs=mask_kwargs["LBA"]),
    }
    link_funs = {
        "DDM": ddm_link_fun(), "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(), "LBA": lba_link_fun(),
    }
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()

    cogformer = CogFormer(
        encoder_input_dim=build_encoder_input_dim(
            args.max_num_regressors, args.max_num_categories, args.keep_intercept
        ),
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
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
        self_attn_padding_mask=args.self_attn_padding_mask,  # must match finetune run
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    cogformer = torch.compile(cogformer, dynamic=True)

    colors = cogformer_mc_colors()
    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(args.outdir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    n_global = model_class.n_global_params

    # ── Build cases: all four families on the shared six val_scenarios ────────
    # val_scenarios, NOT benchmark_configs. The six val_scenarios are the set every other
    # validation path uses (training val_step, model_class_cf_validate, the zero-shot LBA
    # script), and they include `mixed_predictor`, which benchmark_configs does not have;
    # benchmark_configs in turn carries four tags nothing else validates on. Iterating it
    # here produced a set that could not be compared against any other run.
    training_max_cats = args.max_num_categories
    cases = []

    family_configs = {
        "LBA": _LBA_WITH_PI,
        "DDM": _MODELS_WITH_PI["ddm"],
        "RDM": _MODELS_WITH_PI["rdm"],
        "CDM": _MODELS_WITH_PI["cdm"],
    }

    for model_name, model_cfg in family_configs.items():
        mf = model_families[model_name]
        model_id = model_class.model_index[model_name]
        link_fn = link_funs[model_name]
        global_indices = model_class.local_to_global[model_name]
        intrinsic_params = (LBA_LOCAL_PARAMS if model_name == "LBA"
                            else model_cfg["intrinsic_params"])

        for cfg_name, design_config, scenario_opts in model_cfg["val_scenarios"]:
            max_cats, min_cats, mix_opts, skip = resolve_val_scenario(
                scenario_opts, training_max_cats
            )
            n_main_effects = sum(1 for k in design_config if k != "1" and ":" not in k)
            if n_main_effects > args.max_num_regressors:
                print(f"Skipping {model_name} / {cfg_name}: needs {n_main_effects} "
                      f"regressors > max_num_regressors={args.max_num_regressors}")
                continue
            if skip:
                print(f"Skipping {model_name} / {cfg_name}: needs max_num_categories="
                      f"{max_cats} > training {training_max_cats}")
                continue

            print(f"Preparing {model_name} / {cfg_name}")
            seed_case(args.seed, model_name, cfg_name)

            free, fixed, fixed_vals = infer_free_fixed_intrinsics(
                design_config, intrinsic_params, model_cfg["default_fixed_values"]
            )
            if args.data_dir is not None:
                # Paired mode: score the exact datasets model_family_bf_validate wrote,
                # so BF / original / finetuned all see identical simulations. Mirrors
                # model_class_cf_validate's branch -- crucially the lift to global space
                # below stays OUTSIDE this if/else, because load_bf_validation_data
                # returns param_matrices in LOCAL space just as batch_sample does.
                data_path = Path(args.data_dir) / f"{model_name.lower()}_{cfg_name}_data.npz"
                if not data_path.exists():
                    print(f"Skipping {model_name} / {cfg_name}: paired mode needs "
                          f"{data_path}, which does not exist (val_scenarios covers "
                          f"cases the BF baselines do not, e.g. mixed_predictor).")
                    continue
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
                samples = load_bf_validation_data(
                    data_path=data_path,
                    design_config=design_config,
                    intrinsic_params=intrinsic_params,
                    model_id=model_id,
                    batch_size=args.batch_size,
                    max_dm_cols=max_dm_cols,
                )
                samples["max_num_regressors"] = args.max_num_regressors
                # The scenario's resolved max_cats, NOT args.max_num_categories: the BF
                # parameter grid has len(design_config) rows, and build_position_codes
                # expands by block_width = max_num_categories - 1. These benchmark
                # scenarios pin max_cats=2 (block_width 1), giving the 4 rows the BF grid
                # actually has; passing the training max of 4 yields 16 and fails to
                # broadcast. It also keeps the grid equal to the BF side, which is what
                # model_family_c2st's guard requires.
                samples["max_num_categories"] = max_cats
                # batch_sample emits per-element position_codes; the BF loader cannot,
                # so hand the adapter the design config and let it rebuild them via its
                # documented fallback. Without this the paired branch dies in
                # Adapter._resolve_position_codes.
                samples["design_configs"] = [design_config] * args.batch_size
            else:
                samples = mf.batch_sample(
                    design_config=design_config,
                    batch_size=args.batch_size,
                    num_obs=args.num_obs,
                    mask_randomizer_kwargs={"free_intrinsics": free, "fixed_intrinsics": fixed,
                                            "fixed_values": fixed_vals,
                                            "global_params": GLOBAL_PARAMS},
                    max_num_regressors=args.max_num_regressors,
                    max_num_categories=max_cats,
                    min_num_categories=min_cats,
                    keep_intercept=args.keep_intercept,
                    add_interaction=args.add_interaction,
                    flatten_param_outputs=True,
                    fixed_config=True,
                    link_fun=link_fn,
                    pad_to_max_num_categories=training_max_cats,
                    **mix_opts,
                )
            pm_g, mask_g = model_class.lift_to_global_space(
                model_name, samples["param_matrices"], samples["param_masks"]
            )
            samples["param_matrices"] = pm_g
            samples["param_masks"] = mask_g
            samples["n_params"] = n_global
            samples["model_ids"] = np.full(args.batch_size, model_id, dtype=np.int64)

            n_rows = pm_g.shape[1] // n_global
            true_set = pm_g.reshape(args.batch_size, n_rows, n_global)[:, :, global_indices]
            params_mask = mask_g.reshape(args.batch_size, n_rows, n_global)[0][:, global_indices]

            outdir = Path(args.outdir) / model_name.lower() / cfg_name
            outdir.mkdir(parents=True, exist_ok=True)
            cases.append({
                "label": model_name.lower(), "cfg_name": cfg_name,
                "model_name": model_name, "design_config": design_config,
                "intrinsic_params": intrinsic_params,
                "variable_names": model_cfg["variable_names"],
                "samples": samples, "true_set": true_set, "params_mask": params_mask,
                "n_rows": n_rows, "global_indices": global_indices, "outdir": outdir,
            })

    print(f"Prepared {len(cases)} cases "
          f"({len(family_configs)} families x {len(cases)//max(1,len(family_configs))} scenarios)")

    # ── Phase 2: ODE integration ───────────────────────────────────────────────
    for case in cases:
        print(f"Sampling: {case['label']} / {case['cfg_name']}")
        seed_case(args.seed, case["model_name"], case["cfg_name"], salt="ode")
        adapted = adapter.adapt(case["samples"], device=device)
        raw = chunked_sample(
            cogformer, adapted,
            num_samples=args.num_samples,
            steps=args.num_sample_steps,
            solver=args.solver,
            samples_chunk_size=args.samples_chunk_size,
        )
        case["pred_set"] = raw.reshape(
            args.batch_size, args.num_samples, case["n_rows"], n_global
        )[:, :, :, case["global_indices"]]

        # Persist immediately rather than at the top of phase 3. Sampling is the
        # expensive part and lives only in memory until written, so a failure anywhere
        # in phase-3 plotting used to discard every case not yet plotted -- one
        # AttributeError in a posterior plot once cost a full 7-hour run.
        tag = f"{case['label']}_{case['cfg_name']}_ft"
        np.savez(pred_dir / f"{tag}_pred.npz",
                 pred_set=case["pred_set"], true_set=case["true_set"],
                 params_mask=case["params_mask"])
        logging.info(f"[saved] {pred_dir / f'{tag}_pred.npz'}")

    # ── Phase 3: figures (predictions are already written in phase 2) ──────────
    for case in cases:
        label         = case["label"]
        cfg_name      = case["cfg_name"]
        design_config = case["design_config"]
        intrinsic_params = case["intrinsic_params"]
        variable_names   = case["variable_names"]
        true_set      = case["true_set"]
        pred_set      = case["pred_set"]
        params_mask   = case["params_mask"]
        outdir        = case["outdir"]
        tag = f"{label}_{cfg_name}_ft"
        # max_num_regressors is forwarded to every plot and metric call below, not
        # only to the sampler. Without it expand_design_config_to_mask cannot account
        # for the rows an interaction block occupies and invents phantom main effects
        # (u_3, u_4, u_5 ...) to cover them, so `u_1:u_2` is labelled and COLOURED as
        # a main effect. The cell values are unaffected -- this is purely which name
        # sits beside a row -- but every interaction/mixed_predictor figure and
        # metrics.csv this script wrote before 2026-09-15 carries the wrong names.


        for fn, name, kwargs in [
            (adaptive_recovery,      "recovery",  {}),
            (adaptive_coverage,      "coverage",  {}),
            (adaptive_ecdf,          "ecdf",      {"difference": True}),
            (plot_adaptive_metrics,  "metrics",   {"skip_log_gamma": args.skip_log_gamma}),
        ]:
            fig = fn(
                true=true_set, pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                max_num_regressors=args.max_num_regressors,
                parameter_mask=params_mask,
                variable_names=variable_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
                global_params=GLOBAL_PARAMS,
                **kwargs,
            )
            fig.savefig(outdir / f"{tag}_{name}.pdf", bbox_inches="tight")
            plt.close(fig)
            logging.info(f"[saved] {outdir / f'{tag}_{name}.pdf'}")

        # Global parameters (e.g. pi) are estimated jointly but plotted separately.
        for gfn, name, gkwargs in [
            (global_recovery,     "recovery", {}),
            (global_coverage,     "coverage", {}),
            (global_ecdf,         "ecdf",     {"difference": True}),
            (plot_global_metrics, "metrics",  {"skip_log_gamma": args.skip_log_gamma}),
        ]:
            gfig = gfn(
                true=true_set, pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=args.max_num_categories,
                max_num_regressors=args.max_num_regressors,
                global_params=GLOBAL_PARAMS,
                parameter_mask=params_mask,
                variable_names=variable_names,
                global_color=colors["intercept"],
                **gkwargs,
            )
            if gfig is not None:
                gfig.savefig(outdir / f"{tag}_{name}_global.pdf", bbox_inches="tight")
                plt.close(gfig)
                logging.info(f"[saved] {outdir / f'{tag}_{name}_global.pdf'}")

        metrics_df = compute_adaptive_metrics(
            true=true_set, pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=args.max_num_categories,
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
            intrinsic_params=intrinsic_params, max_num_categories=args.max_num_categories,
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
