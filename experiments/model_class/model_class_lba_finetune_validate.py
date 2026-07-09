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
    add_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim, chunked_sample
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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/figures/model_class/cf/lba_finetune/")
    p.add_argument("--pred_dir", type=str, default="./outputs/data/model_class/cf/lba_finetune/")

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

    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"],
                           mask_randomizer_kwargs=mask_kwargs["DDM"]),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"],
                           mask_randomizer_kwargs=mask_kwargs["RDM"]),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"],
                           mask_randomizer_kwargs=mask_kwargs["CDM"]),
        "LBA": ModelFamily(model=LBA(), name="LBA", prior=lba_priors(),
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
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
    }
    n_global = model_class.n_global_params

    # ── Build cases: LBA + DDM/RDM/CDM ────────────────────────────────────────
    cases = []

    # LBA benchmarks (model_id = LBA = 3)
    lba_benchmarks = dict(_LBA_WITH_PI["benchmark_configs"])
    if not args.include_full:
        lba_benchmarks.pop("full", None)
    lba_model_id = model_class.model_index["LBA"]
    lba_global_indices = model_class.local_to_global["LBA"]
    lba_link = link_funs["LBA"]

    for cfg_name, design_config in lba_benchmarks.items():
        print(f"Preparing LBA / {cfg_name}")
        free, fixed, fixed_vals = infer_free_fixed_intrinsics(
            design_config, LBA_LOCAL_PARAMS, _LBA_WITH_PI["default_fixed_values"]
        )
        samples = model_families["LBA"].batch_sample(
            design_config=design_config,
            batch_size=args.batch_size,
            num_obs=args.num_obs,
            mask_randomizer_kwargs={"free_intrinsics": free, "fixed_intrinsics": fixed,
                                    "fixed_values": fixed_vals, "global_params": GLOBAL_PARAMS},
            flatten_param_outputs=True,
            fixed_config=True,
            link_fun=lba_link,
            **model_family_config,
        )
        pm_g, mask_g = model_class.lift_to_global_space("LBA", samples["param_matrices"],
                                                         samples["param_masks"])
        samples["param_matrices"] = pm_g
        samples["param_masks"] = mask_g
        samples["n_params"] = n_global
        samples["model_ids"] = np.full(args.batch_size, lba_model_id, dtype=np.int64)

        n_rows = pm_g.shape[1] // n_global
        true_set = pm_g.reshape(args.batch_size, n_rows, n_global)[:, :, lba_global_indices]
        params_mask = mask_g.reshape(args.batch_size, n_rows, n_global)[0][:, lba_global_indices]

        outdir = Path(args.outdir) / "lba" / cfg_name
        outdir.mkdir(parents=True, exist_ok=True)
        cases.append({
            "label": "lba", "cfg_name": cfg_name, "design_config": design_config,
            "intrinsic_params": LBA_LOCAL_PARAMS,
            "variable_names": _LBA_WITH_PI["variable_names"],
            "samples": samples, "true_set": true_set, "params_mask": params_mask,
            "n_rows": n_rows, "global_indices": lba_global_indices, "outdir": outdir,
        })

    # DDM/RDM/CDM benchmarks (forgetting check)
    for model_name, model_cfg in {
        "DDM": _MODELS_WITH_PI["ddm"],
        "RDM": _MODELS_WITH_PI["rdm"],
        "CDM": _MODELS_WITH_PI["cdm"],
    }.items():
        mf = model_families[model_name]
        model_id = model_class.model_index[model_name]
        link_fn = link_funs[model_name]
        global_indices = model_class.local_to_global[model_name]
        intrinsic_params = model_cfg["intrinsic_params"]
        benchmarks = dict(model_cfg["benchmark_configs"])
        if not args.include_full:
            benchmarks.pop("full", None)

        for cfg_name, design_config in benchmarks.items():
            print(f"Preparing {model_name} / {cfg_name}")
            free, fixed, fixed_vals = infer_free_fixed_intrinsics(
                design_config, intrinsic_params, model_cfg["default_fixed_values"]
            )
            samples = mf.batch_sample(
                design_config=design_config,
                batch_size=args.batch_size,
                num_obs=args.num_obs,
                mask_randomizer_kwargs={"free_intrinsics": free, "fixed_intrinsics": fixed,
                                        "fixed_values": fixed_vals, "global_params": GLOBAL_PARAMS},
                flatten_param_outputs=True,
                fixed_config=True,
                link_fun=link_fn,
                **model_family_config,
            )
            pm_g, mask_g = model_class.lift_to_global_space(model_name,
                                                             samples["param_matrices"],
                                                             samples["param_masks"])
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
                "design_config": design_config,
                "intrinsic_params": intrinsic_params,
                "variable_names": model_cfg["variable_names"],
                "samples": samples, "true_set": true_set, "params_mask": params_mask,
                "n_rows": n_rows, "global_indices": global_indices, "outdir": outdir,
            })

    # ── Phase 2: ODE integration ───────────────────────────────────────────────
    for case in cases:
        print(f"Sampling: {case['label']} / {case['cfg_name']}")
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

    # ── Phase 3: save predictions and figures ─────────────────────────────────
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

        np.savez(pred_dir / f"{tag}_pred.npz",
                 pred_set=pred_set, true_set=true_set, params_mask=params_mask)
        logging.info(f"[saved] {pred_dir / f'{tag}_pred.npz'}")

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
            global_params=GLOBAL_PARAMS, parameter_mask=params_mask,
            variable_names=variable_names, skip_log_gamma=args.skip_log_gamma,
        )
        if not metrics_glob_df.empty:
            metrics_glob_df.to_csv(outdir / f"{tag}_metrics_global.csv")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
