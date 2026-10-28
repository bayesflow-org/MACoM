import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun
from cogformer.simulator.design_utils import (DESIGN_GENERATORS, build_parameter_mask, build_design_matrix,
                                              sample_covering_design_configs)
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery, global_ensemble_recovery
from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage, global_ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf, global_ensemble_ecdf
from cogformer.diagnostics.plot.ensemble_metrics import ensemble_metrics, global_ensemble_metrics

from ..configs import (MODELS as _BASE_MODELS, ALL_PARAMS,
                       ALL_VARIABLE_NAMES as ALL_PARAMS_VARIABLE_NAMES)
from ..utils import (add_global_params, infer_free_fixed_intrinsics,
                     build_encoder_input_dim, chunked_sample,
                     add_interaction_prior_args, add_structure_conditioning_arg)

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

# Training designs always include the global params (a pi-less mask is OOD and
# biases recovery downward), so the random configs must carry pi too.
_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}

DEFAULT_FIXED_VALUES = {
    m.upper(): cfg["default_fixed_values"] for m, cfg in _BASE_MODELS.items()
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str,
                   default="./outputs/figures/model_class/cf/ensemble/")
    p.add_argument("--n_configs", type=int, default=12, help="Total random configs (4 per family by default; use 8 for 2-3 per family)")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    # Design-level size of the SAMPLED random configs — independent of the
    # --max_num_* architecture params, which must match the checkpoint and only
    # set the padded mask layout.
    p.add_argument("--num_regressors", type=int, default=2,
                   help="Regressors per random design config (padded up to max_num_regressors)")
    p.add_argument("--num_categories", type=int, default=2,
                   help="Exact category count K for categorical regressors (padded up to max_num_categories)")
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fix_params", type=str, default="",
                   help="Comma-separated intrinsics held at their default fixed value in "
                        "EVERY config (e.g. s_v,s_tau): left out of the sampled designs, "
                        "and their columns dropped from the plots.")
    p.add_argument("--design_generator", choices=DESIGN_GENERATORS, default="mixture",
                   help="Generator for the sampled evaluation designs.")
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint", "rk4"])
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help="Samples per ODE loop; peak GPU memory scales with batch_size * samples_chunk_size.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--free_prob", type=float, default=0.5)
    p.add_argument("--families", type=str, default="DDM,RDM,CDM",
                   help="Comma-separated families to include, in plot order. A single "
                        "family drops the parameter columns that family does not have "
                        "and lays the mask thumbnails out in one row instead of stacked.")
    p.add_argument("--max_coverage_attempts", type=int, default=20000,
                   help="Rejection budget for drawing a covering set of design configs. "
                        "Every config is a genuine random draw; the SET is redrawn until "
                        "its union covers every estimable cell, so the ensemble grid has no "
                        "N/A cells without any deterministic config. Config construction is "
                        "pure combinatorics (~0.1 ms), so rejection is nearly free -- only "
                        "the accepted set is ever simulated. Measured ~7 draws per family.")
    p.add_argument("--metrics", action="store_true", default=False)
    # Architecture (must match checkpoint)
    add_interaction_prior_args(p)
    add_structure_conditioning_arg(p)
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
                   help="Must match the flag the checkpoint was trained with. Use "
                        "--no-self_attn_padding_mask for checkpoints trained mask-off.")
    return p.parse_args()


def build_model_class():
    """The ModelClass the ensemble is defined over. No checkpoint, no GPU.

    Factored out of main so ``replot_ensemble`` can rebuild the parameter space the
    saved arrays are indexed by without repeating the construction -- a second copy
    is how a replot silently ends up labelling columns differently from the run that
    produced them.
    """
    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"]),
    }
    link_funs = {
        "DDM": ddm_link_fun(),
        "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(),
    }
    return model_families, ModelClass(model_families=model_families,
                                      link_funs=link_funs)


# The pi grids put one config per panel, all showing the same parameter, so they share
# one scale and one set of ticks, labelled on the first column and bottom row only:
# per-panel scales make a config whose pi runs to 0.08 look like one running to 0.15.
# Recovery calls this share_limits (its x and y are the same quantity); coverage and
# ECDF share_axes. replot_ensemble imports this so a replot draws the same layout.
GLOBAL_SHARE = {"recovery": {"share_limits": True},
                "coverage": {"share_axes": True},
                "ecdf": {"share_axes": True}}


def ensemble_colors(n_configs):
    """Per-config palette. Deterministic in n_configs, so a replot reproduces it."""
    return sns.husl_palette(n_configs, h=0.05, l=0.82)


@torch.no_grad()
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_families, model_class = build_model_class()
    adapter = Adapter()

    n_global_params = model_class.n_global_params  # 9 (8 intrinsics + pi)

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, args.keep_intercept
    )
    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_structure_conditioning=args.encoder_structure_conditioning,
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
        self_attn_padding_mask=args.self_attn_padding_mask,  # must match training run
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    cogformer = torch.compile(cogformer, dynamic=True)

    # Configs are SEQUENCED by family (not shuffled): all of DDM, then RDM, then CDM,
    # in --families order. The mask-thumbnail legend is laid out in the same order, so a
    # shuffle here would scramble the legend grouping.
    model_names_ordered = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in model_names_ordered if f not in model_families]
    if unknown:
        raise ValueError(f"--families contains unknown families {unknown}; "
                         f"available: {list(model_families)}")
    n_families = len(model_names_ordered)
    base, remainder = divmod(args.n_configs, n_families)
    per_family = {name: base + (1 if j < remainder else 0)
                  for j, name in enumerate(model_names_ordered)}
    sampled_model_names = []
    for name in model_names_ordered:
        sampled_model_names.extend([name] * per_family[name])

    # One covering SET per family, drawn before any simulation. Rejection sampling
    # replaces the old --saturate_last, which made the last config of each family a
    # deterministic saturated config and left the ensemble a mixture of random and
    # constructed designs. Raising free_prob instead does not work: coverage caps near
    # 71% however high it goes, because slopes are drawn from the intercept set, so
    # intercept_prob becomes the binding constraint.
    # --fix_params: a parameter the designs never mention is held at its default
    # fixed value by infer_free_fixed_intrinsics below, so pinning it is just leaving
    # it out of the parameter list the configs are drawn over.
    fix_params = {p.strip() for p in args.fix_params.split(",") if p.strip()}
    covering_configs = {}
    for name in model_names_ordered:
        default_fixed = DEFAULT_FIXED_VALUES[name]
        not_fixable = fix_params - set(default_fixed)
        if not_fixable:
            raise ValueError(f"--fix_params {sorted(not_fixable)} have no default fixed "
                             f"value in {name} ({sorted(default_fixed)})")
        local = [p for p in model_class.model_families[name].intrinsic_params
                 if p not in fix_params]
        covering_configs[name] = sample_covering_design_configs(
            n_configs=per_family[name],
            intrinsic_params=local,
            num_regressors=args.num_regressors,
            free_intrinsics=[p for p in local if p not in default_fixed],
            fixed_intrinsics=[p for p in default_fixed if p not in fix_params],
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
            interaction_policy=args.interaction_policy,
            interaction_prob=args.interaction_prob,
            free_prob=args.free_prob,
            global_params=GLOBAL_PARAMS,
            max_attempts=args.max_coverage_attempts,
        )
        logging.info(f"{name}: drew {per_family[name]} covering design configs")

    true_list, pred_list, masks = [], [], []
    design_configs_out, labels, categorical_regressors_list = [], [], []
    colors = ensemble_colors(args.n_configs)

    # Per-model counter for label numbering: "DDM | 1", "RDM | 1", "DDM | 2", ...
    model_counters = {name: 0 for name in model_names_ordered}

    for i, model_name in enumerate(sampled_model_names):
        model_counters[model_name] += 1
        label = f"{model_name} | {model_counters[model_name]}"
        logging.info(f"Config {i + 1}/{args.n_configs}: {label}")

        mf = model_class.model_families[model_name]
        model_id = model_class.model_index[model_name]
        intrinsic_params_local = mf.intrinsic_params
        link_fun = model_class.link_funs[model_name]
        default_fixed = DEFAULT_FIXED_VALUES[model_name]

        dc = covering_configs[model_name][model_counters[model_name] - 1]

        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            dc, intrinsic_params_local, default_fixed
        )

        samples = mf.batch_sample(
            design_config=dc,
            batch_size=args.batch_size,
            num_obs=args.num_obs,
            mask_randomizer_kwargs={
                "free_intrinsics": free_intr,
                "fixed_intrinsics": fixed_intr,
                "fixed_values": fixed_vals,
            },
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.num_categories,
            min_num_categories=args.num_categories,
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
            flatten_param_outputs=True,
            fixed_config=True,
            link_fun=link_fun,
            pad_to_max_num_categories=args.max_num_categories,
            design_generator=args.design_generator,
        )
        samples["model_ids"] = np.full(args.batch_size, model_id, dtype=np.int64)

        # Lift local param positions to global space before adapting
        samples["param_matrices"], samples["param_masks"] = \
            model_class.lift_to_global_space(
                model_name,
                samples["param_matrices"],
                samples["param_masks"],
            )
        samples["n_params"] = n_global_params

        adapted = adapter.adapt(samples, device=device)

        pred_set = chunked_sample(
            cogformer, adapted, num_samples=args.num_samples,
            steps=args.num_sample_steps, solver=args.solver,
            samples_chunk_size=args.samples_chunk_size,
        )

        # Keep in global 8-param space — no local extraction
        n_rows = adapted["param_matrices"].shape[1] // n_global_params
        true_global = adapted["param_matrices"].detach().cpu().numpy()
        true_global = true_global.reshape(args.batch_size, n_rows, n_global_params)
        pred_global = pred_set.reshape(args.batch_size, args.num_samples, n_rows, n_global_params)
        mask_global = adapted["param_masks"].detach().cpu().numpy()
        mask_global = mask_global.reshape(args.batch_size, n_rows, n_global_params)[0]

        true_list.append(true_global)
        pred_list.append(pred_global)
        masks.append(mask_global)
        design_configs_out.append(dc)
        labels.append(label)
        # Element 0's realized continuous/categorical assignment (plots use mask[0]).
        # Dummy coding drops the base category c_0, so even a K=2 categorical carries
        # a c_1 contrast — labels must come from the discrete mask, not slot counts.
        # Restrict to regressors this design actually declares. The discrete mask is
        # sized to max_num_regressors (the padded encoder budget), so slicing to it
        # reports a continuous/categorical verdict for padding regressors the design
        # never used -- harmless for labels, since those rows are dead, but wrong in
        # the saved manifest, where it reads as "u_3 was categorical" for a two-
        # regressor design.
        design_main_keys = {k for k in dc if k != "1" and ":" not in k}
        disc = np.asarray(samples["discrete_masks"][0][:args.max_num_regressors])
        categorical_regressors_list.append(
            [f"u_{i + 1}" for i, v in enumerate(disc)
             if v == 1.0 and f"u_{i + 1}" in design_main_keys]
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Persist BEFORE plotting. Sampling is the expensive half and previously lived only
    # in the rendered PDFs, so the design configs an ensemble actually used were not
    # recoverable afterwards and the run could not be reproduced or re-plotted.
    # Per-config keys rather than stacked arrays: row counts differ between configs.
    arrays = {}
    for i, (t, q, m) in enumerate(zip(true_list, pred_list, masks)):
        arrays[f"true_{i}"] = t
        arrays[f"pred_{i}"] = q
        arrays[f"mask_{i}"] = m
    np.savez_compressed(
        outdir / "model_class_ensemble_data.npz",
        n_configs=len(true_list),
        labels=np.array(labels, dtype=object),
        families=np.array(sampled_model_names, dtype=object),
        design_configs=json.dumps(design_configs_out),
        categorical_regressors=json.dumps(categorical_regressors_list),
        # Columns are saved unsliced (global space); a replot drops these itself.
        fix_params=json.dumps(sorted(fix_params)),
        **arrays,
    )
    manifest = pd.DataFrame([
        {
            "index": i,
            "family": fam,
            "label": lab,
            "categorical_regressors": ";".join(cats) if cats else "",
            **{f"cells[{k}]": ",".join(v) for k, v in dc.items() if v},
        }
        for i, (fam, lab, dc, cats) in enumerate(
            zip(sampled_model_names, labels, design_configs_out, categorical_regressors_list))
    ])
    manifest.to_csv(outdir / "model_class_ensemble_configs.csv", index=False)
    logging.info(f"[saved] {outdir / 'model_class_ensemble_data.npz'}")
    logging.info(f"[saved] {outdir / 'model_class_ensemble_configs.csv'}")

    # The global param space is the ordered union across families, which places pi
    # BEFORE the family-specific params (v_diff, v_theta) — build labels by name
    # rather than assuming the pi-less ALL_PARAMS ordering.
    global_param_names = model_class.global_param_names
    _label_map = dict(zip(ALL_PARAMS, ALL_PARAMS_VARIABLE_NAMES))
    _label_map.update(dict(zip(GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)))

    # Single-family runs drop the columns that family does not have (e.g. v_diff and
    # v_theta for DDM), and --fix_params the columns it pins; both would otherwise be
    # all-N/A. Everything downstream lives in
    # the global parameter space, so slice the data and the names together.
    if len(model_names_ordered) == 1 or fix_params:
        fam_params = set().union(*(model_families[f].intrinsic_params
                                   for f in model_names_ordered)) - fix_params
        keep_cols = [i for i, p in enumerate(global_param_names) if p in fam_params]
        global_param_names = [global_param_names[i] for i in keep_cols]
        true_list = [t[..., keep_cols] for t in true_list]
        pred_list = [q[..., keep_cols] for q in pred_list]
        masks = [m[:, keep_cols] for m in masks]
        logging.info(f"{','.join(model_names_ordered)}: keeping {len(keep_cols)} parameter columns "
                     f"{global_param_names}")

    variable_names = [_label_map[p] for p in global_param_names]

    # Thumbnail layout: with several families, two per row so each family occupies whole
    # rows (DDM 1-2 / DDM 3-4 / RDM 1-2 / ...). With one family, a single row.
    thumb_per_row = 2 if len(model_names_ordered) > 1 else max(len(labels), 1)

    shared_kwargs = dict(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=global_param_names,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=variable_names,
        max_num_regressors=args.max_num_regressors,
        categorical_regressors_list=categorical_regressors_list,
        global_params=GLOBAL_PARAMS,
    )
    color_kwargs = {"colors": colors, "labels": labels}

    # thumb_per_row applies to the mask-thumbnail legend, which only the non-global
    # plots draw; the global panels use a plain colour legend and metrics has no legend.
    thumb_kw = {"thumb_per_row": thumb_per_row}
    plotters = [
        ("recovery", ensemble_recovery, global_ensemble_recovery, color_kwargs, thumb_kw),
        ("coverage", ensemble_coverage, global_ensemble_coverage, color_kwargs, thumb_kw),
        ("ecdf", ensemble_ecdf, global_ensemble_ecdf, color_kwargs, thumb_kw),
    ]
    if args.metrics:
        plotters.append(("metrics", ensemble_metrics, global_ensemble_metrics, {}, {}))

    for plot_name, plot_fn, global_fn, extra, main_only in plotters:
        logging.info(f"Generating ensemble {plot_name} plot...")
        fig = plot_fn(**shared_kwargs, **extra, **main_only)
        if fig is not None:
            p = outdir / f"model_class_ensemble_{plot_name}.pdf"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            logging.info(f"[saved] {p}")

        # Global params (pi): one panel per config, wrapped at 6 per row, on one
        # shared scale (GLOBAL_SHARE).
        glob_fig = global_fn(**shared_kwargs, **extra, **GLOBAL_SHARE.get(plot_name, {}))
        if glob_fig is not None:
            p = outdir / f"model_class_ensemble_{plot_name}_global.pdf"
            glob_fig.savefig(p, bbox_inches="tight")
            plt.close(glob_fig)
            logging.info(f"[saved] {p}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
