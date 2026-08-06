import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun
from cogformer.simulator.design_utils import build_parameter_mask, build_design_matrix, build_random_design_config
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery, global_ensemble_recovery
from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage, global_ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf, global_ensemble_ecdf
from cogformer.diagnostics.plot.ensemble_metrics import ensemble_metrics, global_ensemble_metrics

from ..configs import (MODELS as _BASE_MODELS, ALL_PARAMS,
                       ALL_VARIABLE_NAMES as ALL_PARAMS_VARIABLE_NAMES)
from ..utils import (add_global_params, infer_free_fixed_intrinsics,
                     build_encoder_input_dim, chunked_sample)

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


def build_gap_filling_design_config(
    prev_configs: list[dict],
    intrinsic_params: list[str],
    num_regressors: int,
    keep_intercept: bool,
    add_interaction: bool,
    global_params: list[str] | None = None,
) -> dict:
    """Design config that covers every (block, parameter) cell the previous configs missed.

    The ensemble grid keeps a row if ANY config activates it, so a cell goes N/A whenever
    one config opens a row that no other config covers. Sampling configs independently
    cannot guarantee coverage; this closes the union deterministically.

    Every legal cell is filled, not just the gaps ("and then some"): the intercept carries
    all parameters -- which is also what makes any slope legal, since
    ``build_random_design_config`` only permits slopes for parameters present in the
    intercept -- and each slope block carries every non-global parameter. Global params
    (e.g. pi) stay intercept-only, matching the sampler.

    prev_configs is accepted so the intent stays legible and so this can be tightened to a
    true complement later; the saturated config is a superset of any complement.
    """
    global_set = set(global_params or ())
    config: dict[str, list[str]] = {}
    if keep_intercept:
        config["1"] = list(intrinsic_params)

    main_keys = [f"u_{r + 1}" for r in range(num_regressors)]
    slope_params = [p for p in intrinsic_params if p not in global_set]
    for key in main_keys:
        config[key] = list(slope_params)

    if add_interaction:
        for i in range(len(main_keys)):
            for j in range(i + 1, len(main_keys)):
                config[f"{main_keys[i]}:{main_keys[j]}"] = list(slope_params)
    return config


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
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--samples_chunk_size", type=int, default=None,
                   help="Samples per ODE loop; peak GPU memory scales with batch_size * samples_chunk_size.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--free_prob", type=float, default=0.5)
    p.add_argument("--families", type=str, default="DDM,RDM,CDM",
                   help="Comma-separated families to include, in plot order. A single "
                        "family drops the parameter columns that family does not have "
                        "and lays the mask thumbnails out in one row instead of stacked.")
    p.add_argument("--saturate_last", action=argparse.BooleanOptionalAction, default=True,
                   help="Make the last config of each family cover every (block, parameter) "
                        "cell the earlier ones missed, so the combined grid has no N/A cells. "
                        "Pass --no-saturate_last to sample all configs randomly.")
    p.add_argument("--metrics", action="store_true", default=False)
    # Architecture (must match checkpoint)
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
                   help="Must match the flag the checkpoint was trained with. Use "
                        "--no-self_attn_padding_mask for checkpoints trained mask-off.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()

    n_global_params = model_class.n_global_params  # 9 (8 intrinsics + pi)

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

    true_list, pred_list, masks = [], [], []
    design_configs_out, labels, categorical_regressors_list = [], [], []
    colors = sns.husl_palette(args.n_configs, h=0.05, l=0.82)

    # Per-model counter for label numbering: "DDM | 1", "RDM | 1", "DDM | 2", ...
    model_counters = {name: 0 for name in model_names_ordered}
    configs_by_family = {name: [] for name in model_names_ordered}

    for i, model_name in enumerate(sampled_model_names):
        model_counters[model_name] += 1
        label = f"{model_name} | {model_counters[model_name]}"
        logging.info(f"Config {i + 1}/{args.n_configs}: {label}")

        mf = model_class.model_families[model_name]
        model_id = model_class.model_index[model_name]
        intrinsic_params_local = mf.intrinsic_params
        link_fun = link_funs[model_name]
        default_fixed = DEFAULT_FIXED_VALUES[model_name]

        is_last_of_family = model_counters[model_name] == per_family[model_name]
        if args.saturate_last and is_last_of_family:
            dc = build_gap_filling_design_config(
                prev_configs=configs_by_family[model_name],
                intrinsic_params=intrinsic_params_local,
                num_regressors=args.num_regressors,
                keep_intercept=args.keep_intercept,
                add_interaction=args.add_interaction,
                global_params=GLOBAL_PARAMS,
            )
        else:
            dc = build_random_design_config(
                intrinsic_params=intrinsic_params_local,
                num_regressors=args.num_regressors,
                free_intrinsics=[p for p in intrinsic_params_local if p not in default_fixed],
                fixed_intrinsics=list(default_fixed.keys()),
                keep_intercept=args.keep_intercept,
                add_interaction=args.add_interaction,
                free_prob=args.free_prob,
                global_params=GLOBAL_PARAMS,
            )
        configs_by_family[model_name].append(dc)

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
            steps=args.num_sample_steps,
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
        disc = np.asarray(samples["discrete_masks"][0][:args.max_num_regressors])
        categorical_regressors_list.append(
            [f"u_{i + 1}" for i, v in enumerate(disc) if v == 1.0]
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The global param space is the ordered union across families, which places pi
    # BEFORE the family-specific params (v_diff, v_theta) — build labels by name
    # rather than assuming the pi-less ALL_PARAMS ordering.
    global_param_names = model_class.global_param_names
    _label_map = dict(zip(ALL_PARAMS, ALL_PARAMS_VARIABLE_NAMES))
    _label_map.update(dict(zip(GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)))

    # Single-family runs drop the columns that family does not have (e.g. v_diff and
    # v_theta for DDM), which would otherwise be all-N/A. Everything downstream lives in
    # the global parameter space, so slice the data and the names together.
    if len(model_names_ordered) == 1:
        only = model_names_ordered[0]
        fam_params = set(model_families[only].intrinsic_params)
        keep_cols = [i for i, p in enumerate(global_param_names) if p in fam_params]
        global_param_names = [global_param_names[i] for i in keep_cols]
        true_list = [t[..., keep_cols] for t in true_list]
        pred_list = [q[..., keep_cols] for q in pred_list]
        masks = [m[:, keep_cols] for m in masks]
        logging.info(f"Single family {only}: keeping {len(keep_cols)} parameter columns "
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

        # Global params (pi): one panel per config, wrapped at 6 per row.
        glob_fig = global_fn(**shared_kwargs, **extra)
        if glob_fig is not None:
            p = outdir / f"model_class_ensemble_{plot_name}_global.pdf"
            glob_fig.savefig(p, bbox_inches="tight")
            plt.close(glob_fig)
            logging.info(f"[saved] {p}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
