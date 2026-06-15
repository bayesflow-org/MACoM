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
from cogformer.diagnostics.plot.ensemble_recovery import ensemble_recovery
from cogformer.diagnostics.plot.ensemble_coverage import ensemble_coverage
from cogformer.diagnostics.plot.ensemble_ecdf import ensemble_ecdf
from cogformer.diagnostics.plot.ensemble_metrics import ensemble_metrics

from ..configs import MODELS as _BASE_MODELS, ALL_VARIABLE_NAMES as ALL_PARAMS_VARIABLE_NAMES
from ..utils import infer_free_fixed_intrinsics, build_encoder_input_dim, chunked_sample


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

    def _no_pi(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "pi"}

    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=_no_pi(ddm_priors()),
                           intrinsic_params=_BASE_MODELS["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=_no_pi(rdm_priors()),
                           intrinsic_params=_BASE_MODELS["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=_no_pi(cdm_priors()),
                           intrinsic_params=_BASE_MODELS["cdm"]["intrinsic_params"]),
    }
    link_funs = {
        "DDM": _no_pi(ddm_link_fun()),
        "RDM": _no_pi(rdm_link_fun()),
        "CDM": _no_pi(cdm_link_fun()),
    }
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()

    n_global_params = model_class.n_global_params  # 8

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

    # Build balanced model-name list: floor(n_configs / n_families) per family,
    # distributing any remainder to the first families.
    model_names_ordered = list(model_families.keys())
    n_families = len(model_names_ordered)
    base, remainder = divmod(args.n_configs, n_families)
    sampled_model_names = []
    for j, name in enumerate(model_names_ordered):
        sampled_model_names.extend([name] * (base + (1 if j < remainder else 0)))
    rng.shuffle(sampled_model_names)

    true_list, pred_list, masks = [], [], []
    design_configs_out, labels = [], []
    colors = sns.husl_palette(args.n_configs, h=0.05, l=0.82)

    # Per-model counter for label numbering: "DDM | 1", "RDM | 1", "DDM | 2", ...
    model_counters = {name: 0 for name in model_names_ordered}

    for i, model_name in enumerate(sampled_model_names):
        model_counters[model_name] += 1
        label = f"{model_name} | {model_counters[model_name]}"
        logging.info(f"Config {i + 1}/{args.n_configs}: {label}")

        mf = model_class.model_families[model_name]
        model_id = model_class.model_index[model_name]
        intrinsic_params_local = mf.intrinsic_params
        link_fun = link_funs[model_name]
        default_fixed = DEFAULT_FIXED_VALUES[model_name]

        dc = build_random_design_config(
            intrinsic_params=intrinsic_params_local,
            num_regressors=args.max_num_regressors,
            free_intrinsics=[p for p in intrinsic_params_local if p not in default_fixed],
            fixed_intrinsics=list(default_fixed.keys()),
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
            free_prob=args.free_prob,
        )

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
            max_num_categories=args.max_num_categories,
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
            flatten_param_outputs=True,
            fixed_config=True,
            link_fun=link_fun,
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    global_param_names = model_class.global_param_names
    variable_names = ALL_PARAMS_VARIABLE_NAMES

    logging.info("Generating ensemble recovery plot...")
    fig = ensemble_recovery(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=global_param_names,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=variable_names,
        colors=colors,
        labels=labels,
    )
    p = outdir / "model_class_ensemble_recovery.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {p}")

    logging.info("Generating ensemble coverage plot...")
    fig = ensemble_coverage(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=global_param_names,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=variable_names,
        colors=colors,
        labels=labels,
    )
    p = outdir / "model_class_ensemble_coverage.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {p}")

    logging.info("Generating ensemble ECDF plot...")
    fig = ensemble_ecdf(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=global_param_names,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=variable_names,
        colors=colors,
        labels=labels,
    )
    p = outdir / "model_class_ensemble_ecdf.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {p}")

    if args.metrics:
        logging.info("Generating ensemble metrics heatmap...")
        fig = ensemble_metrics(
            true_list=true_list,
            pred_list=pred_list,
            design_configs=design_configs_out,
            intrinsic_params=global_param_names,
            max_num_categories=args.max_num_categories,
            parameter_masks=masks,
            variable_names=variable_names,
        )
        p = outdir / "model_class_ensemble_metrics.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {p}")

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
