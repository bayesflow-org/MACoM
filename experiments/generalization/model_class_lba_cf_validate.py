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
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors, lba_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun, lba_link_fun
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_mc_colors

from ..configs import MODELS as _BASE_MODELS, LBA_CONFIG
from ..utils import add_global_params, infer_free_fixed_intrinsics, build_encoder_input_dim


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
    p.add_argument("--pred_dir", type=str, default="./outputs/data/generalization/model_class_lba/cf/")

    # Which model embedding to use during inference (default: RDM=1)
    p.add_argument("--eval_model_id", type=int, default=1,
                   help="Model ID to condition on (0=DDM, 1=RDM, 2=CDM). "
                        "Ignored when --all_model_ids is set.")
    p.add_argument("--all_model_ids", action="store_true", default=False,
                   help="Evaluate LBA under each of the 3 known model IDs.")

    # Benchmark cases
    p.add_argument("--include_full", action="store_true", default=False)

    # Validation settings
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--solver", type=str, default="rk4", choices=["euler", "midpoint", "rk4"])
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")

    # Architecture (must match the checkpoint)
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

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
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

    # Standalone LBA family for simulation only
    lba_family = ModelFamily(
        model=LBA(),
        name="LBA",
        prior=lba_priors(),
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
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()

    colors = cogformer_mc_colors()

    benchmarks = dict(_LBA_WITH_PI["benchmark_configs"])
    if not args.include_full:
        benchmarks.pop("full", None)

    model_family_config = {
        "max_num_regressors": args.max_num_regressors,
        "max_num_categories": args.max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
    }

    eval_ids = (
        list(model_class.model_index.values())
        if args.all_model_ids
        else [args.eval_model_id]
    )
    id_to_name = {v: k for k, v in model_class.model_index.items()}

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for cfg_name, design_config in benchmarks.items():
        print(f"Generating LBA data: {cfg_name}")

        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            design_config, LBA_LOCAL_PARAMS, _LBA_WITH_PI["default_fixed_values"]
        )

        test_samples = lba_family.batch_sample(
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
            **model_family_config,
        )

        pm_global, mask_global = lift_lba_to_global(
            test_samples["param_matrices"],
            test_samples["param_masks"],
        )
        test_samples["param_matrices"] = pm_global
        test_samples["param_masks"] = mask_global
        test_samples["n_params"] = N_GLOBAL_PARAMS

        n_rows = pm_global.shape[1] // N_GLOBAL_PARAMS
        true_set_full = pm_global.reshape(args.batch_size, n_rows, N_GLOBAL_PARAMS)
        mask_full = mask_global.reshape(args.batch_size, n_rows, N_GLOBAL_PARAMS)

        true_set = true_set_full[:, :, LBA_TO_GLOBAL]
        params_mask = mask_full[0][:, LBA_TO_GLOBAL]

        for model_id in eval_ids:
            id_label = id_to_name[model_id]
            print(f"  Running inference with model_id={model_id} ({id_label})")

            test_samples_copy = dict(test_samples)
            test_samples_copy["model_ids"] = np.full(args.batch_size, model_id, dtype=np.int64)

            adapted = adapter.adapt(test_samples_copy, device=device)

            pred_set = cogformer.sample(
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                steps=args.num_sample_steps,
                num_samples=args.num_samples,
                solver=args.solver,
                model_ids=adapted["model_ids"],
            )
            pred_set = pred_set.reshape(
                args.batch_size, args.num_samples, n_rows, N_GLOBAL_PARAMS
            )[:, :, :, LBA_TO_GLOBAL]

            tag = f"lba_{cfg_name}_mid{model_id}"
            np.savez(
                pred_dir / f"{tag}_cf_pred.npz",
                pred_set=pred_set,
                true_set=true_set,
                params_mask=params_mask,
            )
            logging.info(f"[saved] {pred_dir / f'{tag}_cf_pred.npz'}")

            outdir = Path(args.outdir) / cfg_name / f"mid_{id_label.lower()}"
            outdir.mkdir(parents=True, exist_ok=True)

            variable_names = _LBA_WITH_PI["variable_names"]
            intrinsic_params = LBA_LOCAL_PARAMS

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

            coverage_fig = adaptive_coverage(
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
            coverage_fig.savefig(outdir / f"{tag}_coverage.pdf", bbox_inches="tight")
            plt.close(coverage_fig)

            ecdf_fig = adaptive_ecdf(
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
            ecdf_fig.savefig(outdir / f"{tag}_ecdf.pdf", bbox_inches="tight")
            plt.close(ecdf_fig)

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
