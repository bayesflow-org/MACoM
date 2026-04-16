import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.design_utils import build_parameter_mask, build_design_matrix, build_random_design_config
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.ensemble_metrics import ensemble_metrics

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import MODELS as _BASE_MODELS, ALL_VARIABLE_NAMES as ALL_PARAMS_VARIABLE_NAMES
from utils import infer_free_fixed_intrinsics, build_encoder_input_dim


DEFAULT_FIXED_VALUES = {
    m.upper(): cfg["default_fixed_values"] for m, cfg in _BASE_MODELS.items()
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str,
                   default="./cogformer/experiments/figures/fm/model_class/ensemble/")
    p.add_argument("--n_configs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=2)
    p.add_argument("--max_num_categories", type=int, default=2)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
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
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors()),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors()),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors()),
    }
    link_funs = {"DDM": ddm_link_fun(), "RDM": rdm_link_fun(), "CDM": cdm_link_fun()}
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
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()

    model_names_ordered = list(model_families.keys())
    sampled_model_names = [
        model_names_ordered[i]
        for i in rng.integers(0, len(model_names_ordered), size=args.n_configs)
    ]

    true_list, pred_list, masks, design_configs_out = [], [], [], []

    for i, model_name in enumerate(sampled_model_names):
        logging.info(f"Config {i + 1}/{args.n_configs} ({model_name})")

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

        samples["param_matrices"], samples["param_masks"] = \
            model_class.lift_to_global_space(
                model_name,
                samples["param_matrices"],
                samples["param_masks"],
            )

        adapted = adapter.adapt(samples, device=device)

        pred_set = cogformer.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=args.num_sample_steps,
            num_samples=args.num_samples,
            model_ids=adapted["model_ids"],
        )

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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = ensemble_metrics(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs_out,
        intrinsic_params=model_class.global_param_names,
        max_num_categories=args.max_num_categories,
        parameter_masks=masks,
        variable_names=ALL_PARAMS_VARIABLE_NAMES,
    )
    fig_path = outdir / "model_class_ensemble_metrics.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {fig_path}")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
