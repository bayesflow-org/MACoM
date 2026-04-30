import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from cogformer.simulator import ModelFamily
from cogformer.simulator.design_utils import build_parameter_mask, build_design_matrix, build_random_design_config
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.ensemble_metrics import ensemble_metrics

from ..configs import MODELS as _BASE_MODELS
from ..utils import infer_free_fixed_intrinsics, build_encoder_input_dim


_OUTDIRS = {
    "ddm": "./outputs/figures/model_family/cf/ddm/ensemble/",
    "cdm": "./outputs/figures/model_family/cf/cdm/ensemble/",
    "rdm": "./outputs/figures/model_family/cf/rdm/ensemble/",
}

MODELS = {}
for _key, _base in _BASE_MODELS.items():
    _cfg = dict(_base)
    _cfg["outdir"] = _OUTDIRS[_key]
    MODELS[_key] = _cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (defaults to model-specific path)")
    p.add_argument("--n_configs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--add_interaction", action="store_true", default=True)
    p.add_argument("--num_sample_steps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=200)
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
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = MODELS[args.model]
    model_name = args.model
    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]
    free_params = model_cfg["free_params"]
    fixed_params = model_cfg["fixed_params"]
    default_fixed_values = model_cfg["default_fixed_values"]
    link_fn = model_cfg["link_fn"]
    n_cols = len(intrinsic_params)

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
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()

    design_configs = [
        build_random_design_config(
            intrinsic_params=intrinsic_params,
            num_regressors=args.max_num_regressors,
            free_intrinsics=free_params,
            fixed_intrinsics=fixed_params,
            keep_intercept=args.keep_intercept,
            add_interaction=args.add_interaction,
        )
        for _ in range(args.n_configs)
    ]

    true_list, pred_list, masks = [], [], []

    for i, dc in enumerate(design_configs):
        logging.info(f"Config {i + 1}/{args.n_configs}")
        free_intr, fixed_intr, fixed_vals = infer_free_fixed_intrinsics(
            dc, intrinsic_params, default_fixed_values
        )
        samples = model_family.batch_sample(
            **model_family_config,
            mask_randomizer_kwargs={
                "free_intrinsics": free_intr,
                "fixed_intrinsics": fixed_intr,
                "fixed_values": fixed_vals,
            },
            min_num_regressors=0,
            fixed_config=True,
            batch_size=args.batch_size,
            flatten_param_outputs=True,
            design_config=dc,
            link_fun=link_fn(),
        )
        adapted = adapter.adapt(samples)
        for k, v in adapted.items():
            if torch.is_tensor(v):
                adapted[k] = v.to(device)

        true = adapted["param_matrices"].detach().cpu().numpy()
        n_rows = true.shape[1] // n_cols
        true = true.reshape(args.batch_size, n_rows, n_cols)

        pred = cogformer.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=args.num_sample_steps,
            num_samples=args.num_samples,
        )
        pred = pred.reshape(args.batch_size, args.num_samples, n_rows, n_cols)

        mask = adapted["param_masks"].detach().cpu().numpy()
        mask = mask.reshape((args.batch_size, n_rows, n_cols))[0]

        true_list.append(true)
        pred_list.append(pred)
        masks.append(mask)

    outdir = Path(args.outdir) if args.outdir else Path(model_cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    fig = ensemble_metrics(
        true_list=true_list,
        pred_list=pred_list,
        design_configs=design_configs,
        intrinsic_params=intrinsic_params,
        max_num_categories=model_family_config["max_num_categories"],
        parameter_masks=masks,
        variable_names=variable_names,
    )
    fig_path = outdir / f"{model_name}_ensemble_metrics.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {fig_path}")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
