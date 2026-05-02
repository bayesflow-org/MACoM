"""Apply a ModelFamily CDM CogFormer to real CDM data (Kvam et al., 2019).

Each subject is processed individually with their exact trial count (no NaN padding).
Posteriors are saved as (num_subjects, num_samples, n_rows, n_params) arrays.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import cdm_link_fun

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, build_encoder_input_dim
from .data_utils import (
    load_cdm_data,
    build_real_design_matrix,
    build_real_param_mask,
    build_subject_batch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/data_application_cdm.csv")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/data/real_data/cdm/mf/")

    # Design
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)
    p.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction, default=True,
                   help="Include pi (contamination) in inference.")

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])

    # Architecture (must match training)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=64)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_dropout", type=float, default=0.0)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg = _BASE_MODELS["cdm"]
    if args.estimate_global_params:
        model_cfg = add_global_params(base_cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    else:
        model_cfg = base_cfg

    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names   = model_cfg["variable_names"]
    design_config    = model_cfg["benchmark_configs"]["interaction"]
    n_params         = len(intrinsic_params)

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

    adapter = Adapter()

    param_mask = build_real_param_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_regressors=args.max_num_regressors,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )
    max_num_cols = args.max_num_regressors * (args.max_num_regressors + 1) // 2 * (args.max_num_categories - 1) + 1
    n_rows = max_num_cols

    subject_data, subject_ids = load_cdm_data(args.data)
    logging.info(f"Loaded {len(subject_ids)} subjects from {args.data}")

    all_posteriors = []

    for sid in subject_ids:
        d = subject_data[sid]
        num_obs = len(d["rts"])
        print(f"Subject {sid}: {num_obs} trials")

        regressors = {"u_1": d["u_1"], "u_2": d["u_2"], "u_1:u_2": d["u_1:u_2"]}
        dm = build_real_design_matrix(
            regressors=regressors,
            design_config=design_config,
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.max_num_categories,
            keep_intercept=args.keep_intercept,
        )

        batch = build_subject_batch(
            rts=d["rts"], choices=d["choices"],
            design_matrix=dm, param_mask=param_mask,
            n_params=n_params,
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.max_num_categories,
            num_alternatives=2,
        )

        adapted = adapter.adapt(batch, device=device)

        pred = cogformer.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=args.num_sample_steps,
            num_samples=args.num_samples,
            solver=args.solver,
        )
        # pred: (1, num_samples, n_rows * n_params) → (num_samples, n_rows, n_params)
        pred = pred.reshape(args.num_samples, n_rows, n_params)
        all_posteriors.append(pred)

    posteriors = np.stack(all_posteriors, axis=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "cdm_mf_posteriors.npz"
    np.savez(
        out_path,
        posteriors=posteriors,
        subject_ids=np.array(subject_ids),
        intrinsic_params=np.array(intrinsic_params),
        variable_names=np.array(variable_names),
        design_config_keys=np.array(list(design_config.keys())),
    )
    logging.info(f"[saved] {out_path}  shape={posteriors.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
