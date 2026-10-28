"""Apply a ModelClass CDM CogFormer to real CDM data (Kvam et al., 2019).

Applies a DDM+RDM+CDM-trained ModelClass CogFormer to real CDM data.
Each subject is processed individually (batch_size=1). Posteriors are saved in
global parameter space as (num_subjects, num_samples, n_rows, n_global_params).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from ..utils import add_structure_conditioning_arg
from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, build_encoder_input_dim, chunked_sample
from .data_utils import (
    load_cdm_data,
    build_real_design_matrix,
    build_real_param_mask,
    build_subject_batch,
)


def parse_args():
    p = argparse.ArgumentParser()
    add_structure_conditioning_arg(p)
    p.add_argument("--data", type=str, default="./data/data_application_cdm.csv")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs/data/real_data/cdm/mc/")

    # Design
    p.add_argument("--max_num_regressors", type=int, default=5)
    p.add_argument("--max_num_categories", type=int, default=5)
    p.add_argument("--keep_intercept", action="store_true", default=True)

    # Inference
    p.add_argument("--num_sample_steps", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--samples_chunk_size", type=int, default=1)
    p.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])

    # Architecture (must match training)
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
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    p.add_argument("--num_models", type=int, default=3)
    p.add_argument("--model_embed_dim", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct model class (same as training) to recover global param space
    _MODELS_WITH_PI = {
        m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
        for m, cfg in _BASE_MODELS.items()
        if m != "lba"
    }

    model_families = {
        "DDM": ModelFamily(
            model=DDM(), name="DDM",
            prior=ddm_priors(),
            intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "a", "z", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
                "global_params": GLOBAL_PARAMS,
            },
        ),
        "RDM": ModelFamily(
            model=RDM(), name="RDM",
            prior=rdm_priors(),
            intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_diff", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
                "global_params": GLOBAL_PARAMS,
            },
        ),
        "CDM": ModelFamily(
            model=CDM(), name="CDM",
            prior=cdm_priors(),
            intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_theta", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {},
                "global_params": GLOBAL_PARAMS,
            },
        ),
    }
    link_funs = {"DDM": ddm_link_fun(), "RDM": rdm_link_fun(), "CDM": cdm_link_fun()}
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)

    n_global_params = model_class.n_global_params
    cdm_model_id = model_class.model_index["CDM"]

    cdm_cfg = _MODELS_WITH_PI["cdm"]

    # speed_condition, difficulty, and their interaction are regressed on
    # drift length (v), threshold (a), and ndt (tau); v_theta/s_v/s_tau are intercept-only
    design_config = {
        "1":       ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
        "u_1":     ["v", "a", "tau"],
        "u_2":     ["v", "a", "tau"],
        "u_1:u_2": ["v", "a", "tau"],
    }

    encoder_input_dim = build_encoder_input_dim(
        args.max_num_regressors, args.max_num_categories, args.keep_intercept
    )
    cogformer = CogFormer(
        encoder_structure_conditioning=args.encoder_structure_conditioning,
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
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=args.num_models,
        model_embed_dim=args.model_embed_dim,
    ).to(device)

    state = torch.load(Path(args.checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()

    adapter = Adapter()

    # Build local param mask then lift to global space (shared across all subjects)
    local_mask_flat = build_real_param_mask(
        design_config=design_config,
        intrinsic_params=cdm_cfg["intrinsic_params"],
        max_num_regressors=args.max_num_regressors,
        max_num_categories=args.max_num_categories,
        keep_intercept=args.keep_intercept,
    )
    local_pm = local_mask_flat[np.newaxis]  # (1, max_num_cols * local_n_params)
    _, param_mask_global = model_class.lift_to_global_space("CDM", local_pm, local_pm)
    param_mask_global = param_mask_global[0]  # (max_num_cols * n_global_params,)
    n_rows = len(local_mask_flat) // len(cdm_cfg["intrinsic_params"])

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
            design_matrix=dm, param_mask=param_mask_global,
            n_params=n_global_params,
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.max_num_categories,
            design_config=design_config,
            num_alternatives=2,
            model_id=cdm_model_id,
        )

        adapted = adapter.adapt(batch, device=device)

        pred = chunked_sample(
            cogformer, adapted,
            num_samples=args.num_samples,
            steps=args.num_sample_steps,
            solver=args.solver,
            samples_chunk_size=args.samples_chunk_size,
        )
        # pred: (1, num_samples, n_rows * n_global_params) → (num_samples, n_rows, n_global_params)
        pred = pred.reshape(args.num_samples, n_rows, n_global_params)
        all_posteriors.append(pred)

    posteriors = np.stack(all_posteriors, axis=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "cdm_mc_posteriors.npz"
    np.savez(
        out_path,
        posteriors=posteriors,
        subject_ids=np.array(subject_ids),
        global_param_names=np.array(model_class.global_param_names),
        cdm_intrinsic_params=np.array(cdm_cfg["intrinsic_params"]),
        cdm_variable_names=np.array(cdm_cfg["variable_names"]),
        cdm_local_to_global=np.array(model_class.local_to_global["CDM"]),
        design_config_keys=np.array(list(design_config.keys())),
    )
    logging.info(f"[saved] {out_path}  shape={posteriors.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
