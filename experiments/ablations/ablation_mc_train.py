import torch
import wandb
import argparse
from pathlib import Path

import numpy as np

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from ..utils import add_global_params, build_encoder_input_dim
from ..configs import MODELS as _BASE_MODELS
from ..model_class.model_class_cf_train import CogFormerTrainer as ModelClassTrainer, MODEL_CONFIGS

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CogFormer ablation study — model embedding (ModelClass: DDM + CDM + RDM)."
    )
    parser.add_argument("--use_wandb", action="store_true")

    # Architecture
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=128)
    parser.add_argument("--model_embed_dim", type=int, default=8)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)

    # Ablation-specific
    parser.add_argument(
        "--no_model_embedding",
        action="store_true",
        default=False,
        help="Disable the model-identity embedding, so the network receives no explicit signal "
             "about which SSM generated the data. (default: False)",
    )

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)

    # Training schedule
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--val_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)

    # Inference
    parser.add_argument("--fm_sample_steps", type=int, default=200)
    parser.add_argument("--fm_num_samples", type=int, default=100)

    # Design space
    parser.add_argument("--max_num_regressors", type=int, default=5)
    parser.add_argument("--max_num_categories", type=int, default=5)
    parser.add_argument("--keep_intercept", action="store_true", default=True)
    parser.add_argument("--add_interaction", action="store_true", default=True)

    # Output path overrides
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_prefix", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_num_regressors = args.max_num_regressors
    max_num_categories = args.max_num_categories
    num_models = 3

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
    }
    train_sample_config = {
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "fixed_config": False,
    }

    encoder_input_dim = build_encoder_input_dim(max_num_regressors, max_num_categories, args.keep_intercept)

    train_config = {
        "epochs": args.epochs,
        "batch_size": args.train_batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.lr,
        "gradient_clip_norm": 5.0,
        "device": device,
        "model_family_config": model_family_config,
        "train_sample_config": train_sample_config,
    }
    val_config = {
        "fm_sample_steps": args.fm_sample_steps,
        "fm_num_samples": args.fm_num_samples,
        "batch_size": args.val_batch_size,
        "num_obs": args.val_num_obs,
        "model_family_config": model_family_config,
    }

    # When no_model_embedding is set, num_models=None suppresses the embedding table.
    effective_num_models = None if args.no_model_embedding else num_models

    cogformer_config = {
        "encoder_input_dim": encoder_input_dim,
        "encoder_num_layers": args.encoder_num_layers,
        "decoder_num_layers": args.decoder_num_layers,
        "encoder_num_heads": args.encoder_num_heads,
        "decoder_num_heads": args.decoder_num_heads,
        "num_seeds": args.num_seeds,
        "seed_dim": args.seed_dim,
        "proj_dim": args.proj_dim,
        "dropout": args.dropout,
        "layer_dropout": args.layer_dropout,
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
        "num_models": effective_num_models,
        "model_embed_dim": args.model_embed_dim,
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
    }

    model_families = {
        "DDM": ModelFamily(
            model=DDM(), name="DDM",
            prior=ddm_priors(),
            intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "a", "z", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"],
                "fixed_values": {},
                "global_params": GLOBAL_PARAMS,
            },
        ),
        "RDM": ModelFamily(
            model=RDM(), name="RDM",
            prior=rdm_priors(),
            intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_diff", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"],
                "fixed_values": {},
                "global_params": GLOBAL_PARAMS,
            },
        ),
        "CDM": ModelFamily(
            model=CDM(), name="CDM",
            prior=cdm_priors(),
            intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics": ["v", "v_theta", "a", "tau"],
                "fixed_intrinsics": ["s_v", "s_tau"],
                "fixed_values": {},
                "global_params": GLOBAL_PARAMS,
            },
        ),
    }
    link_funs = {
        "DDM": ddm_link_fun(),
        "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(),
    }

    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()

    if args.use_wandb:
        wandb.init(
            project="cogformer-model-class-ablation",
            tags=["CogFormer", "ModelClass", "Ablation", "ModelEmbedding"],
            config={**train_config, "cogformer": cogformer_config},
        )

    emb_tag = "no_emb" if args.no_model_embedding else f"emb{args.model_embed_dim}"
    ckpt_prefix = args.checkpoint_prefix or "cogformer_mc_abl"
    checkpoint_path = (
        f"{ckpt_prefix}_{emb_tag}"
        f"_el{cogformer_config['encoder_num_layers']}"
        f"_dl{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )
    fig_path = (
        f"mc_abl_{emb_tag}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
    )

    checkpoint_dir = args.checkpoint_dir or "./outputs/checkpoints/ablations/model_embedding/"
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    trainer = ModelClassTrainer(
        net=cogformer,
        model_class=model_class,
        adapter=adapter,
        use_wandb=args.use_wandb,
    )

    trainer.train(
        train_config=train_config,
        val_config=val_config,
        checkpoint_path=checkpoint_path,
        fig_path=fig_path,
    )

    if args.use_wandb:
        trainer.finish()
