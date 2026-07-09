import torch
try:
    import wandb
except Exception:
    wandb = None
import argparse

from cogformer.simulator import ModelFamily
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from ..utils import (add_global_params, strip_global_params, build_encoder_input_dim,
                     add_architecture_args, add_training_args)
from ..configs import GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..model_family.model_family_cf_train import CogFormerTrainer, MODELS
from ..timing import RunTimer


def parse_args():
    parser = argparse.ArgumentParser(
        description="CogFormer ablation studies — model family (single SSM)."
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    add_architecture_args(parser)
    parser.add_argument("--seed_dim", type=int, default=64)

    # Ablation-specific architecture args
    parser.add_argument(
        "--decoder_layer_design",
        type=str,
        default="mixed_attention",
        choices=["mixed_attention", "self_attention", "cross_attention"],
        help="Decoder attention layer design. (default: mixed_attention)",
    )
    parser.add_argument(
        "--mab_first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For mixed_attention: whether cross-attention (MAB) precedes self-attention (SAB). "
             "(default: True)",
    )
    parser.add_argument(
        "--time_freq_trainable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether Fourier frequencies in the time embedding are learned. (default: True)",
    )
    parser.add_argument(
        "--block_film",
        action="store_true",
        default=False,
        help="Apply per-block time conditioning after every decoder attention block, in "
             "addition to the output head. Mechanism set by --block_conditioning. (default: False)",
    )
    parser.add_argument(
        "--output_conditioning",
        type=str,
        default="concat",
        choices=["concat", "film"],
        help="Output-head time conditioning: 'concat' (small MLP over [hidden; time]) or "
             "'film' (feature-wise affine modulation). (default: concat)",
    )
    parser.add_argument(
        "--block_conditioning",
        type=str,
        default="concat",
        choices=["concat", "film"],
        help="Per-block time conditioning mechanism when --block_film is set: 'concat' (residual "
             "small MLP over [hidden; time]) or 'film' (affine modulation). (default: concat)",
    )

    add_training_args(parser)
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=500)
    parser.add_argument("--val_samples_chunk_size", type=int, default=None,
                        help="Samples per ODE loop during validation; peak GPU memory scales with "
                             "batch_size * val_samples_chunk_size.")
    parser.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--val_epoch_interval", type=int, default=None)
    parser.add_argument("--fm_sample_steps", type=int, default=50)
    parser.add_argument("--fm_num_samples", type=int, default=200)
    parser.add_argument("--fm_solver", type=str, default="midpoint",
                        choices=["euler", "midpoint", "rk4"])

    # Output path overrides
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override the model-default checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default=None,
        help="Override the model-default checkpoint prefix.",
    )
    parser.add_argument(
        "--fig_subdir",
        type=str,
        default=None,
        help="Override the model-default figure subdirectory (under outputs/figures/).",
    )

    return parser.parse_args()


def _build_variant_tag(args) -> str:
    """Construct a short descriptor of the non-default ablation choices."""
    parts = []
    if args.decoder_layer_design == "mixed_attention":
        if not args.mab_first:
            parts.append("sab_first")
    else:
        parts.append(args.decoder_layer_design.replace("_attention", ""))
    if args.block_film:
        parts.append("bfilm" if args.block_conditioning == "film" else "bmlp")
    if args.output_conditioning != "concat":
        parts.append("ofilm")
    if not args.time_freq_trainable:
        parts.append("fixed_freq")
    return ("_" + "_".join(parts)) if parts else ""


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = dict(MODELS[args.model])
    if not args.estimate_global_params:
        model_cfg = strip_global_params(model_cfg)
    if args.val_epoch_interval is not None:
        model_cfg["val_epoch_interval"] = args.val_epoch_interval
    if args.checkpoint_dir is not None:
        model_cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.checkpoint_prefix is not None:
        model_cfg["checkpoint_prefix"] = args.checkpoint_prefix
    if args.fig_subdir is not None:
        model_cfg["fig_subdir"] = args.fig_subdir

    max_num_regressors = args.max_num_regressors
    max_num_categories = args.max_num_categories

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": args.keep_intercept,
        "add_interaction": args.add_interaction,
    }

    train_params_kwargs = {
        "free_intrinsics": model_cfg["free_params"],
        "fixed_intrinsics": model_cfg["fixed_params"],
        "fixed_values": model_cfg["default_fixed_values"],
        "global_params": model_cfg.get("global_params"),
    }
    val_params_kwargs = {
        "free_intrinsics": model_cfg["intrinsic_params"],
        "fixed_intrinsics": [],
        "fixed_values": {},
    }

    train_sample_config = {
        "mask_randomizer_kwargs": train_params_kwargs,
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "fixed_config": False,
    }
    val_sample_config = {
        "mask_randomizer_kwargs": val_params_kwargs,
        "min_num_regressors": 2,
        "num_obs": args.num_obs,
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
        "fm_solver": args.fm_solver,
        "batch_size": args.val_batch_size,
        "val_samples_chunk_size": args.val_samples_chunk_size,
        "model_family_config": model_family_config,
        "val_sample_config": val_sample_config,
        "device": device,
    }

    decoder_layer_kwargs = {}
    if args.decoder_layer_design == "mixed_attention":
        decoder_layer_kwargs = {"mab_first": args.mab_first}

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
        "decoder_layer_design": args.decoder_layer_design,
        "decoder_layer_kwargs": decoder_layer_kwargs,
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
        "time_freq_trainable": args.time_freq_trainable,
        "block_film": args.block_film,
        "output_conditioning": args.output_conditioning,
        "block_conditioning": args.block_conditioning,
    }

    if args.use_wandb:
        wandb.init(
            project=model_cfg["wandb_project"],
            tags=["CogFormer", args.model.upper(), "Ablation"],
            config={**train_config, "cogformer": cogformer_config},
        )

    model_family = ModelFamily(
        model=model_cfg["model_cls"](),
        name=model_cfg["name"],
        prior=model_cfg["prior_fn"](),
        mask_randomizer_kwargs=train_params_kwargs,
    )
    adapter = Adapter()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    cogformer = CogFormer(**cogformer_config).to(device).train()
    cogformer = torch.compile(cogformer, dynamic=True, mode=args.compile_mode)

    trainer = CogFormerTrainer(
        model_family=model_family,
        model_cfg=model_cfg,
        adapter=adapter,
        net=cogformer,
        use_wandb=args.use_wandb,
    )

    ckpt_prefix = model_cfg["checkpoint_prefix"]
    variant_tag = _build_variant_tag(args)

    checkpoint_path = (
        f"{ckpt_prefix}{variant_tag}"
        f"_el{cogformer_config['encoder_num_layers']}"
        f"_dl{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_o{val_sample_config['num_obs']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )
    fig_path = checkpoint_path.replace(".pt", "_test.pdf")

    is_dry_run = args.dry_run_epochs is not None
    total_epochs = train_config["epochs"]
    if is_dry_run:
        train_config["epochs"] = args.dry_run_epochs

    with RunTimer(f"abl_mf_{args.model}{variant_tag}", dry_run=is_dry_run,
                  total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
        trainer.train(
            train_config=train_config,
            val_config=val_config,
            checkpoint_path=checkpoint_path,
            fig_path=fig_path,
        )
    if args.use_wandb:
        wandb.finish()
