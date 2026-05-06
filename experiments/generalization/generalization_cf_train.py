import torch
try:
    import wandb
except Exception:
    wandb = None
import logging
import argparse

from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from cogformer.simulator import ModelFamily
from cogformer.simulator.design_utils import split_design_space
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, strip_global_params, build_encoder_input_dim, chunked_sample
from ..timing import RunTimer

HOLDOUT_TYPES = ["random", "variability", "interaction", "sparsity"]

MODELS = {
    key: add_global_params(base, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for key, base in _BASE_MODELS.items()
}

_CHECKPOINT_DIRS = {
    "ddm": "./outputs/checkpoints/generalization/cf/ddm/",
    "cdm": "./outputs/checkpoints/generalization/cf/cdm/",
    "rdm": "./outputs/checkpoints/generalization/cf/rdm/",
}
_FIG_SUBDIRS = {
    "ddm": "generalization/cf/ddm",
    "cdm": "generalization/cf/cdm",
    "rdm": "generalization/cf/rdm",
}
_CHECKPOINT_PREFIXES = {
    "ddm": "cogformer_mixed_attn",
    "cdm": "cogformer_cdm_mixed_attn",
    "rdm": "cogformer_rdm_mixed_attn",
}


class CogFormerTrainer:
    def __init__(self, net, model_cfg, model_family, adapter, holdout_type,
                 val_epoch_interval=1000, use_wandb=False):
        self.net = net
        self.model_cfg = model_cfg
        self.model_family = model_family
        self.adapter = adapter
        self.holdout_type = holdout_type
        self.val_epoch_interval = val_epoch_interval
        self.use_wandb = use_wandb

    def train(self, train_config, val_config, checkpoint_path="cogformer_gen.pt", fig_path="fig.pdf"):
        global_step = 0
        optimizer = AdamW(self.net.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])

        for epoch in range(train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {epoch + 1}/{train_config['epochs']}",
                miniters=100,
            )
            for _ in range(train_config["steps_per_epoch"]):
                loss, current_lr = self.train_step(train_config, optimizer, scheduler)
                if self.use_wandb:
                    wandb.log(
                        {"train/loss": loss, "opt/lr": current_lr, "epoch": epoch + 1},
                        step=global_step,
                    )
                global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            if (epoch + 1) % self.val_epoch_interval == 0:
                self.val_step(val_config, global_step, fig_path)

            scheduler.step()
            pbar.close()

        checkpoint_dir = Path(self.model_cfg["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), checkpoint_dir / checkpoint_path)
        logging.info(f"[saved] {checkpoint_dir / checkpoint_path}")

    def train_step(self, config, optimizer, scheduler):
        link_fn = self.model_cfg["link_fn"]
        train_samples = self.model_family.batch_sample(
            **config["model_family_config"],
            **config["train_sample_config"],
            prior=self.model_family.prior,
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
            link_fun=link_fn(),
            context=self.model_cfg.get("train_context"),
            free_prob=config.get("train_free_prob"),
            allowed_slope_patterns=config.get("train_slope_patterns"),
        )
        adapted = self.adapter.adapt(train_samples, device=config["device"])
        optimizer.zero_grad()
        device_type = config["device"].type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
            pred_velocity, target_velocity = self.net(
                adapted["param_matrices"][..., None],
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
            )
            L = self.net.compute_loss(pred_velocity, target_velocity, adapted["param_masks"])
        L.backward()
        if config["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), config["gradient_clip_norm"])
        optimizer.step()
        return L.detach().item(), scheduler.get_last_lr()[0]

    @torch.no_grad()
    def val_step(self, config, global_step, fig_path):
        model_name = self.model_cfg["name"].lower()
        params = self.model_cfg["intrinsic_params"]
        param_names = self.model_cfg["variable_names"]
        link_fn = self.model_cfg["link_fn"]
        fig_subdir = self.model_cfg["fig_subdir"]
        colors = cogformer_fm_colors()
        default_max_cats = config["model_family_config"]["max_num_categories"]

        base = Path(f"./outputs/figures/{fig_subdir}/{self.holdout_type}")
        base.mkdir(parents=True, exist_ok=True)

        self.net.eval()

        for tag, design_config, scenario_opts in self.model_cfg["val_scenarios"]:
            max_num_categories = scenario_opts.get("max_num_categories", default_max_cats)
            model_family_cfg = {**config["model_family_config"], "max_num_categories": max_num_categories}
            test_samples = self.model_family.batch_sample(
                **model_family_cfg,
                **config["val_sample_config"],
                batch_size=config["batch_size"],
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=link_fn(),
            )
            adapted = self.adapter.adapt(test_samples, device=config["device"])

            true_set = adapted["param_matrices"].detach().cpu().numpy()
            params_mask = adapted["param_masks"].detach().cpu().numpy()
            n_cols = len(params)
            n_rows = true_set.shape[1] // n_cols
            true_set = true_set.reshape(config["batch_size"], n_rows, n_cols)

            pred_set = chunked_sample(
                self.net, adapted,
                num_samples=config["fm_num_samples"],
                steps=config["fm_sample_steps"],
                solver=config.get("fm_solver", "euler"),
                samples_chunk_size=config.get("val_samples_chunk_size"),
            )
            pred_set = pred_set.reshape(
                config["batch_size"], config["fm_num_samples"], n_rows, n_cols
            )
            params_mask = params_mask.reshape((config["batch_size"], n_rows, n_cols))[0]

            fig = adaptive_recovery(
                true_set, pred_set,
                design_config=design_config,
                intrinsic_params=params,
                max_num_categories=max_num_categories,
                parameter_mask=params_mask,
                variable_names=param_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out = base / Path(fig_path).with_stem(f"{Path(fig_path).stem}_{tag}")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            logging.info(f"[saved] {out}")

            if self.use_wandb:
                wandb.log({f"val/recovery_{tag}": wandb.Image(fig)}, step=global_step)

        self.net.train()

    @staticmethod
    def finish():
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--holdout_type", type=str, required=True, choices=HOLDOUT_TYPES,
                        help="Which design structures to withhold from training.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    # Architecture
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=64)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)

    # Training schedule
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--val_epoch_interval", type=int, default=1000)

    # Design space
    parser.add_argument("--max_num_regressors", type=int, default=5)
    parser.add_argument("--max_num_categories", type=int, default=5)
    parser.add_argument("--keep_intercept", action="store_true", default=True)

    # Inference
    parser.add_argument("--fm_sample_steps", type=int, default=50)
    parser.add_argument("--fm_num_samples", type=int, default=200)
    parser.add_argument("--fm_solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--val_samples_chunk_size", type=int, default=None,
                        help="Samples per ODE loop during validation; peak GPU memory scales with "
                             "batch_size * val_samples_chunk_size.")

    # Holdout-specific
    parser.add_argument("--split_seed", type=int, default=0,
                        help="[random] Seed for split_design_space.")
    parser.add_argument("--split_ratio", type=float, default=0.5,
                        help="[random] Fraction of slope patterns assigned to training set.")
    parser.add_argument("--sparse_free_prob", type=float, default=0.2,
                        help="[sparsity] Fixed free_prob used during training.")

    parser.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry_run_epochs", type=int, default=None,
                        help="If set, train for this many epochs only and record extrapolated time.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = MODELS[args.model]
    if not args.estimate_global_params:
        model_cfg = strip_global_params(model_cfg)

    model_cfg = dict(model_cfg)
    model_cfg["checkpoint_dir"] = _CHECKPOINT_DIRS[args.model]
    model_cfg["checkpoint_prefix"] = _CHECKPOINT_PREFIXES[args.model]
    model_cfg["fig_subdir"] = _FIG_SUBDIRS[args.model]

    max_num_regressors = args.max_num_regressors
    max_num_categories = args.max_num_categories
    keep_intercept = args.keep_intercept

    # ── Holdout-specific training constraints ─────────────────────────────────

    # Base train mask kwargs: all intrinsic params free (unrestricted intercept)
    train_mask_kwargs = {
        "free_intrinsics": model_cfg["free_params"],
        "fixed_intrinsics": model_cfg["fixed_params"],
        "fixed_values": model_cfg["default_fixed_values"],
        "global_params": model_cfg.get("global_params"),
    }
    train_free_prob = None        # None → Beta(2,2) sampling
    train_slope_patterns = None   # None → unconstrained
    add_interaction_train = True

    if args.holdout_type == "random":
        # Restrict slopes to set A; set B is the holdout for validation.
        set_a, _ = split_design_space(
            free_intrinsics=model_cfg["free_params"],
            split_ratio=args.split_ratio,
            seed=args.split_seed,
        )
        train_slope_patterns = set_a
        logging.info(f"[random holdout] Training on {len(set_a)} slope patterns.")

    elif args.holdout_type == "variability":
        # s_v / s_tau never appear as slopes during training.
        # (free_params already excludes them — this is the default, made explicit.)
        train_mask_kwargs["free_intrinsics"] = model_cfg["free_params"]
        logging.info("[variability holdout] Variability params excluded from slopes during training.")

    elif args.holdout_type == "interaction":
        # No interaction terms during training.
        add_interaction_train = False
        logging.info("[interaction holdout] Interaction terms withheld from training.")

    elif args.holdout_type == "sparsity":
        # Fixed low free_prob; model sees only sparse designs during training.
        train_free_prob = args.sparse_free_prob
        logging.info(f"[sparsity holdout] Training with fixed free_prob={train_free_prob}.")

    # ── Configs ───────────────────────────────────────────────────────────────

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "add_interaction": add_interaction_train,
    }

    train_sample_config = {
        "mask_randomizer_kwargs": train_mask_kwargs,
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "fixed_config": False,
    }

    val_params_kwargs = {
        "free_intrinsics": model_cfg["intrinsic_params"],
        "fixed_intrinsics": [],
        "fixed_values": {},
    }
    val_sample_config = {
        "mask_randomizer_kwargs": val_params_kwargs,
        "min_num_regressors": 2,
        "num_obs": args.num_obs,
        "fixed_config": False,
    }

    encoder_input_dim = build_encoder_input_dim(max_num_regressors, max_num_categories, keep_intercept)

    train_config = {
        "epochs": args.epochs,
        "batch_size": args.train_batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.lr,
        "gradient_clip_norm": 5.0,
        "device": device,
        "model_family_config": model_family_config,
        "train_sample_config": train_sample_config,
        "train_free_prob": train_free_prob,
        "train_slope_patterns": train_slope_patterns,
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
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
    }

    ckpt_prefix = model_cfg["checkpoint_prefix"]
    checkpoint_path = (
        f"{ckpt_prefix}_gen_{args.holdout_type}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_o{val_sample_config['num_obs']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )
    fig_path = (
        f"{ckpt_prefix}_gen_{args.holdout_type}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}_test.pdf"
    )

    if args.use_wandb:
        wandb.init(
            project=f"cogformer-gen-{args.model}-{args.holdout_type}",
            tags=["CogFormer", args.model.upper(), "Generalization", args.holdout_type],
            config={**train_config, "cogformer": cogformer_config},
        )

    train_mask_kwargs_for_family = {
        k: v for k, v in train_mask_kwargs.items()
        if k in ("free_intrinsics", "fixed_intrinsics", "fixed_values", "global_params")
    }
    model_family = ModelFamily(
        model=model_cfg["model_cls"](),
        name=model_cfg["name"],
        prior=model_cfg["prior_fn"](),
        mask_randomizer_kwargs=train_mask_kwargs_for_family,
    )
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()

    trainer = CogFormerTrainer(
        net=cogformer,
        model_cfg=model_cfg,
        model_family=model_family,
        adapter=adapter,
        holdout_type=args.holdout_type,
        val_epoch_interval=args.val_epoch_interval,
        use_wandb=args.use_wandb,
    )

    is_dry_run = args.dry_run_epochs is not None
    total_epochs = train_config["epochs"]
    if is_dry_run:
        train_config["epochs"] = args.dry_run_epochs

    with RunTimer(f"gen_cf_{args.model}_{args.holdout_type}", dry_run=is_dry_run,
                  total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
        trainer.train(
            train_config=train_config,
            val_config=val_config,
            checkpoint_path=checkpoint_path,
            fig_path=fig_path,
        )

    if args.use_wandb:
        trainer.finish()
