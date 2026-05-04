import torch
import wandb
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
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params, strip_global_params, build_encoder_input_dim
from ..timing import RunTimer


# CF train uses global params (pi) and adds script-specific train keys
def _build_train_config(base: dict, val_epoch_interval: int, wandb_project: str,
                        checkpoint_dir: str, checkpoint_prefix: str, fig_subdir: str) -> dict:
    cfg = add_global_params(base, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    cfg.update({
        "val_epoch_interval": val_epoch_interval,
        "wandb_project": wandb_project,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_prefix": checkpoint_prefix,
        "fig_subdir": fig_subdir,
    })
    return cfg


MODELS = {
    "ddm": _build_train_config(
        _BASE_MODELS["ddm"],
        val_epoch_interval=2000,
        wandb_project="cogformer-fm-ddm",
        checkpoint_dir="./outputs/checkpoints/model_family/cf/ddm/",
        checkpoint_prefix="cogformer_ddm_mf",
        fig_subdir="model_family/cf/ddm",
    ),
    "cdm": _build_train_config(
        _BASE_MODELS["cdm"],
        val_epoch_interval=2000,
        wandb_project="cogformer-fm-cdm",
        checkpoint_dir="./outputs/checkpoints/model_family/cf/cdm/",
        checkpoint_prefix="cogformer_cdm_mf",
        fig_subdir="model_family/cf/cdm",
    ),
    "rdm": _build_train_config(
        _BASE_MODELS["rdm"],
        val_epoch_interval=2000,
        wandb_project="cogformer-fm-rdm",
        checkpoint_dir="./outputs/checkpoints/model_family/cf/rdm/",
        checkpoint_prefix="cogformer_rdm_mf",
        fig_subdir="model_family/cf/rdm",
    ),
}


class CogFormerTrainer:
    def __init__(
        self,
        net,
        model_cfg,
        model_family=None,
        adapter=None,
        use_wandb=False,
    ):
        self.net = net
        self.model_cfg = model_cfg
        self.adapter = adapter
        self.use_wandb = use_wandb
        if model_family is not None:
            self.model_family = model_family
        else:
            self.model_family = ModelFamily(
                name=model_cfg["name"],
                model=model_cfg["model_cls"](),
                prior=model_cfg["prior_fn"](),
            )

    def train(
        self,
        train_config,
        val_config,
        checkpoint_path="cogformer_fm.pt",
        fig_path="fig.pdf",
    ):
        global_step = 0
        optimizer = AdamW(self.net.parameters(), lr=train_config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])
        val_interval = self.model_cfg["val_epoch_interval"]

        for epoch in range(train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {epoch + 1}/{train_config['epochs']}",
                miniters=100,
            )
            for _ in range(train_config["steps_per_epoch"]):
                loss, current_lr = self.train_step(
                    config=train_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                if self.use_wandb:
                    wandb.log(
                        {"train/loss": loss, "opt/lr": current_lr, "epoch": epoch + 1},
                        step=global_step,
                    )
                global_step += 1
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

            if (epoch + 1) % val_interval == 0:
                self.val_step(val_config, global_step, fig_path)

            scheduler.step()
            pbar.close()

        checkpoint_dir = Path(self.model_cfg["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), checkpoint_dir / checkpoint_path)

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
        )
        adapted = self.adapter.adapt(
            train_samples,
            device=config["device"],
        )
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
        max_num_categories = config["model_family_config"]["max_num_categories"]
        base = Path(f"./outputs/figures/{fig_subdir}")

        recovery_dir = base / "recovery"
        posterior_dir = base / "test_posterior"
        coverage_dir = base / "coverage"
        metrics_dir = base / "metrics"
        for d in (recovery_dir, posterior_dir, coverage_dir, metrics_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.net.eval()

        for tag, design_config in self.model_cfg["val_scenarios"]:
            test_samples = self.model_family.batch_sample(
                **config["model_family_config"],
                **config["val_sample_config"],
                batch_size=config["batch_size"],
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=link_fn(),
            )
            adapted = self.adapter.adapt(
                test_samples,
                device=config["device"],
            )

            true_set = adapted["param_matrices"].detach().cpu().numpy()
            params_mask = adapted["param_masks"].detach().cpu().numpy()

            n_cols = len(params)
            n_rows = true_set.shape[1] // n_cols
            true_set = true_set.reshape(config["batch_size"], n_rows, n_cols)

            chunk_size = config.get("val_sample_chunk_size", config["batch_size"])
            pred_chunks = []
            for c_start in range(0, config["batch_size"], chunk_size):
                c_end = min(c_start + chunk_size, config["batch_size"])
                chunk = self.net.sample(
                    adapted["input_data"][c_start:c_end],
                    adapted["param_indices"][c_start:c_end],
                    adapted["regressor_indices"][c_start:c_end],
                    adapted["param_masks"][c_start:c_end],
                    steps=config["fm_sample_steps"],
                    num_samples=config["fm_num_samples"],
                    solver=config.get("fm_solver", "euler"),
                )
                pred_chunks.append(chunk)
            pred_set = np.concatenate(pred_chunks, axis=0)
            pred_set = pred_set.reshape(
                config["batch_size"], config["fm_num_samples"], n_rows, n_cols
            )
            params_mask = params_mask.reshape((config["batch_size"], n_rows, n_cols))[0]

            recovery = adaptive_recovery(
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
            out_recovery = recovery_dir / Path(fig_path).with_stem(f"{Path(fig_path).stem}_{tag}")
            recovery.savefig(out_recovery, bbox_inches="tight")

            posterior = adaptive_posterior(
                samples=pred_set[0],
                design_config=design_config,
                intrinsic_params=params,
                max_num_categories=max_num_categories,
                parameter_mask=params_mask,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
                unfold=False,
            )
            out_posterior = posterior_dir / f"{model_name}_benchmark_test_posterior_{tag}.pdf"
            posterior.savefig(out_posterior, bbox_inches="tight")

            coverage = adaptive_coverage(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=params,
                variable_names=param_names,
                max_num_categories=max_num_categories,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out_coverage = coverage_dir / f"{model_name}_benchmark_test_coverage_{tag}.pdf"
            coverage.savefig(out_coverage, bbox_inches="tight")

            metrics_fig = plot_adaptive_metrics(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=params,
                max_num_categories=max_num_categories,
                parameter_mask=params_mask,
                variable_names=param_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out_metrics = metrics_dir / f"{model_name}_benchmark_test_metrics_{tag}.pdf"
            metrics_fig.savefig(out_metrics, bbox_inches="tight")

            if self.use_wandb:
                wandb.log(
                    {
                        f"val/recovery_{tag}": wandb.Image(recovery),
                        f"val/posterior_{tag}": wandb.Image(posterior.fig),
                        f"val/coverage_{tag}": wandb.Image(coverage),
                        f"val/metrics_{tag}": wandb.Image(metrics_fig),
                    },
                    step=global_step,
                )
                plt.close(recovery)
                plt.close(posterior.fig)
                plt.close(coverage)
                plt.close(metrics_fig)

        self.net.train()

    @staticmethod
    def finish():
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Benchmark model to train (ddm, cdm, rdm)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    # Network dimensions
    parser.add_argument("--encoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_seeds", type=int, default=32)
    parser.add_argument("--seed_dim", type=int, default=64)
    parser.add_argument("--time_embedding_dim", type=int, default=32)
    parser.add_argument("--pos_embedding_dim", type=int, default=32)
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)
    # Training
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    # Inference
    parser.add_argument("--fm_sample_steps", type=int, default=50)
    parser.add_argument("--fm_num_samples", type=int, default=200)
    parser.add_argument("--fm_solver", type=str, default="midpoint",
                        choices=["euler", "midpoint", "rk4"],
                        help="ODE solver for flow matching sampling")
    parser.add_argument("--val_sample_chunk_size", type=int, default=100,
                        help="Batch chunk size for val sampling to avoid GPU OOM")
    parser.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include global parameters (e.g. contamination probability pi) "
                             "in estimation. Use --no-estimate-global-params to disable.")
    parser.add_argument("--val_epoch_interval", type=int, default=None,
                        help="Override the model-default validation interval (epochs between val steps).")
    parser.add_argument("--dry_run_epochs", type=int, default=None,
                        help="If set, train for this many epochs only and record extrapolated time.")
    # Design space
    parser.add_argument("--max_num_regressors", type=int, default=5)
    parser.add_argument("--max_num_categories", type=int, default=5)
    parser.add_argument("--keep_intercept", action="store_true", default=True)
    parser.add_argument("--add_interaction", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = MODELS[args.model]
    if not args.estimate_global_params:
        model_cfg = strip_global_params(model_cfg)
    if args.val_epoch_interval is not None:
        model_cfg = dict(model_cfg)
        model_cfg["val_epoch_interval"] = args.val_epoch_interval
    use_wandb = args.use_wandb

    max_num_regressors = args.max_num_regressors
    max_num_categories = args.max_num_categories
    keep_intercept = args.keep_intercept

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
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
    }

    val_config = {
        "fm_sample_steps": args.fm_sample_steps,
        "fm_num_samples": args.fm_num_samples,
        "fm_solver": args.fm_solver,
        "batch_size": args.val_batch_size,
        "val_sample_chunk_size": args.val_sample_chunk_size,
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

    logging.info(
        f"Training {args.model.upper()} with {cogformer_config['decoder_layer_design']} over "
        f"{train_config['epochs']} epochs, {train_config['steps_per_epoch']} steps/epoch, "
        f"batch size {train_config['batch_size']}."
    )

    if use_wandb:
        wandb.init(
            project=model_cfg["wandb_project"],
            tags=["CogFormer", args.model.upper(), "ModelFamily"],
            config={**train_config, **{"cogformer": cogformer_config}},
        )

    model_family = ModelFamily(
        model=model_cfg["model_cls"](),
        name=model_cfg["name"],
        prior=model_cfg["prior_fn"](),
        mask_randomizer_kwargs=train_params_kwargs,
    )
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()

    trainer = CogFormerTrainer(
        model_family=model_family,
        model_cfg=model_cfg,
        adapter=adapter,
        net=cogformer,
        use_wandb=use_wandb,
    )

    ckpt_prefix = model_cfg["checkpoint_prefix"]
    checkpoint_path = (
        f"{ckpt_prefix}"
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
        f"{ckpt_prefix}"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_o{val_sample_config['num_obs']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}_test.pdf"
    )

    is_dry_run = args.dry_run_epochs is not None
    total_epochs = train_config["epochs"]
    if is_dry_run:
        train_config["epochs"] = args.dry_run_epochs

    with RunTimer(f"mf_cf_{args.model}", dry_run=is_dry_run,
                  total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
        trainer.train(
            train_config=train_config,
            val_config=val_config,
            checkpoint_path=checkpoint_path,
            fig_path=fig_path,
        )
    trainer.finish()
