import os
import threading
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
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import (add_global_params, strip_global_params, build_encoder_input_dim,
                     chunked_sample, PrefetchQueue, train_step,
                     add_architecture_args, add_training_args)
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
        val_epoch_interval=2500,
        wandb_project="cogformer-fm-ddm",
        checkpoint_dir="./outputs/checkpoints/model_family/cf/ddm/",
        checkpoint_prefix="cogformer_ddm_mf",
        fig_subdir="model_family/cf/ddm",
    ),
    "cdm": _build_train_config(
        _BASE_MODELS["cdm"],
        val_epoch_interval=2500,
        wandb_project="cogformer-fm-cdm",
        checkpoint_dir="./outputs/checkpoints/model_family/cf/cdm/",
        checkpoint_prefix="cogformer_cdm_mf",
        fig_subdir="model_family/cf/cdm",
    ),
    "rdm": _build_train_config(
        _BASE_MODELS["rdm"],
        val_epoch_interval=2500,
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
        self._local = threading.local()
        if model_family is not None:
            self.model_family = model_family
        else:
            self.model_family = ModelFamily(
                name=model_cfg["name"],
                model=model_cfg["model_cls"](),
                prior=model_cfg["prior_fn"](),
            )
        if model_cfg.get("holdout_configs"):
            self.model_family.set_holdout_configs(model_cfg["holdout_configs"])

    def _prepare_val_data(self, val_config):
        """Pre-generate fixed validation datasets for all scenarios once."""
        link_fun = self.model_cfg["link_fn"]()
        n_cols = len(self.model_cfg["intrinsic_params"])
        training_max_cats = val_config["model_family_config"]["max_num_categories"]
        self._val_data = []
        for tag, design_config, scenario_opts in self.model_cfg["val_scenarios"]:
            max_cats = scenario_opts.get("max_num_categories", training_max_cats)
            model_family_cfg = {**val_config["model_family_config"], "max_num_categories": max_cats}
            samples = self.model_family.batch_sample(
                **model_family_cfg,
                **val_config["val_sample_config"],
                batch_size=val_config["batch_size"],
                flatten_param_outputs=True,
                design_config=design_config,
                link_fun=link_fun,
                pad_to_max_num_categories=training_max_cats,
            )
            adapted = self.adapter.adapt(samples, device=val_config["device"])
            true_flat = adapted["param_matrices"].detach().cpu().numpy()
            n_rows = true_flat.shape[1] // n_cols
            true_set = true_flat.reshape(val_config["batch_size"], n_rows, n_cols)
            params_mask = adapted["param_masks"].detach().cpu().numpy()
            params_mask = params_mask.reshape(val_config["batch_size"], n_rows, n_cols)[0]
            self._val_data.append({
                "tag": tag,
                "design_config": design_config,
                "scenario_opts": scenario_opts,
                "adapted": adapted,
                "true_set": true_set,
                "params_mask": params_mask,
                "n_rows": n_rows,
            })

    def train(
        self,
        train_config,
        val_config,
        checkpoint_path="cogformer_fm.pt",
        fig_path="fig.pdf",
    ):
        self._prepare_val_data(val_config)

        global_step = 0
        optimizer = AdamW(self.net.parameters(), lr=train_config["learning_rate"], fused=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=train_config["epochs"])
        val_interval = self.model_cfg["val_epoch_interval"]
        device = train_config["device"]

        with PrefetchQueue(lambda: self._simulate_and_adapt(train_config)) as prefetch:
            for epoch in range(train_config["epochs"]):
                pbar = tqdm(
                    total=train_config["steps_per_epoch"],
                    desc=f"Epoch {epoch + 1}/{train_config['epochs']}",
                    miniters=100,
                )
                for _ in range(train_config["steps_per_epoch"]):
                    adapted_cpu = prefetch.next()
                    adapted = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in adapted_cpu.items()
                    }
                    loss, current_lr = train_step(self.net, adapted, train_config, optimizer, scheduler)
                    if self.use_wandb:
                        wandb.log(
                            {"train/loss": loss, "opt/lr": current_lr, "epoch": epoch + 1},
                            step=global_step,
                        )
                    global_step += 1
                    pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.2e}")
                    pbar.update(1)

                if (epoch + 1) % val_interval == 0:
                    checkpoint_dir = Path(self.model_cfg["checkpoint_dir"])
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.net.state_dict(), checkpoint_dir / checkpoint_path)
                    self.val_step(val_config, global_step, fig_path)

                scheduler.step()
                pbar.close()

        checkpoint_dir = Path(self.model_cfg["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), checkpoint_dir / checkpoint_path)

    def _get_sim_mf(self):
        """Return a per-thread ModelFamily so 3 outer workers don't contend on one pool."""
        if not hasattr(self._local, "mf"):
            n_outer = 3
            n_inner = max(1, (os.cpu_count() or 1) // n_outer)
            mf = ModelFamily(
                name=self.model_cfg["name"],
                model=self.model_cfg["model_cls"](),
                prior=self.model_cfg["prior_fn"](),
                num_workers=n_inner,
            )
            if self.model_cfg.get("holdout_configs"):
                mf.set_holdout_configs(self.model_cfg["holdout_configs"])
            self._local.mf = mf
        return self._local.mf

    def _simulate_and_adapt(self, config):
        mf = self._get_sim_mf()
        link_fn = self.model_cfg["link_fn"]
        samples = mf.batch_sample(
            **config["model_family_config"],
            **config["train_sample_config"],
            prior=mf.prior,
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
            link_fun=link_fn(),
            context=self.model_cfg.get("train_context"),
        )
        adapted = self.adapter.adapt(samples, device="cpu")
        return {k: v.pin_memory() if isinstance(v, torch.Tensor) else v
                for k, v in adapted.items()}

    @torch.no_grad()
    def val_step(self, config, global_step, fig_path):
        model_name = self.model_cfg["name"].lower()
        params = self.model_cfg["intrinsic_params"]
        param_names = self.model_cfg["variable_names"]
        fig_subdir = self.model_cfg["fig_subdir"]

        colors = cogformer_fm_colors()
        default_max_cats = config["model_family_config"]["max_num_categories"]
        rec_dir = Path(f"./outputs/figures/{fig_subdir}") / "recovery"
        rec_dir.mkdir(parents=True, exist_ok=True)

        self.net.eval()
        torch.cuda.empty_cache()

        for scenario in self._val_data:
            tag           = scenario["tag"]
            design_config = scenario["design_config"]
            scenario_opts = scenario["scenario_opts"]
            adapted       = scenario["adapted"]
            true_set      = scenario["true_set"]
            params_mask   = scenario["params_mask"]
            n_rows        = scenario["n_rows"]
            n_cols        = len(params)

            pred_set = chunked_sample(
                self.net, adapted,
                num_samples=config["fm_num_samples"],
                steps=config["fm_sample_steps"],
                solver=config.get("fm_solver", "euler"),
                samples_chunk_size=config.get("val_samples_chunk_size"),
            )
            pred_set = pred_set.reshape(config["batch_size"], config["fm_num_samples"], n_rows, n_cols)

            rec_fig = adaptive_recovery(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=params,
                max_num_categories=default_max_cats,
                parameter_mask=params_mask,
                variable_names=param_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )
            out_rec = rec_dir / f"{model_name}_val_recovery_{tag}.pdf"
            rec_fig.savefig(out_rec, bbox_inches="tight")

            if self.use_wandb:
                wandb.log(
                    {f"val/recovery_{tag}": wandb.Image(rec_fig)},
                    step=global_step,
                )
            plt.close(rec_fig)

        self.net.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Benchmark model to train (ddm, cdm, rdm)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    add_architecture_args(parser)
    parser.add_argument("--encoder_embed_dim", type=int, default=64)
    parser.add_argument("--ffn_expansion_factor", type=float, default=4.0,
                        help="FFN intermediate width multiplier.")
    parser.add_argument("--seed_dim", type=int, default=64)
    add_training_args(parser)
    parser.add_argument("--num_obs", type=int, default=500)
    parser.add_argument("--max_num_obs", type=int, default=1000)
    parser.add_argument("--val_batch_size", type=int, default=50)
    parser.add_argument("--fm_sample_steps", type=int, default=100)
    parser.add_argument("--fm_num_samples", type=int, default=200)
    parser.add_argument("--fm_solver", type=str, default="euler",
                        choices=["euler", "midpoint", "rk4"],
                        help="ODE solver for flow matching sampling")
    parser.add_argument("--val_samples_chunk_size", type=int, default=4,
                        help="Samples per ODE loop during validation; peak GPU memory scales with "
                             "batch_size * val_samples_chunk_size.")
    parser.add_argument("--estimate_global_params", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Include global parameters (e.g. contamination probability pi) "
                             "in estimation. Use --no-estimate-global-params to disable.")
    parser.add_argument("--val_epoch_interval", type=int, default=None,
                        help="Override the model-default validation interval (epochs between val steps).")
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
        "fixed_values": model_cfg["default_fixed_values"],
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
        "encoder_embed_dim": args.encoder_embed_dim,
        "ffn_expansion_factor": args.ffn_expansion_factor,
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

    torch.set_float32_matmul_precision("high")

    model_family = ModelFamily(
        model=model_cfg["model_cls"](),
        name=model_cfg["name"],
        prior=model_cfg["prior_fn"](),
        mask_randomizer_kwargs=train_params_kwargs,
        num_workers=max(1, (os.cpu_count() or 1) // 2),
    )
    adapter = Adapter()
    cogformer = CogFormer(**cogformer_config).to(device).train()
    cogformer = torch.compile(cogformer, dynamic=True, mode=args.compile_mode)

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
    if use_wandb:
        wandb.finish()
