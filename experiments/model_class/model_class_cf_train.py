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

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS
from ..utils import (add_global_params, build_encoder_input_dim, chunked_sample,
                     infer_free_fixed_intrinsics, PrefetchQueue, train_step,
                     add_architecture_args, add_training_args)
from ..timing import RunTimer


GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}

# val_step uses the "interaction" benchmark config as the fixed validation design
MODEL_CONFIGS = {
    m.upper(): {
        "intrinsic_params": cfg["intrinsic_params"],
        "variable_names": cfg["variable_names"],
        "default_fixed_values": cfg["default_fixed_values"],
        "val_design_config": cfg["benchmark_configs"]["interaction"],
    }
    for m, cfg in _MODELS_WITH_PI.items()
}

CHECKPOINT_DIR = "./outputs/checkpoints/model_class/cf/"


class CogFormerTrainer:
    def __init__(self, net, model_class, model_class_config, adapter, use_wandb=False, val_epoch_interval=2000,
                 num_sim_workers=3):
        self.net = net
        self.model_class = model_class      # main instance used in val_step
        self.model_class_config = model_class_config  # config dict for per-thread reconstruction
        self.adapter = adapter
        self.use_wandb = use_wandb
        self.val_epoch_interval = val_epoch_interval
        self.num_sim_workers = num_sim_workers
        self._local = threading.local()

    def _get_sim_mc(self):
        """Return a per-thread ModelClass so outer workers don't contend on one instance."""
        if not hasattr(self._local, "mc"):
            n_outer = self.num_sim_workers
            n_inner = max(1, (os.cpu_count() or 1) // n_outer)
            families = {}
            for name, cfg in self.model_class_config["families"].items():
                families[name] = ModelFamily(
                    model=cfg["model_cls"](),
                    name=name,
                    prior=cfg["prior_fn"](),
                    intrinsic_params=cfg["intrinsic_params"],
                    mask_randomizer_kwargs=cfg["mask_randomizer_kwargs"],
                    num_workers=n_inner,
                )
            link_funs = {k: fn() for k, fn in self.model_class_config["link_funs"].items()}
            self._local.mc = ModelClass(model_families=families, link_funs=link_funs)
        return self._local.mc

    def _simulate_and_adapt(self, config):
        mc = self._get_sim_mc()
        samples = mc.batch_sample(
            **config["model_family_config"],
            **config["train_sample_config"],
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
        )
        adapted = self.adapter.adapt(samples, device="cpu")
        return {k: v.pin_memory() if isinstance(v, torch.Tensor) else v
                for k, v in adapted.items()}

    def _prepare_val_data(self, val_config):
        """Pre-generate fixed validation datasets for all model families once."""
        device = val_config["device"]
        model_family_config = val_config["model_family_config"]
        n_global_params = self.model_class.n_global_params
        self._val_data = {}
        for model_name, model_cfg in MODEL_CONFIGS.items():
            mf = self.model_class.model_families[model_name]
            model_id = self.model_class.model_index[model_name]
            link_fun = self.model_class.link_funs[model_name]
            design_config = model_cfg["val_design_config"]
            intrinsic_params = model_cfg["intrinsic_params"]

            free, fixed, fixed_vals = infer_free_fixed_intrinsics(
                design_config, intrinsic_params, model_cfg["default_fixed_values"]
            )
            mask_kwargs = {"free_intrinsics": free, "fixed_intrinsics": fixed, "fixed_values": fixed_vals}

            test_samples = mf.batch_sample(
                design_config=design_config,
                batch_size=val_config["batch_size"],
                num_obs=val_config["num_obs"],
                max_num_regressors=model_family_config["max_num_regressors"],
                max_num_categories=model_family_config["max_num_categories"],
                keep_intercept=model_family_config["keep_intercept"],
                mask_randomizer_kwargs=mask_kwargs,
                flatten_param_outputs=True,
                fixed_config=True,
                link_fun=link_fun,
            )
            test_samples["model_ids"] = np.full(val_config["batch_size"], model_id, dtype=np.int64)
            test_samples["param_matrices"], test_samples["param_masks"] = \
                self.model_class.lift_to_global_space(
                    model_name, test_samples["param_matrices"], test_samples["param_masks"])
            test_samples["n_params"] = n_global_params
            adapted = self.adapter.adapt(test_samples, device=device)

            global_indices = self.model_class.local_to_global[model_name]
            batch_size = val_config["batch_size"]
            true_flat = adapted["param_matrices"].detach().cpu().numpy()
            n_rows = true_flat.shape[1] // n_global_params
            true_set = true_flat.reshape(batch_size, n_rows, n_global_params)[:, :, global_indices]
            params_mask = adapted["param_masks"].detach().cpu().numpy()
            params_mask = params_mask.reshape(batch_size, n_rows, n_global_params)[0][:, global_indices]

            self._val_data[model_name] = {
                "adapted": adapted,
                "true_set": true_set,
                "params_mask": params_mask,
                "n_rows": n_rows,
                "global_indices": global_indices,
                "design_config": design_config,
                "intrinsic_params": intrinsic_params,
                "variable_names": model_cfg["variable_names"],
            }

    def train(self, train_config, val_config, checkpoint_path="cogformer_model_class_fm.pt",
              fig_path="fig", resume=False, resume_start_epoch=0):
        self._prepare_val_data(val_config)
        checkpoint_dir = Path(CHECKPOINT_DIR)
        full_ckpt_path = checkpoint_dir / checkpoint_path
        device = train_config["device"]

        # ── optional resume ────────────────────────────────────────────────────
        start_epoch = 0
        saved_optimizer_state = None
        saved_scheduler_state = None
        if resume and full_ckpt_path.exists():
            ckpt = torch.load(full_ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.net.load_state_dict(ckpt["state_dict"])
                start_epoch = ckpt.get("epoch", resume_start_epoch)
                saved_optimizer_state = ckpt.get("optimizer_state")
                saved_scheduler_state = ckpt.get("scheduler_state")
            else:
                # old format: raw state_dict OrderedDict
                self.net.load_state_dict(ckpt)
                start_epoch = resume_start_epoch
            logging.info(f"[resumed] from epoch {start_epoch}, checkpoint: {full_ckpt_path}")

        # ── optimizer + scheduler (LR resumes at the right point) ─────────────
        optimizer = AdamW(self.net.parameters(), lr=train_config["learning_rate"], fused=True)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=train_config["epochs"], last_epoch=start_epoch - 1
        )
        if saved_optimizer_state is not None:
            optimizer.load_state_dict(saved_optimizer_state)
        if saved_scheduler_state is not None:
            scheduler.load_state_dict(saved_scheduler_state)

        global_step = start_epoch * train_config["steps_per_epoch"]

        with PrefetchQueue(lambda: self._simulate_and_adapt(train_config),
                           n_workers=self.num_sim_workers) as prefetch:
            for epoch in range(start_epoch, train_config["epochs"]):
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

                if (epoch + 1) % self.val_epoch_interval == 0:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "state_dict": self.net.state_dict(),
                        "epoch": epoch + 1,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    }, full_ckpt_path)
                    self.val_step(val_config, global_step, fig_path)

                scheduler.step()
                pbar.close()

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "epoch": train_config["epochs"],
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, full_ckpt_path)
        logging.info(f"[saved] {full_ckpt_path}")

    @torch.no_grad()
    def val_step(self, config, global_step, fig_path):
        self.net.eval()
        torch.cuda.empty_cache()

        n_global_params = self.model_class.n_global_params
        colors = cogformer_fm_colors()
        figures_dir = Path("./outputs/figures/model_class/cf/recovery")
        figures_dir.mkdir(parents=True, exist_ok=True)

        for model_name, val in self._val_data.items():
            adapted       = val["adapted"]
            true_set      = val["true_set"]
            params_mask   = val["params_mask"]
            n_rows        = val["n_rows"]
            global_indices = val["global_indices"]
            design_config  = val["design_config"]
            intrinsic_params = val["intrinsic_params"]
            variable_names   = val["variable_names"]

            batch_size = config["batch_size"]

            pred_set = chunked_sample(
                self.net, adapted,
                num_samples=config["fm_num_samples"],
                steps=config["fm_sample_steps"],
                solver=config.get("fm_solver", "euler"),
                samples_chunk_size=config.get("val_samples_chunk_size"),
            )
            pred_set = pred_set.reshape(
                batch_size, config["fm_num_samples"], n_rows, n_global_params
            )[:, :, :, global_indices]

            recovery_fig = adaptive_recovery(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=config["model_family_config"]["max_num_categories"],
                parameter_mask=params_mask,
                variable_names=variable_names,
                intercept_color=colors["intercept"],
                main_effect_color=colors["main_effect"],
                interaction_color=colors["interaction"],
            )

            fig_name = f"{model_name.lower()}_{fig_path}_recovery.pdf"
            recovery_fig.savefig(figures_dir / fig_name, bbox_inches="tight")
            logging.info(f"[saved] {figures_dir / fig_name}")

            if self.use_wandb:
                wandb.log({f"val/{model_name}/recovery": wandb.Image(recovery_fig)}, step=global_step)

            plt.close(recovery_fig)

        self.net.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint if it exists.")
    parser.add_argument("--resume_start_epoch", type=int, default=0,
                        help="Epoch to resume from when checkpoint lacks epoch metadata (old format).")

    add_architecture_args(parser)
    parser.add_argument("--encoder_embed_dim", type=int, default=64)
    parser.add_argument("--ffn_expansion_factor", type=float, default=4.0)
    parser.add_argument("--seed_dim", type=int, default=128)
    parser.add_argument("--model_embed_dim", type=int, default=8)
    parser.add_argument("--num_sim_workers", type=int, default=None,
                        help="Parallel CPU simulation workers; each gets cpu_count//num_sim_workers numba threads. "
                             "Defaults to max(3, cpu_count // 8) — ~8 numba threads/worker on any machine.")
    add_training_args(parser)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--val_num_obs", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=500)
    parser.add_argument("--val_epoch_interval", type=int, default=2500,
                        help="Epochs between validation + checkpoint saves.")
    parser.add_argument("--fm_sample_steps", type=int, default=100)
    parser.add_argument("--fm_num_samples", type=int, default=100)
    parser.add_argument("--fm_solver", type=str, default="euler",
                        choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--val_samples_chunk_size", type=int, default=4,
                        help="Samples per ODE loop during validation; peak GPU memory scales with "
                             "batch_size * val_samples_chunk_size.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    num_sim_workers = args.num_sim_workers if args.num_sim_workers is not None else max(3, (os.cpu_count() or 24) // 8)

    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_num_regressors = args.max_num_regressors
    max_num_categories = args.max_num_categories
    keep_intercept = args.keep_intercept
    num_models = 3

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept": keep_intercept,
        "add_interaction": args.add_interaction,
    }

    train_sample_config = {
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
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
        "val_samples_chunk_size": args.val_samples_chunk_size,
        "batch_size": args.val_batch_size,
        "num_obs": args.val_num_obs,
        "model_family_config": model_family_config,
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
        "time_embedding_dim": args.time_embedding_dim,
        "pos_embedding_dim": args.pos_embedding_dim,
        "num_models": num_models,
        "model_embed_dim": args.model_embed_dim,
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
    }

    logging.info(
        f"Training ModelClass CogFormer: {train_config['epochs']} epochs, "
        f"{train_config['steps_per_epoch']} steps/epoch, "
        f"batch_size={train_config['batch_size']}, device={device}"
    )

    mask_randomizer_kwargs = {
        "DDM": {"free_intrinsics": ["v", "a", "z", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {}, "global_params": GLOBAL_PARAMS},
        "RDM": {"free_intrinsics": ["v", "v_diff", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {}, "global_params": GLOBAL_PARAMS},
        "CDM": {"free_intrinsics": ["v", "v_theta", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {}, "global_params": GLOBAL_PARAMS},
    }

    # Config dict used by trainer to reconstruct per-thread ModelClass instances
    model_class_config = {
        "families": {
            "DDM": {"model_cls": DDM, "prior_fn": ddm_priors, "intrinsic_params": _MODELS_WITH_PI["ddm"]["intrinsic_params"], "mask_randomizer_kwargs": mask_randomizer_kwargs["DDM"]},
            "RDM": {"model_cls": RDM, "prior_fn": rdm_priors, "intrinsic_params": _MODELS_WITH_PI["rdm"]["intrinsic_params"], "mask_randomizer_kwargs": mask_randomizer_kwargs["RDM"]},
            "CDM": {"model_cls": CDM, "prior_fn": cdm_priors, "intrinsic_params": _MODELS_WITH_PI["cdm"]["intrinsic_params"], "mask_randomizer_kwargs": mask_randomizer_kwargs["CDM"]},
        },
        "link_funs": {"DDM": ddm_link_fun, "RDM": rdm_link_fun, "CDM": cdm_link_fun},
    }

    # Main ModelClass instance used for validation
    model_families = {
        name: ModelFamily(
            model=cfg["model_cls"](), name=name,
            prior=cfg["prior_fn"](),
            intrinsic_params=cfg["intrinsic_params"],
            mask_randomizer_kwargs=cfg["mask_randomizer_kwargs"],
        )
        for name, cfg in model_class_config["families"].items()
    }
    link_funs = {k: fn() for k, fn in model_class_config["link_funs"].items()}
    model_class = ModelClass(model_families=model_families, link_funs=link_funs)

    adapter = Adapter()

    torch.set_float32_matmul_precision("high")
    cogformer = CogFormer(**cogformer_config).to(device).train()
    cogformer = torch.compile(cogformer, dynamic=True, mode=args.compile_mode)

    if args.use_wandb:
        wandb.init(
            project="cogformer-model-class",
            tags=["CogFormer", "ModelClass", "DDM", "RDM", "CDM"],
            config={**train_config, "cogformer": cogformer_config},
        )

    checkpoint_path = (
        f"cogformer_model_class_fm"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_b{train_config['batch_size']}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )

    fig_path = (
        f"model_class_fm"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
    )

    trainer = CogFormerTrainer(
        net=cogformer,
        model_class=model_class,
        model_class_config=model_class_config,
        adapter=adapter,
        use_wandb=args.use_wandb,
        val_epoch_interval=args.val_epoch_interval,
        num_sim_workers=num_sim_workers,
    )

    is_dry_run = args.dry_run_epochs is not None
    total_epochs = train_config["epochs"]
    if is_dry_run:
        train_config["epochs"] = args.dry_run_epochs

    with RunTimer("mc_cf", dry_run=is_dry_run,
                  total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
        trainer.train(
            train_config=train_config,
            val_config=val_config,
            checkpoint_path=checkpoint_path,
            fig_path=fig_path,
            resume=args.resume,
            resume_start_epoch=args.resume_start_epoch,
        )

    if args.use_wandb:
        wandb.finish()
