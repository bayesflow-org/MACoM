import os
import torch
import torch.distributed as dist
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
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery, global_recovery
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf, global_ecdf
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS
from ..utils import (add_global_params, build_encoder_input_dim, chunked_sample,
                     infer_free_fixed_intrinsics, ModelClassDataset, make_sim_loader, train_step,
                     add_architecture_args, add_training_args, recommend_sim_workers,
                     add_sim_backend_arg, apply_sim_backend, add_decoder_groups_arg,
                     resolve_val_scenario)
from ..timing import RunTimer


GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

_MODELS_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
    if m != "lba"
}

# val_step validates all six per-model scenarios shared with model_family
# (intercept_only, fixed, regressed, fixed_regressed, interaction, mixed_predictor),
# each with its own category controls (exact-K etc.), so recovery plots are
# homogeneous per batch and match the model_family study.
MODEL_CONFIGS = {
    m.upper(): {
        "intrinsic_params": cfg["intrinsic_params"],
        "variable_names": cfg["variable_names"],
        "default_fixed_values": cfg["default_fixed_values"],
        "val_scenarios": cfg["val_scenarios"],
    }
    for m, cfg in _MODELS_WITH_PI.items()
}

CHECKPOINT_DIR = "./outputs/checkpoints/model_class/cf/"


class CogFormerTrainer:
    def __init__(self, net, model_class, model_class_config, adapter, use_wandb=False, val_epoch_interval=2000,
                 early_val_epoch=None, num_sim_workers=3, world_size=1, rank=0, fig_dir=None,
                 val_scenario_tags=None):
        self.net = net
        self.model_class = model_class      # main instance used in val_step
        self.model_class_config = model_class_config  # config dict for per-thread reconstruction
        self.adapter = adapter
        self.use_wandb = use_wandb
        self.val_epoch_interval = val_epoch_interval
        self.fig_dir = fig_dir
        self.val_scenario_tags = val_scenario_tags
        # One-off early validation + checkpoint (safety check on long runs), in
        # addition to the regular val_epoch_interval schedule.
        self.early_val_epoch = early_val_epoch
        self.num_sim_workers = num_sim_workers
        self.world_size = world_size
        self.rank = rank
        self.is_distributed = world_size > 1
    def _prepare_val_data(self, val_config):
        """Pre-generate fixed validation datasets for all families and scenarios once.

        Mirrors model_family: each scenario pins its own category count (exact-K,
        pinned continuous/categorical mix, etc.) so every sample in a batch shares one
        parameter mask — no padding cells leak into the recovery scatter.
        """
        device = val_config["device"]
        model_family_config = val_config["model_family_config"]
        n_global_params = self.model_class.n_global_params
        batch_size = val_config["batch_size"]
        training_max_cats = model_family_config["max_num_categories"]
        # Run-level validation category controls (CLI); default ceiling = training max,
        # default floor = 2. Scenarios may override or pin exactly (exact_num_categories).
        default_max_cats = val_config.get("val_max_num_categories") or training_max_cats
        default_min_cats = val_config.get("val_min_num_categories") or 2
        self._val_data = []
        for model_name, model_cfg in MODEL_CONFIGS.items():
            mf = self.model_class.model_families[model_name]
            model_id = self.model_class.model_index[model_name]
            link_fun = self.model_class.link_funs[model_name]
            intrinsic_params = model_cfg["intrinsic_params"]
            global_indices = self.model_class.local_to_global[model_name]

            for tag, design_config, scenario_opts in model_cfg["val_scenarios"]:
                if self.val_scenario_tags is not None and tag not in self.val_scenario_tags:
                    continue
                max_cats, min_cats, mix_opts, skip = resolve_val_scenario(
                    scenario_opts, training_max_cats,
                    default_max_cats=default_max_cats, default_min_cats=default_min_cats,
                )
                # Encoder input dim is sized from the training max; a wider scenario
                # would crash val_step, so skip it rather than break the whole run.
                if skip:
                    logging.warning(
                        f"[val] skipping scenario '{model_name}/{tag}': needs "
                        f"max_num_categories={max_cats} > training {training_max_cats}."
                    )
                    continue

                free, fixed, fixed_vals = infer_free_fixed_intrinsics(
                    design_config, intrinsic_params, model_cfg["default_fixed_values"]
                )
                mask_kwargs = {"free_intrinsics": free, "fixed_intrinsics": fixed, "fixed_values": fixed_vals}

                test_samples = mf.batch_sample(
                    design_config=design_config,
                    batch_size=batch_size,
                    num_obs=val_config["num_obs"],
                    max_num_regressors=model_family_config["max_num_regressors"],
                    max_num_categories=max_cats,
                    min_num_categories=min_cats,
                    keep_intercept=model_family_config["keep_intercept"],
                    mask_randomizer_kwargs=mask_kwargs,
                    flatten_param_outputs=True,
                    fixed_config=True,
                    link_fun=link_fun,
                    pad_to_max_num_categories=training_max_cats,
                    **mix_opts,
                )
                test_samples["model_ids"] = np.full(batch_size, model_id, dtype=np.int64)
                test_samples["param_matrices"], test_samples["param_masks"] = \
                    self.model_class.lift_to_global_space(
                        model_name, test_samples["param_matrices"], test_samples["param_masks"])
                test_samples["n_params"] = n_global_params
                adapted = self.adapter.adapt(test_samples, device=device)

                true_flat = adapted["param_matrices"].detach().cpu().numpy()
                n_rows = true_flat.shape[1] // n_global_params
                true_set = true_flat.reshape(batch_size, n_rows, n_global_params)[:, :, global_indices]
                params_mask = adapted["param_masks"].detach().cpu().numpy()
                params_mask = params_mask.reshape(batch_size, n_rows, n_global_params)[0][:, global_indices]

                self._val_data.append({
                    "model_name": model_name,
                    "tag": tag,
                    "adapted": adapted,
                    "true_set": true_set,
                    "params_mask": params_mask,
                    "n_rows": n_rows,
                    "global_indices": global_indices,
                    "design_config": design_config,
                    "intrinsic_params": intrinsic_params,
                    "variable_names": model_cfg["variable_names"],
                    # Contrast-coded regressors for row labeling (e.g. mixed_predictor's u_1)
                    "categorical_regressors": mix_opts.get("categorical_regressors"),
                })

    def train(self, train_config, val_config, checkpoint_path="cogformer_model_class_fm.pt",
              fig_path="fig", resume=False, resume_start_epoch=0):
        if self.rank == 0:
            self._prepare_val_data(val_config)
        if self.is_distributed:
            dist.barrier()

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
            logging.info(f"[rank {self.rank}] resumed from epoch {start_epoch}, checkpoint: {full_ckpt_path}")

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

        dataset = ModelClassDataset(self.model_class_config, train_config,
                                    self.num_sim_workers, self.world_size)
        loader_iter = iter(make_sim_loader(dataset, self.num_sim_workers))
        for epoch in range(start_epoch, train_config["epochs"]):
            pbar = tqdm(
                total=train_config["steps_per_epoch"],
                desc=f"Epoch {epoch + 1}/{train_config['epochs']}",
                miniters=100,
                disable=self.rank != 0,
            )
            for _ in range(train_config["steps_per_epoch"]):
                adapted_cpu = next(loader_iter)
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

            is_early_val = self.early_val_epoch is not None and (epoch + 1) == self.early_val_epoch
            if (epoch + 1) % self.val_epoch_interval == 0 or is_early_val:
                if self.is_distributed:
                    dist.barrier()
                if self.rank == 0:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    net_to_save = getattr(self.net, 'module', self.net)
                    torch.save({
                        "state_dict": net_to_save.state_dict(),
                        "epoch": epoch + 1,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    }, full_ckpt_path)
                    self.val_step(val_config, global_step, fig_path)
                if self.is_distributed:
                    dist.barrier()

            scheduler.step()
            pbar.close()

        if self.is_distributed:
            dist.barrier()
        if self.rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            net_to_save = getattr(self.net, 'module', self.net)
            torch.save({
                "state_dict": net_to_save.state_dict(),
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
        if self.fig_dir is not None:
            figures_dir = Path(self.fig_dir) / "recovery"
        else:
            figures_dir = Path("./outputs/figures/model_class/cf/recovery")
        figures_dir.mkdir(parents=True, exist_ok=True)

        for val in self._val_data:
            model_name    = val["model_name"]
            tag           = val["tag"]
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
                global_params=GLOBAL_PARAMS,
                max_num_regressors=config["model_family_config"]["max_num_regressors"],
                categorical_regressors=val.get("categorical_regressors"),
            )

            fig_name = f"{model_name.lower()}_{tag}_{fig_path}_recovery.pdf"
            recovery_fig.savefig(figures_dir / fig_name, bbox_inches="tight")
            logging.info(f"[saved] {figures_dir / fig_name}")

            if self.use_wandb:
                wandb.log({f"val/{model_name}/{tag}/recovery": wandb.Image(recovery_fig)}, step=global_step)

            plt.close(recovery_fig)

            # Global parameters (e.g. pi) are estimated jointly but plotted separately.
            global_fig = global_recovery(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=config["model_family_config"]["max_num_categories"],
                global_params=GLOBAL_PARAMS,
                parameter_mask=params_mask,
                variable_names=variable_names,
                global_color=colors["intercept"],
                max_num_regressors=config["model_family_config"]["max_num_regressors"],
                categorical_regressors=val.get("categorical_regressors"),
            )
            if global_fig is not None:
                global_name = f"{model_name.lower()}_{tag}_{fig_path}_recovery_global.pdf"
                global_fig.savefig(figures_dir / global_name, bbox_inches="tight")
                logging.info(f"[saved] {figures_dir / global_name}")
                if self.use_wandb:
                    wandb.log({f"val/{model_name}/{tag}/recovery_global": wandb.Image(global_fig)}, step=global_step)
                plt.close(global_fig)

            ecdf_fig = adaptive_ecdf(
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
                global_params=GLOBAL_PARAMS,
                max_num_regressors=config["model_family_config"]["max_num_regressors"],
                categorical_regressors=val.get("categorical_regressors"),
            )
            ecdf_name = f"{model_name.lower()}_{tag}_{fig_path}_ecdf.pdf"
            ecdf_fig.savefig(figures_dir / ecdf_name, bbox_inches="tight")
            logging.info(f"[saved] {figures_dir / ecdf_name}")

            if self.use_wandb:
                wandb.log({f"val/{model_name}/{tag}/ecdf": wandb.Image(ecdf_fig)}, step=global_step)

            plt.close(ecdf_fig)

            global_ecdf_fig = global_ecdf(
                true=true_set,
                pred=pred_set,
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=config["model_family_config"]["max_num_categories"],
                global_params=GLOBAL_PARAMS,
                parameter_mask=params_mask,
                variable_names=variable_names,
                global_color=colors["intercept"],
                max_num_regressors=config["model_family_config"]["max_num_regressors"],
                categorical_regressors=val.get("categorical_regressors"),
            )
            if global_ecdf_fig is not None:
                global_ecdf_name = f"{model_name.lower()}_{tag}_{fig_path}_ecdf_global.pdf"
                global_ecdf_fig.savefig(figures_dir / global_ecdf_name, bbox_inches="tight")
                logging.info(f"[saved] {figures_dir / global_ecdf_name}")
                if self.use_wandb:
                    wandb.log({f"val/{model_name}/{tag}/ecdf_global": wandb.Image(global_ecdf_fig)}, step=global_step)
                plt.close(global_ecdf_fig)

        self.net.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint if it exists.")
    parser.add_argument("--resume_start_epoch", type=int, default=0,
                        help="Epoch to resume from when checkpoint lacks epoch metadata (old format).")

    add_architecture_args(parser)
    parser.add_argument("--self_attn_padding_mask", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Mask padded parameter tokens in decoder self-attention (default). "
                             "Pass --no-self_attn_padding_mask to drop it: ~1.48x faster/step "
                             "(FlashAttention fires) with near-neutral recovery per the A/B ablation. "
                             "Finetune/inference scripts must be run with the matching flag.")
    # BayesFlow-style attention options, ON by default (adopted 2026-09-28 after the A/B;
    # see CogFormer.__init__). Defaults here MUST equal the class defaults: this script
    # passes them explicitly, so a False here would silently train the old architecture
    # while every script that relies on the class default expects the new one.
    parser.add_argument("--gated_residual", action=argparse.BooleanOptionalAction, default=True,
                        help="Zero-initialised learnable scale on every attention/FFN residual branch.")
    parser.add_argument("--pre_pool_ffn", action=argparse.BooleanOptionalAction, default=True,
                        help="Gated row-wise FFN on the encoder tokens before PMA pooling.")
    parser.add_argument("--weight_init", type=str, default="orthogonal", choices=["default", "orthogonal"])
    parser.add_argument("--encoder_embed_dim", type=int, default=64)
    parser.add_argument("--ffn_expansion_factor", type=float, default=4.0)
    parser.add_argument("--seed_dim", type=int, default=128)
    parser.add_argument("--model_embed_dim", type=int, default=8)
    parser.add_argument("--num_sim_workers", type=int, default=None,
                        help="Parallel CPU simulation workers; each gets cpu_count//num_sim_workers numba threads. "
                             "Defaults to recommend_sim_workers(): cpu_count - 4 on numba, <= 6 on gpu.")
    add_sim_backend_arg(parser)
    add_decoder_groups_arg(parser)
    add_training_args(parser)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--val_num_obs", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=100,
                        help="Validation datasets per scenario. Sets the width of the SBC "
                             "confidence band (~1/sqrt(N)); the standalone validate/ensemble "
                             "scripts use 200.")
    parser.add_argument("--val_epoch_interval", type=int, default=2500,
                        help="Epochs between validation + checkpoint saves.")
    parser.add_argument("--early_val_epoch", type=int, default=None,
                        help="Run one extra validation + checkpoint at the end of this epoch "
                             "(safety check on long runs), in addition to --val_epoch_interval.")
    parser.add_argument("--val_max_num_categories", type=int, default=None,
                        help="Category ceiling for validation scenarios that don't pin their own "
                             "'max_num_categories'. Defaults to the training --max_num_categories.")
    parser.add_argument("--val_min_num_categories", type=int, default=None,
                        help="Category floor for validation scenarios (default 2). Set equal to "
                             "--val_max_num_categories to validate at an exact number of categories.")
    parser.add_argument("--fm_sample_steps", type=int, default=100)
    parser.add_argument("--fm_num_samples", type=int, default=100,
                        help="Posterior draws per validation dataset. Sets SBC rank "
                             "resolution: the rank statistic can only take "
                             "fm_num_samples + 1 distinct values.")
    parser.add_argument("--fm_solver", type=str, default="euler",
                        choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--val_samples_chunk_size", type=int, default=4,
                        help="Samples per ODE loop during validation; peak GPU memory scales with "
                             "batch_size * val_samples_chunk_size.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override the default checkpoint output directory (CHECKPOINT_DIR). The "
                             "checkpoint filename does NOT encode max_num_categories/max_num_regressors, "
                             "so use this to keep a non-default design space from clobbering existing checkpoints.")
    parser.add_argument("--fig_dir", type=str, default=None,
                        help="Override the default figure output directory (hardcoded "
                             "./outputs/figures/model_class/cf otherwise). Recovery/ecdf PDFs land in "
                             "<fig_dir>/recovery/. Filenames encode scenario tag and architecture but not "
                             "batch_size/epochs, so use this to keep a non-default run from clobbering "
                             "another run's plots.")
    parser.add_argument("--val_scenario_tags", type=str, default=None,
                        help="Comma-separated subset of val_scenario tags to validate on, applied to every "
                             "model family (e.g. 'intercept_only,fixed_regressed,interaction'). Defaults to "
                             "all scenarios in each family's val_scenarios list.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint_dir is not None:
        CHECKPOINT_DIR = args.checkpoint_dir
    sim_backend = apply_sim_backend(args)
    num_sim_workers = args.num_sim_workers if args.num_sim_workers is not None else recommend_sim_workers(sim_backend)

    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True

    # DDP: torchrun sets LOCAL_RANK when launching with --nproc_per_node > 1
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0
    if is_distributed:
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        world_size = 1
        rank = 0
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
        "design_generator": args.design_generator,
        "coefficient_prior": args.coefficient_prior,
        "coefficient_prior_kwargs": {"order_scale": args.coefficient_order_scale},
    }

    train_sample_config = {
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "num_obs_bucket": args.num_obs_bucket,
        "fixed_config": False,
    }

    encoder_input_dim = build_encoder_input_dim(max_num_regressors, max_num_categories, keep_intercept)

    # Each process handles a local slice of the total batch; DDP averages gradients
    local_batch = max(1, args.train_batch_size // world_size)

    train_config = {
        "epochs": args.epochs,
        "batch_size": local_batch,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.lr,
        "gradient_clip_norm": 5.0,
        "decoder_groups": args.decoder_groups,
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
        "val_max_num_categories": args.val_max_num_categories,
        "val_min_num_categories": args.val_min_num_categories,
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
        "self_attn_padding_mask": args.self_attn_padding_mask,
        "gated_residual": args.gated_residual,
        "pre_pool_ffn": args.pre_pool_ffn,
        "weight_init": args.weight_init,
    }

    logging.info(
        f"[rank {rank}] Training ModelClass CogFormer: {train_config['epochs']} epochs, "
        f"{train_config['steps_per_epoch']} steps/epoch, "
        f"local_batch={local_batch} (total={args.train_batch_size}, world={world_size}), device={device}"
    )

    # fixed_values must carry the default_fixed_values (s_v/s_tau -> 0.0): when a random
    # design excludes an intrinsic, ModelFamily.sample() injects this value instead of
    # link_fun(0) = softplus(0) ~= 0.693. Validation scenarios inject 0 via
    # infer_free_fixed_intrinsics, so an empty dict here trains the net on s_v/s_tau ~= 0.693
    # whenever they are masked out — biasing all params below the diagonal in the
    # fixed/fixed_regressed recovery plots.
    mask_randomizer_kwargs = {
        "DDM": {"free_intrinsics": ["v", "a", "z", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": _MODELS_WITH_PI["ddm"]["default_fixed_values"], "global_params": GLOBAL_PARAMS},
        "RDM": {"free_intrinsics": ["v", "v_diff", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": _MODELS_WITH_PI["rdm"]["default_fixed_values"], "global_params": GLOBAL_PARAMS},
        "CDM": {"free_intrinsics": ["v", "v_theta", "a", "tau"], "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": _MODELS_WITH_PI["cdm"]["default_fixed_values"], "global_params": GLOBAL_PARAMS},
    }

    # Config dict used by trainer to reconstruct per-thread ModelClass instances
    model_class_config = {
        "families": {
            "DDM": {"model_cls": DDM, "prior_fn": ddm_priors, "intrinsic_params": _MODELS_WITH_PI["ddm"]["intrinsic_params"], "mask_randomizer_kwargs": mask_randomizer_kwargs["DDM"]},
            "RDM": {"model_cls": RDM, "prior_fn": rdm_priors, "intrinsic_params": _MODELS_WITH_PI["rdm"]["intrinsic_params"], "mask_randomizer_kwargs": mask_randomizer_kwargs["RDM"]},
            "CDM": {"model_cls": CDM, "prior_fn": cdm_priors, "intrinsic_params": _MODELS_WITH_PI["cdm"]["intrinsic_params"], "mask_randomizer_kwargs": mask_randomizer_kwargs["CDM"]},
        },
        "link_funs": {"DDM": ddm_link_fun, "RDM": rdm_link_fun, "CDM": cdm_link_fun},
        # RDM trains over 2-4 alternatives (random per batch element), matching the
        # model_family training distribution. Validation scenarios keep the default
        # context (2 alternatives), which is in-distribution.
        "train_contexts": {"RDM": RDM.build_random_context},
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
    model_class = ModelClass(model_families=model_families, link_funs=link_funs,
                             train_contexts=model_class_config.get("train_contexts"))

    adapter = Adapter()

    torch.set_float32_matmul_precision("high")
    cogformer = CogFormer(**cogformer_config).to(device).train()
    cogformer = torch.compile(cogformer, dynamic=True, mode=args.compile_mode)
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        cogformer = DDP(cogformer, device_ids=[local_rank])

    use_wandb = args.use_wandb and (rank == 0)
    if use_wandb:
        wandb.init(
            project="cogformer-model-class",
            tags=["CogFormer", "ModelClass", "DDM", "RDM", "CDM"],
            config={**train_config, "cogformer": cogformer_config,
                    "total_batch_size": args.train_batch_size, "world_size": world_size},
        )

    # checkpoint name encodes total batch size, not per-process local_batch
    checkpoint_path = (
        f"cogformer_model_class_fm"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
        f"_s{cogformer_config['num_seeds']}"
        f"_d{cogformer_config['seed_dim']}"
        f"_b{args.train_batch_size}"
        f"_e{train_config['epochs']}"
        f"_t{train_config['steps_per_epoch']}.pt"
    )

    fig_path = (
        f"model_class_fm"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
    )

    val_scenario_tags = (
        {t.strip() for t in args.val_scenario_tags.split(",")}
        if args.val_scenario_tags is not None else None
    )

    trainer = CogFormerTrainer(
        net=cogformer,
        model_class=model_class,
        model_class_config=model_class_config,
        adapter=adapter,
        use_wandb=use_wandb,
        val_epoch_interval=args.val_epoch_interval,
        early_val_epoch=args.early_val_epoch,
        num_sim_workers=num_sim_workers,
        world_size=world_size,
        rank=rank,
        fig_dir=args.fig_dir,
        val_scenario_tags=val_scenario_tags,
    )

    is_dry_run = args.dry_run_epochs is not None
    total_epochs = train_config["epochs"]
    if is_dry_run:
        train_config["epochs"] = args.dry_run_epochs

    if rank == 0:
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
    else:
        trainer.train(
            train_config=train_config,
            val_config=val_config,
            checkpoint_path=checkpoint_path,
            fig_path=fig_path,
            resume=args.resume,
            resume_start_epoch=args.resume_start_epoch,
        )

    if use_wandb:
        wandb.finish()
    if is_distributed:
        dist.destroy_process_group()
