"""Fine-tune a pretrained DDM/RDM/CDM ModelClass CogFormer on LBA.

The pretrained checkpoint (num_models=3, model_ids DDM=0 RDM=1 CDM=2) is
loaded and the model embedding table is extended to num_models=4.  LBA's
embedding (row 3) is warm-started from RDM's (row 1) since they share the
same parameter space.  All other weights are loaded exactly.  Fine-tuning
then trains only on LBA samples using the full global parameter space
established by the base training run.
"""
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
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors, lba_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun, lba_link_fun
from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.utils.plot_utils import cogformer_fm_colors

from ..configs import MODELS as _BASE_MODELS
from ..utils import add_global_params, build_encoder_input_dim
from ..timing import RunTimer


GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]

# model_ids in the pretrained checkpoint
_PRETRAINED_IDS = {"DDM": 0, "RDM": 1, "CDM": 2}
LBA_MODEL_ID = 3

_ALL_WITH_PI = {
    m: add_global_params(cfg, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
    for m, cfg in _BASE_MODELS.items()
}

LBA_VAL_CONFIG = {
    "intrinsic_params": _ALL_WITH_PI["lba"]["intrinsic_params"],
    "variable_names":   _ALL_WITH_PI["lba"]["variable_names"],
    "val_design_config": _ALL_WITH_PI["lba"]["benchmark_configs"]["interaction"],
}


def _load_pretrained(net: CogFormer, checkpoint_path: str, device: torch.device) -> None:
    """Load pretrained weights, extending the model embedding table by one row for LBA.

    LBA's new embedding (row 3) is initialised from RDM's (row 1) since they
    share the same intrinsic parameter space.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_embedding.weight" in state:
        w = state["model_embedding.weight"]           # (3, model_embed_dim)
        new_w = torch.empty(w.shape[0] + 1, w.shape[1], dtype=w.dtype)
        new_w[:w.shape[0]] = w
        new_w[w.shape[0]] = w[_PRETRAINED_IDS["RDM"]]  # warm-start from RDM
        state["model_embedding.weight"] = new_w
    net.load_state_dict(state, strict=True)
    logging.info(f"[loaded] {checkpoint_path} — model_embedding extended to num_models=4")


class LBAFinetuner:
    def __init__(self, net, model_class, lba_family, lba_link_fun, adapter, use_wandb=False):
        self.net = net
        self.model_class = model_class
        self.lba_family = lba_family
        self.lba_link_fun = lba_link_fun
        self.adapter = adapter
        self.use_wandb = use_wandb

    def _sample_and_adapt(self, config):
        samples = self.lba_family.batch_sample(
            **config["model_family_config"],
            **config["train_sample_config"],
            batch_size=config["batch_size"],
            flatten_param_outputs=True,
            link_fun=self.lba_link_fun,
        )
        samples["param_matrices"], samples["param_masks"] = \
            self.model_class.lift_to_global_space(
                "LBA",
                samples["param_matrices"],
                samples["param_masks"],
            )
        samples["n_params"] = self.model_class.n_global_params
        samples["model_ids"] = np.full(config["batch_size"], LBA_MODEL_ID, dtype=np.int64)
        return self.adapter.adapt(samples)

    def train(self, train_config, val_config, checkpoint_path="cogformer_lba_finetune.pt", fig_path="fig"):
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

            if (epoch + 1) % train_config["val_interval"] == 0:
                self.val_step(val_config, global_step, fig_path)

            scheduler.step()
            pbar.close()

        checkpoint_dir = Path("./outputs/checkpoints/model_class/cf/lba_finetune/")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), checkpoint_dir / checkpoint_path)
        logging.info(f"[saved] {checkpoint_dir / checkpoint_path}")

    def train_step(self, config, optimizer, scheduler):
        adapted = self._sample_and_adapt(config)

        optimizer.zero_grad()
        device_type = config["device"].type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
            pred_velocity, target_velocity = self.net(
                adapted["param_matrices"][..., None],
                adapted["input_data"],
                adapted["param_indices"],
                adapted["regressor_indices"],
                adapted["param_masks"],
                model_ids=adapted["model_ids"],
            )
            L = self.net.compute_loss(pred_velocity, target_velocity, adapted["param_masks"])
        L.backward()

        if config["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), config["gradient_clip_norm"])

        optimizer.step()
        return L.detach().item(), scheduler.get_last_lr()[0]

    @torch.no_grad()
    def val_step(self, config, global_step, fig_path):
        self.net.eval()

        model_family_config = config["model_family_config"]
        n_global = self.model_class.n_global_params
        global_indices = self.model_class.local_to_global["LBA"]
        colors = cogformer_fm_colors()

        intrinsic_params = LBA_VAL_CONFIG["intrinsic_params"]
        variable_names   = LBA_VAL_CONFIG["variable_names"]
        design_config    = LBA_VAL_CONFIG["val_design_config"]

        test_samples = self.lba_family.batch_sample(
            design_config=design_config,
            batch_size=config["batch_size"],
            num_obs=config["num_obs"],
            max_num_regressors=model_family_config["max_num_regressors"],
            max_num_categories=model_family_config["max_num_categories"],
            keep_intercept=model_family_config["keep_intercept"],
            flatten_param_outputs=True,
            link_fun=self.lba_link_fun,
        )
        test_samples["param_matrices"], test_samples["param_masks"] = \
            self.model_class.lift_to_global_space(
                "LBA",
                test_samples["param_matrices"],
                test_samples["param_masks"],
            )
        test_samples["n_params"] = n_global
        test_samples["model_ids"] = np.full(config["batch_size"], LBA_MODEL_ID, dtype=np.int64)

        adapted = self.adapter.adapt(test_samples)

        pred_set = self.net.sample(
            adapted["input_data"],
            adapted["param_indices"],
            adapted["regressor_indices"],
            adapted["param_masks"],
            steps=config["fm_sample_steps"],
            num_samples=config["fm_num_samples"],
            model_ids=adapted["model_ids"],
        )

        batch_size = config["batch_size"]
        true_set = adapted["param_matrices"].detach().cpu().numpy()
        n_rows = true_set.shape[1] // n_global
        true_set = true_set.reshape(batch_size, n_rows, n_global)[:, :, global_indices]
        pred_set = pred_set.reshape(batch_size, config["fm_num_samples"], n_rows, n_global)[:, :, :, global_indices]
        params_mask = adapted["param_masks"].detach().cpu().numpy()
        params_mask = params_mask.reshape(batch_size, n_rows, n_global)[0][:, global_indices]

        recovery_fig = adaptive_recovery(
            true=true_set,
            pred=pred_set,
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            max_num_categories=model_family_config["max_num_categories"],
            parameter_mask=params_mask,
            variable_names=variable_names,
            intercept_color=colors["intercept"],
            main_effect_color=colors["main_effect"],
            interaction_color=colors["interaction"],
        )

        figures_dir = Path("./outputs/figures/model_class/cf/lba_finetune/recovery")
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_name = f"lba_{fig_path}_recovery.pdf"
        recovery_fig.savefig(figures_dir / fig_name, bbox_inches="tight")
        logging.info(f"[saved] {figures_dir / fig_name}")

        if self.use_wandb:
            wandb.log({"val/LBA/recovery": wandb.Image(recovery_fig)}, step=global_step)

        plt.close(recovery_fig)
        self.net.train()

    @staticmethod
    def finish():
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to the pretrained DDM/RDM/CDM ModelClass checkpoint.")
    parser.add_argument("--use_wandb", action="store_true")

    # Architecture — must match the pretrained checkpoint exactly
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

    # Optimizer — smaller defaults than pretraining
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer_dropout", type=float, default=0.05)

    # Training schedule
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=500)
    parser.add_argument("--val_num_obs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=500)

    # Flow matching
    parser.add_argument("--fm_sample_steps", type=int, default=100)
    parser.add_argument("--fm_num_samples", type=int, default=100)

    parser.add_argument("--dry_run_epochs", type=int, default=None)

    # Design space — must match the pretrained checkpoint
    parser.add_argument("--max_num_regressors", type=int, default=5)
    parser.add_argument("--max_num_categories", type=int, default=5)
    parser.add_argument("--keep_intercept", action="store_true", default=True)
    parser.add_argument("--add_interaction", action="store_true", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_num_regressors = args.max_num_regressors
    max_num_categories = args.max_num_categories
    keep_intercept     = args.keep_intercept

    model_family_config = {
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "keep_intercept":     keep_intercept,
        "add_interaction":    args.add_interaction,
    }

    train_sample_config = {
        "min_num_regressors": 0,
        "min_num_obs": args.min_num_obs,
        "max_num_obs": args.max_num_obs,
        "fixed_config": False,
    }

    encoder_input_dim = build_encoder_input_dim(max_num_regressors, max_num_categories, keep_intercept)

    train_config = {
        "epochs":            args.epochs,
        "batch_size":        args.train_batch_size,
        "steps_per_epoch":   args.steps_per_epoch,
        "learning_rate":     args.lr,
        "gradient_clip_norm": 5.0,
        "val_interval":      args.val_interval,
        "device":            device,
        "model_family_config": model_family_config,
        "train_sample_config": train_sample_config,
    }

    val_config = {
        "fm_sample_steps":   args.fm_sample_steps,
        "fm_num_samples":    args.fm_num_samples,
        "batch_size":        args.val_batch_size,
        "num_obs":           args.val_num_obs,
        "model_family_config": model_family_config,
    }

    cogformer_config = {
        "encoder_input_dim":   encoder_input_dim,
        "encoder_num_layers":  args.encoder_num_layers,
        "decoder_num_layers":  args.decoder_num_layers,
        "encoder_num_heads":   args.encoder_num_heads,
        "decoder_num_heads":   args.decoder_num_heads,
        "num_seeds":           args.num_seeds,
        "seed_dim":            args.seed_dim,
        "proj_dim":            args.proj_dim,
        "dropout":             args.dropout,
        "layer_dropout":       args.layer_dropout,
        "time_embedding_dim":  args.time_embedding_dim,
        "pos_embedding_dim":   args.pos_embedding_dim,
        "num_models":          4,          # DDM=0 RDM=1 CDM=2 LBA=3
        "model_embed_dim":     args.model_embed_dim,
        "decoder_layer_design": "mixed_attention",
        "decoder_layer_kwargs": {"mab_first": True},
    }

    lba_with_pi = _ALL_WITH_PI["lba"]

    # Build all four model families so ModelClass computes the same global
    # parameter space as the pretrained run (insertion order: DDM→RDM→CDM→LBA).
    model_families = {
        "DDM": ModelFamily(
            model=DDM(), name="DDM",
            prior=ddm_priors(),
            intrinsic_params=_ALL_WITH_PI["ddm"]["intrinsic_params"],
        ),
        "RDM": ModelFamily(
            model=RDM(), name="RDM",
            prior=rdm_priors(),
            intrinsic_params=_ALL_WITH_PI["rdm"]["intrinsic_params"],
        ),
        "CDM": ModelFamily(
            model=CDM(), name="CDM",
            prior=cdm_priors(),
            intrinsic_params=_ALL_WITH_PI["cdm"]["intrinsic_params"],
        ),
        "LBA": ModelFamily(
            model=LBA(), name="LBA",
            prior=lba_priors(),
            intrinsic_params=lba_with_pi["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics":  lba_with_pi["free_params"],
                "fixed_intrinsics": lba_with_pi["fixed_params"],
                "fixed_values":     lba_with_pi["default_fixed_values"],
                "global_params":    GLOBAL_PARAMS,
            },
        ),
    }

    link_funs = {
        "DDM": ddm_link_fun(), "RDM": rdm_link_fun(),
        "CDM": cdm_link_fun(), "LBA": lba_link_fun(),
    }

    model_class = ModelClass(model_families=model_families, link_funs=link_funs)
    lba_family  = model_families["LBA"]
    lba_lf      = link_funs["LBA"]

    logging.info(
        f"Global param space ({model_class.n_global_params}): {model_class.global_param_names}"
    )
    logging.info(
        f"LBA local→global indices: {model_class.local_to_global['LBA']}"
    )

    cogformer = CogFormer(**cogformer_config).to(device)
    _load_pretrained(cogformer, args.pretrained_checkpoint, device)
    cogformer.train()

    adapter = Adapter()

    checkpoint_path = (
        f"cogformer_lba_finetune"
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
        f"lba_finetune"
        f"_l{cogformer_config['decoder_num_layers']}"
        f"_h{cogformer_config['decoder_num_heads']}"
        f"_p{cogformer_config['proj_dim']}"
    )

    if args.use_wandb:
        wandb.init(
            project="cogformer-model-class",
            tags=["CogFormer", "FineTune", "LBA"],
            config={**train_config, "cogformer": cogformer_config,
                    "pretrained": args.pretrained_checkpoint},
        )

    trainer = LBAFinetuner(
        net=cogformer,
        model_class=model_class,
        lba_family=lba_family,
        lba_link_fun=lba_lf,
        adapter=adapter,
        use_wandb=args.use_wandb,
    )

    is_dry_run = args.dry_run_epochs is not None
    total_epochs = train_config["epochs"]
    if is_dry_run:
        train_config["epochs"] = args.dry_run_epochs

    with RunTimer("lba_finetune", dry_run=is_dry_run,
                  total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
        trainer.train(
            train_config=train_config,
            val_config=val_config,
            checkpoint_path=checkpoint_path,
            fig_path=fig_path,
        )

    if args.use_wandb:
        trainer.finish()
