"""Encoder representation experiment: FFI inference with frozen CogFormer encoder + BayesFlow.

The CogFormer encoder (trained on DDM+RDM+CDM+LBA) is frozen and used as a
summary network. BayesFlow FlowMatching is trained on top of the pooled encoder
output to infer FFI parameters.

Two cases:
  intercept_only  — all FFI params at intercept, no regressors
  intercept_v     — intercept params + u_1 modulates drift rate v
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import logging
import numpy as np
import torch
import keras
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.ffi import FFI
from cogformer.simulator.benchmarks.ssms.priors import ffi_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ffi_link_fun
from cogformer.networks.transformers.cogformer import CogFormer

from ..utils import build_encoder_input_dim
from ..timing import RunTimer


INTRINSIC_PARAMS = ["v", "v_diff", "k", "beta", "a", "tau", "s_v", "s_tau"]
VARIABLE_NAMES   = [r"$v$", r"$v_\mathrm{diff}$", r"$\kappa$", r"$\beta$",
                    r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

DESIGN_CONFIGS = {
    "intercept_only": {
        "1": INTRINSIC_PARAMS,
    },
    "intercept_v": {
        "1": INTRINSIC_PARAMS,
        "u_1": ["v"],
    },
}

MASK_RANDOMIZER_KWARGS = {
    "free_intrinsics": ["v", "v_diff", "k", "beta", "a", "tau"],
    "fixed_intrinsics": ["s_v", "s_tau"],
    "fixed_values": {},
}

MAX_NUM_REGRESSORS = 5
MAX_NUM_CATEGORIES = 5
MAX_ALTERNATIVES   = 4
NUM_ALTERNATIVES   = 2

CKPT_DIR    = Path("./outputs/checkpoints/encoder_rep/bf/ffi")
DATA_DIR    = Path("./outputs/data/encoder_rep/ffi")
FIGURES_DIR = Path("./outputs/figures/encoder_rep/ffi")


class CogFormerEncoderSummary:
    """Wraps a frozen CogFormer encoder to produce mean-pooled seed summaries."""

    def __init__(self, cogformer: CogFormer, device: torch.device):
        self.encoder = cogformer.encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.device = device

        n_total = MAX_NUM_REGRESSORS * (MAX_NUM_REGRESSORS + 1) // 2
        self.max_num_cols = n_total * (MAX_NUM_CATEGORIES - 1) + 1
        self._choices_denom = float(max(NUM_ALTERNATIVES - 1, 1))
        self._alt_val = float(NUM_ALTERNATIVES - 1) / float(max(MAX_ALTERNATIVES - 1, 1))

    @torch.no_grad()
    def summarize(
        self,
        design_matrices: np.ndarray,  # (B, T, max_num_cols)
        rts: np.ndarray,               # (B, T, 1)
        choices: np.ndarray,           # (B, T, 1)
    ) -> np.ndarray:                   # (B, seed_dim)
        batch_size, num_obs = design_matrices.shape[:2]
        choices_norm = choices / self._choices_denom
        alt_col = np.full((batch_size, num_obs, 1), self._alt_val, dtype=np.float32)
        input_data = np.concatenate([design_matrices, rts, choices_norm, alt_col], axis=-1)
        x = torch.from_numpy(input_data).to(dtype=torch.float32, device=self.device)
        tokens = self.encoder(x)   # (B, n_seeds, seed_dim)
        return tokens.mean(dim=1).cpu().numpy()  # (B, seed_dim)


class FFISimulator(bf.simulators.Simulator):

    def __init__(self, encoder_summary: CogFormerEncoderSummary, case: str, num_obs: int = 500):
        self.encoder_summary = encoder_summary
        self.design_config = DESIGN_CONFIGS[case]
        self.num_obs = num_obs
        self.model_family = ModelFamily(
            model=FFI(),
            prior=ffi_priors(),
            intrinsic_params=INTRINSIC_PARAMS,
            mask_randomizer_kwargs=MASK_RANDOMIZER_KWARGS,
        )

    def sample(self, batch_size, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=self.num_obs,
            design_config=self.design_config,
            max_num_regressors=MAX_NUM_REGRESSORS,
            max_num_categories=MAX_NUM_CATEGORIES,
            keep_intercept=True,
            fixed_config=True,
            flatten_param_outputs=True,
            link_fun=ffi_link_fun(),
        )

        summaries = self.encoder_summary.summarize(
            design_matrices=samples["design_matrices"],
            rts=samples["sim_data"]["rts"],
            choices=samples["sim_data"]["choices"],
        )

        return {
            "summaries": summaries.astype(np.float32),
            "params":    samples["param_matrices"].astype(np.float32),
            "masks":     samples["param_masks"].astype(np.float32),
        }


def build_adapter():
    return (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .rename("summaries", "summary_variables")
        .rename("params", "inference_variables")
    )


def ckpt_path(case: str) -> str:
    return str(CKPT_DIR / case / "ffi_encoder_rep_fm")


def train(args):
    device = torch.device(args.encoder_device)

    cogformer = _load_cogformer(args, device)
    encoder_summary = CogFormerEncoderSummary(cogformer, device)
    simulator = FFISimulator(encoder_summary, case=args.case, num_obs=args.num_obs)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=build_adapter(),
        inference_network=bf.networks.FlowMatching(),
        checkpoint_filepath=ckpt_path(args.case),
    )

    logging.info(f"[FFI/{args.case}] Training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    workflow.fit_online(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )
    logging.info(f"[FFI/{args.case}] Training complete.")


def validate(args):
    device = torch.device(args.encoder_device)

    cogformer = _load_cogformer(args, device)
    encoder_summary = CogFormerEncoderSummary(cogformer, device)
    simulator = FFISimulator(encoder_summary, case=args.case, num_obs=args.num_obs)

    approximator = keras.saving.load_model(ckpt_path(args.case) + "/model.keras")

    val_sims = simulator.sample(args.val_batch_size)
    summaries = val_sims["summaries"]
    true_params = val_sims["params"]
    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)

    post_draws = approximator.sample(
        conditions={"summaries": summaries},
        num_samples=args.num_samples,
    )
    pred_params = post_draws["params"]

    true_active = true_params[:, active_idx]
    pred_active = pred_params[:, :, active_idx]

    out_dir = DATA_DIR / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "ffi_encoder_rep_data.npz",
        summaries=summaries,
        true_params=true_active,
        pred_params=pred_active,
        param_masks=masks,
    )
    logging.info(f"[FFI/{args.case}] Saved data to {out_dir}")

    fig_dir = FIGURES_DIR / args.case
    fig_dir.mkdir(parents=True, exist_ok=True)

    active_names = [VARIABLE_NAMES[i] for i in range(len(INTRINSIC_PARAMS)) if active_idx[i]]

    _plot_recovery(true_active, pred_active, active_names, fig_dir / "ffi_recovery.pdf")
    logging.info(f"[FFI/{args.case}] Saved figures to {fig_dir}")


def _plot_recovery(true, pred, names, path):
    n = true.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, j, name in zip(axes, range(n), names):
        ax.scatter(true[:, j], pred[:, :, j].mean(axis=0), alpha=0.4, s=10)
        lo = min(true[:, j].min(), pred[:, :, j].min())
        hi = max(true[:, j].max(), pred[:, :, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Est. {name}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _load_cogformer(args, device):
    encoder_input_dim = build_encoder_input_dim(MAX_NUM_REGRESSORS, MAX_NUM_CATEGORIES, keep_intercept=True)
    cogformer = CogFormer(
        encoder_input_dim=encoder_input_dim,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        num_seeds=args.num_seeds,
        seed_dim=args.seed_dim,
        proj_dim=args.proj_dim,
        dropout=0.0,
        layer_dropout=0.0,
        time_embedding_dim=args.time_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        num_models=4,
        model_embed_dim=args.model_embed_dim,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
    ).to(device)
    state = torch.load(Path(args.cogformer_checkpoint), map_location=device)
    cogformer.load_state_dict(state)
    cogformer.eval()
    return cogformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["train", "validate"])
    p.add_argument("--case", type=str, default="intercept_only",
                   choices=list(DESIGN_CONFIGS.keys()))
    p.add_argument("--cogformer_checkpoint", type=str, required=True,
                   help="Path to the encoder_rep CogFormer checkpoint (.pt)")
    p.add_argument("--encoder_device", type=str, default="cpu",
                   help="Device for CogFormer encoder (cpu or cuda)")

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--dry_run_epochs", type=int, default=None,
                   help="If set, train for this many epochs only and record extrapolated time.")

    # Validation
    p.add_argument("--val_batch_size", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=500)

    # CogFormer architecture (must match checkpoint)
    p.add_argument("--encoder_num_layers", type=int, default=8)
    p.add_argument("--decoder_num_layers", type=int, default=8)
    p.add_argument("--encoder_num_heads", type=int, default=8)
    p.add_argument("--decoder_num_heads", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=32)
    p.add_argument("--seed_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--model_embed_dim", type=int, default=8)
    p.add_argument("--time_embedding_dim", type=int, default=32)
    p.add_argument("--pos_embedding_dim", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.mode == "train":
        is_dry_run = args.dry_run_epochs is not None
        total_epochs = args.epochs
        if is_dry_run:
            args.epochs = args.dry_run_epochs
        with RunTimer(f"ffi_bf_{args.case}", dry_run=is_dry_run,
                      total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
            train(args)
    else:
        validate(args)
