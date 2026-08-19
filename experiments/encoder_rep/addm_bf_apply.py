"""Encoder representation experiment: aDDM inference with frozen CogFormer encoder + BayesFlow.

The CogFormer encoder is frozen and used as a summary network; BayesFlow
FlowMatching is trained on the pooled encoder output to infer aDDM parameters.
aDDM is a probe: the encoder never saw it, and unlike LCA/FFI its key parameter
(omega, the attentional discount) is driven by a within-trial process the encoder
has no channel for. What survives into the summary is the question.

Use the ORIGINAL (non-finetuned) model-class checkpoint, so the encoder is the
DDM+RDM+CDM model and every probe family is cleanly out-of-distribution.

Cases (deliberately close to intercept-only — there is no fixation covariate):
  intercept_only  — all aDDM params at intercept, no regressors
  intercept_v     — intercept params + u_1 modulates v

first_fixation drives omega's identifiability, so it is a CLI switch:
  random  — standard aDDM; the first-fixation advantage averages out, leaving
            omega visible only through RT-distribution shape
  item0   — gaze always starts on item 0, turning that advantage into a
            systematic choice bias recoverable from marginal choice proportions
Measured at 500 trials/dataset, v_diff=0: P(choose 0) runs 0.578 -> 0.506 across
omega 0 -> 1 under "item0", and is flat (~0.50) under "random".
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import logging
import numpy as np
try:
    import wandb
except Exception:
    wandb = None
import torch
import keras
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.addm import ADDM
from cogformer.simulator.benchmarks.ssms.priors import addm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import addm_link_fun
from cogformer.networks.transformers.cogformer import CogFormer

from ..utils import build_encoder_input_dim
from ..timing import RunTimer


INTRINSIC_PARAMS = ["v", "v_diff", "omega", "a", "tau", "s_v", "s_tau"]
VARIABLE_NAMES   = [r"$v$", r"$v_\mathrm{diff}$", r"$\omega$",
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
    "free_intrinsics": ["v", "v_diff", "omega", "a", "tau"],
    "fixed_intrinsics": ["s_v", "s_tau"],
    "fixed_values": {},
}

# MUST match the geometry the checkpoint was trained at: build_encoder_input_dim
# gives r2c2->9, r2c3->14, r4c4->72, r5c5->186, and load_state_dict fails on any
# mismatch at encoder.input_embedding. The model-class checkpoints are r4c4.
MAX_NUM_REGRESSORS = 4
MAX_NUM_CATEGORIES = 4
MAX_ALTERNATIVES   = 4
NUM_ALTERNATIVES   = 2

CKPT_DIR    = Path("./outputs/checkpoints/encoder_rep/bf/addm")
DATA_DIR    = Path("./outputs/data/encoder_rep/addm")
FIGURES_DIR = Path("./outputs/figures/encoder_rep/addm")


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
        # The encoder expects FIVE observation channels, matching Adapter.adapt:
        #   [design, rt, choices_normalized, alt_col, favored_idx, theta]
        # favored_idx is RDM-only and theta is CDM-only, so both are zero for aDDM —
        # but they must still be present or the input is 2 columns short and
        # encoder.input_embedding rejects it.
        fav_col = np.zeros((batch_size, num_obs, 1), dtype=np.float32)
        theta_col = np.zeros((batch_size, num_obs, 1), dtype=np.float32)
        input_data = np.concatenate(
            [design_matrices, rts, choices_norm, alt_col, fav_col, theta_col], axis=-1
        )
        x = torch.from_numpy(input_data).to(dtype=torch.float32, device=self.device)
        tokens = self.encoder(x)   # (B, n_seeds, seed_dim)
        return tokens.mean(dim=1).cpu().numpy()  # (B, seed_dim)


class ADDMSimulator(bf.simulators.Simulator):

    def __init__(self, encoder_summary: CogFormerEncoderSummary, case: str, num_obs: int = 500,
                 first_fixation: str = "item0"):
        self.encoder_summary = encoder_summary
        self.design_config = DESIGN_CONFIGS[case]
        self.num_obs = num_obs
        self.model_family = ModelFamily(
            model=ADDM(first_fixation=first_fixation),
            prior=addm_priors(),
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
            # build_link_functions requires link_fun keys to be a subset of
            # intrinsic_params; addm_link_fun carries "pi", which this probe does
            # not infer (contamination is off here), so drop anything unlisted.
            link_fun={k: f for k, f in addm_link_fun().items() if k in INTRINSIC_PARAMS},
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
        # The summary is PRECOMPUTED by the frozen CogFormer encoder, so there is no
        # BayesFlow summary network. "summary_variables" is reserved for input a
        # summary network consumes; a precomputed vector is an inference condition.
        .rename("summaries", "inference_conditions")
        .rename("params", "inference_variables")
    )


def ckpt_path(case: str) -> str:
    return str(CKPT_DIR / case / "addm_encoder_rep_fm")


def train(args):
    device = torch.device(args.encoder_device)

    cogformer = _load_cogformer(args, device)
    encoder_summary = CogFormerEncoderSummary(cogformer, device)
    simulator = ADDMSimulator(encoder_summary, case=args.case, num_obs=args.num_obs,
                              first_fixation=args.first_fixation)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=build_adapter(),
        inference_network=bf.networks.FlowMatching(),
        checkpoint_filepath=ckpt_path(args.case),
    )

    logging.info(f"[aDDM/{args.case}] Training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    workflow.fit_online(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )
    logging.info(f"[aDDM/{args.case}] Training complete.")


def validate(args):
    device = torch.device(args.encoder_device)

    cogformer = _load_cogformer(args, device)
    encoder_summary = CogFormerEncoderSummary(cogformer, device)
    simulator = ADDMSimulator(encoder_summary, case=args.case, num_obs=args.num_obs,
                              first_fixation=args.first_fixation)

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
        out_dir / "addm_encoder_rep_data.npz",
        summaries=summaries,
        true_params=true_active,
        pred_params=pred_active,
        param_masks=masks,
    )
    logging.info(f"[aDDM/{args.case}] Saved data to {out_dir}")

    fig_dir = FIGURES_DIR / args.case
    fig_dir.mkdir(parents=True, exist_ok=True)

    active_names = [VARIABLE_NAMES[i] for i in range(len(INTRINSIC_PARAMS)) if active_idx[i]]

    _plot_recovery(true_active, pred_active, active_names, fig_dir / "addm_recovery.pdf")
    logging.info(f"[aDDM/{args.case}] Saved figures to {fig_dir}")


def _plot_recovery(true, pred, names, path):
    n = true.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, j, name in zip(axes, range(n), names):
        # pred is (n_datasets, n_draws, n_params): average over DRAWS (axis 1) to get
        # one posterior mean per dataset. axis=0 averages over datasets and yields
        # n_draws points, which cannot be plotted against n_datasets truths.
        ax.scatter(true[:, j], pred[:, :, j].mean(axis=1), alpha=0.4, s=10)
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
        num_models=args.num_models,
        model_embed_dim=args.model_embed_dim,
        decoder_layer_design="mixed_attention",
        decoder_layer_kwargs={"mab_first": True},
    ).to(device)
    raw = torch.load(Path(args.cogformer_checkpoint), map_location=device, weights_only=False)
    # model-class checkpoints wrap the weights and carry a torch.compile prefix
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    cogformer.load_state_dict(state)
    cogformer.eval()
    return cogformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["train", "validate"])
    p.add_argument("--case", type=str, default="intercept_only",
                   choices=list(DESIGN_CONFIGS.keys()))
    p.add_argument("--first_fixation", type=str, default="item0",
                   choices=["random", "item0", "item1"],
                   help="Gaze-onset rule. 'item0' makes omega recoverable from marginal "
                        "choice proportions; 'random' is standard aDDM but leaves omega "
                        "identifiable only through RT-distribution shape.")
    p.add_argument("--num_models", type=int, default=3,
                   help="Family count the checkpoint was trained with (original=3, "
                        "LBA-finetuned=4). Only sizes model_embedding, which the encoder "
                        "never touches, but load_state_dict is strict about it.")
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
    p.add_argument("--use_wandb", action="store_true")

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
        if args.use_wandb:
            wandb.init(
                project="cogformer-encoder-rep",
                tags=["CogFormer", "LCA", "BF", "EncoderRep"],
                config={
                    "case": args.case,
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size": args.batch_size,
                    "num_obs": args.num_obs,
                },
            )
        is_dry_run = args.dry_run_epochs is not None
        total_epochs = args.epochs
        if is_dry_run:
            args.epochs = args.dry_run_epochs
        with RunTimer(f"addm_bf_{args.case}", dry_run=is_dry_run,
                      total_epochs=total_epochs, dry_run_epochs=args.dry_run_epochs):
            train(args)
        if args.use_wandb:
            wandb.finish()
    else:
        validate(args)
