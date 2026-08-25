"""Baseline for the aDDM encoder-representation probe: BayesFlow end-to-end.

Companion to ``addm_bf_apply.py``. That script freezes CogFormer's encoder and
trains only FlowMatching on top of it; this one replaces the frozen encoder with
a BayesFlow SetTransformer trained JOINTLY with FlowMatching on the same
simulator, design config and budget. The contrast is therefore

    frozen transferred summary  (CogFormer encoder, DDM+RDM+CDM, never saw aDDM)
    vs
    learned task-specific summary  (SetTransformer, trained on aDDM directly)

Both see byte-identical per-trial inputs — the same 72-channel vector
``[design, rt, choices_normalized, alt_col, favored_idx, theta]`` that
``Adapter.adapt`` builds for CogFormer — so the only thing that differs is how
the set of trials is summarised.

NETWORK CONFIGURATION
---------------------
Both networks are left at BayesFlow's OWN defaults, as retuned on the
``best-defaults`` branch -- no architecture kwargs are passed at all. That
branch retunes summary and inference defaults jointly (SetTransformer moves to
Pre-LN RMSNorm, SwiGLU feedforwards and gated residual branches; the
FlowMatching time-MLP to five 256-wide residual blocks with mish/he_normal),
so overriding any single value silently re-opens a choice its authors made as
a package. It also drops ``mlp_depths``/``mlp_widths`` in favour of
``expansion_factor``/``glu_variant``, so the pre-2.0.12 override block this
script used to carry no longer has a meaning to port.

SetTransformer (not TimeSeriesTransformer) remains the right family: trials are
exchangeable and the simulator emits no time index -- RT is a value per trial,
not an axis.

TRAINING BUDGET
---------------
500 epochs x 100 steps x batch 64, byte-identical to ``addm_bf_apply.py``, so
both arms see the same number of simulated datasets (3.2M). This matters: an
earlier default of 2000 x 100 x 256 gave this arm 16x the data of the arm it is
supposed to be compared against, which flatters the baseline on exactly the
axis the probe is testing.

FRAMING
-------
The arms are matched on training budget and on per-trial inputs, NOT on
parameter count. Write it up as "each arm at its library's recommended
defaults, same data budget" -- not as "capacity-matched to CogFormer".
"""
import os

os.environ["KERAS_BACKEND"] = "torch"
# Expandable segments let the CUDA caching allocator grow/shrink segments instead
# of binning fixed-size blocks, which is what OOMed this script at batch 256 with
# 1.67 GiB reserved-but-unallocated. Set here rather than inherited from
# ``experiments.utils`` because this module never imports it; setdefault respects
# an explicit user override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import bayesflow as bf

from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.addm import ADDM
from cogformer.simulator.benchmarks.ssms.priors import addm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import addm_link_fun

from ..timing import RunTimer

INTRINSIC_PARAMS = ["v", "v_diff", "omega", "a", "tau", "s_v", "s_tau"]
VARIABLE_NAMES = [r"$v$", r"$v_\mathrm{diff}$", r"$\omega$",
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

# Must match addm_bf_apply.py so both arms see identical inputs.
MAX_NUM_REGRESSORS = 4
MAX_NUM_CATEGORIES = 4
MAX_ALTERNATIVES = 4
NUM_ALTERNATIVES = 2

CKPT_DIR = Path("./outputs/checkpoints/encoder_rep/bf_baseline/addm")
FIGURES_DIR = Path("./outputs/figures/encoder_rep/addm_baseline")

# Left deliberately empty: BayesFlow's own defaults are the configuration under
# test here. See module docstring, NETWORK CONFIGURATION.
SUMMARY_NET_KWARGS: dict = {}
INFERENCE_NET_KWARGS: dict = {}


class ADDMBaselineSimulator(bf.simulators.Simulator):
    """Emits raw per-trial data for the SetTransformer to summarise itself."""

    def __init__(self, case: str, num_obs: int = 500, first_fixation: str = "item0"):
        self.design_config = DESIGN_CONFIGS[case]
        self.num_obs = num_obs
        self.model_family = ModelFamily(
            model=ADDM(first_fixation=first_fixation),
            prior=addm_priors(),
            intrinsic_params=INTRINSIC_PARAMS,
            mask_randomizer_kwargs=MASK_RANDOMIZER_KWARGS,
        )
        self._choices_denom = max(NUM_ALTERNATIVES - 1, 1)
        self._alt_val = (NUM_ALTERNATIVES - 1) / max(MAX_ALTERNATIVES - 1, 1)

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
            link_fun={k: f for k, f in addm_link_fun().items() if k in INTRINSIC_PARAMS},
        )

        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        b, t = design_matrices.shape[:2]

        # Identical channel layout to Adapter.adapt / CogFormerEncoderSummary.summarize:
        # favored_idx is RDM-only and theta is CDM-only, so both are zero for aDDM,
        # but they must be present for the input width to match the other arm.
        choices_norm = choices / self._choices_denom
        alt_col = np.full((b, t, 1), self._alt_val, dtype=np.float32)
        zeros = np.zeros((b, t, 1), dtype=np.float32)
        data = np.concatenate(
            [design_matrices, rts, choices_norm, alt_col, zeros, zeros], axis=-1
        )

        return {
            "data": data.astype(np.float32),
            "params": samples["param_matrices"].astype(np.float32),
            "masks": samples["param_masks"].astype(np.float32),
        }


def build_adapter():
    return (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        # Unlike the frozen-encoder arm, the summary here is LEARNED, so the raw
        # per-trial set is a summary_variable rather than an inference_condition.
        .rename("data", "summary_variables")
        .rename("params", "inference_variables")
    )


def build_summary_network():
    return bf.networks.SetTransformer(**SUMMARY_NET_KWARGS)


def ckpt_path(case: str, first_fixation: str) -> str:
    # first_fixation is in the path: the probe script keys only on case, so a
    # second run with a different gaze rule silently overwrites the first.
    return str(CKPT_DIR / case / first_fixation / "addm_baseline_fm")


def train(args):
    simulator = ADDMBaselineSimulator(case=args.case, num_obs=args.num_obs,
                                      first_fixation=args.first_fixation)
    summary_net = build_summary_network()

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=build_adapter(),
        inference_network=bf.networks.FlowMatching(**INFERENCE_NET_KWARGS),
        summary_network=summary_net,
        checkpoint_filepath=ckpt_path(args.case, args.first_fixation),
    )

    logging.info(
        f"[aDDM-baseline/{args.case}] SetTransformer + FlowMatching, "
        f"{args.epochs} epochs x {args.steps_per_epoch} steps, batch {args.batch_size}"
    )
    # NB: the parameter is num_batches_per_epoch, not steps_per_epoch. The latter
    # is not a fit_online argument -- it falls through **kwargs into keras' fit()
    # while the dataset keeps its own default of 100 batches per epoch, so
    # --steps_per_epoch silently did nothing except when left at 100.
    workflow.fit_online(
        epochs=args.epochs,
        num_batches_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )
    n_summary = summary_net.count_params()
    logging.info(f"[aDDM-baseline/{args.case}] summary network params: {n_summary:,}")
    logging.info(f"[aDDM-baseline/{args.case}] Training complete.")


def validate(args):
    simulator = ADDMBaselineSimulator(case=args.case, num_obs=args.num_obs,
                                      first_fixation=args.first_fixation)
    approximator = keras.saving.load_model(
        ckpt_path(args.case, args.first_fixation) + "/model.keras"
    )

    val_sims = simulator.sample(args.val_batch_size)
    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)

    post_draws = approximator.sample(
        conditions={"data": val_sims["data"]},
        num_samples=args.num_samples,
    )
    true_active = val_sims["params"][:, active_idx]
    pred_active = post_draws["params"][:, :, active_idx]

    outdir = FIGURES_DIR / args.case / args.first_fixation
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "addm_baseline_data.npz",
        true_params=true_active,
        pred_params=pred_active,
        param_masks=masks,
    )
    logging.info(f"[aDDM-baseline/{args.case}] Saved data to {outdir}")

    names = [VARIABLE_NAMES[i] for i in range(len(INTRINSIC_PARAMS)) if active_idx[i]]
    n_params = len(names)
    plots = {
        "recovery": bf.diagnostics.plots.recovery,
        "calibration_ecdf": bf.diagnostics.plots.calibration_ecdf,
        "coverage": bf.diagnostics.plots.coverage,
        "z_score_contraction": bf.diagnostics.plots.z_score_contraction,
    }
    for name, fn in plots.items():
        # Several BayesFlow plotters size their axis grid independently of the
        # variable count and then iterate every axis, overrunning ranks[:, j].
        fig = fn(estimates=pred_active, targets=true_active, variable_names=names,
                 num_row=1, num_col=n_params)
        path = outdir / f"addm_baseline_{args.case}_{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {path}")

    contraction = np.asarray(bf.diagnostics.metrics.posterior_contraction(
        estimates=pred_active, targets=true_active)["values"]).ravel()
    zscore = np.asarray(bf.diagnostics.metrics.posterior_z_score(
        estimates=pred_active, targets=true_active)["values"]).ravel()
    rmse = np.asarray(bf.diagnostics.metrics.root_mean_squared_error(
        estimates=pred_active, targets=true_active)["values"]).ravel()

    import pandas as pd
    active_names = [p for i, p in enumerate(INTRINSIC_PARAMS) if active_idx[i]]
    df = pd.DataFrame({"param": active_names, "contraction": contraction,
                       "z_score": zscore, "nrmse": rmse})
    df.to_csv(outdir / f"addm_baseline_{args.case}_bf_metrics.csv",
              index=False, float_format="%.4f")
    logging.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def parse_args():
    p = argparse.ArgumentParser(
        description="BayesFlow-only baseline (SetTransformer summary) for the aDDM probe."
    )
    p.add_argument("--mode", type=str, required=True, choices=["train", "validate"])
    p.add_argument("--case", type=str, default="intercept_only",
                   choices=list(DESIGN_CONFIGS.keys()))
    p.add_argument("--first_fixation", type=str, default="item0",
                   choices=["random", "item0", "item1"],
                   help="Must match the frozen-encoder arm for the comparison to hold.")
    p.add_argument("--epochs", type=int, default=500)        # == addm_bf_apply.py
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)     # == addm_bf_apply.py
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--val_batch_size", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=100)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.mode == "train":
        with RunTimer(f"addm_bf_baseline_{args.case}", total_epochs=args.epochs):
            train(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
