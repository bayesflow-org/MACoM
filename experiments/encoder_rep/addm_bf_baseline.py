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
    # Trial-varying item values -- the design real aDDM experiments actually use,
    # where each trial shows a different pair of items whose values are KNOWN
    # (from a pre-experiment bidding task) and therefore enter as regressors.
    #   v_diff_t = b_d0 + b_d1 * u_1t   (the value DIFFERENCE channel)
    #   v_t      = b_v0 + b_v1 * u_2t   (the common-value / SUM channel)
    # This is what removes the flat direction WITHOUT pinning v. Under
    # intercept_only the data see only mu = (1+w)/2*v_diff and
    # Delta = (1-w)(2v + v_diff), two functionals for three parameters. Here
    # v_diff enters BOTH channels with the same b_d1, and the ratio of its
    # Delta-effect to its mu-effect is 2(1-w)/(1+w) -- free of b_d1, so omega is
    # structurally identified. Whether that is PRACTICALLY strong at 500 trials
    # is the empirical question this case exists to answer; the Delta channel is
    # still second-order, which is what defeated the v-fixed variant.
    "varying_values": {
        "1": INTRINSIC_PARAMS,
        "u_1": ["v_diff"],
        "u_2": ["v"],
    },
}

# Extra batch_sample kwargs per case. Item values are CONTINUOUS quantities, but
# build_discrete_mask assigns each regressor continuous/categorical at random
# (discrete_prob=0.5) unless told otherwise -- an empty categorical_regressors
# list pins both value regressors continuous.
CASE_SAMPLE_KWARGS = {
    "varying_values": {"categorical_regressors": []},
}

MASK_RANDOMIZER_KWARGS = {
    "free_intrinsics": ["v", "v_diff", "omega", "a", "tau"],
    "fixed_intrinsics": ["s_v", "s_tau"],
    "fixed_values": {},
}


# ── v-fixed variant ───────────────────────────────────────────────────────────
# With gaze unobserved and item values constant across trials, the data depend on
# (v, v_diff, omega) only through two functionals -- the effective drift
#     mu    = (1 + omega)/2 * v_diff
# and the attentional swing
#     Delta = (1 - omega) * (2v + v_diff),
# which drives both the first-fixation choice bias and the extra within-trial
# drift variance. Three parameters, two functionals: exactly one flat direction,
# and v is the coordinate that spans it -- it never enters mu at all. Measured at
# 200k trials, a=1, tau=0.3, s_v=s_tau=0, first_fixation=item0:
#     v=1.00 v_diff=0.500 omega=0.25 -> P(c=1)=0.5713, RT q10/50/90 .554/1.007/2.173
#     v=1.67 v_diff=0.417 omega=0.50 -> P(c=1)=0.5719,               .554/1.006/2.167
#     v=3.57 v_diff=0.357 omega=0.75 -> P(c=1)=0.5727,               .553/1.006/2.172
# Pinning v makes (v_diff, omega) <-> (mu, Delta) a bijection; omega then moves
# P(c=1) by 0.17-0.28 across [0, 1]. This is also the scientifically correct move:
# in Krajbich's design item values come from a pre-experiment bidding task, i.e.
# they are known covariates, not free parameters.
#
# V_FIXED is the PRIOR MEDIAN of v on the natural scale -- shifted_softplus(0.6),
# 0.6 being the prior location -- so the marginal data distribution stays as close
# as possible to the v-free variant. Fixed-value injection in ModelFamily.sample()
# BYPASSES the link function, so this is a post-link value, not an unconstrained one.
V_FIXED = 1.0375

# S_V_FIXED is the prior median of s_v on the natural scale -- shifted_softplus(-1.0),
# -1.0 being the prior location -- chosen exactly as V_FIXED was.
#
# WHY pinning s_v is a different trade from pinning v. Under intercept_only with
# constant item values the data see only mu = (1+omega)/2 * v_diff and
# Delta = (1-omega)(2v + v_diff): two functionals for three parameters, one flat
# direction, which is what --fix_v removes -- at the cost of v no longer being
# INFERRED. The across-trial drift variance is a third functional,
# s_v^2 * (1 + omega^2). Pinning s_v turns that into a constraint on omega, which
# closes the same system from the other side and leaves v free to be recovered.
#
# Measured before running: s_v is itself only weakly recovered (r = 0.22-0.34 across
# the 7-dim and varying-values arms) and its within-posterior correlation with both
# omega and v is ~0.00, so the channel this reasoning leans on is real but thin.
# That is the empirical question these variants exist to answer.
S_V_FIXED = 0.3133

# Parameters that may be pinned, with the post-link value injected for each.
PINNABLE = {"v": V_FIXED, "s_v": S_V_FIXED}


def _pinned(fix_v: bool, fix_s_v: bool) -> list:
    return [q for q, on in (("v", fix_v), ("s_v", fix_s_v)) if on]


def resolve_case(case: str, fix_v: bool, fix_s_v: bool = False):
    """Design config + mask-randomizer kwargs for a case, with v optionally pinned.

    Dropping "v" from the design config leaves its column of the parameter mask
    empty, which is exactly the condition under which ModelFamily.sample() injects
    fixed_values["v"] instead of running the link function. The inference target
    goes from 7 active cells to 6 (v_diff, omega, a, tau, s_v, s_tau).

    Train and validate MUST call this with the same fix_v, or their simulated data
    distributions diverge -- the same failure mode as the empty-fixed_values bug.
    """
    pins = _pinned(fix_v, fix_s_v)
    if not pins:
        return DESIGN_CONFIGS[case], MASK_RANDOMIZER_KWARGS
    if case != "intercept_only":
        raise ValueError(
            f"pinning {pins} is only defined for intercept_only, got {case!r}: "
            "intercept_v regresses v on u_1, and pinning it would leave that "
            "term empty."
        )
    return (
        {"1": [q for q in INTRINSIC_PARAMS if q not in pins]},
        {
            "free_intrinsics": [q for q in MASK_RANDOMIZER_KWARGS["free_intrinsics"]
                                if q not in pins],
            "fixed_intrinsics": MASK_RANDOMIZER_KWARGS["fixed_intrinsics"] + pins,
            "fixed_values": {q: PINNABLE[q] for q in pins},
        },
    )


_VARIANT_SUFFIX = {(): "", ("v",): "_vfixed", ("s_v",): "_svfixed",
                   ("v", "s_v"): "_vsvfixed"}


def _variant(p: Path, fix_v: bool, fix_s_v: bool = False) -> Path:
    """Route the v-fixed run to its own output tree, leaving 7-dim results in place."""
    suf = _VARIANT_SUFFIX[tuple(_pinned(fix_v, fix_s_v))]
    return p.with_name(p.name + suf) if suf else p

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

    def __init__(self, case: str, num_obs: int = 500, first_fixation: str = "item0",
                 fix_v: bool = False, fix_s_v: bool = False):
        self.design_config, mask_kwargs = resolve_case(case, fix_v, fix_s_v)
        self.num_obs = num_obs
        self.extra_sample_kwargs = CASE_SAMPLE_KWARGS.get(case, {})
        self.model_family = ModelFamily(
            model=ADDM(first_fixation=first_fixation),
            prior=addm_priors(),
            intrinsic_params=INTRINSIC_PARAMS,
            mask_randomizer_kwargs=mask_kwargs,
        )
        self._choices_denom = max(NUM_ALTERNATIVES - 1, 1)
        self._alt_val = (NUM_ALTERNATIVES - 1) / max(MAX_ALTERNATIVES - 1, 1)

    def assemble(self, design_matrices, rts, choices):
        """Build the per-trial input tensor both arms consume.

        Factored out of sample() so the representational analysis encodes through the
        SAME code path rather than a re-implementation -- a channel-order divergence
        between the two would show up as a representational difference and be
        indistinguishable from a real one.

        Identical channel layout to Adapter.adapt / CogFormerEncoderSummary.summarize:
        favored_idx is RDM-only and theta is CDM-only, so both are zero for aDDM, but
        they must be present or the input width does not match.
        """
        b, t = design_matrices.shape[:2]
        choices_norm = choices / self._choices_denom
        alt_col = np.full((b, t, 1), self._alt_val, dtype=np.float32)
        zeros = np.zeros((b, t, 1), dtype=np.float32)
        return np.concatenate(
            [design_matrices, rts, choices_norm, alt_col, zeros, zeros], axis=-1
        ).astype(np.float32)

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
            **self.extra_sample_kwargs,
        )

        design_matrices = samples["design_matrices"]
        rts = samples["sim_data"]["rts"]
        choices = samples["sim_data"]["choices"]
        b, t = design_matrices.shape[:2]

        data = self.assemble(design_matrices, rts, choices)

        param_masks = samples["param_masks"].astype(np.float32)
        # Infer ONLY the active cells. The full grid is 67 design rows x 7
        # parameters = 469, of which intercept_only activates 7 (1.49%); the rest
        # are structurally zero in every dataset. Handing all 469 to the flow
        # spends ~98.5% of the flow-matching MSE learning to annihilate base
        # noise on constant dimensions, drives 462 of them through the
        # standardizer's degenerate std==0 path, and makes the training loss
        # useless as a comparison statistic. The full grid exists because
        # CogFormer's DECODER emits one coefficient per design cell -- but
        # neither arm of this probe uses that decoder, so nothing requires it.
        active = param_masks[0].astype(bool)   # constant within a case (fixed_config=True)
        return {
            "data": data.astype(np.float32),
            "params": samples["param_matrices"].astype(np.float32)[:, active],
            "masks": param_masks,
        }


def active_labels(mask_row):
    """Names for the active cells of a flattened 67x7 design grid.

    The flat index is ``row * n_params + param``. The previous code indexed a
    469-long mask with range(7), which is correct only when the active cells are
    exactly the intercept row -- true for intercept_only, wrong for intercept_v,
    where u_1 activates a cell in a later row.
    """
    n = len(INTRINSIC_PARAMS)
    idx = np.where(np.asarray(mask_row).astype(bool))[0]
    plain, pretty = [], []
    for i in idx:
        row, par = divmod(int(i), n)
        suffix = "" if row == 0 else f"@r{row}"
        plain.append(INTRINSIC_PARAMS[par] + suffix)
        pretty.append(VARIABLE_NAMES[par] if row == 0
                      else VARIABLE_NAMES[par][:-1] + r"^{(" + str(row) + r")}$")
    return plain, pretty


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


def build_inference_network(subnet_dropout=None, time_power_law_alpha=None):
    """FlowMatching at library defaults, with optional single-setting overrides.

    subnet_kwargs is merged into the library's TIME_MLP defaults rather than
    replacing them, so overriding dropout leaves widths, fourier_scale,
    activation and the rest untouched. Both knobs exist because main and the
    best-defaults branch disagree on exactly these values, and a comparison
    between the two is only interpretable if they can be held equal.

    time_power_law_alpha sets the training-time distribution of the flow-matching
    interpolation time: ``t = u ** (1 + alpha)`` for ``u ~ U(0,1)``, so
    ``E[t] = 1 / (2 + alpha)``. Since ``x = t*x1 + (1-t)*x0`` with x1 the target
    and x0 noise, t=1 is the data end -- where the velocity field learns to
    CONTRACT onto the posterior. alpha=0.5 (branch default) pushes sampling
    toward t=0 and starves that region; alpha=0.0 (main default) is uniform.
    This is the leading suspect for the branch's over-dispersed posteriors:
    means that track the truth at r~0.98 with sd(z)~0.2 instead of 1.0.
    """
    kwargs = dict(INFERENCE_NET_KWARGS)
    if subnet_dropout is not None:
        kwargs["subnet_kwargs"] = {**kwargs.get("subnet_kwargs", {}),
                                   "dropout": subnet_dropout}
    if time_power_law_alpha is not None:
        kwargs["time_power_law_alpha"] = time_power_law_alpha
    return bf.networks.FlowMatching(**kwargs)


def build_summary_network():
    return bf.networks.SetTransformer(**SUMMARY_NET_KWARGS)


def _sample_chunked(approximator, key: str, conditions, num_samples: int, chunk: int):
    """Posterior draws in dataset-sized chunks, concatenated.

    ``approximator.sample`` flattens (n_datasets, num_samples) into ONE batch, so
    500 datasets x 500 draws builds a 250,000 x 256 activation inside
    FlowMatching's FiLM layer and OOMs a 31.4 GiB card at ~29 GiB. Chunking over
    datasets keeps peak memory linear in ``chunk`` and leaves the draws per
    dataset -- and therefore every downstream metric -- byte-identical.
    """
    outs = []
    for i in range(0, len(conditions), chunk):
        draws = approximator.sample(
            conditions={key: conditions[i:i + chunk]}, num_samples=num_samples
        )
        outs.append(np.asarray(draws["params"]))
    return np.concatenate(outs, axis=0)


def ckpt_path(case: str, first_fixation: str, tag: str = "", fix_v: bool = False,
              fix_s_v: bool = False) -> str:
    # first_fixation is in the path: the probe script keys only on case, so a
    # second run with a different gaze rule silently overwrites the first.
    # tag serves the same purpose for anything the CLI cannot express -- notably
    # which BayesFlow defaults the run used, since that is a property of the
    # installed library rather than of any argument.
    return str(_variant(CKPT_DIR, fix_v, fix_s_v) / case / (first_fixation + _suffix(tag))
               / "addm_baseline_fm")


def _suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def train(args):
    simulator = ADDMBaselineSimulator(case=args.case, num_obs=args.num_obs,
                                      first_fixation=args.first_fixation,
                                      fix_v=args.fix_v, fix_s_v=args.fix_s_v)
    summary_net = build_summary_network()

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=build_adapter(),
        inference_network=build_inference_network(args.subnet_dropout,
                                                  args.time_power_law_alpha),
        summary_network=summary_net,
        checkpoint_filepath=ckpt_path(args.case, args.first_fixation, args.tag,
                                      args.fix_v, args.fix_s_v),
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
                                      first_fixation=args.first_fixation,
                                      fix_v=args.fix_v, fix_s_v=args.fix_s_v)
    approximator = keras.saving.load_model(
        ckpt_path(args.case, args.first_fixation, args.tag, args.fix_v,
                  args.fix_s_v) + "/model.keras"
    )

    val_sims = simulator.sample(args.val_batch_size)
    masks = val_sims["masks"]

    pred_active = _sample_chunked(approximator, "data", val_sims["data"],
                                  args.num_samples, args.sample_chunk_size)
    # The simulator now emits only the active cells, so no masking here.
    true_active = val_sims["params"]

    outdir = (_variant(FIGURES_DIR, args.fix_v, args.fix_s_v) / args.case
              / (args.first_fixation + _suffix(args.tag)))
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "addm_baseline_data.npz",
        true_params=true_active,
        pred_params=pred_active,
        param_masks=masks,
    )
    logging.info(f"[aDDM-baseline/{args.case}] Saved data to {outdir}")

    active_names, names = active_labels(masks[0])
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
    p.add_argument("--fix_s_v", action="store_true",
                   help="Pin across-trial drift variability s_v to S_V_FIXED instead of "
                        "inferring it. Combine with --fix_v to pin both. Each combination "
                        "writes to its own tree (_svfixed / _vsvfixed). intercept_only only.")
    p.add_argument("--fix_v", action="store_true",
                   help="Pin the common value v to V_FIXED instead of inferring it, "
                        "removing the (v, v_diff, omega) flat direction. Drops the "
                        "target from 7 active cells to 6 and routes all outputs to a "
                        "*_vfixed tree. intercept_only only.")
    p.add_argument("--subnet_dropout", type=float, default=None,
                   help="Override the inference subnet's dropout. Default None keeps the "
                        "installed library's value, which differs between BayesFlow main "
                        "(0.05) and the best-defaults branch (0.0).")
    p.add_argument("--time_power_law_alpha", type=float, default=None,
                   help="Override the flow-matching interpolation-time exponent. Default None "
                        "keeps the installed library's value: 0.5 on the best-defaults branch "
                        "(biased toward the noise end), 0.0 on main (uniform). Pass 0 to test "
                        "whether that default explains the branch's over-dispersed posteriors.")
    p.add_argument("--tag", type=str, default="",
                   help="Suffix for the checkpoint and figure directories. Use it to keep "
                        "runs that differ only in the installed BayesFlow apart, e.g. "
                        "--tag best_defaults vs --tag main_defaults.")
    p.add_argument("--epochs", type=int, default=500)        # == addm_bf_apply.py
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)     # == addm_bf_apply.py
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--sample_chunk_size", type=int, default=50,
                   help="Datasets per approximator.sample call at validation. The call "
                        "flattens datasets x draws into one batch, so 500x500 OOMs a "
                        "31 GiB card; chunking changes peak memory only, not results.")
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
