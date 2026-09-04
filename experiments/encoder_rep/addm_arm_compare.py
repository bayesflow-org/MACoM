"""Shared-validation comparison of the two aDDM encoder-representation arms.

The per-arm validate steps each draw their OWN simulated datasets, so their
outputs cannot be compared dataset-by-dataset. This script simulates ONE
validation set and pushes it through both trained approximators, which makes two
sharper comparisons available:

  (a) EXPECTED LOG POSTERIOR DENSITY AT THE TRUE THETA -- a proper scoring rule.
      Unlike C2ST it is signed, so it RANKS the arms; it penalises bias and
      mis-dispersion together. ContinuousApproximator.log_prob folds in the
      adapter's log_det_jac, so densities are in the ORIGINAL parameter space and
      the two arms' absolute values are comparable despite each carrying its own
      standardizer statistics. Datasets are shared, so the arms are compared
      PAIRED.

  (b) PER-DATASET C2ST on the joint posterior draws, which catches differences in
      shape, correlation structure and tails that the per-parameter contraction /
      r summaries cannot see.

WHY NOT cogformer.diagnostics.metric.adaptive_c2st
--------------------------------------------------
``compute_joint_c2st`` pools draws across datasets
(``pred[:, :, active].reshape(-1, n_active)``). That is degenerate for this
comparison. Both arms are SBC-calibrated, and a calibrated posterior averaged
over the prior predictive returns the PRIOR. So both pooled sets are ~the prior
and C2ST -> 0.5 by construction, however much the per-dataset posteriors
disagree -- a confident-looking null that means nothing. Only the per-dataset
statistic is informative. ``compute_c2st_accuracy`` (the two-sample primitive) is
reused as-is; it is only the pooling wrapper that is wrong here.

THE NULL
--------
C2ST accuracy is biased ABOVE 0.5 in finite samples with a flexible classifier
(the shared primitive uses an MLP sized 2**ceil(log2(10*d)) per layer -- 64x64 at
d=6, against 500+500 points). The arm-vs-arm accuracies are therefore compared
against an EMPIRICAL null built by running the identical code path on two
INDEPENDENT sampling calls from the SAME arm. Same dimension, same draw count,
same classifier, same folds -- so the null carries the identical finite-sample
bias, and only the excess over it is evidence of a real difference.

INTERPRETATION TRAP
-------------------
For a parameter neither arm identifies (omega, s_v under intercept_only), both
posteriors are ~the prior, so C2ST reads 0.5 and log_prob agrees -- trivially,
because both arms return the same prior, not because the summaries agree. The
joint statistic is therefore also reported over the IDENTIFIED cells only,
selected by contraction on this shared validation set.
"""
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from cogformer.diagnostics.metric.adaptive_c2st import compute_c2st_accuracy

from .addm_bf_apply import (
    CogFormerEncoderSummary, _load_cogformer, _sample_chunked,
    active_labels, ckpt_path as frozen_ckpt_path, FROZEN_COLOR,
    _pinned, _VARIANT_SUFFIX,
)
from .addm_bf_baseline import (
    ADDMBaselineSimulator, ckpt_path as baseline_ckpt_path,
)

BASELINE_COLOR = "#132a70"   # BayesFlow's default, as used by the baseline arm
OUT_ROOT = Path("./outputs/data/encoder_rep/addm_compare")
FIG_ROOT = Path("./outputs/figures/encoder_rep/addm_compare")


def build_shared_validation(args, device):
    """One simulated validation set, rendered for BOTH arms.

    ADDMBaselineSimulator's ``data`` tensor is byte-identical to the input
    CogFormerEncoderSummary assembles internally, so encoding that same tensor is
    what guarantees the arms see the same trials rather than merely the same
    distribution.
    """
    sim = ADDMBaselineSimulator(case=args.case, num_obs=args.num_obs,
                                first_fixation=args.first_fixation, fix_v=args.fix_v,
                                fix_s_v=args.fix_s_v)
    out = sim.sample(args.n_val)
    raw, true_params, masks = out["data"], out["params"], out["masks"]

    cogformer = _load_cogformer(args, device)
    summaries = CogFormerEncoderSummary(cogformer, device).summarize_raw(raw)
    logging.info(f"shared validation: {raw.shape[0]} datasets x {raw.shape[1]} trials, "
                 f"{true_params.shape[1]} active cells; summary dim {summaries.shape[1]}")
    return raw, summaries, true_params, masks


def contraction(true, pred):
    return 1.0 - (pred.std(axis=1) ** 2).mean(axis=0) / true.var(axis=0)


def c2st_one(a, b, seed):
    """Joint C2ST for one dataset. Returns held-out accuracy (0.5 == identical)."""
    return compute_c2st_accuracy(samples_a=a, samples_b=b, random_state=seed)


def run_c2st(draws_a, draws_b, idx, cols, n_jobs, label):
    logging.info(f"[c2st/{label}] {len(idx)} datasets x {draws_a.shape[1]} draws, "
                 f"{len(cols)} dims, n_jobs={n_jobs}")
    accs = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(c2st_one)(draws_a[i][:, cols], draws_b[i][:, cols], int(i))
        for i in idx
    )
    return np.asarray(accs, dtype=float)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = torch.device(args.encoder_device)
    tag = args.out_tag or (args.case + _VARIANT_SUFFIX[tuple(_pinned(args.fix_v,
                                                                    args.fix_s_v))])
    outdir, figdir = OUT_ROOT / tag, FIG_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    raw, summaries, true_params, masks = build_shared_validation(args, device)
    plain, pretty = active_labels(masks[0])

    frozen = keras.saving.load_model(
        frozen_ckpt_path(args.case, args.fix_v, args.fix_s_v) + "/model.keras")
    baseline = keras.saving.load_model(
        baseline_ckpt_path(args.case, args.first_fixation, args.tag, args.fix_v,
                           args.fix_s_v) + "/model.keras")

    # ── posterior draws on the SHARED datasets ────────────────────────────────
    dr_f = _sample_chunked(frozen, "summaries", summaries, args.num_samples,
                           args.sample_chunk_size)
    dr_b = _sample_chunked(baseline, "data", raw, args.num_samples,
                           args.sample_chunk_size)
    # Second independent call from the FROZEN arm: the empirical null. Same
    # dimension, same draw count, same classifier -> same finite-sample bias.
    dr_null = _sample_chunked(frozen, "summaries", summaries, args.num_samples,
                              args.sample_chunk_size)

    # ── (a) log posterior density at the truth, PAIRED over shared datasets ───
    lp_f = frozen.log_prob({"summaries": summaries, "params": true_params,
                            "masks": masks})
    lp_b = baseline.log_prob({"data": raw, "params": true_params, "masks": masks})
    lp_f, lp_b = np.asarray(lp_f).ravel(), np.asarray(lp_b).ravel()
    d = lp_f - lp_b
    se = d.std(ddof=1) / np.sqrt(len(d))
    logging.info(
        f"\nlog q(theta_true | x), shared {len(lp_f)} datasets:\n"
        f"  frozen   {lp_f.mean():8.3f} +/- {lp_f.std(ddof=1)/np.sqrt(len(lp_f)):.3f}\n"
        f"  baseline {lp_b.mean():8.3f} +/- {lp_b.std(ddof=1)/np.sqrt(len(lp_b)):.3f}\n"
        f"  paired diff (frozen - baseline) {d.mean():+.3f} +/- {se:.3f} "
        f"(z = {d.mean()/se:+.2f})")

    # ── per-parameter contraction on the shared set, to pick identified cells ─
    c_f, c_b = contraction(true_params, dr_f), contraction(true_params, dr_b)
    per_param = pd.DataFrame({"param": plain, "contr_frozen": c_f, "contr_baseline": c_b})
    logging.info("\n" + per_param.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    identified = [j for j in range(len(plain))
                  if min(c_f[j], c_b[j]) >= args.identified_threshold]
    logging.info(f"identified cells (min contraction >= {args.identified_threshold}): "
                 f"{[plain[j] for j in identified]}")

    # ── (b) per-dataset C2ST, joint and identified-only, each against its null ─
    rng = np.random.default_rng(0)
    idx = rng.choice(len(true_params), size=min(args.n_c2st, len(true_params)),
                     replace=False)
    all_cols = list(range(len(plain)))
    subsets = {"joint": all_cols}
    if identified and len(identified) < len(all_cols):
        subsets["identified"] = identified

    rows = []
    for name, cols in subsets.items():
        arm = run_c2st(dr_f, dr_b, idx, cols, args.n_jobs, f"{name}/arms")
        null = run_c2st(dr_f, dr_null, idx, cols, args.n_jobs, f"{name}/null")
        excess = arm.mean() - null.mean()
        se_e = np.sqrt(arm.var(ddof=1)/len(arm) + null.var(ddof=1)/len(null))
        rows.append({"subset": name, "n_dims": len(cols), "n_datasets": len(idx),
                     "c2st_arms": arm.mean(), "c2st_null": null.mean(),
                     "excess": excess, "se": se_e, "z": excess/se_e})
        np.savez(outdir / f"c2st_{name}.npz", arms=arm, null=null, idx=idx)
        logging.info(f"[c2st/{name}] arms {arm.mean():.4f}  null {null.mean():.4f}  "
                     f"excess {excess:+.4f} +/- {se_e:.4f}  (z = {excess/se_e:+.2f})")

    c2st_df = pd.DataFrame(rows)
    c2st_df.to_csv(outdir / "c2st_summary.csv", index=False, float_format="%.4f")
    per_param.to_csv(outdir / "contraction_shared.csv", index=False, float_format="%.4f")
    pd.DataFrame({"logprob_frozen": lp_f, "logprob_baseline": lp_b}).to_csv(
        outdir / "logprob_paired.csv", index=False, float_format="%.4f")
    np.savez(outdir / "shared_validation.npz", true_params=true_params, masks=masks,
             draws_frozen=dr_f, draws_baseline=dr_b, logprob_frozen=lp_f,
             logprob_baseline=lp_b)

    _plot(rows, outdir, figdir, lp_f, lp_b, tag)
    logging.info(f"[saved] {outdir}")


def _plot(rows, outdir, figdir, lp_f, lp_b, tag):
    n = len(rows)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 3.6))
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, rows):
        arm = np.load(outdir / f"c2st_{row['subset']}.npz")
        bins = np.linspace(0.4, 1.0, 40)
        ax.hist(arm["null"], bins=bins, alpha=0.55, label="null (frozen vs frozen)",
                color="grey")
        ax.hist(arm["arms"], bins=bins, alpha=0.55, label="frozen vs baseline",
                color=FROZEN_COLOR)
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_title(f"C2ST — {row['subset']} ({row['n_dims']}d)")
        ax.set_xlabel("held-out accuracy")
        ax.legend(fontsize=7)
    ax = axes[-1]
    ax.scatter(lp_b, lp_f, s=9, alpha=0.4, color=FROZEN_COLOR)
    lo = min(lp_b.min(), lp_f.min()); hi = max(lp_b.max(), lp_f.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(r"baseline  $\log q(\theta_{true}\mid x)$")
    ax.set_ylabel(r"frozen  $\log q(\theta_{true}\mid x)$")
    ax.set_title("paired log posterior density")
    fig.tight_layout()
    path = figdir / f"addm_arm_compare_{tag}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", type=str, default="intercept_only")
    p.add_argument("--first_fixation", type=str, default="item0")
    p.add_argument("--fix_v", action="store_true")
    p.add_argument("--fix_s_v", action="store_true",
                   help="Pin s_v (combine with --fix_v to pin both). Must match the "
                        "flags the checkpoints were trained with.")
    p.add_argument("--tag", type=str, default="vfixed_e500",
                   help="Baseline arm's checkpoint tag.")
    p.add_argument("--out_tag", type=str, default=None)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--sample_chunk_size", type=int, default=50)
    p.add_argument("--n_c2st", type=int, default=200,
                   help="Datasets to run C2ST on (each fits an MLP with 5-fold CV).")
    p.add_argument("--n_jobs", type=int, default=12)
    p.add_argument("--identified_threshold", type=float, default=0.5,
                   help="Min contraction in BOTH arms for a cell to count as identified.")
    p.add_argument("--cogformer_checkpoint", type=str, required=True)
    p.add_argument("--encoder_device", type=str, default="cuda")
    p.add_argument("--num_models", type=int, default=3)
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
    main()
