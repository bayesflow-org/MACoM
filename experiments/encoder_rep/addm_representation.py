"""RSA and CKA on CogFormer's encoder representation — aDDM.

Experiment 4's posterior-level results say what the two arms DECODE. This says what
they REPRESENT. That distinction is not cosmetic: the frozen CogFormer summary is
128-dimensional and the learned SetTransformer summary is 16-dimensional, so they
share no coordinate system and no posterior metric can compare them directly.

APPLICATIONS
------------
1. Frozen vs learned geometry. Do a transferred summary and a task-specific one
   organise the SAME datasets the same way? This is the universal-representation
   claim stated directly rather than inferred from posterior agreement.
2. What the representation encodes. Compare the encoder RDM against one model RDM per
   parameter. This can separate two things the posteriors cannot: a parameter absent
   from the representation, versus present but not decoded. omega has posterior
   contraction ~0.0 under intercept_only; whether its structure is in the encoder is a
   different question with a different answer.
3. Cross-family geometry (see --families). The encoder was trained on DDM+RDM+CDM and
   never saw aDDM, so aDDM is the held-out family.

METRIC CHOICE
-------------
RSA with CROSSNOBIS is primary. Plain RDM correlations are inflated by measurement
noise, and the inflation depends on each system's noise level, so it does NOT cancel
between systems of different dimensionality. Crossnobis is unbiased but needs repeated
measurements per condition, which here are exact and cheap: hold theta fixed and
re-simulate.

CKA is the cross-check, not a co-equal number. Linear CKA is dominated by a few
high-variance directions, so it can read high while the representations differ on
everything those directions do not span.

THE NULL THAT MATTERS
---------------------
Comparing 128-d against 16-d gives a score partly determined by the dimensionality gap
alone. Measured on isotropic Gaussian data, a RANDOM 128->16 projection scores
CKA = 0.33 -- not 0. Both metrics are therefore reported against a
dimensionality-matched null built from random projections of the frozen summary, and
only the excess over that null is evidence of shared structure.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

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
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare

from cogformer.simulator.benchmarks.ssms.link_funs import addm_link_fun
from .representation import simulate_repeats, cka
from .addm_bf_apply import (CogFormerEncoderSummary, _load_cogformer, INTRINSIC_PARAMS,
                            VARIABLE_NAMES, resolve_case, FROZEN_COLOR,
                            _pinned, _VARIANT_SUFFIX)
from .addm_bf_baseline import (ADDMBaselineSimulator, MAX_NUM_REGRESSORS,
                               MAX_NUM_CATEGORIES, CASE_SAMPLE_KWARGS,
                               ckpt_path as baseline_ckpt_path, active_labels)

BASELINE_COLOR = "#132a70"
OUT = Path("./outputs/data/encoder_rep/representation")
FIG = Path("./outputs/figures/encoder_rep/representation")


def build_dataset(summaries):
    """rsatoolbox Dataset from (n_cond, n_rep, dim) summaries.

    obs_descriptors carry the condition index (what an RDM entry is between) and the
    repeat index, which calc_rdm uses as the cross-validation fold for crossnobis.
    """
    n_cond, n_rep, dim = summaries.shape
    return Dataset(
        measurements=summaries.reshape(n_cond * n_rep, dim),
        obs_descriptors={
            "condition": np.repeat(np.arange(n_cond), n_rep),
            "fold": np.tile(np.arange(n_rep), n_cond),
        },
        channel_descriptors={"channel": np.arange(dim)},
    )


def rdm_of(summaries, method="crossnobis"):
    ds = build_dataset(summaries)
    kw = {"descriptor": "condition"}
    if method == "crossnobis":
        kw["cv_descriptor"] = "fold"
    return calc_rdm(ds, method=method, **kw)


def parameter_rdm(theta_col):
    """Model RDM from one parameter: absolute difference between conditions."""
    v = np.asarray(theta_col, dtype=float).reshape(-1, 1)
    ds = Dataset(measurements=v,
                 obs_descriptors={"condition": np.arange(len(v))},
                 channel_descriptors={"channel": np.array([0])})
    return calc_rdm(ds, method="euclidean", descriptor="condition")


def rdm_regression(encoder_rdm, param_rdms, names):
    """Regress the encoder RDM on ALL parameter RDMs jointly.

    The MARGINAL correlation between the encoder RDM and one parameter's RDM is
    diluted by every other parameter's contribution to the same geometry: measured at
    smoke size, `a` had posterior contraction 0.983 yet a marginal RSA of only 0.166.
    That is dilution, not absence.

    Standardising each RDM vector and solving jointly gives each parameter's UNIQUE
    contribution, holding the others fixed -- which is the quantity that answers
    "is this parameter represented?". Reported alongside the marginal so the gap
    between them is visible rather than hidden by a choice of estimator.
    """
    y = np.asarray(encoder_rdm.get_vectors()[0], dtype=float)
    X = np.column_stack([np.asarray(r.get_vectors()[0], dtype=float) for r in param_rdms])
    y = (y - y.mean()) / (y.std() + 1e-12)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.sum((y - X @ beta) ** 2) / np.sum(y ** 2)
    return dict(zip(names, beta)), float(r2)


def matched_null(frozen_sum, target_dim, n_draws, method, rng):
    """Dimensionality-matched null: random projections of the frozen summary.

    Answers 'what score would ANY 16-d readout of this 128-d representation get?', so
    the reported similarity is judged against the dimensionality gap rather than 0.
    """
    n_cond, n_rep, dim = frozen_sum.shape
    rdm_f = rdm_of(frozen_sum, method)
    flat = frozen_sum.reshape(-1, dim)
    rsa_null, cka_null = [], []
    for _ in range(n_draws):
        W = rng.normal(size=(dim, target_dim)) / np.sqrt(dim)
        proj = (flat @ W).reshape(n_cond, n_rep, target_dim)
        rsa_null.append(float(compare(rdm_f, rdm_of(proj, method), method="corr")[0, 0]))
        cka_null.append(cka(frozen_sum.mean(1), proj.mean(1)))
    return np.array(rsa_null), np.array(cka_null)


def encode_all(reps, frozen_enc, baseline_ap, sim):
    """Encode every (condition, repeat) dataset with BOTH encoders.

    Both consume the identical 72-channel per-trial tensor, so any geometry difference
    is the encoder's, not the input's.
    """
    n_cond, n_rep = len(reps), len(reps[0])
    raw = []
    for cond in reps:
        for obs in cond:
            raw.append(sim.assemble(obs))
    raw = np.stack(raw)                                  # (n_cond*n_rep, T, 72)
    fro = frozen_enc.summarize_raw(raw)
    with torch.no_grad():
        lea = keras.ops.convert_to_numpy(baseline_ap.summary_network(
            keras.ops.convert_to_tensor(raw)))
    return (fro.reshape(n_cond, n_rep, -1),
            np.asarray(lea).reshape(n_cond, n_rep, -1))


def gram_of(summaries):
    """Centred linear Gram matrix K = XX^T of the per-condition means.

    CKA compares centred Gram matrices, so K is to CKA what the RDM is to RSA.
    """
    x = summaries.mean(1)
    x = x - x.mean(0)
    return x @ x.T


def _square_heatmap(M, title, color, cbar_label, outpath, cmap, vmin=None, vmax=None):
    """One matrix, one figure, square pixels.

    These were panels in a four-across strip, which forced them to ~3 inches wide;
    at 200x200 conditions that put four condition rows in a single pixel column and
    the block structure was unreadable. Standalone they get room to resolve it.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                   aspect="equal")
    ax.set_title(title, color=color, fontsize=11, pad=8)
    ax.set_xlabel("condition"); ax.set_ylabel("condition")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_rdm(rdm, title, color, outpath):
    """One representational dissimilarity matrix."""
    _square_heatmap(rdm.get_matrices()[0], title, color,
                    "crossnobis dissimilarity", outpath, cmap="viridis")


def plot_gram(summaries, title, color, outpath):
    """One centred Gram matrix, symmetric diverging scale about zero.

    vmax is the 98th percentile of |K| rather than its max: a handful of extreme
    entries otherwise compress the whole map into the middle of the colormap. The
    colorbar is labelled so the clip is stated rather than implied.
    """
    K = gram_of(summaries)
    v = np.percentile(np.abs(K), 98)
    _square_heatmap(K, title, color, "centred inner product (clipped at |98th pct|)",
                    outpath, cmap="RdBu_r", vmin=-v, vmax=v)


def plot_rdm_correspondence(rdm_f, rdm_l, score, outpath):
    """Frozen against learned dissimilarity, one point per condition pair."""
    f = rdm_f.get_vectors()[0]; l = rdm_l.get_vectors()[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(f, l, s=6, alpha=0.3, color=FROZEN_COLOR, edgecolors="none")
    ax.set_xlabel("frozen RDM (crossnobis)"); ax.set_ylabel("learned RDM")
    ax.set_title(f"RDM correspondence   r = {score:.3f}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def _unit_gram(summaries):
    """Centred Gram scaled to unit Frobenius norm over its OFF-DIAGONAL entries.

    Required before the two arms can be differenced at all. The frozen summary is
    128-d and the learned one 16-d, and their raw Gram norms differ by up to 2x
    (measured: ratio 0.89 on varying_values, 1.98 on intercept_only_vfixed), so a raw
    K_learned - K_frozen would be dominated by that scale factor and would show "one
    encoder has larger inner products", not "the two organise conditions differently".

    CKA is invariant to isotropic scaling by construction, so a residual meant to
    explain a CKA value has to be invariant too. Normalising over the off-diagonal
    matches the unbiased HSIC estimator, which zeroes the diagonal before computing.
    """
    K = gram_of(summaries)
    iu = np.triu_indices_from(K, k=1)
    K = K / np.linalg.norm(K[iu])
    np.fill_diagonal(K, 0.0)
    return K


def plot_gram_residual(fro, lea, outpath):
    """Learned minus frozen, after each is scaled to unit norm.

    Positive (red) means the LEARNED encoder places that pair of conditions closer
    together than the frozen one does; negative (blue) the reverse. The two Gram
    heatmaps show what each arm encodes; this shows only where they disagree, which
    is the part the CKA scalar compresses into one number.

    The Frobenius norm of this residual is not free: for unit-norm matrices
    ||A - B||_F = sqrt(2(1 - <A,B>)), and <A,B> is the alignment CKA reports. It is
    printed in the title as a consistency check against cka_vs_null.pdf.
    """
    A_f, A_l = _unit_gram(fro), _unit_gram(lea)
    R = A_l - A_f
    iu = np.triu_indices_from(R, k=1)
    # Upper triangle only, matching how _unit_gram normalised: summing the full
    # symmetric matrix counts every pair twice and reports 2x the alignment.
    align = float(A_f[iu] @ A_l[iu])
    fro_norm = float(np.linalg.norm(R[iu]))

    v = np.percentile(np.abs(R[iu]), 98)
    _square_heatmap(
        R, f"learned - frozen residual\n"
           f"$\\|\\Delta K\\|_F$ = {fro_norm:.3f}   (alignment {align:.3f})",
        "0.25", "normalised inner-product difference (clipped at |98th pct|)",
        outpath, cmap="RdBu_r", vmin=-v, vmax=v)


def plot_gram_correspondence(fro, lea, cka_obs, outpath):
    """Frozen against learned centred inner product, one point per condition PAIR.

    The Gram analogue of plot_rdm_correspondence, and the honest visual for CKA:
    CKA compares centred Gram matrices, so this scatter is the alignment the scalar
    summarises. The two heatmaps show each Gram separately and leave the reader to
    compare 200x200 cells by eye; this shows the correspondence directly.

    Only OFF-DIAGONAL entries are plotted, matching the unbiased HSIC estimator in
    representation.cka, which zeroes the diagonal before computing anything. Keeping
    the diagonal would add n points at the squared-norm scale -- an order of magnitude
    above every off-diagonal entry -- that contribute nothing to the reported number.

    Two statistics are annotated because they are NOT the same thing. CKA is the
    reported quantity; the entrywise Pearson r is what the eye reads off this cloud.
    They differ because CKA is an uncentred alignment with HSIC's bias correction,
    so quoting only one invites reading the other off the picture.
    """
    Kf, Kl = gram_of(fro), gram_of(lea)
    iu = np.triu_indices_from(Kf, k=1)
    f, l = Kf[iu], Kl[iu]
    r = float(np.corrcoef(f, l)[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    # Gram entries are signed and centred on zero, unlike the all-positive RDM
    # dissimilarities, so the axes through the origin carry meaning here.
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.scatter(f, l, s=6, alpha=0.25, color=FROZEN_COLOR, edgecolors="none")
    ax.set_xlabel(f"frozen Gram ({fro.shape[-1]}-d), off-diagonal")
    ax.set_ylabel(f"learned Gram ({lea.shape[-1]}-d), off-diagonal")
    ax.set_title(f"Gram correspondence   CKA = {cka_obs:.3f}   (entrywise r = {r:.3f})",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_rsa_vs_null(score, null_rsa, cka_obs, null_cka, outpath):
    """Observed RSA against the dimensionality-matched null, with CKA for reference."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.hist(null_rsa, bins=20, color="grey", alpha=.7, label="matched null (RSA)")
    ax.axvline(score, color=FROZEN_COLOR, lw=2, label=f"observed RSA {score:.3f}")
    ax.axvline(np.mean(null_cka), color="grey", ls=":", lw=1.5,
               label=f"null CKA {np.mean(null_cka):.3f}")
    ax.axvline(cka_obs, color=BASELINE_COLOR, lw=2, ls="--",
               label=f"observed CKA {cka_obs:.3f}")
    ax.set_title("RSA vs dimensionality-matched null", fontsize=11)
    ax.set_xlabel("similarity"); ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_cka_vs_null(cka_obs, null_cka, outpath):
    """Observed CKA against the same null, with the z-score in the title."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.hist(null_cka, bins=20, color="grey", alpha=.75,
            label=f"matched null {np.mean(null_cka):.3f}")
    ax.axvline(cka_obs, color=BASELINE_COLOR, lw=2, label=f"observed {cka_obs:.3f}")
    z = (cka_obs - np.mean(null_cka)) / (np.std(null_cka) + 1e-12)
    ax.set_title(f"CKA vs dimensionality-matched null   (z = {z:+.2f})", fontsize=11)
    ax.set_xlabel("linear CKA"); ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_representation_set(rdm_f, rdm_l, fro, lea, score, null_rsa,
                            cka_obs, null_cka, figdir):
    """Every representation figure, one plot per file.

    These were two multi-panel strips (app1_frozen_vs_learned, cka_gram_matrices).
    Splitting them means each matrix can be read at a usable size, and a figure can
    be dropped into a document without cropping three neighbours off it.
    """
    d_f, d_l = fro.shape[-1], lea.shape[-1]
    plot_rdm(rdm_f, f"frozen CogFormer ({d_f}-d)", FROZEN_COLOR,
             figdir / "rdm_frozen.pdf")
    plot_rdm(rdm_l, f"learned SetTransformer ({d_l}-d)", BASELINE_COLOR,
             figdir / "rdm_learned.pdf")
    plot_gram(fro, f"frozen Gram ({d_f}-d)", FROZEN_COLOR,
              figdir / "gram_frozen.pdf")
    plot_gram(lea, f"learned Gram ({d_l}-d)", BASELINE_COLOR,
              figdir / "gram_learned.pdf")
    plot_gram_residual(fro, lea, figdir / "gram_residual.pdf")
    plot_rdm_correspondence(rdm_f, rdm_l, score, figdir / "rdm_correspondence.pdf")
    plot_gram_correspondence(fro, lea, cka_obs, figdir / "gram_correspondence.pdf")
    plot_rsa_vs_null(score, null_rsa, cka_obs, null_cka, figdir / "rsa_vs_null.pdf")
    plot_cka_vs_null(cka_obs, null_cka, figdir / "cka_vs_null.pdf")


def plot_app2(df, outpath):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df)), 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.beta_frozen, 0.4, color=FROZEN_COLOR, label="frozen CogFormer")
    ax.bar(x + 0.2, df.beta_learned, 0.4, color=BASELINE_COLOR, label="learned SetTransformer")
    ax.plot(x - 0.2, df.rsa_frozen, "o", ms=3, color="k", label="marginal RSA (diluted)")
    ax.plot(x + 0.2, df.rsa_learned, "o", ms=3, color="k")
    for i, c in enumerate(df.contraction_frozen):
        if c == c:                                    # skip NaN
            top = max(df.beta_frozen[i], df.beta_learned[i], df.rsa_frozen[i])
            ax.annotate(f"contr\n{c:.2f}", (i, top + .02), ha="center",
                        fontsize=6, color="dimgrey")
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(df.param, rotation=30, ha="right")
    ax.set_ylabel("standardised regression weight on encoder RDM")
    ax.set_title("What the representation encodes: unique contribution per parameter\n"
                 "(bars = joint regression, dots = diluted marginal, text = posterior contraction)",
                 fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    tag = variant_tag(args.case, args.fix_v, args.fix_s_v)
    outdir, figdir = OUT / tag, FIG / tag
    outdir.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.encoder_device)

    sim = ADDMBaselineSimulator(case=args.case, num_obs=args.num_obs,
                                first_fixation=args.first_fixation, fix_v=args.fix_v,
                                fix_s_v=args.fix_s_v)
    dc, mk = resolve_case(args.case, args.fix_v, args.fix_s_v)
    logging.info(f"simulating {args.n_conditions} conditions x {args.n_repeats} repeats")
    samples, reps = simulate_repeats(
        sim.model_family, sim.model_family.model,
        n_conditions=args.n_conditions, n_repeats=args.n_repeats, num_obs=args.num_obs,
        design_config=dc, mask_kwargs=mk,
        link_fun={k: f for k, f in addm_link_fun().items() if k in INTRINSIC_PARAMS},
        max_num_regressors=MAX_NUM_REGRESSORS, max_num_categories=MAX_NUM_CATEGORIES,
        extra_sample_kwargs=CASE_SAMPLE_KWARGS.get(args.case), seed=args.seed)

    designs = samples["design_matrices"]
    frozen_enc = CogFormerEncoderSummary(_load_cogformer(args, device), device)
    baseline_ap = keras.saving.load_model(
        baseline_ckpt_path(args.case, args.first_fixation, args.tag, args.fix_v,
                           args.fix_s_v) + "/model.keras")

    n_cond, n_rep = len(reps), len(reps[0])
    raw = np.stack([sim.assemble(designs[i:i+1], o["rts"][None], o["choices"][None])[0]
                    for i, cond in enumerate(reps) for o in cond])
    fro = frozen_enc.summarize_raw(raw).reshape(n_cond, n_rep, -1)
    with torch.no_grad():
        lea = np.asarray(keras.ops.convert_to_numpy(
            baseline_ap.summary_network(keras.ops.convert_to_tensor(raw))))
    lea = lea.reshape(n_cond, n_rep, -1)
    logging.info(f"summaries: frozen {fro.shape}  learned {lea.shape}")

    rdm_f, rdm_l = rdm_of(fro), rdm_of(lea)

    # ---- application 1: frozen vs learned geometry -------------------------------
    score = float(compare(rdm_f, rdm_l, method="corr")[0, 0])
    cka_obs = cka(fro.mean(1), lea.mean(1))
    rng = np.random.default_rng(args.seed)
    null_rsa, null_cka = matched_null(fro, lea.shape[-1], args.n_null, "crossnobis", rng)
    logging.info(f"[app1] RSA r={score:.4f}  null {null_rsa.mean():.4f}+/-{null_rsa.std():.4f}"
                 f"  | CKA {cka_obs:.4f}  null {null_cka.mean():.4f}+/-{null_cka.std():.4f}")
    plot_representation_set(rdm_f, rdm_l, fro, lea, score, null_rsa,
                            cka_obs, null_cka, figdir)

    # ---- application 2: what the representation encodes ---------------------------
    active = samples["param_masks"][0].astype(bool)
    plain, _ = active_labels(samples["param_masks"][0])
    theta = samples["param_matrices"][:, active]
    contr = _load_contraction(args.case, args.fix_v, plain, args.fix_s_v)
    param_rdms = [parameter_rdm(theta[:, j]) for j in range(len(plain))]
    beta_f, r2_f = rdm_regression(rdm_f, param_rdms, plain)
    beta_l, r2_l = rdm_regression(rdm_l, param_rdms, plain)
    logging.info(f"[app2] joint RDM regression R^2: frozen {r2_f:.3f}  learned {r2_l:.3f}")
    rows = []
    for j, name in enumerate(plain):
        rows.append({"param": name,
                     "rsa_frozen": float(compare(rdm_f, param_rdms[j], method="corr")[0, 0]),
                     "rsa_learned": float(compare(rdm_l, param_rdms[j], method="corr")[0, 0]),
                     "beta_frozen": beta_f[name], "beta_learned": beta_l[name],
                     "contraction_frozen": contr.get(name, np.nan)})
    df = pd.DataFrame(rows)
    logging.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(outdir / "app2_parameter_rsa.csv", index=False, float_format="%.4f")
    plot_app2(df, figdir / "app2_what_is_encoded.pdf")

    # null_rsa / null_cka are stored so the figures can be regenerated from this
    # file alone. They are reproducible from `frozen` and the seed, but recomputing
    # them costs an RDM per draw, which is the slow part of a pure replot.
    np.savez_compressed(outdir / "summaries.npz", frozen=fro, learned=lea,
                        theta=theta, param_names=np.array(plain, dtype=object),
                        null_rsa=null_rsa, null_cka=null_cka,
                        rsa=np.float64(score), cka=np.float64(cka_obs))
    pd.DataFrame([{"rsa": score, "rsa_null_mean": null_rsa.mean(),
                   "rsa_null_sd": null_rsa.std(), "cka": cka_obs,
                   "cka_null_mean": null_cka.mean(), "cka_null_sd": null_cka.std(),
                   "n_conditions": n_cond, "n_repeats": n_rep,
                   "dim_frozen": fro.shape[-1], "dim_learned": lea.shape[-1]}]
                 ).to_csv(outdir / "app1_geometry.csv", index=False, float_format="%.4f")
    logging.info(f"[saved] {outdir}  |  {figdir}")


def variant_tag(case, fix_v, fix_s_v):
    """`<case>` plus the pinning suffix, matching addm_bf_apply._variant."""
    return case + _VARIANT_SUFFIX[tuple(_pinned(fix_v, fix_s_v))]


def _load_contraction(case, fix_v, names, fix_s_v=False):
    """Posterior contraction per cell, for annotating application 2."""
    p = (Path("outputs/data/encoder_rep") /
         ("addm" + _VARIANT_SUFFIX[tuple(_pinned(fix_v, fix_s_v))]) /
         case / f"addm_{case}_bf_metrics.csv")
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    col = "param" if "param" in d.columns else d.columns[0]
    return dict(zip(d[col], d["contraction"])) if "contraction" in d.columns else {}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", type=str, default="varying_values")
    p.add_argument("--fix_v", action="store_true")
    p.add_argument("--fix_s_v", action="store_true",
                   help="Pin s_v (combine with --fix_v to pin both). Must match the "
                        "flags the checkpoints were trained with.")
    p.add_argument("--first_fixation", type=str, default="item0")
    p.add_argument("--tag", type=str, default="varying_e500",
                   help="Baseline arm checkpoint tag.")
    p.add_argument("--n_conditions", type=int, default=200,
                   help="Parameter vectors = RDM conditions.")
    p.add_argument("--n_repeats", type=int, default=4,
                   help="Independent datasets per theta. >=2 required for crossnobis; "
                        "more repeats reduce the noise floor of the unbiased estimate.")
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--n_null", type=int, default=30,
                   help="Random projections for the dimensionality-matched null.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cogformer_checkpoint", type=str, required=True)
    p.add_argument("--encoder_device", type=str, default="cuda")
    p.add_argument("--num_models", type=int, default=3)
    for k, v in [("encoder_num_layers", 8), ("decoder_num_layers", 8),
                 ("encoder_num_heads", 8), ("decoder_num_heads", 8), ("num_seeds", 32),
                 ("seed_dim", 128), ("proj_dim", 256), ("model_embed_dim", 8),
                 ("time_embedding_dim", 32), ("pos_embedding_dim", 32)]:
        p.add_argument(f"--{k}", type=int, default=v)
    return p.parse_args()


if __name__ == "__main__":
    main()
