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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare

from cogformer.simulator.benchmarks.ssms.link_funs import addm_link_fun
from cogformer.utils.plot_utils import cogformer_fm_colors
from .representation import simulate_repeats, cka
from .addm_bf_apply import (CogFormerEncoderSummary, _load_cogformer, INTRINSIC_PARAMS,
                            VARIABLE_NAMES, resolve_case, FROZEN_COLOR, BASELINE_COLOR,
                            _pinned, _VARIANT_SUFFIX)
from .addm_bf_baseline import (ADDMBaselineSimulator, MAX_NUM_REGRESSORS,
                               MAX_NUM_CATEGORIES, CASE_SAMPLE_KWARGS,
                               ckpt_path as baseline_ckpt_path, active_labels)

# BASELINE_COLOR is imported from addm_bf_apply (single definition).
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


def _unit_gram(summaries):
    """Centred Gram scaled to unit Frobenius norm over its OFF-DIAGONAL entries.

    Required before the two arms can be differenced at all. The frozen summary is
    128-d and the learned one 16-d, and their raw Gram norms differ by up to 3.6x
    (measured: ratio 0.89 on varying_values, 3.58 on intercept_only_vsvfixed), so a raw
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


# Correspondence scatter. Taken from the shared palette rather than hardcoded, and
# deliberately NOT an arm colour: the cloud is a joint plot of both arms, so borrowing
# either one's colour would imply it belongs to that arm.
SCATTER_COLOR = cogformer_fm_colors()["interaction"]   # #FF6969


def _shared_ticks(ax, ticks, labels=None):
    """Identical tick positions on both axes.

    Matplotlib locates x and y independently, which on a square 200x200 map gave
    y every 25 and x every 50, and on the scatter 9 y-ticks against 5 x-ticks. Both
    axes carry the same quantity here, so they must carry the same ticks.
    """
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)


# Its own, smaller set. This figure packs four panels plus a colorbar into one
# \textwidth row, so the shared FS_TITLE/FS_LABEL/FS_TICK -- sized for a figure with
# one or two panels -- crowd it: at 15 pt the four titles nearly touch, and the tick
# numbers compete with the matrices they annotate for attention they do not deserve.
FS_MAT_TITLE, FS_MAT_LABEL, FS_MAT_TICK, FS_MAT_CBTICK = 12, 10, 8.5, 8
# (title, label, tick, colorbar tick, annotation). Passed as a unit so a caller that
# draws this block at reduced width -- the combined panel puts it in half a row -- can
# scale every size in it with one argument instead of five.
MAT_FS = (FS_MAT_TITLE, FS_MAT_LABEL, FS_MAT_TICK, FS_MAT_CBTICK, 10)


def _fig_size_in(fig):
    """Size in inches of a Figure OR a SubFigure.

    SubFigure has no get_size_inches; its bbox is in display units of the parent, so
    dividing by the shared dpi gives the inches this block actually occupies -- which
    is what the scatter's square aspect has to be computed against.
    """
    return tuple(np.asarray(fig.bbox.size) / fig.dpi)


def _matrix_scatter_panel(mats, titles, colors, cbar_label, xy, scatter_title,
                          stat_text, axis_label, scatter_labels, outpath, cmap,
                          target=None, fs=MAT_FS, height_ratio=5.1, hspace=0.60,
                          width_ratios=(3.4, 0.34, 1.05), gap_in=0.825,
                          divider_in=0.143, margins=None, axes_override=None):
    """Three heatmaps on ONE shared scale plus a full-height correspondence scatter.

    Draws into ``target`` (a Figure or SubFigure) instead of a new figure when one is
    given. With ``target`` set nothing is saved and the block is laid out inside
    whatever area the caller allocated, so the combined three-part panel reuses this
    block rather than keeping a second copy of the layout.

    Shared by the RSA and CKA panels so their layout cannot drift apart -- the two
    diverged once already when a hand-rolled recovery plot silently overwrote the
    BayesFlow one, and duplicated layout code is how that happens.

    ``mats`` is (first, second, first - second); that difference is expected to read
    fainter than its two inputs, which is the shared structure showing rather than a
    rendering artifact. It is labelled "Difference", not "Residual": nothing is
    fitted here, it is a plain subtraction of two matrices on one shared scale.

    The colormap is DIVERGING (Spectral_r, red positive) and the limits are symmetric
    about zero, so zero lands on the colormap's centre in all three panels and they
    stay directly comparable. viridis was tried first and rejected: it is monotonic, so
    values clustered near zero all mapped to the same mid-range teal, and the residual
    -- whose entire content is which DIRECTION each pair moved -- read as near-uniform.

    Spectral_r is a ColorBrewer palette, not perceptually uniform, so apparent contrast
    does not scale linearly with magnitude. That is an accepted trade here: these panels
    are read for structure, and the value a reader needs precisely is the scalar in the
    correspondence panel, not a cell estimated off the colorbar.
    """
    M0 = mats[0]
    n = M0.shape[0]
    iu = np.triu_indices(n, k=1)
    v = max(np.percentile(np.abs(M[iu]), 98) for M in mats)

    # Gap between the x-labels and the colorbar. Measured on the LABEL's tight bbox,
    # not the axes box: the label sits below the box, so an axes-to-axes measurement
    # overstates the clearance and lets the two collide. At 4.7/17.5 with hspace 0.42
    # the clearance was 0.008 in -- the colorbar touching "Simulated dataset". Raising
    # hspace alone buys clearance out of the maps themselves (0.75 gives 0.21 in but
    # shrinks each square 1.73 -> 1.57 in), so the height ratio goes up with it: at
    # 5.1 / hspace 0.60 the clearance is 0.19 in with the squares still 1.73 in, for
    # 0.26 in of extra figure height.
    FS_T, FS_L, FS_TK, FS_CB, FS_AN = fs
    own = target is None
    fig = (plt.figure(figsize=(FIG_W_WIDE, FIG_W_WIDE * height_ratio / 17.5))
           if own else target)
    if axes_override is not None:
        # The caller has already placed every axes -- used by the combined panel,
        # which sizes this block's maps to match the geometry block's panels and so
        # cannot let a gridspec decide them. The scatter is then left exactly where it
        # was put; the square-and-full-height repositioning below is skipped.
        axes, cax, ax = axes_override
        axes = list(axes)
    else:
        # margins go to the gridspec rather than subplots_adjust: the scatter is
        # repositioned by hand afterwards, and subplots_adjust would move it back.
        outer = fig.add_gridspec(1, 3, width_ratios=list(width_ratios), wspace=0.10,
                                 **(margins or {}))
        maps_gs = outer[0, 0].subgridspec(2, 3, height_ratios=[1.0, 0.055],
                                          hspace=hspace, wspace=0.14)
        axes = [fig.add_subplot(maps_gs[0, i]) for i in range(3)]
        cax = fig.add_subplot(maps_gs[1, :])
        ax = fig.add_subplot(outer[0, 2])

    # imshow centres cell i at i and ends the axis at n - 0.5, so np.arange(0, n, 50)
    # stops at 150 and the last 50 draws run off with nothing marking where the matrix
    # ends. The terminal tick goes on the axis EDGE (n - 0.5, the outer boundary of
    # cell n-1) and is labelled n, so it reads as "after the 200th draw" rather than
    # mislabelling cell 199 as 200.
    step = np.arange(0, n, 50)
    map_ticks = np.append(step.astype(float), n - 0.5)
    map_labels = [str(int(t)) for t in step] + [str(n)]
    for a, M, title, color in zip(axes, mats, titles, colors):
        im = a.imshow(M, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest",
                      aspect="equal")
        a.set_title(title, color=color, fontsize=FS_T, pad=8)
        a.set_xlabel(axis_label, fontsize=FS_L)
        _shared_ticks(a, map_ticks, map_labels)
        a.tick_params(labelsize=FS_TK)
        a.set_xlim(-0.5, n - 0.5); a.set_ylim(n - 0.5, -0.5)
    axes[0].set_ylabel(axis_label, fontsize=FS_L)
    for a in axes[1:3]:
        a.set_yticklabels([])

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=FS_L)
    cb.ax.tick_params(labelsize=FS_CB)

    x, y = xy
    # Limits from the DATA rather than symmetric about zero, but identical on both
    # axes so y = x still renders at 45 degrees. Gram entries are near-symmetric so
    # this changes nothing there; z-scored dissimilarities have a floor, and forcing
    # symmetry left the whole lower-left of that panel empty.
    lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.45", lw=1, zorder=1)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.scatter(x, y, s=5, alpha=0.30, color=SCATTER_COLOR, edgecolors="none", zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    sc_ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    _shared_ticks(ax, sc_ticks[(sc_ticks >= lo) & (sc_ticks <= hi)])
    ax.tick_params(labelsize=FS_TK)
    ax.set_xlabel(scatter_labels[0], fontsize=FS_L)
    ax.set_ylabel(scatter_labels[1], fontsize=FS_L)
    ax.set_title(scatter_title, fontsize=FS_T, pad=8)
    # The statistic sits inside the panel, lower right, rather than in the title:
    # it belongs to the cloud, and the titles across the row then read as a plain
    # list of what each panel shows. Axes coordinates, so it follows the box when
    # the panel is repositioned below.
    ax.text(0.97, 0.045, stat_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_AN)
    ax.spines[["top", "right"]].set_visible(False)

    # Size the scatter to the FULL height of the left block -- top of the maps down to
    # the bottom of the colorbar -- and make it square by setting its position
    # directly. aspect="equal" cannot do this: it shrinks the box to fit whatever
    # vertical space the title and tick labels happen to leave. With the box square
    # and xlim == ylim, the y = x reference still renders at 45 degrees.
    fig.canvas.draw()
    top = axes[0].get_position().y1
    bot = cax.get_position().y0
    if axes_override is not None:
        # Divider only: span the maps block as placed, and leave the scatter alone.
        # Midway between the two blocks' INK, not between their axes boxes. The maps'
        # terminal "200" overhangs the box on the right and the scatter's y-label and
        # ticks hang off it on the left, by different amounts -- splitting the boxes
        # put the rule underneath that y-label.
        R = fig.canvas.get_renderer()
        inv = fig.transSubfigure.inverted()
        x_maps = inv.transform((axes[-1].get_tightbbox(R).x1, 0))[0]
        x_scat = inv.transform((ax.get_tightbbox(R).x0, 0))[0]
        xd = (x_maps + x_scat) / 2
        fig.add_artist(Line2D([xd, xd], [cax.get_position().y0, top],
                              transform=fig.transSubfigure,
                              color="0.72", linewidth=1.1, zorder=10))
        return
    H = top - bot
    figW, figH = _fig_size_in(fig)
    W = H * figH / figW                      # equal side lengths in INCHES
    # gap and divider offset are given in INCHES, not figure fractions. They used to
    # be fractions (0.075 and 0.013), which is the same thing only while the block is
    # 11 in wide: in the combined panel's 6.4 in subfigure that 0.075 shrank from
    # 0.83 in to 0.48 in and the scatter's y-label landed on the Difference map.
    gap = gap_in / figW
    x0 = cax.get_position().x1 + gap
    # A square at the block's full height does not always fit the width left over. It
    # does at 11 x 3.2 in; it does not in a subfigure that is relatively taller. Rather
    # than let it run off the right edge, cap the side length and hang the square from
    # the TOP of the maps, which keeps it aligned with the row it belongs to.
    if x0 + W > 1.0:
        W = 1.0 - x0
        H = W * figW / figH
        bot = top - H
    ax.set_aspect("auto")
    ax.set_position([x0, bot, W, H])

    # Divider: the left block shares one color scale and one normalization; the
    # scatter is a different kind of plot of the same numbers. Placed just clear of
    # the block so the scatter's y-LABEL stays on the scatter's side of the rule.
    xd = cax.get_position().x1 + divider_in / figW
    fig.add_artist(Line2D([xd, xd], [cax.get_position().y0, top],
                          transform=fig.transSubfigure,
                          color="0.72", linewidth=1.1, zorder=10))

    if own:
        fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def plot_gram_panel(fro, lea, cka_obs, outpath, **kw):
    """CKA side: two centred Gram matrices, their difference, and the scatter.

    Each Gram is scaled to unit Frobenius norm over its off-diagonal, then BOTH are
    multiplied by one shared constant that puts them in [-1, 1]. A common factor
    cancels from every ratio, so the scatter's alignment is still exactly the reported
    CKA and the residual still satisfies ||R|| / ||A|| = sqrt(2(1 - CKA)).

    This is NOT cosine normalization, which divides each entry by sqrt(K_ii K_jj) and
    shifts the alignment 1-2 points off the reported CKA in variant-dependent
    directions. Entries stay SIGNED -- about half the off-diagonals are negative and
    zero means orthogonal -- so the map is diverging and centred on zero.
    """
    A_f, A_l = _unit_gram(fro), _unit_gram(lea)
    iu = np.triu_indices_from(A_f, k=1)
    A_f, A_l = (A_f, A_l) / np.abs(np.concatenate([A_f[iu], A_l[iu]])).max()
    _matrix_scatter_panel(
        # Set Transformer first, so the Difference (Set Transformer - CogFormer) reads
        # as first minus second, matching the order it is displayed in.
        (A_l, A_f, A_l - A_f),
        ("Set Transformer", "CogFormer", "Difference"),
        (BASELINE_COLOR, FROZEN_COLOR, "0.25"),
        "Summary similarity", (A_f[iu], A_l[iu]),
        "Correspondence", f"CKA = {cka_obs:.3f}", "Simulated dataset",
        ("CogFormer", "Set Transformer"), outpath, "Spectral_r", **kw)


def _z_rdm(rdm):
    """RDM z-scored over its off-diagonal -- the normalization Pearson r applies.

    The two RDMs are ~21x apart in scale (crossnobis units depend on the
    representation), so they cannot share a color scale untransformed. RSA is a
    Pearson correlation, which centres AND scales, so z-scoring is the transform that
    makes the picture show exactly what the reported r measures: it reproduces
    corr(z_f, z_l) = r and ||z_l - z_f|| / sqrt(n) = sqrt(2(1 - r)).

    A consequence worth stating: z-scored dissimilarities are SIGNED, centred on the
    mean dissimilarity rather than on zero, so these maps read "more/less dissimilar
    than typical" and use a diverging colormap. The raw RDM's "0 = identical" anchor
    is not preserved -- it cannot be, by any transform that also makes the two arms
    comparable.
    """
    M = rdm.get_matrices()[0].astype(float)
    iu = np.triu_indices_from(M, k=1)
    v = M[iu]
    Z = (M - v.mean()) / (v.std() + 1e-12)
    np.fill_diagonal(Z, 0.0)
    return Z


def plot_rdm_panel(rdm_f, rdm_l, score, outpath, **kw):
    """RSA side, mirroring plot_gram_panel exactly: two RDMs, difference, scatter."""
    Z_f, Z_l = _z_rdm(rdm_f), _z_rdm(rdm_l)
    iu = np.triu_indices_from(Z_f, k=1)
    Z_f, Z_l = (Z_f, Z_l) / np.abs(np.concatenate([Z_f[iu], Z_l[iu]])).max()
    _matrix_scatter_panel(
        (Z_l, Z_f, Z_l - Z_f),
        ("Set Transformer", "CogFormer", "Difference"),
        (BASELINE_COLOR, FROZEN_COLOR, "0.25"),
        "Dissimilarity, z-scored", (Z_f[iu], Z_l[iu]),
        "Correspondence", f"r = {score:.3f}", "Simulated dataset",
        ("CogFormer", "Set Transformer"), outpath, "Spectral_r", **kw)


def plot_rsa_vs_null(score, null_rsa, cka_obs, null_cka, outpath):
    """Observed RSA against the dimensionality-matched null, with CKA for reference."""
    fig, ax = plt.subplots(figsize=(FIG_W_SMALL, FIG_W_SMALL * 0.75))
    ax.hist(null_rsa, bins=20, color="grey", alpha=.7, label="matched null (RSA)")
    ax.axvline(score, color=FROZEN_COLOR, lw=2, label=f"observed RSA {score:.3f}")
    ax.axvline(np.mean(null_cka), color="grey", ls=":", lw=1.5,
               label=f"null CKA {np.mean(null_cka):.3f}")
    ax.axvline(cka_obs, color=BASELINE_COLOR, lw=2, ls="--",
               label=f"observed CKA {cka_obs:.3f}")
    ax.set_title("RSA vs dimensionality-matched null", fontsize=FS_TITLE)
    ax.set_xlabel("Similarity"); ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS_LEG, frameon=False)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


# The CKA panel is a single histogram with a long title; at the shared sizes the
# title wraps past the axes and the tick numbers dominate a plot whose content is
# one distribution. Its own, smaller set.
FS_CKA_TITLE, FS_CKA_LABEL, FS_CKA_TICK = 11, 9, 8


def plot_cka_vs_null(cka_obs, null_cka, outpath):
    """Observed CKA against the same null, with the z-score in the title."""
    fig, ax = plt.subplots(figsize=(FIG_W_SMALL, FIG_W_SMALL * 0.75))
    ax.hist(null_cka, bins=20, color="grey", alpha=.75,
            label=f"matched null {np.mean(null_cka):.3f}")
    ax.axvline(cka_obs, color=BASELINE_COLOR, lw=2, label=f"observed {cka_obs:.3f}")
    z = (cka_obs - np.mean(null_cka)) / (np.std(null_cka) + 1e-12)
    ax.set_title(f"CKA vs dimensionality-matched null   (z = {z:+.2f})", fontsize=FS_CKA_TITLE)
    ax.set_xlabel("Linear CKA"); ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS_CKA_LABEL, frameon=False)
    ax.tick_params(labelsize=FS_CKA_TICK)
    ax.xaxis.label.set_fontsize(FS_CKA_LABEL)
    ax.yaxis.label.set_fontsize(FS_CKA_LABEL)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


# SIZED FOR THE PAGE. Target is a 5.5 in LaTeX \\textwidth; the figures are drawn at
# twice that so \\includegraphics scales by 0.5 and every size below lands at half its
# nominal value. The old 17.5 in width scaled by 0.31 instead, which put a 12 pt title
# at under 4 pt on the page.
FIG_W_WIDE, FIG_W_SMALL = 11.0, 5.6
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, FS_ANNOT = 15, 13, 11, 11, 10


def plot_representation_set(rdm_f, rdm_l, fro, lea, score, null_rsa,
                            cka_obs, null_cka, figdir):
    """Every representation figure, one plot per file.

    These were two multi-panel strips (app1_frozen_vs_learned, cka_gram_matrices).
    Splitting them means each matrix can be read at a usable size, and a figure can
    be dropped into a document without cropping three neighbours off it.
    """
    plot_rdm_panel(rdm_f, rdm_l, score, figdir / "rdm_rsa.pdf")
    plot_gram_panel(fro, lea, cka_obs, figdir / "gram_cka.pdf")
    plot_rsa_vs_null(score, null_rsa, cka_obs, null_cka, figdir / "rsa_vs_null.pdf")
    plot_cka_vs_null(cka_obs, null_cka, figdir / "cka_vs_null.pdf")


def plot_app2(df, outpath):
    fig, ax = plt.subplots(figsize=(max(FIG_W_SMALL, 0.9 * len(df)), 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.beta_frozen, 0.4, color=FROZEN_COLOR, label="frozen CogFormer")
    ax.bar(x + 0.2, df.beta_learned, 0.4, color=BASELINE_COLOR, label="learned SetTransformer")
    ax.plot(x - 0.2, df.rsa_frozen, "o", ms=3, color="k", label="marginal RSA (diluted)")
    ax.plot(x + 0.2, df.rsa_learned, "o", ms=3, color="k")
    for i, c in enumerate(df.contraction_frozen):
        if c == c:                                    # skip NaN
            top = max(df.beta_frozen[i], df.beta_learned[i], df.rsa_frozen[i])
            ax.annotate(f"contr\n{c:.2f}", (i, top + .02), ha="center",
                        fontsize=FS_ANNOT - 2, color="dimgrey")
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(df.param, rotation=30, ha="right")
    ax.set_ylabel("Standardized regression weight on encoder RDM")
    ax.set_title("What the representation encodes: unique contribution per parameter\n"
                 "(bars = joint regression, dots = diluted marginal, text = posterior contraction)",
                 fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEG)
    ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
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
