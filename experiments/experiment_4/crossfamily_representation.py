"""Application 3: cross-family representational geometry of a frozen CogFormer encoder.

The encoder was trained on DDM + RDM + CDM. aDDM and LBA it never saw. If the encoder
learned a UNIVERSAL representation of evidence-accumulation data rather than three
memorised family-specific schemes, its geometry on the held-out families should look
like its geometry on the trained ones.

TWO ANALYSES, BECAUSE RSA NEEDS SHARED CONDITIONS
-------------------------------------------------
An RDM is a matrix over CONDITIONS, and conditions are not shared across families -- a
DDM dataset and an aDDM dataset are not the same stimulus -- so RDMs cannot simply be
correlated across families. Instead:

  (a) PER-FAMILY SCORE. For each family, how well does the encoder RDM predict the
      PARAMETER RDM? That is one scalar per family, and the scalars are comparable
      because the parameter RDM is built from the five parameters every family shares
      (v, a, tau, s_v, s_tau). Trained families set the reference; if the held-out
      family scores far lower, the representational scheme does not transfer.

  (b) POOLED EMBEDDING. Put every family's datasets in ONE RDM and ask what the
      geometry organises by -- family identity, or parameters -- and crucially where
      the held-out families fall: inside the manifold spanned by the trained three,
      or off it. Drawn twice, side by side, for the encoder before and after LBA
      finetuning, from a single shared MDS fit so the panels are comparable.

The per-family score (a) and the between-family CKA table stay on the ORIGINAL encoder
and the original four families, so their numbers remain comparable with earlier runs;
LBA enters the pooled embedding only.

MATCHED DESIGNS
---------------
All families are simulated at the same case (intercept-only), num_obs and regressor
budget. Without that, a family difference in the design distribution would masquerade
as a difference in representational geometry.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rsatoolbox.rdm import calc_rdm, compare
from scipy.stats import gaussian_kde
from sklearn.manifold import MDS

from cogformer.adapter import Adapter
from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms import DDM, RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.addm import ADDM
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.priors import (ddm_priors, rdm_priors,
                                                        cdm_priors, addm_priors,
                                                        lba_priors)
from cogformer.simulator.benchmarks.ssms.link_funs import (ddm_link_fun, rdm_link_fun,
                                                           cdm_link_fun, addm_link_fun,
                                                           lba_link_fun)
from cogformer.utils.plot_utils import FAMILY_PALETTES
from experiments.configs import MODELS as _BASE
from experiments.utils import add_global_params
from .representation import simulate_repeats, assemble_input, cka
from .addm_representation import rdm_of, parameter_rdm, build_dataset
from .addm_bf_apply import CogFormerEncoderSummary, _load_cogformer

SHARED = ["v", "a", "tau", "s_v", "s_tau"]      # present in all five families
TRAINED = ["ddm", "rdm", "cdm"]
HELD_OUT = ["addm", "lba"]                      # neither was in the encoder's training mix
POOLED = TRAINED + HELD_OUT
LBA_PARAMS = ["v", "v_diff", "a", "tau", "s_v", "s_tau"]

# Family hues come from the project-wide registry so that "orange is LBA" means the
# same thing here as in every other figure. aDDM has no registry entry; teal is the
# only hue the four registered families leave unclaimed. Held-out families also take
# a different MARKER, so the trained/held-out split survives greyscale printing and
# does not rest on hue alone.
COLORS = {f: FAMILY_PALETTES[f]()["main_effect"] for f in ("ddm", "rdm", "cdm", "lba")}
COLORS["addm"] = "#0E7C86"
MARKERS = {f: ("D" if f in HELD_OUT else "o") for f in POOLED}

OUT = Path("./outputs/data/encoder_rep/representation/crossfamily")
FIG = Path("./outputs/figures/encoder_rep/representation/crossfamily")

SPECS = {
    "ddm":  (DDM,  ddm_priors,  ddm_link_fun),
    "rdm":  (RDM,  rdm_priors,  rdm_link_fun),
    "cdm":  (CDM,  cdm_priors,  cdm_link_fun),
    "addm": (ADDM, addm_priors, addm_link_fun),
    "lba":  (LBA,  lba_priors,  lba_link_fun),
}


def _family_cmap(color):
    """White-to-family-colour ramp, transparent at the low end.

    A filled contour needs a colormap, and a per-family one keeps the background
    reading as that family rather than as a generic heat scale. Transparent at the
    bottom so the panels do not turn into solid blocks where a family is sparse.
    """
    rgb = mcolors.to_rgb(color)
    return mcolors.LinearSegmentedColormap.from_list(
        "fam", [(*rgb, 0.0), (*rgb, 0.18), (*rgb, 0.34)])


def family_setup(key, fixed_A=0.5):
    cls, prior_fn, link_fn = SPECS[key]
    if key == "addm":
        params = ["v", "v_diff", "omega", "a", "tau", "s_v", "s_tau"]
        model = cls(first_fixation="item0")
    elif key == "lba":
        # fixed_A pins the start-point range so that `a` is the THRESHOLD, which is
        # what the finetuned checkpoint was trained against. Leaving A free makes `a`
        # the start-point range instead -- the bug that held zero-shot recovery at
        # 0.24 until it was found (0.86 once fixed).
        params = LBA_PARAMS
        model = cls(fixed_A=fixed_A)
    else:
        params = add_global_params(_BASE[key], ["pi"], [r"$\pi$"])["intrinsic_params"]
        model = cls()
    mf = ModelFamily(model=model, prior=prior_fn(), intrinsic_params=params)
    link = {k: v for k, v in link_fn().items() if k in params}
    return mf, params, link


def simulate_family(key, n_cond, n_rep, num_obs, seed, fixed_A=0.5):
    """Simulate repeats for one family and return the RAW per-trial tensors.

    Separated from encoding so that several encoders can be applied to the SAME data.
    Simulating once per encoder does NOT work even with a fixed seed: the numba kernels
    draw from numba's own RNG, which np.random.seed does not control, so two calls
    produce different datasets. Comparing encoder A on dataset X against encoder B on
    dataset Y drives every similarity to ~0 regardless of the encoders.
    """
    mf, params, link = family_setup(key, fixed_A)
    dc = {"1": params}
    mk = {"free_intrinsics": params, "fixed_intrinsics": [], "fixed_values": {}}
    samples, reps = simulate_repeats(
        mf, mf.model, n_conditions=n_cond, n_repeats=n_rep, num_obs=num_obs,
        design_config=dc, mask_kwargs=mk, link_fun=link,
        max_num_regressors=4, max_num_categories=4, seed=seed)
    # simulate_repeats captures ONE condition per simulate() call, so each obs holds a
    # single condition's trials. Slice the batch-level fields to match rather than
    # handing assemble_input a full batch against a single-condition obs.
    def one(i, obs):
        sub = dict(samples)
        sub["design_matrices"] = samples["design_matrices"][i:i + 1]
        na = np.asarray(samples.get("num_alternatives", np.full(n_cond, 2)))
        sub["num_alternatives"] = np.atleast_1d(na)[i:i + 1]
        ptc = samples.get("per_trial_context", {}) or {}
        sub["per_trial_context"] = {k: np.asarray(v)[i:i + 1] for k, v in ptc.items()}
        o = {"rts": obs["rts"][None], "choices": obs["choices"][None]}
        return assemble_input(sub, o)[0]

    raw = np.stack([one(i, obs) for i, cond in enumerate(reps) for obs in cond])
    # Intercept-only design: flat index is row * n_params + param, so the intercept
    # row's cells are simply the first len(params) entries.
    theta = {p: samples["param_matrices"][:, params.index(p)] for p in SHARED}
    return raw, theta


def encode_family(key, enc, n_cond, n_rep, num_obs, seed, fixed_A=0.5):
    """Convenience wrapper: simulate one family and encode with a single encoder."""
    raw, theta = simulate_family(key, n_cond, n_rep, num_obs, seed, fixed_A)
    return enc.summarize_raw(raw).reshape(n_cond, n_rep, -1), theta


def shared_parameter_rdm(theta):
    """One RDM from the five parameters every family has, standardised.

    Standardising matters: the families' priors put v, a, tau on different scales
    (measured v ranges: DDM [0.01,1.76], RDM [-0.09,1.00], CDM [0.42,1.60]). Without it
    the parameter RDM would be dominated by whichever parameter that family's prior
    happens to spread widest, and the per-family scores would not be comparable.
    """
    from rsatoolbox.data import Dataset
    M = np.column_stack([theta[p] for p in SHARED]).astype(float)
    M = (M - M.mean(0)) / (M.std(0) + 1e-12)
    ds = Dataset(measurements=M,
                 obs_descriptors={"condition": np.arange(M.shape[0])},
                 channel_descriptors={"channel": np.arange(M.shape[1])})
    return calc_rdm(ds, method="euclidean", descriptor="condition")


def plot_scores(df, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = [COLORS[f] for f in df.family]
    ax.bar(df.family, df.rsa_encoder_vs_params, color=cols)
    ref = df[df.family.isin(TRAINED)].rsa_encoder_vs_params
    ax.axhspan(ref.min(), ref.max(), color="grey", alpha=.18,
               label="range over TRAINED families")
    ax.axhline(ref.mean(), color="grey", ls="--", lw=1, label="trained mean")
    ax.set_ylabel("RSA: encoder RDM vs shared-parameter RDM")
    ax.set_title("Does the representational scheme transfer to the held-out family?",
                 fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def noise_ceiling(summ):
    """Split-half CKA/RSA within a family: the same conditions, different noise.

    THE CONTROL THAT MAKES THE OFF-DIAGONAL INTERPRETABLE. Raw CKA(x, x) is 1.0 by
    construction, so a matrix with 1.0 on the diagonal tells you nothing about whether
    near-zero off-diagonals mean "orthogonal geometry" or "this metric cannot detect
    similarity across independently simulated stimulus sets at all".

    Splitting the repeats gives two independent MEASUREMENTS of the same conditions.
    If the split-half score is high, the metric works and a near-zero off-diagonal is
    a real statement about geometry. If it is low, the off-diagonals are uninformative
    and the orthogonality reading must be withdrawn.
    """
    n_rep = summ.shape[1]
    if n_rep < 2:
        return float("nan"), float("nan")
    h = n_rep // 2
    a, b = summ[:, :h], summ[:, h:2 * h]
    rsa = float(compare(rdm_of(a), rdm_of(b), method="corr")[0, 0]) if h >= 2 else float("nan")
    return rsa, cka(a.mean(1), b.mean(1))


def between_family_cka(summ_by_fam):
    """CKA between every pair of families, diagonal replaced by the noise ceiling.

    No longer plotted -- the matrix panel was dropped from the pooled figure -- but
    still computed and written to CSV, because it is what the orthogonality claim
    rests on. Diagonal = split-half ceiling, not the trivial CKA(x, x) = 1.
    """
    fams = list(summ_by_fam)
    M = np.zeros((len(fams), len(fams)))
    for i, a in enumerate(fams):
        for j, b in enumerate(fams):
            M[i, j] = cka(summ_by_fam[a].mean(1), summ_by_fam[b].mean(1))
    for i, f in enumerate(fams):
        M[i, i] = noise_ceiling(summ_by_fam[f])[1]
    return fams, M


DISPLAY = {"ddm": "DDM", "rdm": "RDM", "cdm": "CDM", "addm": "aDDM", "lba": "LBA"}


def _fam_label(fam):
    return f"{DISPLAY.get(fam, fam.upper())}{' (held out)' if fam in HELD_OUT else ''}"


# Font sizes for the pooled figure. Separate from anything else here because this
# is the one figure of the set that goes in the paper at full width.
FS_PANEL_TITLE, FS_AXIS_LABEL, FS_TICK, FS_LEGEND = 18, 14, 11, 12.5
# The panel title stays a step above the axis label: the titles say which encoder
# a side is, the labels only name the MDS axes.


def geometry_summary(emb, fam_lab, enc_lab, fams, encs):
    """What the MDS panels show, as numbers.

    A scatter can only be read qualitatively -- "the families look separate", "LBA
    barely moved" -- and those are exactly the claims worth pinning down. Four
    quantities, all in the shared MDS frame so they are comparable across panels:

      spread          mean distance of a family's points to its own centroid.
      nearest         distance from its centroid to the closest OTHER family's.
      separation      nearest / spread. Above ~1 the family occupies its own
                      region; near or below 1 it overlaps its neighbour.
      centroid_shift  how far the family's centroid moved between the two
                      encoders. This is the persistence number: small means
                      finetuning left that family's geometry where it was.

    The shift is only meaningful because both panels come from ONE MDS fit; with
    per-panel fits it would measure the arbitrary rotation between them.
    """
    cent = {(e, f): emb[(enc_lab == e) & (fam_lab == f)].mean(0)
            for e in encs for f in fams}
    rows = []
    for e in encs:
        for f in fams:
            pts = emb[(enc_lab == e) & (fam_lab == f)]
            c = cent[(e, f)]
            spread = float(np.linalg.norm(pts - c, axis=1).mean())
            others = [float(np.linalg.norm(c - cent[(e, g)])) for g in fams if g != f]
            nearest = min(others) if others else float("nan")
            shift = (float(np.linalg.norm(cent[(encs[0], f)] - cent[(encs[1], f)]))
                     if len(encs) > 1 else float("nan"))
            rows.append({"encoder": e, "family": f, "n": int(len(pts)),
                         "spread": spread, "nearest_family": nearest,
                         "separation": nearest / spread if spread else float("nan"),
                         "centroid_shift": shift})
    df = pd.DataFrame(rows)
    # Shift relative to spread: 0.1 means the family moved a tenth of its own width,
    # which is the scale on which "it did not really move" is a fair statement.
    df["shift_over_spread"] = df.centroid_shift / df.spread
    return df


def plot_pooled(summ_by_enc, outpath, seed=0, csv_path=None):
    """MDS of the pooled condition means, one panel per encoder.

    Left is the encoder as trained on DDM + RDM + CDM; right is the same encoder
    after LBA finetuning. Both panels carry every family, so the question the figure
    answers is where the two HELD-OUT families (aDDM, LBA) sit relative to the
    trained three, and whether finetuning moved them.

    ONE MDS FIT, BOTH PANELS. An MDS embedding is arbitrary up to rotation and
    reflection, so fitting each panel separately would put the two in unrelated
    frames and any apparent movement between them would be an artefact of the fit.
    The two encoders' points are embedded together and then split, which makes the
    panels share axes -- and the standardisation is likewise computed once, over the
    pooled matrix, so neither encoder gets its own scaling.

    Returns the geometry summary table, and writes it to `csv_path` when given.
    """
    emb, fam_lab, enc_lab, fams, encs = pooled_embedding(summ_by_enc, seed)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 5.2))
    draw_pooled(fig, ax, emb, fam_lab, enc_lab, fams, encs)
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

    df = geometry_summary(emb, fam_lab, enc_lab, fams, encs)
    if csv_path is not None:
        df.to_csv(csv_path, index=False, float_format="%.4f")
        logging.info(f"[saved] {csv_path}")
    logging.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def pooled_embedding(summ_by_enc, seed=0):
    """The shared MDS fit, split out from the drawing.

    Separate because the combined three-part panel draws the same embedding into a
    subfigure and must not refit it: MDS is arbitrary up to rotation and reflection,
    so a second fit would put the composite's panels in a different frame from the
    standalone figure's and the two would silently disagree.
    """
    encs = list(summ_by_enc)
    fams = list(summ_by_enc[encs[0]])
    blocks, fam_lab, enc_lab = [], [], []
    for e in encs:
        for fam in fams:
            s = summ_by_enc[e][fam]
            blocks.append(s.mean(1))
            fam_lab += [fam] * s.shape[0]
            enc_lab += [e] * s.shape[0]
    X = np.concatenate(blocks)
    fam_lab, enc_lab = np.array(fam_lab), np.array(enc_lab)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    emb = MDS(n_components=2, random_state=seed, n_init=4,
              normalized_stress="auto").fit_transform(Xs)
    return emb, fam_lab, enc_lab, fams, encs


def draw_pooled(fig, ax, emb, fam_lab, enc_lab, fams, encs, fs=None,
                legend_anchor=-0.02, tight_rect=(0, 0.10, 1, 1), adjust=None,
                marker_size=14, contour_lw=0.5, legend_markerscale=1.8,
                legend_ncol=None):
    """Draw the two panels and the legend into an existing figure or subfigure.

    ``fig`` may be a SubFigure, which has no tight_layout -- pass ``adjust`` (a
    subplots_adjust kwargs dict) in that case and ``tight_rect`` is ignored.

    ``marker_size`` is in points^2 and so does NOT follow the panel: at the combined
    panel's ~40%% width the default 14 covers about six times as much of the panel,
    the five clouds merge into one mass and the density contours underneath stop
    being visible at all. A caller drawing small must shrink it.
    """
    FS_T, FS_L, FS_TK, FS_LG = fs or (FS_PANEL_TITLE, FS_AXIS_LABEL,
                                      FS_TICK, FS_LEGEND)
    titles = {"original": "Before finetuning", "finetuned": "After finetuning"}
    # MDS output is centred, so symmetric limits give both axes the same span; with a
    # square box that makes the DATA aspect equal too, which a distance embedding
    # needs -- a stretched axis misreports how far apart two families are.
    lim = float(np.abs(emb).max()) * 1.08
    grid = np.linspace(-lim, lim, 160)
    GX, GY = np.meshgrid(grid, grid)
    mesh = np.vstack([GX.ravel(), GY.ravel()])

    for k, e in enumerate(encs):
        for fam in fams:
            m = (fam_lab == fam) & (enc_lab == e)
            pts = emb[m]
            # Density contour UNDER the points. Where a family's cloud overlaps
            # another's the scatter alone just reads as mixed colour; the contour
            # shows where each family's mass actually is, which is the thing being
            # compared between panels.
            if len(pts) > 5:
                try:
                    dens = gaussian_kde(pts.T)(mesh).reshape(GX.shape)
                    # Levels start well above zero. A KDE is positive everywhere, so
                    # contouring from 0 paints every family across the whole panel
                    # and five overlapping tails turn the background grey. Starting
                    # at a fraction of the peak draws only where the family's mass
                    # actually is.
                    hi = float(dens.max())
                    levels = np.linspace(0.22 * hi, hi, 5)
                    ax[k].contourf(GX, GY, dens, levels=levels, zorder=0,
                                   cmap=_family_cmap(COLORS[fam]), extend="neither")
                    ax[k].contour(GX, GY, dens, levels=levels[1:], zorder=1,
                                  colors=[COLORS[fam]], linewidths=contour_lw,
                                  alpha=0.45)
                except Exception:
                    pass          # singular covariance: scatter alone still stands
            ax[k].scatter(pts[:, 0], pts[:, 1], s=marker_size, alpha=.65,
                          color=COLORS[fam],
                          marker=MARKERS[fam], linewidths=0, label=_fam_label(fam),
                          zorder=3)
        ax[k].set_title(titles.get(e, e), fontsize=FS_T)
        ax[k].set_xlabel("MDS 1", fontsize=FS_L)
        ax[k].set_xlim(-lim, lim); ax[k].set_ylim(-lim, lim)
        ax[k].tick_params(labelsize=FS_TK)
        ax[k].set_box_aspect(1)
        for side in ("top", "right"):
            ax[k].spines[side].set_visible(False)
    ax[0].set_ylabel("MDS 2", fontsize=FS_L)
    # Shared y axis. Both panels come from ONE MDS fit and already carry identical
    # limits, so the right panel's tick numbers are a verbatim repeat of the left's;
    # printing them twice invites the reader to check whether they differ.
    ax[1].set_yticklabels([])

    handles, labels = ax[0].get_legend_handles_labels()
    # No suptitle: the panel titles already say which encoder each side is, and the
    # rest belongs in the caption.
    if adjust is not None:
        fig.subplots_adjust(**adjust)
    else:
        fig.tight_layout(rect=list(tight_rect))
    # Anchored BELOW the figure, not just near its bottom. Measured on the tight
    # bboxes, 0.01 put the legend 0.09 in INTO the "MDS 1" labels; -0.02 clears them.
    # savefig with bbox_inches="tight" grows the canvas to take it in, so a negative
    # anchor costs nothing but the gap it buys.
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, legend_anchor),
               ncol=legend_ncol or len(fams), fontsize=FS_LG, frameon=False,
               markerscale=legend_markerscale)
    return ax


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.encoder_device)

    # Two encoders, same architecture, both 128-d out. `model_embedding` is the only
    # thing the finetune resized (3 -> 4) and the encoder never touches it, so the
    # summaries are directly comparable.
    orig_num_models = a.num_models
    enc_orig = CogFormerEncoderSummary(_load_cogformer(a, dev), dev)
    a.num_models = a.finetuned_num_models
    a.cogformer_checkpoint = a.finetuned_checkpoint
    enc_ft = CogFormerEncoderSummary(_load_cogformer(a, dev), dev)
    a.num_models = orig_num_models
    logging.info(f"loaded both encoders (original num_models={orig_num_models}, "
                 f"finetuned={a.finetuned_num_models})")

    shp = (a.n_conditions, a.n_repeats, -1)
    summ = {"original": {}, "finetuned": {}}
    rows = []
    for fam in POOLED:
        logging.info(f"[{fam}] simulating {a.n_conditions}x{a.n_repeats} and encoding")
        # Simulated ONCE and encoded twice. Re-simulating per encoder does not work
        # even at a fixed seed -- the numba kernels use their own RNG -- and comparing
        # two encoders on two different datasets drives every similarity to ~0.
        raw, th = simulate_family(fam, a.n_conditions, a.n_repeats, a.num_obs,
                                  a.seed, a.fixed_A)
        s = enc_orig.summarize_raw(raw).reshape(shp)
        summ["original"][fam] = s
        summ["finetuned"][fam] = enc_ft.summarize_raw(raw).reshape(shp)
        if fam == "lba":
            continue                        # per-family transfer table is unchanged
        score = float(compare(rdm_of(s), shared_parameter_rdm(th), method="corr")[0, 0])
        nc_rsa, nc_cka = noise_ceiling(s)
        rows.append({"family": fam, "trained": fam in TRAINED,
                     "rsa_encoder_vs_params": score,
                     "noise_ceiling_rsa": nc_rsa, "noise_ceiling_cka": nc_cka})
        logging.info(f"[{fam}] RSA encoder-vs-shared-params = {score:.4f}  "
                     f"| split-half ceiling: RSA {nc_rsa:.4f} CKA {nc_cka:.4f}")
    df = pd.DataFrame(rows)
    ref = df[df.trained].rsa_encoder_vs_params
    held = float(df[~df.trained].rsa_encoder_vs_params.iloc[0])
    logging.info(f"\ntrained families: {ref.mean():.4f} +/- {ref.std():.4f}  "
                 f"| held-out aDDM: {held:.4f}")
    df.to_csv(OUT / "app3_per_family_rsa.csv", index=False, float_format="%.4f")
    plot_scores(df, FIG / "app3_transfer_scores.pdf")
    plot_pooled(summ, FIG / "app3_pooled_geometry.pdf", a.seed,
                csv_path=OUT / "app3_geometry_summary.csv")
    # CKA matrix keeps the original encoder and the original four families, so the
    # CSV stays comparable with the runs that produced the orthogonality claim.
    fams, M = between_family_cka({f: summ["original"][f] for f in TRAINED + ["addm"]})
    pd.DataFrame(M, index=fams, columns=fams).to_csv(OUT / "app3_between_family_cka.csv",
                                                     float_format="%.4f")
    np.savez_compressed(
        OUT / "app3_summaries.npz",
        **{f: v for f, v in summ["original"].items()},
        **{f"{f}__finetuned": v for f, v in summ["finetuned"].items()})
    logging.info(f"[saved] {OUT} | {FIG}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n_conditions", type=int, default=200)
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cogformer_checkpoint", type=str, required=True,
                   help="Encoder as trained on DDM + RDM + CDM (left panel).")
    p.add_argument("--finetuned_checkpoint", type=str, required=True,
                   help="LBA-finetuned encoder (right panel).")
    p.add_argument("--fixed_A", type=float, default=0.5,
                   help="LBA start-point range, pinned to match the finetuned "
                        "checkpoint. See family_setup.")
    p.add_argument("--encoder_device", type=str, default="cuda")
    p.add_argument("--num_models", type=int, default=3)
    p.add_argument("--finetuned_num_models", type=int, default=4)
    for k, v in [("encoder_num_layers", 8), ("decoder_num_layers", 8),
                 ("encoder_num_heads", 8), ("decoder_num_heads", 8), ("num_seeds", 32),
                 ("seed_dim", 128), ("proj_dim", 256), ("model_embed_dim", 8),
                 ("time_embedding_dim", 32), ("pos_embedding_dim", 32)]:
        p.add_argument(f"--{k}", type=int, default=v)
    return p.parse_args()


if __name__ == "__main__":
    main()
