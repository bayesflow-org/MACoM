"""The three aDDM/encoder figures as one page panel.

    row 1   recovery, both arms, 2 x 5
    row 2   pooled cross-family geometry   |   CKA Gram block

Every block is drawn by the SAME function that draws its standalone figure --
``combined_recovery.draw_grid``, ``crossfamily_representation.draw_pooled`` and
``addm_representation.plot_gram_panel`` -- into a matplotlib SubFigure rather than a
figure of its own. Nothing about what the blocks SHOW is re-derived here; if a block
changes, this panel changes with it.

A note on the type sizes. The convention everywhere else is to draw at 11 in, twice a
5.5 in \\textwidth, so \\includegraphics scales by 0.5 and a 12 pt label lands at 6 pt
on the page. That still holds for the panel as a whole, but the two bottom blocks now
occupy roughly 40%% and 60%% of the width they had as standalone figures while their
text cannot shrink by the same factor without becoming illegible. The sizes below are
therefore reduced by ~15%%, not by the ~45%% the geometry would imply: the bottom
panels are deliberately text-heavy relative to their plotting area, which is the price
of putting three full-width figures in one.

SubFigure has no tight_layout, so each block is positioned with explicit margins.

    python -m experiments.experiment_4.combined_panel
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .combined_recovery import ROWS, load_arm, draw_grid, parse_args as rec_args
from .crossfamily_representation import pooled_embedding, draw_pooled, POOLED
from .addm_representation import plot_gram_panel, gram_of, _unit_gram

FIG = Path("outputs/figures/encoder_rep")

# Block heights in inches. Recovery is 2 rows of square panels at 11/5 in wide plus
# its labels; the bottom row is sized to the geometry block, which is the taller of
# the two (square panels plus a title, an x-label AND a legend row).
H_TOP, H_BOTTOM = 4.62, 2.98
# Bottom split. The CKA block carries four panels plus a colorbar against the
# geometry's two, so it gets the larger share.
W_SPLIT = (0.375, 0.625)

# ~0.85 of the standalone sizes; see the module docstring.
FS_GEOM = (14.0, 11.0, 9.0, 10.0)             # title, label, tick, legend
FS_CKA = (10.0, 8.5, 7.0, 6.5, 8.5)           # title, label, tick, cbar tick, annot

# Explicit margins, one dict per block, in that block's subfigure coordinates.
ADJ_REC = dict(left=0.052, right=0.995, top=0.950, bottom=0.095,
               wspace=0.34, hspace=0.30)
ADJ_GEOM = dict(left=0.125, right=0.995, top=0.895, bottom=0.315, wspace=0.10)
# Marker area in points^2, and the legend's compensating magnification. Both follow
# the panel, unlike the font sizes: a dot has no legibility floor the way type does,
# so there is no reason not to shrink it with the plotting area.
GEOM_MARKER, GEOM_LEG_SCALE, GEOM_CONTOUR_LW = 5.0, 3.2, 0.4
# Three columns, not five. A single row of five entries is ~4.6 in wide -- wider than
# the 4.1 in block it belongs to -- so it ran under the CKA block and collided with
# that colorbar's first tick label. Two rows of three keep it over its own block.
GEOM_LEG_NCOL = 3
# The CKA block's own gridspec margins. Its standalone figure leaves the matplotlib
# defaults; here the block has to start higher and end lower to use the row.
MARGINS_CKA = dict(left=0.075, right=0.995, top=0.895, bottom=0.175)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--geometry_npz", type=str,
                   default="outputs/finals/experiment-4/representation/crossfamily"
                           "/app3_summaries.npz")
    p.add_argument("--rep_npz", type=str,
                   default="outputs/finals/experiment-4/representation"
                           "/intercept_only_vsvfixed/summaries.npz")
    p.add_argument("--out", type=str,
                   default=str(FIG / "addm_compare/intercept_only_vsvfixed"
                                     "/addm_arm_compare_intercept_only_vsvfixed"
                                     "_panel.pdf"))
    p.add_argument("--fig_width", type=float, default=11.0)
    p.add_argument("--seed", type=int, default=0,
                   help="Must match the geometry run's seed so the MDS frame here is "
                        "the same one the standalone figure uses.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args()
    rec = rec_args()                       # the recovery block's style defaults

    arms = {"frozen": load_arm(rec.frozen_data),
            "baseline": load_arm(rec.baseline_data)}
    names = arms["frozen"][2]
    if arms["baseline"][2] != names:
        raise ValueError(f"arms differ: {names} vs {arms['baseline'][2]}")

    g = np.load(a.geometry_npz)
    summ = {"original": {f: g[f] for f in POOLED if f in g.files},
            "finetuned": {f: g[f + "__finetuned"] for f in POOLED
                          if f + "__finetuned" in g.files}}
    if not summ["finetuned"]:
        raise SystemExit(f"{a.geometry_npz} has no finetuned arrays")

    r = np.load(a.rep_npz, allow_pickle=True)
    fro, lea = r["frozen"], r["learned"]
    cka_obs = float(r["cka"])

    fig = plt.figure(figsize=(a.fig_width, H_TOP + H_BOTTOM))
    top, bottom = fig.subfigures(2, 1, height_ratios=[H_TOP, H_BOTTOM])
    left, right = bottom.subfigures(1, 2, width_ratios=list(W_SPLIT))

    axes = top.subplots(len(ROWS), len(names))
    draw_grid(axes, arms, names, rec)
    top.subplots_adjust(**ADJ_REC)

    emb, fam_lab, enc_lab, fams, encs = pooled_embedding(summ, a.seed)
    gax = left.subplots(1, 2)
    draw_pooled(left, gax, emb, fam_lab, enc_lab, fams, encs,
                fs=FS_GEOM, legend_anchor=0.005, adjust=ADJ_GEOM,
                marker_size=GEOM_MARKER, contour_lw=GEOM_CONTOUR_LW,
                legend_markerscale=GEOM_LEG_SCALE, legend_ncol=GEOM_LEG_NCOL)

    # outpath is ignored when target is given; the block draws and returns.
    plot_gram_panel(fro, lea, cka_obs, None, target=right, fs=FS_CKA,
                    margins=MARGINS_CKA)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {out}")


if __name__ == "__main__":
    main()
