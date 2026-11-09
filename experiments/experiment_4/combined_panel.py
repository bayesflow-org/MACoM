"""The three aDDM/encoder figures as one page panel.

    row 1   recovery, both arms, 2 x 5
    -------------------------------------------- horizontal rule
    row 2   pooled cross-family geometry   |   CKA Gram block

Every block is drawn by the SAME function that draws its standalone figure --
``combined_recovery.draw_grid``, ``crossfamily_representation.draw_pooled`` and
``addm_representation.plot_gram_panel`` -- into a matplotlib SubFigure rather than a
figure of its own. Nothing about what the blocks SHOW is re-derived here; if a block
changes, this panel changes with it.

THE BOTTOM ROW IS PLACED BY HAND, NOT BY GRIDSPEC. Its six plotting areas -- two MDS
panels, three Gram maps, one correspondence scatter -- all have to come out the same
size, and they belong to two blocks with different internal structure and different
margin needs. Nested gridspecs cannot express "these six squares are equal" across
that boundary: each block divides only the width it was given, so equal panels would
mean hand-tuning two width splits against each other until they happened to agree.
Instead the common side S is solved from the width budget (`_bottom_layout`), every
axes is positioned in inches, and the blocks are handed axes they did not create.

Besides the combined page, the two rows are also written on their own
(``<out>_recovery.pdf``, ``<out>_representation.pdf``) for layouts that place them
apart; they are drawn by the same code at the same width, so their scale matches.

Type sizes are shared by all three bottom panels (``FS_BOTTOM``) for the same reason:
with the panels equal, different title sizes across them read as a hierarchy that
isn't there.

    python -m experiments.experiment_4.combined_panel
"""
from experiments.paths import RESULTS_ROOT
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .combined_recovery import ROWS, load_arm, draw_grid, parse_args as rec_args
from .crossfamily_representation import (pooled_embedding, draw_pooled, POOLED,
                                        _stacked_legend)
from .addm_representation import plot_gram_panel

FIG = (RESULTS_ROOT / "figures/encoder_rep")

FIG_W = 11.0
H_TOP = 4.62
ADJ_REC = dict(left=0.052, right=0.995, top=0.950, bottom=0.135,
               wspace=0.34, hspace=0.30)

# One size per role, shared by all three bottom blocks: title, axis label, tick,
# legend, colorbar tick, in-panel annotation.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, FS_CBTICK, FS_ANNOT = 11.5, 9.5, 7.5, 9.5, 7.5, 8.5
FS_GEOM = (FS_TITLE, FS_LABEL, FS_TICK, FS_LEG)
FS_CKA = (FS_TITLE, FS_LABEL, FS_TICK, FS_CBTICK, FS_ANNOT)

# Marker area in points^2 and the legend's compensating magnification. These DO follow
# the panel, unlike the font sizes: a dot has no legibility floor the way type does,
# and at the standalone's s=14 the five clouds merge into one mass and the density
# contours underneath stop being visible.
GEOM_MARKER, GEOM_LEG_SCALE, GEOM_CONTOUR_LW = 5.0, 3.2, 0.4
# Three trained families on the first line, the two held-out ones on the second --
# drawn as two SEPARATE legends so the rows are not forced onto a shared column grid.
# The long labels are all on the second row, and a single ncol=3 legend sizes its
# columns to them, leaving the first row's three short entries spread through columns
# they do not fill.
GEOM_LEG_ROWS = (3, 2)
# Tightened from matplotlib's defaults so the block fits the combined width of the two
# MDS panels (2S + GAP_GEOM). At the defaults it is 3.96 in against a 2.83 in target
# and runs out under the CKA block.
GEOM_LEG_KW = dict(columnspacing=0.5, handletextpad=0.25, handlelength=0.6)

# --- bottom-row width budget, inches ----------------------------------------------
# Everything that is NOT one of the six squares. Measured from the standalone figures
# at these font sizes rather than guessed; _bottom_layout asserts the leftovers are
# positive so a font change that breaks the budget fails loudly instead of overlapping.
L_GEOM = 0.52       # "MDS 2" + its tick labels
GAP_GEOM = 0.22     # between the two MDS panels (the right one has no y ticks)
BLOCK_GAP = 0.46    # between the geometry block and the CKA block
L_CKA = 0.56        # "Simulated dataset" + its tick labels
# Wide enough for the tick labels that sit ON the shared edge: every map is labelled
# 0..200, so a narrow gap runs one map's terminal "200" straight into the next map's
# "0" and the pair reads as "2000".
GAP_MAP = 0.30
# Divider + the scatter's own y-label and ticks. The rule goes midway between the two
# blocks' ink, so this has to hold the maps' "200" overhang, the rule, and the
# scatter's y-label with clearance on both sides of the rule.
GAP_SCATTER = 0.86
R_PAD = 0.06

# --- bottom-row height budget, inches ---------------------------------------------
TOP_PAD = 0.34      # panel titles, and clearance under the row divider
FOOT_PAD = 0.52     # x tick labels + x label, below every panel, and clearance above the colorbar
CB_H = 0.085        # colorbar bar itself
CB_LABEL = 0.30     # "Summary similarity" under it
# The geometry legend's two rows, measured at FS_LEG with GEOM_LEG_ROW_GAP between
# them. The footer allowance is the TALLER of the two footers, not the colorbar's:
# budgeting for the colorbar alone put the legend's second row below the figure, and
# bbox_inches="tight" then silently grew the canvas to take it in.
LEG_BLOCK = 0.52
GEOM_LEG_ROW_GAP = 0.06
FOOTER = max(CB_H + CB_LABEL, LEG_BLOCK)
BOT_PAD = 0.04


def _bottom_layout(fig_w=FIG_W):
    """Common panel side S and the x of every column, in inches."""
    fixed = (L_GEOM + GAP_GEOM + BLOCK_GAP + L_CKA + 2 * GAP_MAP
             + GAP_SCATTER + R_PAD)
    S = (fig_w - fixed) / 6.0
    if S <= 0:
        raise ValueError(f"no width left for panels: fixed={fixed:.2f} in of {fig_w}")
    xs = {}
    x = L_GEOM
    xs["geom"] = [x, x + S + GAP_GEOM]
    x = xs["geom"][1] + S + BLOCK_GAP + L_CKA
    xs["maps"] = [x + i * (S + GAP_MAP) for i in range(3)]
    xs["scatter"] = xs["maps"][2] + S + GAP_SCATTER
    return S, xs


def _rect(x_in, y_in, w_in, h_in, fig_w, fig_h):
    return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--geometry_npz", type=str,
                   default=str(RESULTS_ROOT / "finals/experiment-4/representation/crossfamily"
                           "/app3_summaries.npz"))
    p.add_argument("--rep_npz", type=str,
                   default=str(RESULTS_ROOT / "finals/experiment-4/representation"
                           "/frozen_vs_learned/summaries.npz"))
    p.add_argument("--out", type=str,
                   default=str(FIG / "addm_compare/intercept_only_vsvfixed"
                                     "/addm_arm_compare_intercept_only_vsvfixed"
                                     "_panel.pdf"))
    p.add_argument("--fig_width", type=float, default=FIG_W)
    p.add_argument("--seed", type=int, default=0,
                   help="Must match the geometry run's seed so the MDS frame here is "
                        "the same one the standalone figure uses.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args()
    # Only combined_recovery's defaults: its parser must not see this script's argv.
    rec = rec_args([])

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
    fro, lea, cka_obs = r["frozen"], r["learned"], float(r["cka"])

    # Height follows from the solved panel side, so the row is exactly as tall as its
    # contents need -- no slack to trim afterwards.
    S, xs = _bottom_layout(a.fig_width)
    h_bottom = TOP_PAD + S + FOOT_PAD + FOOTER + BOT_PAD
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(a.fig_width, H_TOP + h_bottom))
    top, bottom = fig.subfigures(2, 1, height_ratios=[H_TOP, h_bottom])
    _draw_top(top, arms, names, rec)
    _draw_bottom(bottom, summ, (fro, lea, cka_obs), a.seed, a.fig_width, h_bottom, S, xs)
    _row_divider(fig, top, bottom)
    _save(fig, out)

    # The same two blocks as separate files, for a page layout that places them apart
    # (e.g. with a diagram between or beside them). Each is drawn by the same code at
    # the same width, so at equal \includegraphics width they match the combined
    # panel's scale exactly.
    fig = plt.figure(figsize=(a.fig_width, H_TOP))
    _draw_top(fig, arms, names, rec)
    _save(fig, out.with_name(out.stem + "_recovery.pdf"))
    fig = plt.figure(figsize=(a.fig_width, h_bottom))
    _draw_bottom(fig, summ, (fro, lea, cka_obs), a.seed, a.fig_width, h_bottom, S, xs)
    _save(fig, out.with_name(out.stem + "_representation.pdf"))


def _save(fig, path):
    w, h = fig.get_size_inches()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {path} ({w:.2f} x {h:.2f} in)")


def _draw_top(target, arms, names, rec):
    """Recovery block, both arms, into a Figure or SubFigure."""
    axes = target.subplots(len(ROWS), len(names))
    draw_grid(axes, arms, names, rec)
    target.subplots_adjust(**ADJ_REC)


def _draw_bottom(target, summ, gram, seed, W, Hb, S, xs):
    """Geometry + CKA block into a Figure or SubFigure of size W x Hb inches."""
    fro, lea, cka_obs = gram
    y_panel = FOOTER + FOOT_PAD + BOT_PAD                # bottom edge of every panel
    y_foot = BOT_PAD + FOOTER                            # top of colorbar / legend

    gax = np.array([target.add_axes(_rect(x, y_panel, S, S, W, Hb))
                    for x in xs["geom"]])
    emb, fam_lab, enc_lab, fams, encs = pooled_embedding(summ, seed)
    # Legend top pinned to the same height as the colorbar's top so the two footers
    # line up across the row.
    draw_pooled(target, gax, emb, fam_lab, enc_lab, fams, encs, fs=FS_GEOM,
                adjust=False, marker_size=GEOM_MARKER, contour_lw=GEOM_CONTOUR_LW,
                legend_markerscale=GEOM_LEG_SCALE, legend_rows=GEOM_LEG_ROWS,
                legend_kw=GEOM_LEG_KW, legend_loc="upper center",
                legend_x=(xs["geom"][0] + S + GAP_GEOM / 2) / W,
                legend_anchor=y_foot / Hb, legend_row_gap=GEOM_LEG_ROW_GAP)

    map_axes = [target.add_axes(_rect(x, y_panel, S, S, W, Hb)) for x in xs["maps"]]
    cax = target.add_axes(_rect(xs["maps"][0], y_foot - CB_H,
                                3 * S + 2 * GAP_MAP, CB_H, W, Hb))
    sax = target.add_axes(_rect(xs["scatter"], y_panel, S, S, W, Hb))
    plot_gram_panel(fro, lea, cka_obs, None, target=target, fs=FS_CKA,
                    axes_override=(map_axes, cax, sax))

    _match_legend_to_colorbar(target, cax, gax[0], GEOM_LEG_ROWS,
                              (xs["geom"][0] + S + GAP_GEOM / 2) / W, y_foot / Hb)


def _match_legend_to_colorbar(fig, cax, src_ax, rows, x, anchor):
    """Redraw the geometry legend so it occupies the colorbar's exact vertical band.

    The two footers are the bottom of their respective blocks and the eye reads them
    as a pair, so they need the same top AND the same overall height -- but one is a
    colorbar plus its label and the other is two rows of legend entries, and no shared
    constant makes those agree. Both are therefore MEASURED after a first draw and the
    legend is rebuilt: its anchor is shifted so its top ink meets the colorbar's, and
    the gap between its rows is set to whatever makes the block the colorbar's height.

    The gap is clamped at zero; if the rows are already taller than the colorbar's ink
    the heights cannot be matched, and the tops still are.
    """
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    legs = list(fig.legends)
    if not legs:
        return
    cb = cax.get_tightbbox(R)
    h_row = max(lg.get_window_extent(R).height for lg in legs)
    top_ink = max(lg.get_window_extent(R).y1 for lg in legs)

    # bbox_to_anchor positions the legend BOX; its ink sits a borderpad inside that,
    # so the offset has to be measured rather than assumed to be zero.
    anchor_disp = fig.transSubfigure.transform((0, anchor))[1]
    offset = anchor_disp - top_ink
    new_anchor = fig.transSubfigure.inverted().transform((0, cb.y1 + offset))[1]

    n = len(rows)
    gap_in = max(0.0, (cb.height - n * h_row) / (n - 1) / fig.dpi) if n > 1 else 0.0

    handles, labels = src_ax.get_legend_handles_labels()
    for lg in legs:
        lg.remove()
    _stacked_legend(fig, handles, labels, rows, x, new_anchor, FS_LEG,
                    GEOM_LEG_SCALE, GEOM_LEG_KW, gap_in)

    fig.canvas.draw()
    lo = min(lg.get_window_extent(R).y0 for lg in fig.legends)
    hi = max(lg.get_window_extent(R).y1 for lg in fig.legends)
    logging.info(f"footers: colorbar {cb.y0 / fig.dpi:.3f}..{cb.y1 / fig.dpi:.3f} in, "
                 f"legend {lo / fig.dpi:.3f}..{hi / fig.dpi:.3f} in")


def _row_divider(fig, top, bottom):
    """Rule between the recovery row and the two blocks under it.

    Placed midway between the lowest ink of the top subfigure and the highest ink of
    the bottom one, measured rather than set to the subfigure boundary -- the boundary
    sits wherever the height ratio put it, which is not generally the middle of the
    visible gap.
    """
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    lo = min(ax.get_tightbbox(R).y0 for ax in top.axes)
    hi = max(ax.get_tightbbox(R).y1 for ax in bottom.axes)
    y = fig.transFigure.inverted().transform((0, (lo + hi) / 2))[1]
    fig.add_artist(Line2D([0.01, 0.99], [y, y], transform=fig.transFigure,
                          color="0.72", linewidth=1.1, zorder=10))


if __name__ == "__main__":
    main()
