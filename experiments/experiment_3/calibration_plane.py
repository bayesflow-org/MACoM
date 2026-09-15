"""Where LBA finetuning takes the posterior: sharpness against calibration.

Two panels on ONE pair of axes-units, because they are two views of the same plane:

  A  one arrow per (scenario x parameter) cell of LBA, from the zero-shot checkpoint
     to the finetuned one.
  B  one trace per model family through the same plane, across the five finetuning
     checkpoints (epochs 0/500/1000/1500/2000).

WHY THIS PLANE, AND NOT NRMSE
-----------------------------
Aggregate NRMSE says LBA improves ~50%, and stops there. It cannot say HOW, because
two opposite things happen at once and cancel in the mean: cells the zero-shot model
was overconfident about get WIDER, and cells it was vague about get SHARPER. The
correlation between a cell's zero-shot contraction and its change in contraction is
r = -0.82 -- finetuning pulls every parameter toward a common sharpness rather than
uniformly improving them. Plotting contraction against calibration error shows that
directly: the zero-shot cloud is scattered across the plane and the finetuned cloud
collapses into a band along the bottom.

The axes are also the two metrics that are REPRODUCIBLE here. This experiment's own
notes record that bf's root_mean_squared_error is stochastic at the 1-2% level on
byte-identical arrays, which is the same size as the forgetting effect it is used to
rule out; posterior contraction and calibration error are deterministic to 4 dp.

Contraction runs the opposite way to calibration error -- higher is better -- so the
good corner is bottom-RIGHT, which the panels mark.

SOURCE: the seeded trajectory lineage only
------------------------------------------
Every number comes from outputs/data/model_class/cf/lba_trajectory, whose epoch 0 IS
the zero-shot condition (pre-finetune checkpoint, LBA's embedding row warm-started by
copying RDM's) and whose epoch 2000 IS the finetuned checkpoint. The finals' separate
`zeroshot/` and `finetuned/` arms come from an earlier UNSEEDED run on a different
validation set -- their true_set does not match this one -- so mixing the two would
put the arrows and the traces on different models and different data. They are not
read here.

    python -m experiments.experiment_3.calibration_plane
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cogformer.utils.plot_utils import family_colors, GUIDE_COLOR

TRAJ = Path("outputs/data/model_class/cf/lba_trajectory")
EPOCHS = [0, 500, 1000, 1500, 2000]
FAMILIES = ["lba", "ddm", "rdm", "cdm"]
CONTR, CALIB = "Posterior Contraction", "Calibration Error"

# Drawn at 2x a 5.5 in \textwidth, as everywhere else in this project.
FIG_W = 11.0
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, FS_ANNOT = 13, 11, 9.5, 9.5, 9

# Arrow colours in panel A: the two opposite movements the mean hides. Deliberately
# NOT family colours -- these split one family's cells by what happened to them.
C_SHARPEN = "#1F9E55"    # contraction up: the model found information it had missed
C_WIDEN = "#C96A04"      # contraction down: the model stopped claiming what it lacked

# Both panels are dense enough that annotations have to sit over data; a white scrim
# keeps the text readable without hiding the points under it.
SCRIM = dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2.0)


def load_cells(root=TRAJ):
    """Per-(epoch, family, scenario, parameter) metrics from the trajectory tree."""
    rows, missing = [], []
    for e in EPOCHS:
        for fam in FAMILIES:
            base = root / f"ep{e}" / fam
            if not base.is_dir():
                missing.append(str(base))
                continue
            for sc in sorted(p for p in base.iterdir() if p.is_dir()):
                f = sc / f"{fam}_{sc.name}_ft_metrics.csv"
                if not f.exists():
                    missing.append(str(f))
                    continue
                d = pd.read_csv(f, index_col=0)
                for p, r in d.iterrows():
                    rows.append(dict(epoch=e, family=fam, scenario=sc.name,
                                     param=p, **r.to_dict()))
    if missing:
        raise SystemExit(f"{len(missing)} missing metric files, first: {missing[0]}")
    return pd.DataFrame(rows)


def panel_a(ax, df):
    """LBA cells, zero-shot -> finetuned, one arrow each."""
    w = (df[(df.family == "lba") & df.epoch.isin([EPOCHS[0], EPOCHS[-1]])]
         .pivot_table(index=["scenario", "param"], columns="epoch",
                      values=[CONTR, CALIB]).dropna())
    x0, x1 = w[(CONTR, EPOCHS[0])], w[(CONTR, EPOCHS[-1])]
    y0, y1 = w[(CALIB, EPOCHS[0])], w[(CALIB, EPOCHS[-1])]
    sharp = (x1 > x0).to_numpy()

    # 80 arrows in one panel is dense by construction -- that IS the spread being
    # shown -- so they are drawn thin and transparent and the endpoints carry the
    # reading: hollow where a cell started, filled where it ended up.
    for i in range(len(w)):
        c = C_SHARPEN if sharp[i] else C_WIDEN
        ax.annotate("", xy=(x1.iloc[i], y1.iloc[i]), xytext=(x0.iloc[i], y0.iloc[i]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=0.55,
                                    alpha=0.32, shrinkA=0, shrinkB=0,
                                    mutation_scale=5))
    for m, c in ((sharp, C_SHARPEN), (~sharp, C_WIDEN)):
        ax.scatter(x0[m], y0[m], s=7, facecolors="none", edgecolors=c,
                   linewidths=0.6, alpha=0.75, zorder=3)
        ax.scatter(x1[m], y1[m], s=7, color=c, alpha=0.9, zorder=4)

    r = np.corrcoef(x0, x1 - x0)[0, 1]
    # Over the data with a scrim rather than tucked into a corner: at 80 arrows there
    # is no corner both panels leave empty, and a scrim keeps the points readable.
    ax.text(0.03, 0.88,
            f"zero-shot sharpness vs its change:  $r$ = {r:+.2f}\n"
            f"calibration error spread  {y0.std():.3f} $\\rightarrow$ {y1.std():.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
            color="0.35", linespacing=1.5, zorder=6, bbox=SCRIM)
    handles = [plt.Line2D([], [], color=C_SHARPEN, marker=">", ls="-", lw=1.1,
                          ms=5, label=f"sharpened ({int(sharp.sum())})"),
               plt.Line2D([], [], color=C_WIDEN, marker=">", ls="-", lw=1.1,
                          ms=5, label=f"widened ({int((~sharp).sum())})")]
    ax.legend(handles=handles, loc="upper right", fontsize=FS_LEG,
          frameon=True, framealpha=0.78, edgecolor="none",
          facecolor="white")
    ax.set_title("Each LBA coefficient, zero-shot to finetuned", fontsize=FS_TITLE)
    return w


def panel_b(ax, df):
    """One trace per family through the plane, epoch 0 -> 2000."""
    g = (df.groupby(["epoch", "family"])[[CONTR, CALIB]].mean()
         .reset_index().sort_values("epoch"))
    # Epoch labels only where they do not collide: the three trained families sit on
    # top of each other and LBA's last four checkpoints are nearly one point, so
    # labelling every epoch would print 500..2000 over itself.
    LABEL_AT = {EPOCHS[0]: (8, 2), EPOCHS[1]: (2, -14), EPOCHS[-1]: (8, 4)}
    paths = {}
    for fam in FAMILIES:
        f = g[g.family == fam]
        col = family_colors(fam)["main_effect"]
        held = fam == "lba"
        ax.plot(f[CONTR], f[CALIB], "-", color=col, lw=1.6 if held else 1.1,
                alpha=0.9, zorder=3 if held else 2)
        ax.scatter(f[CONTR], f[CALIB], s=np.linspace(10, 34, len(f)), color=col,
                   zorder=4 if held else 3,
                   label=f"{fam.upper()}{' (held out)' if held else ''}")
        paths[fam] = float(np.hypot(np.diff(f[CONTR]), np.diff(f[CALIB])).sum())
        if held:
            for e, cx, cy in zip(f.epoch, f[CONTR], f[CALIB]):
                if e in LABEL_AT:
                    ax.annotate(f"epoch {e}" if e == EPOCHS[0] else f"{e}", (cx, cy),
                                textcoords="offset points", xytext=LABEL_AT[e],
                                fontsize=FS_ANNOT, color=col)
    # The path lengths are the claim the panel is making -- that LBA moves and the
    # trained families do not -- so they are printed rather than left to the eye.
    trained = [paths[f] for f in FAMILIES if f != "lba"]
    ax.text(0.03, 0.55,
            f"path length  LBA {paths['lba']:.3f}\n"
            f"trained families {min(trained):.3f}$-${max(trained):.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
            color="0.35", linespacing=1.5, zorder=6, bbox=SCRIM)
    ax.legend(loc="upper right", fontsize=FS_LEG, frameon=True,
          framealpha=0.78, edgecolor="none", facecolor="white")
    ax.set_title("Every family, across finetuning epochs", fontsize=FS_TITLE)
    return g


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=str, default=str(TRAJ))
    p.add_argument("--out", type=str,
                   default="outputs/figures/model_class/cf/lba_trajectory"
                           "/lba_calibration_plane.pdf")
    p.add_argument("--csv_out", type=str,
                   default="outputs/figures/model_class/cf/lba_trajectory"
                           "/lba_calibration_plane.csv")
    a = p.parse_args()

    df = load_cells(Path(a.root))
    logging.info(f"{len(df)} cells: {df.epoch.nunique()} epochs x "
                 f"{df.family.nunique()} families x {df.scenario.nunique()} scenarios")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.42))
    w = panel_a(axes[0], df)
    g = panel_b(axes[1], df)

    for ax in axes:
        ax.set_xlabel("Posterior contraction  (sharper $\\rightarrow$)",
                      fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_box_aspect(1)
    axes[0].set_ylabel("Calibration error  ($\\leftarrow$ better)", fontsize=FS_LABEL)
    axes[1].set_ylabel("Calibration error  ($\\leftarrow$ better)", fontsize=FS_LABEL)
    # NOT shared limits. The panels are the same plane, but B plots family MEANS over
    # 80 cells and so lives in a corner of the range A's individual cells span;
    # forcing A's extent on it left the traces in a fifth of the panel with the rest
    # empty. B's extent is instead drawn back onto A as a rectangle, which states the
    # relationship the shared axes were meant to and costs nothing.
    axes[0].set_xlim(axes[0].get_xlim()[0], 1.0)
    axes[0].set_ylim(-0.015, axes[0].get_ylim()[1])
    bx, by = axes[1].get_xlim(), axes[1].get_ylim()
    axes[0].add_patch(plt.Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                                    fill=False, edgecolor=GUIDE_COLOR, lw=0.8,
                                    ls=(0, (4, 3)), zorder=1))
    axes[0].annotate("extent of panel B", (bx[1], by[1]), textcoords="offset points",
                     xytext=(-3, 5), ha="right", fontsize=FS_ANNOT, color=GUIDE_COLOR)

    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {out}")

    if a.csv_out:
        w.columns = [f"{m}_ep{e}" for m, e in w.columns]
        w.to_csv(a.csv_out, float_format="%.4f")
        logging.info(f"[saved] {a.csv_out}")
    logging.info("\n" + g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
