"""Stack the two aDDM probe arms' recovery plots into one 2 x 5 figure.

Row 0 is the frozen CogFormer encoder, row 1 the learned SetTransformer baseline;
columns are the five active parameters of the v/s_v-fixed variant. The point of
stacking is direct column-wise comparison, so each column's axis limits are shared
across both rows -- with per-arm limits (what ``make_quadratic`` does when each arm
is drawn into its own figure) two panels of the same parameter can be on visibly
different scales, which is exactly the comparison the stacked layout invites the
reader to make by eye.

The drawing loop mirrors ``bf.diagnostics.plots.recovery`` and reuses BayesFlow's
own helpers, rather than calling it twice, because that function always builds its
own figure and takes no axes. Any style change made here must be mirrored in
``addm_diagnostics.py`` (and vice versa) so the combined figure and the two
per-arm figures cannot drift apart.

Replots from each arm's saved npz -- CPU only, no checkpoint, no GPU:

    python -m experiments.experiment_4.combined_recovery
"""
from experiments.paths import RESULTS_ROOT
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesflow.utils import prettify_subplots, add_metric, add_titles_and_labels
from bayesflow.utils.numpy_utils import credible_interval

from .addm_diagnostics import active_labels, ARM_COLOR

# Row order. Baseline on top: it is the reference the decoder arm is read against, so
# it is the row the eye lands on first. The rows carry NO on-figure label -- the arms
# are distinguished by colour only, and the caption has to say which row is which.
ROWS = ["baseline", "frozen"]

FIG = (RESULTS_ROOT / "figures/encoder_rep")
DATA = (RESULTS_ROOT / "data/encoder_rep")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--frozen_data", type=str,
                   default=str(DATA / "addm_vsvfixed/intercept_only"
                                      "/addm_encoder_rep_data.npz"))
    p.add_argument("--baseline_data", type=str,
                   default=str(FIG / "addm_baseline_vsvfixed/intercept_only"
                                     "/item0_vsvfixed_e500/addm_baseline_data.npz"))
    p.add_argument("--out", type=str,
                   default=str(FIG / "addm_compare/intercept_only_vsvfixed"
                                     "/addm_arm_compare_intercept_only_vsvfixed"
                                     "_recovery.pdf"))
    # Same page sizing as addm_diagnostics: drawn at 2x a 5.5 in \textwidth. The
    # per-row height matches the single-arm figure so panels stay the same size.
    p.add_argument("--fig_width", type=float, default=11.0)
    p.add_argument("--row_height", type=float, default=1.05,
                   help="Row height as a fraction of the column width.")
    p.add_argument("--title_fs", type=float, default=15)
    p.add_argument("--label_fs", type=float, default=13)
    p.add_argument("--metric_fs", type=float, default=11)
    p.add_argument("--tick_fs", type=float, default=11)
    p.add_argument("--markersize", type=float, default=2.5)
    p.add_argument("--elinewidth", type=float, default=1.0)
    return p.parse_args(argv)


def load_arm(path):
    d = np.load(Path(path), allow_pickle=True)
    targets = d["true_params"]
    estimates = d["pred_params"]
    _, names = active_labels(d["param_masks"][0])
    if len(names) != targets.shape[1]:
        raise ValueError(f"{path}: mask has {len(names)} active cells but "
                         f"true_params has {targets.shape[1]} columns")
    return targets, estimates, names


def draw_row(axes_row, targets, estimates, color, args, titles=None):
    """One arm's five panels. Kept identical to bf...plots.recovery's inner loop."""
    point = np.median(estimates, axis=1)
    u = credible_interval(estimates, axis=1)
    # credible_interval returns absolute bounds; errorbar wants distances.
    u[0] = point - u[0]
    u[1] = u[1] - point
    for i, ax in enumerate(axes_row):
        ax.errorbar(targets[:, i], point[:, i], yerr=u[..., i], fmt="o",
                    alpha=0.5, color=color, markersize=args.markersize,
                    elinewidth=args.elinewidth)
        corr = np.corrcoef(targets[:, i], point[:, i])[0, 1]
        add_metric(ax=ax, metric_text="$r$", metric_value=corr,
                   metric_fontsize=args.metric_fs)
        if titles is not None:
            ax.set_title(titles[i], fontsize=args.title_fs)
    return point


def draw_grid(axes, arms, names, args):
    """Fill a (len(ROWS) x n_params) axes array. Split out so the combined
    three-part panel can draw the same grid into a subfigure without a second copy
    of the row/limit/prettify sequence."""
    per_arm = []
    for r, arm in enumerate(ROWS):
        targets, estimates, _ = arms[arm]
        # Titles on the top row only -- every column is the same parameter in both.
        point = draw_row(axes[r, :], targets, estimates, ARM_COLOR[arm], args,
                         titles=names if r == 0 else None)
        per_arm.append((targets, point))
    shared_quadratic(axes, per_arm)
    prettify_subplots(axes, num_subplots=axes.size, tick_fontsize=args.tick_fs)
    add_titles_and_labels(axes=axes, num_row=len(ROWS), num_col=len(names),
                          xlabel="Ground truth", ylabel="Estimate",
                          label_fontsize=args.label_fs)
    return per_arm


def shared_quadratic(axes, per_arm):
    """Square, identity-line panels on limits pooled over BOTH arms.

    ``bf.utils.make_quadratic`` derives the range from one arm's data. Here the
    range is the union, so a column's two panels are on one scale and the diagonal
    sits at the same place in both.
    """
    n_col = axes.shape[1]
    for i in range(n_col):
        vals = np.concatenate([np.r_[t[:, i], p[:, i]] for t, p in per_arm])
        lower, upper = float(vals.min()), float(vals.max())
        eps = (upper - lower) * 0.1
        lo, hi = lower - eps, upper + eps
        for ax in axes[:, i]:
            ax.set_xlim((lo, hi))
            ax.set_ylim((lo, hi))
            ax.plot([lower, upper], [lower, upper], color="black",
                    alpha=0.9, linestyle="dashed")
            ax.set_box_aspect(1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    arms = {"frozen": load_arm(args.frozen_data),
            "baseline": load_arm(args.baseline_data)}

    names = arms["frozen"][2]
    if arms["baseline"][2] != names:
        raise ValueError("arms have different active parameters: "
                         f"{names} vs {arms['baseline'][2]}")
    n = len(names)

    # With square panels the width is the binding constraint, so each panel comes
    # out 157 px wide at any row height >= ~1.0 -- the same size as in the per-arm
    # figures. 1.05 is the smallest factor that still leaves an ~18 px ink gap
    # between the rows, close to the ~28 px between the columns; the single-arm
    # 1.15 only adds vertical slack.
    fig, axes = plt.subplots(len(ROWS), n,
                             figsize=(args.fig_width,
                                      args.fig_width / n * args.row_height * 2))
    per_arm = draw_grid(axes, arms, names, args)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {out}")
    # The rows are unlabelled on the figure, so the log is the only place the row
    # order is written down. Keep it.
    for r, arm in enumerate(ROWS):
        t, p = per_arm[r]
        rs = ", ".join(f"{nm}={np.corrcoef(t[:, i], p[:, i])[0, 1]:.3f}"
                       for i, nm in enumerate(names))
        logging.info(f"row {r} ({arm:8}) {rs}")


if __name__ == "__main__":
    main()
