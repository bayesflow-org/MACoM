"""Render BayesFlow diagnostics for the aDDM encoder-representation probe.

Reads the npz written by ``addm_bf_apply.py --mode validate`` and produces the
four standard SBI diagnostics. The in-house cogformer.diagnostics.plot helpers
expect the adaptive row x param grid plus a parameter mask; the probe's saved
arrays are already flattened to active parameters only, which is exactly the
layout BayesFlow's diagnostics take, so those are used directly.

z_score_contraction is the one to read first here: it separates "the encoder
lost this parameter" from "the likelihood never carried it". A parameter with
near-zero contraction has a posterior no narrower than its prior, which is an
identifiability statement, not a representation failure.
"""
import argparse
import inspect
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bayesflow as bf

INTRINSIC_PARAMS = ["v", "v_diff", "omega", "a", "tau", "s_v", "s_tau"]
VARIABLE_NAMES = [r"$v$", r"$v_\mathrm{diff}$", r"$\omega$",
                  r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"]

# Colour for the frozen-encoder arm. The BayesFlow baseline arm keeps the
# library default (#132a70), so the two arms stay visually separable when their
# diagnostics are read side by side.
# Both arm colours come from addm_bf_apply so this module cannot drift from the
# figures it is compared against. FROZEN_COLOR used to be redefined here.
from .addm_bf_apply import FROZEN_COLOR, BASELINE_COLOR

ARM_COLOR = {"frozen": FROZEN_COLOR, "baseline": BASELINE_COLOR}
# Baseline files carry an extra infix so both arms can be regenerated in place.
ARM_PREFIX = {"frozen": "addm", "baseline": "addm_baseline"}

# The four diagnostics for this probe. recovery draws credible-interval error
# bars by default (uncertainty_agg=credible_interval), so uncertainty is shown
# per point rather than as a bare scatter.
#
# BayesFlow spells the data colour differently per plotter, hence the per-plot
# kwargs. calibration_ecdf's fill_color is deliberately left at grey: that band
# is the theoretical reference envelope, not this arm's data, and recolouring it
# would imply it belongs to the frozen arm.
# Only these draw error bars; elinewidth is meaningless to the others and would be
# forwarded into a call that has no use for it.
_ERRORBAR_PLOTS = {"recovery"}


def plots_for(arm):
    """The four diagnostics, coloured for one arm.

    Built per call rather than as a module constant so both arms render through the
    SAME code path -- the two previously diverged when this script drew one arm and
    addm_bf_baseline drew the other. calibration_ecdf's fill_color is deliberately
    left grey: that band is the theoretical reference envelope, not the arm's data.
    """
    c = ARM_COLOR[arm]
    return {
        "recovery": (bf.diagnostics.plots.recovery, {"color": c}),
        "calibration_ecdf": (bf.diagnostics.plots.calibration_ecdf,
                             {"rank_ecdf_color": c}),
        "coverage": (bf.diagnostics.plots.coverage, {"color": c}),
        "z_score_contraction": (bf.diagnostics.plots.z_score_contraction, {"color": c}),
    }


def active_labels(mask_row):
    """Names for the active cells of a flattened 67x7 design grid.

    Flat index is ``row * n_params + param``. Kept in step with the identical
    helper in addm_bf_apply.py / addm_bf_baseline.py -- duplicated rather than
    imported so each script stays independently runnable.
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


def parse_args():
    p = argparse.ArgumentParser()
    # addm_bf_apply.py writes to DATA_DIR (outputs/data/...), NOT to FIGURES_DIR.
    # This default used to point at the figures directory, where an old copy of
    # the npz happened to sit, so a rerun silently produced diagnostics for
    # stale data with no error.
    p.add_argument("--fig_width", type=float, default=11.0,
                   help="Figure width in inches; 2x a 5.5 in text width.")
    p.add_argument("--title_fs", type=float, default=15)
    p.add_argument("--label_fs", type=float, default=13)
    p.add_argument("--metric_fs", type=float, default=11)
    p.add_argument("--tick_fs", type=float, default=11)
    p.add_argument("--markersize", type=float, default=2.5)
    p.add_argument("--elinewidth", type=float, default=1.0,
                   help="Error-bar line width for the plots that draw them.")
    p.add_argument("--data", type=str,
                   default="./outputs/data/encoder_rep/addm/intercept_only/"
                           "addm_encoder_rep_data.npz")
    p.add_argument("--outdir", type=str, default=None,
                   help="Defaults to the directory holding --data.")
    p.add_argument("--case", type=str, default="intercept_only")
    p.add_argument("--skip_metrics", action="store_true",
                   help="Redraw the figures only, leaving the bf_metrics CSV alone. Use "
                        "when RE-plotting: recomputing nrmse from the stored draws "
                        "reproduces contraction and z_score exactly but moves nrmse by "
                        "~0.5%% against the value the original validate run wrote from "
                        "its in-memory arrays, and the CSV shipped with the results "
                        "should stay the one that run produced.")
    p.add_argument("--arm", type=str, default="frozen", choices=["frozen", "baseline"],
                   help="Which arm the npz belongs to: picks the colour and the "
                        "output filename prefix. Both arms replot from their saved "
                        "npz on CPU -- no checkpoint or GPU needed.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir) if args.outdir else data_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(data_path, allow_pickle=True)
    targets = d["true_params"]                 # (n_datasets, n_params)
    estimates = d["pred_params"]               # (n_datasets, n_draws, n_params)
    n_params = targets.shape[1]
    # Names must be read off the saved mask, NOT by slicing the full list: the
    # active cells are not in general a prefix of INTRINSIC_PARAMS. The v-fixed
    # variant drops v, so its 6 active cells are v_diff..s_tau, which
    # VARIABLE_NAMES[:6] would label v..s_v -- every panel shifted by one, with
    # no error raised.
    plain_names, names = active_labels(d["param_masks"][0])
    if len(names) != n_params:
        raise ValueError(
            f"param_masks has {len(names)} active cells but true_params has "
            f"{n_params} columns: {data_path} was written by a different config."
        )
    logging.info(f"{targets.shape[0]} datasets x {estimates.shape[1]} draws x {n_params} params")

    prefix = ARM_PREFIX[args.arm]
    for name, (fn, color_kwargs) in plots_for(args.arm).items():
        # Some BayesFlow plotters size their axis grid independently of the
        # variable count and then iterate every axis, which for 7 params builds
        # 2x4=8 panels and indexes past the end. Pin the layout to one row.
        # SIZED FOR THE PAGE. BayesFlow's defaults assume a figure viewed on screen;
        # at a 5.5 in \textwidth these panels scale to well under a third, so the
        # sizes are set here and the figure drawn at twice its target width.
        #
        # Filtered per plotter: the four do NOT take the same arguments -- coverage
        # has no metric_fontsize, for instance -- so passing one set to all of them
        # raises TypeError on whichever lacks it.
        # With 500 datasets the default marker and error bar overlap into a solid
        # block. Thinner bars and smaller marks keep every point -- nothing is
        # subsampled -- while letting the density show through.
        wanted = dict(figsize=(args.fig_width, args.fig_width / n_params * 1.15),
                      label_fontsize=args.label_fs, title_fontsize=args.title_fs,
                      metric_fontsize=args.metric_fs, tick_fontsize=args.tick_fs,
                      markersize=args.markersize)
        params = inspect.signature(fn).parameters
        accepted = set(params)
        style_kwargs = {k: v for k, v in wanted.items() if k in accepted}
        # elinewidth is not a NAMED parameter of any of these -- it rides the **kwargs
        # each one forwards to ax.errorbar. So it is passed only where such a
        # catch-all exists, and only for plotters that draw error bars at all.
        takes_extra = any(q.kind is inspect.Parameter.VAR_KEYWORD
                          for q in params.values())
        if takes_extra and args.elinewidth is not None and name in _ERRORBAR_PLOTS:
            style_kwargs["elinewidth"] = args.elinewidth
        fig = fn(estimates=estimates, targets=targets, variable_names=names,
                 num_row=1, num_col=n_params, **style_kwargs, **color_kwargs)
        path = outdir / f"{prefix}_{args.case}_{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {path}")

    if args.skip_metrics:
        logging.info("[skip] metrics CSV left untouched (--skip_metrics)")
        return

    # Posterior contraction quantifies what the plots show: contraction ~ 0 means
    # the posterior is no narrower than the prior, i.e. the data carried nothing
    # about that parameter. That is an identifiability result, not an encoder one.
    # Both metrics return one aggregated value per variable, shape (n_params,).
    contraction = np.asarray(bf.diagnostics.metrics.posterior_contraction(
        estimates=estimates, targets=targets)["values"]).ravel()
    zscore = np.asarray(bf.diagnostics.metrics.posterior_z_score(
        estimates=estimates, targets=targets)["values"]).ravel()
    rmse = np.asarray(bf.diagnostics.metrics.root_mean_squared_error(
        estimates=estimates, targets=targets)["values"]).ravel()

    rows = []
    logging.info(f"\n{'param':10} {'contraction':>12} {'z_score':>9} {'nrmse':>8}")
    for j, n in enumerate(plain_names):
        logging.info(f"{n:10} {contraction[j]:12.3f} {zscore[j]:9.3f} {rmse[j]:8.3f}")
        rows.append({"param": n, "contraction": contraction[j],
                     "z_score": zscore[j], "nrmse": rmse[j]})
    import pandas as pd
    csv_path = outdir / f"{prefix}_{args.case}_bf_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    logging.info(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
