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
FROZEN_COLOR = "#6969ff"

# The four diagnostics for this probe. recovery draws credible-interval error
# bars by default (uncertainty_agg=credible_interval), so uncertainty is shown
# per point rather than as a bare scatter.
#
# BayesFlow spells the data colour differently per plotter, hence the per-plot
# kwargs. calibration_ecdf's fill_color is deliberately left at grey: that band
# is the theoretical reference envelope, not this arm's data, and recolouring it
# would imply it belongs to the frozen arm.
PLOTS = {
    "recovery": (bf.diagnostics.plots.recovery,
                 {"color": FROZEN_COLOR}),
    "calibration_ecdf": (bf.diagnostics.plots.calibration_ecdf,
                         {"rank_ecdf_color": FROZEN_COLOR}),
    "coverage": (bf.diagnostics.plots.coverage,
                 {"color": FROZEN_COLOR}),
    "z_score_contraction": (bf.diagnostics.plots.z_score_contraction,
                            {"color": FROZEN_COLOR}),
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
    p.add_argument("--data", type=str,
                   default="./outputs/data/encoder_rep/addm/intercept_only/"
                           "addm_encoder_rep_data.npz")
    p.add_argument("--outdir", type=str, default=None,
                   help="Defaults to the directory holding --data.")
    p.add_argument("--case", type=str, default="intercept_only")
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

    for name, (fn, color_kwargs) in PLOTS.items():
        # Some BayesFlow plotters size their axis grid independently of the
        # variable count and then iterate every axis, which for 7 params builds
        # 2x4=8 panels and indexes past the end. Pin the layout to one row.
        fig = fn(estimates=estimates, targets=targets, variable_names=names,
                 num_row=1, num_col=n_params, **color_kwargs)
        path = outdir / f"addm_{args.case}_{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {path}")

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
    pd.DataFrame(rows).to_csv(outdir / f"addm_{args.case}_bf_metrics.csv",
                              index=False, float_format="%.4f")
    logging.info(f"[saved] {outdir / f'addm_{args.case}_bf_metrics.csv'}")


if __name__ == "__main__":
    main()
