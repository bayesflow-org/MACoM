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

# The four diagnostics for this probe. recovery draws credible-interval error
# bars by default (uncertainty_agg=credible_interval), so uncertainty is shown
# per point rather than as a bare scatter.
PLOTS = {
    "recovery": bf.diagnostics.plots.recovery,
    "calibration_ecdf": bf.diagnostics.plots.calibration_ecdf,
    "coverage": bf.diagnostics.plots.coverage,
    "z_score_contraction": bf.diagnostics.plots.z_score_contraction,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   default="./outputs/figures/encoder_rep/addm/intercept_only/"
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
    names = VARIABLE_NAMES[:n_params]
    logging.info(f"{targets.shape[0]} datasets x {estimates.shape[1]} draws x {n_params} params")

    for name, fn in PLOTS.items():
        # Some BayesFlow plotters size their axis grid independently of the
        # variable count and then iterate every axis, which for 7 params builds
        # 2x4=8 panels and indexes past the end. Pin the layout to one row.
        fig = fn(estimates=estimates, targets=targets, variable_names=names,
                 num_row=1, num_col=n_params)
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
    for j, n in enumerate(INTRINSIC_PARAMS[:n_params]):
        logging.info(f"{n:10} {contraction[j]:12.3f} {zscore[j]:9.3f} {rmse[j]:8.3f}")
        rows.append({"param": n, "contraction": contraction[j],
                     "z_score": zscore[j], "nrmse": rmse[j]})
    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir / f"addm_{args.case}_bf_metrics.csv",
                              index=False, float_format="%.4f")
    logging.info(f"[saved] {outdir / f'addm_{args.case}_bf_metrics.csv'}")


if __name__ == "__main__":
    main()
