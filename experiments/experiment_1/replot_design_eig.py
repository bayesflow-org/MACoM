"""Redraw the design-EIG figure from saved sweep outputs — no GPU, no re-simulation.

The sweep in ``ddm_design_eig.py`` costs ~4 GPU-hours, so changing how its figure
LOOKS must not require running it again. Everything the figure draws is either in
``design_eig.csv`` or derivable from ``design_draws.npz``.

    python -m experiments.experiment_1.replot_design_eig \\
        --datadir outputs/finals/experiment-1/information_gain

``nrmse_se`` is bootstrapped from the saved draws when the CSV predates it, so runs
made before the SE column existed still get a shaded band. The plotting function is
imported rather than copied, so a replot and a real run cannot drift apart.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from .ddm_design_eig import _plot, nrmse_with_se

# The finalized tree renames the sweep's files; accept either naming.
CSV_NAMES = ("design_eig.csv", "information_gain.csv")
DRAWS_NAMES = ("design_draws.npz", "posterior_draws.npz")


def _first_existing(datadir, names, what):
    for n in names:
        p = datadir / n
        if p.exists():
            return p
    raise SystemExit(f"no {what} in {datadir} (looked for {', '.join(names)})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, required=True,
                   help="Directory holding the sweep CSV and, optionally, the draws npz.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Where to write design_eig.pdf. Defaults to --datadir.")
    p.add_argument("--width", type=float, default=5.5,
                   help="Figure width in inches. The default is 2x a half-text-width "
                        "target, so the page scales it by 0.5.")
    p.add_argument("--label_fs", type=float, default=16)
    p.add_argument("--legend_fs", type=float, default=13)
    p.add_argument("--tick_fs", type=float, default=9)
    # One row across the full width. It does make the saved figure wider than the
    # panels alone -- the decoder text and the entries sit side by side and their
    # widths add -- and that width is what the page scale is computed against.
    p.add_argument("--stack_legend", action=argparse.BooleanOptionalAction, default=False,
                   help="Stack the decoder text above the entries instead of beside "
                        "them. Narrower, but two rows.")
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write_csv", action="store_true",
                   help="Write the bootstrapped nrmse_se back into the sweep CSV, so the "
                        "numbers beside the figure match the band drawn on it.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    datadir = Path(a.datadir)
    outdir = Path(a.outdir) if a.outdir else datadir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(_first_existing(datadir, CSV_NAMES, "sweep CSV"))

    if "nrmse_se" not in df.columns or df.nrmse_se.isna().any():
        draws_path = _first_existing(datadir, DRAWS_NAMES, "draws npz")
        logging.info(f"CSV has no nrmse_se; bootstrapping from {draws_path.name}")
        d = np.load(draws_path)
        ses = []
        for _, r in df.iterrows():
            key = f"{r.design}_{int(r.num_obs)}"
            _, se = nrmse_with_se(d[f"{key}_pred"], d[f"{key}_true"],
                                  n_boot=a.n_boot, seed=a.seed)
            ses.append(se)
        df["nrmse_se"] = ses
        if a.write_csv:
            csv_path = _first_existing(datadir, CSV_NAMES, "sweep CSV")
            df.to_csv(csv_path, index=False, float_format="%.4f")
            logging.info(f"[updated] {csv_path} (+nrmse_se)")

    # SIZED FOR HALF A TEXT COLUMN. Target is 2.75 in (half of a 5.5 in \textwidth),
    # drawn at 2x so the page scales it by 0.5 and every size below lands at half its
    # nominal value. MEASURED on the saved PDF, which is what matters: bbox_inches
    # ='tight' sets the final width from the widest artist, so the requested figsize
    # is not the scale factor. At these sizes the page gets 8.3 / 6.8 / 6.2 pt; the
    # sweep's own 9x5 default would put the labels at 4.3 pt.
    _plot(df, outdir, figsize=(a.width, a.width * 0.56),
          label_fs=a.label_fs, legend_fs=a.legend_fs, tick_fs=a.tick_fs,
          stack_legend=a.stack_legend)
    logging.info(f"[saved] {outdir / 'design_eig.pdf'}")


if __name__ == "__main__":
    main()
