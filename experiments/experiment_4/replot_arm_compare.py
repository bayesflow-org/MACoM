"""Redraw the arm-comparison figure from the saved outputs — no GPU.

``addm_arm_compare.py`` has no reuse path: it rebuilds the shared validation set and
re-samples both arms every run, which needs two checkpoints and a GPU, and changes
the draws the figure is made of. Everything the figure draws is already on disk --
``c2st_<subset>.npz`` for the histograms, ``shared_validation.npz`` for the paired
log-density scatter -- and the summary CSV names the subsets and their dimensions.

    python -m experiments.experiment_4.replot_arm_compare \\
        --datadir outputs/data/encoder_rep/addm_compare/intercept_only_vsvfixed \\
        --figdir  outputs/figures/encoder_rep/addm_compare/intercept_only_vsvfixed \\
        --tag intercept_only_vsvfixed

``_plot`` is imported rather than copied, so a replot and a real run cannot drift.
"""
from experiments.paths import RESULTS_ROOT
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from .addm_arm_compare import _plot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datadir", type=str, default=str(RESULTS_ROOT / "data/encoder_rep/addm_compare/intercept_only_vsvfixed"))
    p.add_argument("--figdir", type=str, default=str(RESULTS_ROOT / "figures/encoder_rep/addm_compare/intercept_only_vsvfixed"))
    p.add_argument("--tag", type=str, default="intercept_only_vsvfixed",
                   help="Output stem: addm_arm_compare_<tag>.pdf")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    datadir, figdir = Path(a.datadir), Path(a.figdir)

    summary = datadir / "c2st_summary.csv"
    shared = datadir / "shared_validation.npz"
    for f in (summary, shared):
        if not f.exists():
            raise SystemExit(f"missing {f}; this run did not persist what the figure needs")

    # _plot reads c2st_<subset>.npz itself, so the rows only have to carry the
    # subset name and its dimension -- exactly the columns the summary has.
    df = pd.read_csv(summary)
    rows = df[["subset", "n_dims"]].to_dict("records")
    missing = [r["subset"] for r in rows if not (datadir / f"c2st_{r['subset']}.npz").exists()]
    if missing:
        raise SystemExit(f"c2st npz missing for subset(s): {missing}")

    d = np.load(shared)
    logging.info(f"{len(rows)} subset(s): {[r['subset'] for r in rows]}")
    figdir.mkdir(parents=True, exist_ok=True)
    _plot(rows, datadir, figdir, d["logprob_frozen"], d["logprob_baseline"], a.tag)
    logging.info(f"[saved] {figdir / f'addm_arm_compare_{a.tag}.pdf'}")


if __name__ == "__main__":
    main()
