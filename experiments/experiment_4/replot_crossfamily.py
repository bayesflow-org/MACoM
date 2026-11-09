"""Redraw the pooled cross-family geometry from the saved summaries — no GPU.

``crossfamily_representation.py`` simulates five families and encodes them with two
checkpoints. Re-running it to restyle a figure costs that pass AND changes what the
figure shows: the numba kernels ignore ``np.random.seed``, so a rerun draws different
datasets and the MDS embedding moves. The run persists every encoder summary in
``app3_summaries.npz`` for exactly this reason.

    python -m experiments.experiment_4.replot_crossfamily \\
        --npz outputs/finals/experiment-4/representation/crossfamily/app3_summaries.npz \\
        --figdir outputs/figures/encoder_rep/representation/crossfamily

The plotting function is imported rather than copied, so a replot and a real run
cannot drift apart.
"""
from experiments.paths import RESULTS_ROOT
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from .crossfamily_representation import plot_pooled, POOLED

SUFFIX = "__finetuned"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--npz", type=str, default=str(RESULTS_ROOT / "data/encoder_rep/representation/crossfamily/app3_summaries.npz"))
    p.add_argument("--figdir", type=str, default=str(RESULTS_ROOT / "figures/encoder_rep/representation/crossfamily"))
    p.add_argument("--csv_out", type=str, default=None,
                   help="Where to write the geometry summary table.")
    p.add_argument("--seed", type=int, default=0,
                   help="Must match the run's seed for the MDS fit to reproduce it.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    d = np.load(a.npz)

    original = {f: d[f] for f in POOLED if f in d.files}
    finetuned = {f: d[f + SUFFIX] for f in POOLED if f + SUFFIX in d.files}
    if not original:
        raise SystemExit(f"{a.npz} holds none of {POOLED}")
    if not finetuned:
        raise SystemExit(
            f"{a.npz} has no *{SUFFIX} arrays: it predates the two-encoder figure, so "
            "the post-finetuning panel cannot be drawn from it. Re-run "
            "crossfamily_representation to regenerate it.")
    if set(original) != set(finetuned):
        raise SystemExit(f"families differ between encoders: {sorted(original)} vs "
                         f"{sorted(finetuned)}")
    logging.info(f"{len(original)} families, both encoders: {sorted(original)}")

    figdir = Path(a.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    # The summary CSV lands beside the npz it was derived from, not in figdir: it is
    # data, and finalize files it with the other crossfamily tables.
    csv_path = Path(a.csv_out) if a.csv_out else Path(a.npz).parent / "app3_geometry_summary.csv"
    plot_pooled({"original": original, "finetuned": finetuned},
                figdir / "app3_pooled_geometry.pdf", a.seed, csv_path=csv_path)
    logging.info(f"[saved] {figdir / 'app3_pooled_geometry.pdf'}")


if __name__ == "__main__":
    main()
