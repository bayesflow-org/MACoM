"""Regenerate the aDDM representation figures from a saved summaries.npz.

``addm_representation.py`` needs the CogFormer encoder, the baseline checkpoint and a
fresh simulation, so re-running it to change how a figure LOOKS costs a GPU pass over
data whose numbers are already settled. Everything the figures draw is a function of
the two summary tensors, and those are exactly what the run persisted.

    python -m experiments.encoder_rep.replot_representation \\
        --npz outputs/finals/experiment-4/representation/varying_values/summaries.npz \\
        --figdir outputs/figures/encoder_rep/representation/varying_values

The plotting functions are imported from addm_representation rather than copied, so a
replot and a real run can never drift apart in appearance.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from rsatoolbox.rdm import compare

from .addm_representation import (rdm_of, matched_null, plot_representation_set)
from .representation import cka


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True,
                   help="summaries.npz written by addm_representation.py")
    p.add_argument("--figdir", type=str, required=True)
    p.add_argument("--n_null", type=int, default=20,
                   help="Null draws, used only when the npz predates null persistence.")
    p.add_argument("--seed", type=int, default=0,
                   help="Must match the seed of the original run for a recomputed "
                        "null to reproduce it.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    d = np.load(a.npz, allow_pickle=True)
    fro, lea = d["frozen"], d["learned"]
    logging.info(f"summaries: frozen {fro.shape}  learned {lea.shape}")

    rdm_f, rdm_l = rdm_of(fro), rdm_of(lea)
    score = (float(d["rsa"]) if "rsa" in d.files
             else float(compare(rdm_f, rdm_l, method="corr")[0, 0]))
    cka_obs = float(d["cka"]) if "cka" in d.files else cka(fro.mean(1), lea.mean(1))

    # Older npz files predate null persistence. The null is a deterministic function
    # of the frozen summary and the seed, so it can be rebuilt -- but only a seed
    # matching the original run reproduces that run's histogram exactly.
    if "null_rsa" in d.files and "null_cka" in d.files:
        null_rsa, null_cka = d["null_rsa"], d["null_cka"]
        logging.info(f"null draws read from npz ({len(null_rsa)})")
    else:
        logging.info(f"npz has no stored null; recomputing {a.n_null} draws at seed {a.seed}")
        null_rsa, null_cka = matched_null(fro, lea.shape[-1], a.n_null, "crossnobis",
                                          np.random.default_rng(a.seed))

    logging.info(f"RSA {score:.4f} (null {null_rsa.mean():.4f}) | "
                 f"CKA {cka_obs:.4f} (null {null_cka.mean():.4f})")

    figdir = Path(a.figdir); figdir.mkdir(parents=True, exist_ok=True)
    plot_representation_set(rdm_f, rdm_l, fro, lea, score, null_rsa,
                            cka_obs, null_cka, figdir)
    logging.info(f"[saved] {figdir}")


if __name__ == "__main__":
    main()
