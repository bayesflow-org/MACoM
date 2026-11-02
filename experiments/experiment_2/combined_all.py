"""One PPC figure for all three families, from the cached draws — no re-simulation.

``ppc.py`` runs per family; this loads what those runs cached and draws DDM, RDM and
CDM as one figure. It goes through ``collect_ppc_cached`` with the same arguments the
per-family runs use, so it reads their pickles rather than simulating again — which
also means the combined figure and the per-family ones show the same draws, not two
independent samples of the same posterior.

    python -m experiments.experiment_2.combined_all \\
        --variant mc --max_num_regressors 4 --max_num_categories 4

Writes ``<outdir>/combined_ppc.pdf``; ``--layout wide`` writes
``<outdir>/combined_ppc_wide.pdf`` instead (two rows of ten, DDM and RDM left of CDM,
larger quantile-panel text, for slides).
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from .ppc import (collect_ppc_cached, plot_combined_all, plot_combined_all_wide,
                  DEFAULT_RT_QUANTILES, CACHE_DIR, FIG_DIR)

FAMILIES = ("ddm", "rdm", "cdm")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", type=str, default="mc", choices=["mf", "mc"])
    p.add_argument("--outdir", type=str, default=str(FIG_DIR))
    p.add_argument("--families", type=str, default=",".join(FAMILIES))
    p.add_argument("--n_ppc_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    p.add_argument("--rt_quantiles", type=float, nargs="+",
                   default=list(DEFAULT_RT_QUANTILES))
    p.add_argument("--layout", choices=["stacked", "wide"], default="stacked",
                   help="stacked: the paper figure (four rows); wide: two rows of ten "
                        "with DDM and RDM left of CDM, sized for slides.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    # The geometry is module state in ppc.py and part of the cache key, so it has to
    # be set before any collect call -- exactly as ppc.main does it.
    from . import ppc
    ppc.MAX_NUM_REGRESSORS = a.max_num_regressors
    ppc.MAX_NUM_CATEGORIES = a.max_num_categories

    per_family = {}
    for fam in [f.strip() for f in a.families.split(",") if f.strip()]:
        logging.info(f"loading {fam} ({a.variant})")
        per_family[fam] = collect_ppc_cached(
            model_name=fam, variant=a.variant, n_ppc_samples=a.n_ppc_samples,
            seed=a.seed, no_contamination=False, cache_dir=Path(a.cache_dir),
            refresh=False, fixed_natural=None, posteriors_path=None)

    if a.layout == "wide":
        plot_combined_all_wide(per_family, Path(a.outdir) / "combined_ppc_wide.pdf",
                               quantiles=a.rt_quantiles)
    else:
        plot_combined_all(per_family, Path(a.outdir) / "combined_ppc.pdf",
                          quantiles=a.rt_quantiles)


if __name__ == "__main__":
    main()
