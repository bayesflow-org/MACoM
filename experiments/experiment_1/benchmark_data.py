"""Write the shared benchmark validation datasets (bench5 x DDM/RDM/CDM/LBA).

ONE test set for every arm that scores a benchmark case: CogFormer's per-case
validation (experiment 1), its original/finetuned arms (experiment 3) and the
case-specific BayesFlow baselines (experiment 1a; LBA's in experiment 3, for the
held-out family's amortization gap against the finetuned checkpoint). Each arm loads these files with
--data_dir instead of simulating, so all of them score identical datasets and every
comparison between them is exactly paired.

Generated with the BayesFlow per-case simulator (model_family_bf_validate.ModelFamilyBF),
whose design options are read from val_scenarios (utils.benchmark_mix_opts), so it
draws the same distribution as CogFormer's own validation of each case -- verified
2026-09-27, per-dataset mean RT p >= 0.08 in all 15 cases. Unlike the model-class
sampler it emits only the design columns a case uses, which is the layout both the
BayesFlow networks and load_bf_validation_data expect.

THE FILES ARE THE TEST SET, NOT THE SEED. seed_case pins the parameter draws, but the
RTs/choices come from numba kernels with their own RNG and the regressor designs from an
unseeded generator, so a re-run with the same seed gives the same true parameters and
DIFFERENT datasets (measured 2026-09-27). Pairing therefore holds only between arms that
load these files. Do not regenerate them mid-study; the stage marker guards that.

    python -m experiments.experiment_1.benchmark_data --outdir outputs/data/benchmark/seed0
"""
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import logging
from pathlib import Path

import numpy as np

from ..experiment_1a.model_family_bf_validate import ModelFamilyBF
from ..utils import seed_case
from cogformer.simulator.design_utils import DESIGN_GENERATORS

# LBA uses fixed_A=0.5 via its model_kwargs, like every other LBA arm.
FAMILIES = ["ddm", "rdm", "cdm", "lba"]
BENCH5 = ["intercept_only", "fixed", "regressed", "fixed_regressed", "interaction"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--design_generator", choices=DESIGN_GENERATORS, default="mixture",
                   help="Design generator of the test designs; the BF baselines must train on the same.")
    a = p.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for fam in FAMILIES:
        for case in BENCH5:
            path = out / f"{fam}_{case}_data.npz"
            sim = ModelFamilyBF(model_key=fam, case=case, design_generator=a.design_generator)
            seed_case(a.seed, fam, case)
            d = sim.sample(a.batch_size, num_obs=a.num_obs)
            extra = {}
            if "favored_idx" in d:
                extra["favored_idx"] = d["favored_idx"][..., 0]
            extra["num_alternatives"] = sim.last_num_alternatives
            np.savez(path, rts=d["rts"], choices=d["choices"],
                     design_matrices=d["design_matrices"], true_set=d["params"],
                     param_masks=d["masks"], **extra)
            logging.info(f"[saved] {path}  rts{d['rts'].shape} design{d['design_matrices'].shape} "
                         f"params{d['params'].shape}{' +favored_idx' if extra.get('favored_idx') is not None else ''}")


if __name__ == "__main__":
    main()
