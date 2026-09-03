"""RSA and CKA for LBA: does finetuning move the REPRESENTATION or only the decoder?

Experiment 3 answered the forgetting question with posteriors: LBA NRMSE -48.2%, the
three original families +0.3/+1.0/+0.5%. Two problems with resting on that. The NRMSE
estimator is STOCHASTIC (measured ~1-2% spread on byte-identical arrays), so those
three numbers sit inside their own noise; and the per-dataset C2ST in the same folder
reports excess 0.24-0.37 over null, i.e. the posteriors DO differ on nearly every
dataset. Aggregate NRMSE cannot separate those readings.

This asks the question one level down, where it is decidable. Weight drift already
hints at the answer -- finetuning moved the decoder (relative L2 6.8%) about twice as
much as the encoder (3.3%) -- but weight distance is a poor proxy for representational
change, which is what RSA measures directly.

THREE COMPARISONS
-----------------
1. LBA geometry: original vs finetuned encoder on the SAME LBA datasets. Paired,
   shared conditions -- RSA's ideal setting. Did finetuning reshape the target family's
   representation?
2. What each encoder encodes about LBA parameters (joint RDM regression).
3. FORGETTING: original vs finetuned encoder on the SAME DDM/RDM/CDM datasets. If the
   geometry is unchanged while posteriors differ, finetuning adapted the DECODER and
   left the representation intact -- a materially different claim from "no forgetting".

Both encoders produce 128-d summaries and `model_embedding` (the only thing the
finetune resized, 3 -> 4) is never touched by the encoder, so the two are directly
comparable.

LBA uses fixed_A = 0.5 to match the finetuned checkpoint. Getting that wrong makes `a`
the start-point range rather than the threshold -- the bug that moved zero-shot
recovery 0.24 -> 0.86.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rsatoolbox.rdm import compare

from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.priors import lba_priors
from cogformer.simulator.benchmarks.ssms.link_funs import lba_link_fun
from .representation import simulate_repeats, assemble_input, cka
from .addm_representation import rdm_of, parameter_rdm, rdm_regression
from .crossfamily_representation import (simulate_family, shared_parameter_rdm,
                                         noise_ceiling, SHARED, TRAINED)
from .addm_bf_apply import CogFormerEncoderSummary, _load_cogformer

ORIG_COLOR, FT_COLOR = "#6969ff", "#b2182b"
OUT = Path("./outputs/data/encoder_rep/representation/lba")
FIG = Path("./outputs/figures/encoder_rep/representation/lba")
LBA_PARAMS = ["v", "v_diff", "a", "tau", "s_v", "s_tau"]


def simulate_lba(n_cond, n_rep, num_obs, fixed_A, seed):
    """Raw LBA tensors. Simulated ONCE and shared between encoders -- see
    simulate_family for why re-simulating per encoder silently breaks the comparison."""
    mf = ModelFamily(model=LBA(fixed_A=fixed_A), prior=lba_priors(),
                     intrinsic_params=LBA_PARAMS)
    dc = {"1": LBA_PARAMS}
    mk = {"free_intrinsics": LBA_PARAMS, "fixed_intrinsics": [], "fixed_values": {}}
    link = {k: v for k, v in lba_link_fun().items() if k in LBA_PARAMS}
    samples, reps = simulate_repeats(mf, mf.model, n_conditions=n_cond, n_repeats=n_rep,
                                     num_obs=num_obs, design_config=dc, mask_kwargs=mk,
                                     link_fun=link, max_num_regressors=4,
                                     max_num_categories=4, seed=seed)

    def one(i, obs):
        sub = dict(samples)
        sub["design_matrices"] = samples["design_matrices"][i:i + 1]
        na = np.asarray(samples.get("num_alternatives", np.full(n_cond, 2)))
        sub["num_alternatives"] = np.atleast_1d(na)[i:i + 1]
        ptc = samples.get("per_trial_context", {}) or {}
        sub["per_trial_context"] = {k: np.asarray(v)[i:i + 1] for k, v in ptc.items()}
        return assemble_input(sub, {"rts": obs["rts"][None],
                                    "choices": obs["choices"][None]})[0]

    raw = np.stack([one(i, o) for i, c in enumerate(reps) for o in c])
    theta = samples["param_matrices"][:, :len(LBA_PARAMS)]
    return raw, theta


def plot_forgetting(df, outpath):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(df))
    ax[0].bar(x, df.rsa_orig_vs_ft, color=[FT_COLOR if f == "lba" else ORIG_COLOR
                                           for f in df.family])
    ax[0].plot(x, df.noise_ceiling_rsa, "k_", ms=18, label="split-half ceiling")
    ax[0].set_xticks(x); ax[0].set_xticklabels(df.family)
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("RSA: original vs finetuned encoder")
    ax[0].set_title("Did LBA finetuning move the representation?\n"
                    "(at ceiling = unchanged)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].bar(x - 0.2, df.cka_orig_vs_ft, 0.4, color="#444", label="CKA orig vs finetuned")
    ax[1].bar(x + 0.2, df.noise_ceiling_cka, 0.4, color="lightgrey", label="CKA ceiling")
    ax[1].set_xticks(x); ax[1].set_xticklabels(df.family); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("CKA cross-check", fontsize=10); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.encoder_device)

    a.num_models = 3
    a.cogformer_checkpoint = a.original_checkpoint
    enc_orig = CogFormerEncoderSummary(_load_cogformer(a, dev), dev)
    a.num_models = 4
    a.cogformer_checkpoint = a.finetuned_checkpoint
    enc_ft = CogFormerEncoderSummary(_load_cogformer(a, dev), dev)
    logging.info("loaded both encoders (original num_models=3, finetuned=4)")

    rows = []
    # ---- 1 & 2: LBA itself ------------------------------------------------------
    raw_lba, theta = simulate_lba(a.n_conditions, a.n_repeats, a.num_obs,
                                  a.fixed_A, a.seed)
    shp = (a.n_conditions, a.n_repeats, -1)
    s_o = enc_orig.summarize_raw(raw_lba).reshape(shp)
    s_f = enc_ft.summarize_raw(raw_lba).reshape(shp)
    r_o, r_f = rdm_of(s_o), rdm_of(s_f)
    nc_r, nc_c = noise_ceiling(s_o)
    rows.append({"family": "lba", "trained_on_by_ft": True,
                 "rsa_orig_vs_ft": float(compare(r_o, r_f, method="corr")[0, 0]),
                 "cka_orig_vs_ft": cka(s_o.mean(1), s_f.mean(1)),
                 "noise_ceiling_rsa": nc_r, "noise_ceiling_cka": nc_c})

    prdms = [parameter_rdm(theta[:, j]) for j in range(len(LBA_PARAMS))]
    b_o, r2_o = rdm_regression(r_o, prdms, LBA_PARAMS)
    b_f, r2_f = rdm_regression(r_f, prdms, LBA_PARAMS)
    logging.info(f"[lba app2] joint R^2 original {r2_o:.3f}  finetuned {r2_f:.3f}")
    pd.DataFrame([{"param": p, "beta_original": b_o[p], "beta_finetuned": b_f[p],
                   "rsa_original": float(compare(r_o, prdms[i], method="corr")[0, 0]),
                   "rsa_finetuned": float(compare(r_f, prdms[i], method="corr")[0, 0])}
                  for i, p in enumerate(LBA_PARAMS)]
                 ).to_csv(OUT / "lba_app2_parameter_rsa.csv", index=False,
                          float_format="%.4f")

    # ---- 3: forgetting on the families the finetune was NOT for -------------------
    for fam in TRAINED:
        raw_f, _ = simulate_family(fam, a.n_conditions, a.n_repeats, a.num_obs, a.seed)
        so = enc_orig.summarize_raw(raw_f).reshape(shp)
        sf = enc_ft.summarize_raw(raw_f).reshape(shp)
        ro, rf = rdm_of(so), rdm_of(sf)
        ncr, ncc = noise_ceiling(so)
        rows.append({"family": fam, "trained_on_by_ft": False,
                     "rsa_orig_vs_ft": float(compare(ro, rf, method="corr")[0, 0]),
                     "cka_orig_vs_ft": cka(so.mean(1), sf.mean(1)),
                     "noise_ceiling_rsa": ncr, "noise_ceiling_cka": ncc})
        logging.info(f"[{fam}] orig-vs-ft RSA {rows[-1]['rsa_orig_vs_ft']:.4f} "
                     f"(ceiling {ncr:.4f})  CKA {rows[-1]['cka_orig_vs_ft']:.4f}")

    df = pd.DataFrame(rows)
    logging.info("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(OUT / "lba_forgetting_representation.csv", index=False, float_format="%.4f")
    plot_forgetting(df, FIG / "lba_forgetting_representation.pdf")
    logging.info(f"[saved] {OUT} | {FIG}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--original_checkpoint", type=str, required=True)
    p.add_argument("--finetuned_checkpoint", type=str, required=True)
    p.add_argument("--fixed_A", type=float, default=0.5)
    p.add_argument("--n_conditions", type=int, default=200)
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder_device", type=str, default="cuda")
    for k, v in [("encoder_num_layers", 8), ("decoder_num_layers", 8),
                 ("encoder_num_heads", 8), ("decoder_num_heads", 8), ("num_seeds", 32),
                 ("seed_dim", 128), ("proj_dim", 256), ("model_embed_dim", 8),
                 ("time_embedding_dim", 32), ("pos_embedding_dim", 32)]:
        p.add_argument(f"--{k}", type=int, default=v)
    return p.parse_args()


if __name__ == "__main__":
    main()
