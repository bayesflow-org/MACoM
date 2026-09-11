"""Application 3: cross-family representational geometry of a frozen CogFormer encoder.

The encoder was trained on DDM + RDM + CDM. aDDM it never saw. If the encoder learned a
UNIVERSAL representation of evidence-accumulation data rather than three memorised
family-specific schemes, its geometry on the held-out family should look like its
geometry on the trained ones.

TWO ANALYSES, BECAUSE RSA NEEDS SHARED CONDITIONS
-------------------------------------------------
An RDM is a matrix over CONDITIONS, and conditions are not shared across families -- a
DDM dataset and an aDDM dataset are not the same stimulus -- so RDMs cannot simply be
correlated across families. Instead:

  (a) PER-FAMILY SCORE. For each family, how well does the encoder RDM predict the
      PARAMETER RDM? That is one scalar per family, and the scalars are comparable
      because the parameter RDM is built from the five parameters every family shares
      (v, a, tau, s_v, s_tau). Trained families set the reference; if the held-out
      family scores far lower, the representational scheme does not transfer.

  (b) POOLED EMBEDDING. Put every family's datasets in ONE RDM and ask what the
      geometry organises by -- family identity, or parameters -- and crucially where
      aDDM falls: inside the manifold spanned by the trained three, or off it.

MATCHED DESIGNS
---------------
All families are simulated at the same case (intercept-only), num_obs and regressor
budget. Without that, a family difference in the design distribution would masquerade
as a difference in representational geometry.
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
from rsatoolbox.rdm import calc_rdm, compare
from sklearn.manifold import MDS

from cogformer.adapter import Adapter
from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms import DDM, RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.addm import ADDM
from cogformer.simulator.benchmarks.ssms.priors import (ddm_priors, rdm_priors,
                                                        cdm_priors, addm_priors)
from cogformer.simulator.benchmarks.ssms.link_funs import (ddm_link_fun, rdm_link_fun,
                                                           cdm_link_fun, addm_link_fun)
from experiments.configs import MODELS as _BASE
from experiments.utils import add_global_params
from .representation import simulate_repeats, assemble_input, cka
from .addm_representation import rdm_of, parameter_rdm, build_dataset
from .addm_bf_apply import CogFormerEncoderSummary, _load_cogformer

SHARED = ["v", "a", "tau", "s_v", "s_tau"]      # present in all four families
TRAINED = ["ddm", "rdm", "cdm"]
COLORS = {"ddm": "#1b6ca8", "rdm": "#e08214", "cdm": "#2b8a3e", "addm": "#b2182b"}
OUT = Path("./outputs/data/encoder_rep/representation/crossfamily")
FIG = Path("./outputs/figures/encoder_rep/representation/crossfamily")

SPECS = {
    "ddm":  (DDM,  ddm_priors,  ddm_link_fun),
    "rdm":  (RDM,  rdm_priors,  rdm_link_fun),
    "cdm":  (CDM,  cdm_priors,  cdm_link_fun),
    "addm": (ADDM, addm_priors, addm_link_fun),
}


def family_setup(key):
    cls, prior_fn, link_fn = SPECS[key]
    if key == "addm":
        params = ["v", "v_diff", "omega", "a", "tau", "s_v", "s_tau"]
        model = cls(first_fixation="item0")
    else:
        params = add_global_params(_BASE[key], ["pi"], [r"$\pi$"])["intrinsic_params"]
        model = cls()
    mf = ModelFamily(model=model, prior=prior_fn(), intrinsic_params=params)
    link = {k: v for k, v in link_fn().items() if k in params}
    return mf, params, link


def simulate_family(key, n_cond, n_rep, num_obs, seed):
    """Simulate repeats for one family and return the RAW per-trial tensors.

    Separated from encoding so that several encoders can be applied to the SAME data.
    Simulating once per encoder does NOT work even with a fixed seed: the numba kernels
    draw from numba's own RNG, which np.random.seed does not control, so two calls
    produce different datasets. Comparing encoder A on dataset X against encoder B on
    dataset Y drives every similarity to ~0 regardless of the encoders.
    """
    mf, params, link = family_setup(key)
    dc = {"1": params}
    mk = {"free_intrinsics": params, "fixed_intrinsics": [], "fixed_values": {}}
    samples, reps = simulate_repeats(
        mf, mf.model, n_conditions=n_cond, n_repeats=n_rep, num_obs=num_obs,
        design_config=dc, mask_kwargs=mk, link_fun=link,
        max_num_regressors=4, max_num_categories=4, seed=seed)
    # simulate_repeats captures ONE condition per simulate() call, so each obs holds a
    # single condition's trials. Slice the batch-level fields to match rather than
    # handing assemble_input a full batch against a single-condition obs.
    def one(i, obs):
        sub = dict(samples)
        sub["design_matrices"] = samples["design_matrices"][i:i + 1]
        na = np.asarray(samples.get("num_alternatives", np.full(n_cond, 2)))
        sub["num_alternatives"] = np.atleast_1d(na)[i:i + 1]
        ptc = samples.get("per_trial_context", {}) or {}
        sub["per_trial_context"] = {k: np.asarray(v)[i:i + 1] for k, v in ptc.items()}
        o = {"rts": obs["rts"][None], "choices": obs["choices"][None]}
        return assemble_input(sub, o)[0]

    raw = np.stack([one(i, obs) for i, cond in enumerate(reps) for obs in cond])
    # Intercept-only design: flat index is row * n_params + param, so the intercept
    # row's cells are simply the first len(params) entries.
    theta = {p: samples["param_matrices"][:, params.index(p)] for p in SHARED}
    return raw, theta


def encode_family(key, enc, n_cond, n_rep, num_obs, seed):
    """Convenience wrapper: simulate one family and encode with a single encoder."""
    raw, theta = simulate_family(key, n_cond, n_rep, num_obs, seed)
    return enc.summarize_raw(raw).reshape(n_cond, n_rep, -1), theta


def shared_parameter_rdm(theta):
    """One RDM from the five parameters every family has, standardised.

    Standardising matters: the families' priors put v, a, tau on different scales
    (measured v ranges: DDM [0.01,1.76], RDM [-0.09,1.00], CDM [0.42,1.60]). Without it
    the parameter RDM would be dominated by whichever parameter that family's prior
    happens to spread widest, and the per-family scores would not be comparable.
    """
    from rsatoolbox.data import Dataset
    M = np.column_stack([theta[p] for p in SHARED]).astype(float)
    M = (M - M.mean(0)) / (M.std(0) + 1e-12)
    ds = Dataset(measurements=M,
                 obs_descriptors={"condition": np.arange(M.shape[0])},
                 channel_descriptors={"channel": np.arange(M.shape[1])})
    return calc_rdm(ds, method="euclidean", descriptor="condition")


def plot_scores(df, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = [COLORS[f] for f in df.family]
    ax.bar(df.family, df.rsa_encoder_vs_params, color=cols)
    ref = df[df.family.isin(TRAINED)].rsa_encoder_vs_params
    ax.axhspan(ref.min(), ref.max(), color="grey", alpha=.18,
               label="range over TRAINED families")
    ax.axhline(ref.mean(), color="grey", ls="--", lw=1, label="trained mean")
    ax.set_ylabel("RSA: encoder RDM vs shared-parameter RDM")
    ax.set_title("Does the representational scheme transfer to the held-out family?",
                 fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)


def noise_ceiling(summ):
    """Split-half CKA/RSA within a family: the same conditions, different noise.

    THE CONTROL THAT MAKES THE OFF-DIAGONAL INTERPRETABLE. Raw CKA(x, x) is 1.0 by
    construction, so a matrix with 1.0 on the diagonal tells you nothing about whether
    near-zero off-diagonals mean "orthogonal geometry" or "this metric cannot detect
    similarity across independently simulated stimulus sets at all".

    Splitting the repeats gives two independent MEASUREMENTS of the same conditions.
    If the split-half score is high, the metric works and a near-zero off-diagonal is
    a real statement about geometry. If it is low, the off-diagonals are uninformative
    and the orthogonality reading must be withdrawn.
    """
    n_rep = summ.shape[1]
    if n_rep < 2:
        return float("nan"), float("nan")
    h = n_rep // 2
    a, b = summ[:, :h], summ[:, h:2 * h]
    rsa = float(compare(rdm_of(a), rdm_of(b), method="corr")[0, 0]) if h >= 2 else float("nan")
    return rsa, cka(a.mean(1), b.mean(1))


def plot_pooled(summ_by_fam, outpath, seed=0):
    """MDS of the pooled condition means: where does the held-out family sit?"""
    X, lab = [], []
    for fam, s in summ_by_fam.items():
        X.append(s.mean(1)); lab += [fam] * s.shape[0]
    X = np.concatenate(X); lab = np.array(lab)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    emb = MDS(n_components=2, random_state=seed, n_init=4,
              normalized_stress="auto").fit_transform(Xs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for fam in summ_by_fam:
        m = lab == fam
        ax[0].scatter(emb[m, 0], emb[m, 1], s=14, alpha=.65, color=COLORS[fam],
                      label=f"{fam}{' (held out)' if fam == 'addm' else ''}")
    ax[0].set_title("Pooled encoder geometry (MDS)", fontsize=10)
    ax[0].set_xlabel("MDS 1"); ax[0].set_ylabel("MDS 2"); ax[0].legend(fontsize=8)
    # between-family CKA: is the held-out family's geometry like the trained ones'?
    fams = list(summ_by_fam)
    M = np.zeros((len(fams), len(fams)))
    for i, a in enumerate(fams):
        for j, b in enumerate(fams):
            M[i, j] = cka(summ_by_fam[a].mean(1), summ_by_fam[b].mean(1))
    # Diagonal = split-half noise ceiling, not the trivial CKA(x,x)=1.
    for i, f in enumerate(fams):
        M[i, i] = noise_ceiling(summ_by_fam[f])[1]
    im = ax[1].imshow(M, cmap="magma", vmin=0, vmax=1)
    ax[1].set_xticks(range(len(fams))); ax[1].set_xticklabels(fams)
    ax[1].set_yticks(range(len(fams))); ax[1].set_yticklabels(fams)
    for i in range(len(fams)):
        for j in range(len(fams)):
            ax[1].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                       color="w" if M[i, j] < .6 else "k", fontsize=8)
    ax[1].set_title("CKA of encoder geometry\n(diagonal = split-half noise ceiling)",
                    fontsize=10)
    fig.colorbar(im, ax=ax[1], fraction=.046)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    return fams, M


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    enc = CogFormerEncoderSummary(_load_cogformer(a, torch.device(a.encoder_device)),
                                  torch.device(a.encoder_device))
    summ_by_fam, rows = {}, []
    for fam in TRAINED + ["addm"]:
        logging.info(f"[{fam}] simulating {a.n_conditions}x{a.n_repeats} and encoding")
        s, th = encode_family(fam, enc, a.n_conditions, a.n_repeats, a.num_obs, a.seed)
        summ_by_fam[fam] = s
        score = float(compare(rdm_of(s), shared_parameter_rdm(th), method="corr")[0, 0])
        nc_rsa, nc_cka = noise_ceiling(s)
        rows.append({"family": fam, "trained": fam in TRAINED,
                     "rsa_encoder_vs_params": score,
                     "noise_ceiling_rsa": nc_rsa, "noise_ceiling_cka": nc_cka})
        logging.info(f"[{fam}] RSA encoder-vs-shared-params = {score:.4f}  "
                     f"| split-half ceiling: RSA {nc_rsa:.4f} CKA {nc_cka:.4f}")
    df = pd.DataFrame(rows)
    ref = df[df.trained].rsa_encoder_vs_params
    held = float(df[~df.trained].rsa_encoder_vs_params.iloc[0])
    logging.info(f"\ntrained families: {ref.mean():.4f} +/- {ref.std():.4f}  "
                 f"| held-out aDDM: {held:.4f}")
    df.to_csv(OUT / "app3_per_family_rsa.csv", index=False, float_format="%.4f")
    plot_scores(df, FIG / "app3_transfer_scores.pdf")
    fams, M = plot_pooled(summ_by_fam, FIG / "app3_pooled_geometry.pdf", a.seed)
    pd.DataFrame(M, index=fams, columns=fams).to_csv(OUT / "app3_between_family_cka.csv",
                                                     float_format="%.4f")
    np.savez_compressed(OUT / "app3_summaries.npz",
                        **{f: v for f, v in summ_by_fam.items()})
    logging.info(f"[saved] {OUT} | {FIG}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n_conditions", type=int, default=200)
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--num_obs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cogformer_checkpoint", type=str, required=True)
    p.add_argument("--encoder_device", type=str, default="cuda")
    p.add_argument("--num_models", type=int, default=3)
    for k, v in [("encoder_num_layers", 8), ("decoder_num_layers", 8),
                 ("encoder_num_heads", 8), ("decoder_num_heads", 8), ("num_seeds", 32),
                 ("seed_dim", 128), ("proj_dim", 256), ("model_embed_dim", 8),
                 ("time_embedding_dim", 32), ("pos_embedding_dim", 32)]:
        p.add_argument(f"--{k}", type=int, default=v)
    return p.parse_args()


if __name__ == "__main__":
    main()
