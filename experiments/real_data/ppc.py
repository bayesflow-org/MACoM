"""
Global posterior predictive check for real data.

Pools simulated RT distributions across all subjects and compares them to
observed RT distributions, stratified by experimental condition.

Usage:
    python -m experiments.real_data.ppc --model ddm --variant mf
    python -m experiments.real_data.ppc --model cdm --variant mc
    python -m experiments.real_data.ppc --model rdm --variant mf
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params
from .data_utils import (
    load_ddm_data, load_rdm_data, load_cdm_data,
    build_real_design_matrix, build_real_param_mask,
)

DATA_DIR = Path("./outputs/data/real_data")
FIG_DIR = Path("./outputs/figures/real_data/ppc")

MAX_NUM_REGRESSORS = 5
MAX_NUM_CATEGORIES = 5
KEEP_INTERCEPT = True

# DDM conditions: group trials by stim_type and truth
DDM_DESIGN_CONFIG = {
    "1":       ["v", "a", "z", "tau", "s_v", "s_tau", "pi"],
    "u_1":     ["v", "a", "z", "tau", "s_tau"],
    "u_2":     ["v", "a", "z", "tau", "s_tau"],
    "u_1:u_2": ["v", "a", "z", "tau", "s_tau"],
}

RDM_DESIGN_CONFIG = {
    "1":       ["v", "v_diff", "a", "tau", "s_v", "s_tau", "pi"],
    "u_1":     ["v", "v_diff", "a", "tau"],
    "u_2":     ["v", "v_diff", "a", "tau"],
    "u_1:u_2": ["v", "v_diff", "a", "tau"],
}

CDM_DESIGN_CONFIG = {
    "1":       ["v", "v_theta", "a", "tau", "s_v", "s_tau", "pi"],
    "u_1":     ["v", "a", "tau"],
    "u_2":     ["v", "a", "tau"],
    "u_1:u_2": ["v", "a", "tau"],
}

DDM_CONDITIONS = [
    ("stim_type", "u_1", {-0.5: "Novel", 0.5: "Familiar"}),
    ("truth",     "u_2", {-0.5: "False", 0.5: "True"}),
]
RDM_CONDITIONS = [
    ("congruency", "u_1", {-0.5: "Incongruent", 0.5: "Congruent"}),
    ("session",    "u_2", {-0.5: "Session 0",   0.5: "Session 1"}),
]
CDM_CONDITIONS = [
    ("speed",       "u_1", {-0.5: "Unspeeded", 0.5: "Speeded"}),
    ("difficulty",  "u_2", {1: "Easy", 2: "Medium", 3: "Hard"}),
]

MODEL_REGISTRY = {
    "ddm": {
        "loader":        load_ddm_data,
        "data_path":     "./data/data_application_ddm.csv",
        "design_config": DDM_DESIGN_CONFIG,
        "model_cls":     DDM,
        "link_fun":      ddm_link_fun,
        "conditions":    DDM_CONDITIONS,
        # mf saves "intrinsic_params"; mc saves "global_param_names" + "ddm_local_to_global"
        "mf_param_key":        "intrinsic_params",
        "mc_local_param_key":  "ddm_intrinsic_params",
        "mc_local_to_global":  "ddm_local_to_global",
    },
    "rdm": {
        "loader":        load_rdm_data,
        "data_path":     "./data/data_application_rdm.csv",
        "design_config": RDM_DESIGN_CONFIG,
        "model_cls":     RDM,
        "link_fun":      rdm_link_fun,
        "conditions":    RDM_CONDITIONS,
        "mf_param_key":        "intrinsic_params",
        "mc_local_param_key":  "rdm_intrinsic_params",
        "mc_local_to_global":  "rdm_local_to_global",
    },
    "cdm": {
        "loader":        load_cdm_data,
        "data_path":     "./data/data_application_cdm.csv",
        "design_config": CDM_DESIGN_CONFIG,
        "model_cls":     CDM,
        "link_fun":      cdm_link_fun,
        "conditions":    CDM_CONDITIONS,
        "mf_param_key":        "intrinsic_params",
        "mc_local_param_key":  "cdm_intrinsic_params",
        "mc_local_to_global":  "cdm_local_to_global",
    },
}


def _condition_labels(subject_data, condition_col, label_map):
    """Return per-trial condition labels for a subject given the condition column name."""
    vals = subject_data[condition_col]
    if vals.ndim == 2:
        # Categorical regressors: reconstruct the original level from the dummy columns
        # d2, d3 → difficulty level (1=ref, 2, 3)
        d2 = vals[:, 0].round().astype(int)
        d3 = vals[:, 1].round().astype(int)
        levels = np.ones(len(d2), dtype=int)
        levels[d2 == 1] = 2
        levels[d3 == 1] = 3
        return levels
    else:
        return vals.round(1)


def _simulate_subject(model, link_funs, intrinsic_params, design_matrix, posterior_sample):
    """
    Forward-simulate one set of trials from one posterior sample.

    posterior_sample : (n_rows, n_params) — unconstrained regression coefficients
    design_matrix    : (num_obs, n_rows)
    Returns dict with "rts" (num_obs,) and "choices" (num_obs,).
    """
    predictor = design_matrix @ posterior_sample  # (num_obs, n_params)
    params = {}
    for j, pname in enumerate(intrinsic_params):
        if pname == "pi":
            # pi is the intercept value (scalar from the first row's predictor at intercept col)
            params[pname] = link_funs[pname](predictor[:, j])
        else:
            params[pname] = link_funs[pname](predictor[:, j])

    num_obs = predictor.shape[0]
    prepared = model.prepare_params(params, num_obs=num_obs)
    result = model.simulate(prepared)
    rts = result["rts"].squeeze()
    choices = result["choices"].squeeze()
    valid = rts > 0
    return {"rts": rts[valid], "choices": choices[valid], "valid_mask": valid}


def collect_ppc(
    model_name, variant, n_ppc_samples, seed=0,
):
    rng = np.random.default_rng(seed)
    cfg = MODEL_REGISTRY[model_name]

    # Load real data
    subject_data, subject_ids = cfg["loader"](cfg["data_path"])
    logging.info(f"Loaded {len(subject_ids)} subjects")

    # Load posteriors
    post_file = DATA_DIR / model_name / variant / f"{model_name}_{variant}_posteriors.npz"
    post_npz = np.load(post_file, allow_pickle=True)
    posteriors = post_npz["posteriors"]   # (n_subjects, n_samples, n_rows, n_params_or_global)
    n_samples_available = posteriors.shape[1]

    if variant == "mf":
        intrinsic_params = list(post_npz[cfg["mf_param_key"]])
        local_to_global = None  # posteriors already in local param space
    else:
        # mc posteriors are in global param space; extract local columns for this model
        intrinsic_params = list(post_npz[cfg["mc_local_param_key"]])
        local_to_global = post_npz[cfg["mc_local_to_global"]]  # (n_local_params,) index array
        posteriors = posteriors[:, :, :, local_to_global]       # → (n_subjects, n_samples, n_rows, n_local_params)

    n_params = len(intrinsic_params)

    sample_indices = rng.choice(n_samples_available, size=min(n_ppc_samples, n_samples_available), replace=False)

    # Build model and link functions
    model = cfg["model_cls"]()
    link_funs = cfg["link_fun"]()
    design_config = cfg["design_config"]

    # Build param mask once (shared layout)
    param_mask = build_real_param_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_regressors=MAX_NUM_REGRESSORS,
        max_num_categories=MAX_NUM_CATEGORIES,
        keep_intercept=KEEP_INTERCEPT,
    )
    n_rows = len(param_mask) // n_params

    obs_rts_by_cond = {}    # condition_key → list of observed RT arrays
    ppc_rts_by_cond = {}    # condition_key → list of (n_subjects * n_trials,) arrays, one per sample

    for cond_name, cond_col, label_map in cfg["conditions"]:
        obs_rts_by_cond[cond_name] = {lbl: [] for lbl in label_map.values()}
        ppc_rts_by_cond[cond_name] = {lbl: [[] for _ in sample_indices] for lbl in label_map.values()}

    for i, (sid, post_subj) in enumerate(zip(subject_ids, posteriors)):
        d = subject_data[sid]
        num_obs = len(d["rts"])

        regressors = {k: d[k] for k in d if k not in ("rts", "choices")}
        dm = build_real_design_matrix(
            regressors=regressors,
            design_config=design_config,
            max_num_regressors=MAX_NUM_REGRESSORS,
            max_num_categories=MAX_NUM_CATEGORIES,
            keep_intercept=KEEP_INTERCEPT,
        )  # (num_obs, n_rows)

        # Observed data: bucket into conditions
        for cond_name, cond_col, label_map in cfg["conditions"]:
            cond_vals = _condition_labels(d, cond_col, label_map)
            for val, lbl in label_map.items():
                mask = cond_vals == val
                if mask.any():
                    obs_rts_by_cond[cond_name][lbl].append(d["rts"][mask])

        # PPC: simulate for each sampled posterior draw
        for k, s_idx in enumerate(sample_indices):
            post_sample = post_subj[s_idx].reshape(n_rows, n_params)
            sim = _simulate_subject(model, link_funs, intrinsic_params, dm, post_sample)

            for cond_name, cond_col, label_map in cfg["conditions"]:
                cond_vals = _condition_labels(d, cond_col, label_map)
                valid_cond = cond_vals[sim["valid_mask"]]
                for val, lbl in label_map.items():
                    mask = valid_cond == val
                    if mask.any():
                        ppc_rts_by_cond[cond_name][lbl][k].append(sim["rts"][mask])

        if (i + 1) % 10 == 0:
            logging.info(f"  processed {i + 1}/{len(subject_ids)} subjects")

    # Concatenate lists into arrays
    obs_pooled = {
        cn: {lbl: np.concatenate(v) for lbl, v in cond.items() if v}
        for cn, cond in obs_rts_by_cond.items()
    }
    ppc_pooled = {
        cn: {
            lbl: [np.concatenate(per_sample) for per_sample in cond[lbl] if any(len(a) > 0 for a in per_sample)]
            for lbl in cond
        }
        for cn, cond in ppc_rts_by_cond.items()
    }

    return obs_pooled, ppc_pooled, cfg["conditions"]


def _kde_eval(data, xs):
    if len(data) < 5:
        return np.zeros_like(xs)
    try:
        return gaussian_kde(data, bw_method="silverman")(xs)
    except Exception:
        return np.zeros_like(xs)


def plot_ppc(obs_pooled, ppc_pooled, conditions, model_name, variant, outdir):
    obs_color = "#1a1a2e"
    ppc_color = "#e94560"
    n_cond_groups = len(conditions)

    for cond_name, cond_col, label_map in conditions:
        labels = list(label_map.values())
        n_panels = len(labels)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.2), sharey=False)
        if n_panels == 1:
            axes = [axes]

        for ax, lbl in zip(axes, labels):
            obs = obs_pooled.get(cond_name, {}).get(lbl, np.array([]))
            ppc_samples = ppc_pooled.get(cond_name, {}).get(lbl, [])

            if len(obs) == 0:
                ax.set_visible(False)
                continue

            rt_min = max(0.0, np.percentile(obs, 1))
            rt_max = np.percentile(obs, 99)
            xs = np.linspace(rt_min, rt_max, 300)

            # Posterior predictive band
            if ppc_samples:
                ppc_kdes = np.stack([_kde_eval(s, xs) for s in ppc_samples], axis=0)
                lo = np.percentile(ppc_kdes, 5, axis=0)
                hi = np.percentile(ppc_kdes, 95, axis=0)
                med = np.median(ppc_kdes, axis=0)
                ax.fill_between(xs, lo, hi, color=ppc_color, alpha=0.25, label="PPC 90% CI")
                ax.plot(xs, med, color=ppc_color, lw=1.5, label="PPC median")

            # Observed
            obs_kde = _kde_eval(obs, xs)
            ax.plot(xs, obs_kde, color=obs_color, lw=1.8, label="Observed")

            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("RT (s)", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(
            f"{model_name.upper()} ({variant}) — PPC by {cond_name}",
            fontsize=10,
        )
        fig.tight_layout()

        outdir.mkdir(parents=True, exist_ok=True)
        savepath = outdir / f"{model_name}_{variant}_ppc_{cond_name}.pdf"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {savepath}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["ddm", "rdm", "cdm"])
    p.add_argument("--variant", type=str, default="mf", choices=["mf", "mc"])
    p.add_argument("--n_ppc_samples", type=int, default=50,
                   help="Number of posterior samples to simulate from (subset for speed).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir) if args.outdir else FIG_DIR / args.model
    logging.info(f"Running PPC: model={args.model}, variant={args.variant}, n_ppc_samples={args.n_ppc_samples}")

    obs_pooled, ppc_pooled, conditions = collect_ppc(
        model_name=args.model,
        variant=args.variant,
        n_ppc_samples=args.n_ppc_samples,
        seed=args.seed,
    )
    plot_ppc(obs_pooled, ppc_pooled, conditions, args.model, args.variant, outdir)
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
