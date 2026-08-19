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

# Encoder design geometry. MUST match the checkpoint that produced the posteriors
# (r5c5 for the 2026-06-02 npz, r4c4 for the model_class r4c4 checkpoint). A mismatch
# silently reshapes the posterior into the wrong row layout, so collect_ppc asserts
# the implied row count against the stored posteriors.
MAX_NUM_REGRESSORS = 4
MAX_NUM_CATEGORIES = 4
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


def _simulate_subject(model, link_funs, intrinsic_params, design_matrix, posterior_sample,
                      no_contamination=False, param_mask_2d=None):
    """
    Forward-simulate one set of trials from one posterior sample.

    posterior_sample : (n_rows, n_params) — unconstrained regression coefficients
    design_matrix    : (num_obs, n_rows)
    param_mask_2d    : (n_rows, n_params) 0/1 — which (row, param) cells the
                       decoder was actually trained on. REQUIRED for correctness:
                       inactive cells hold unconstrained flow output that never
                       received gradient, and the matmul below would otherwise sum
                       them into params that are intercept-only in the design
                       (pi, s_v, and for CDM v_theta/s_tau). Leaving it unapplied
                       inflated the effective contamination rate from ~0.4% to ~27%.
    Returns dict with "rts" (num_obs,) and "choices" (num_obs,).
    """
    if param_mask_2d is not None:
        posterior_sample = posterior_sample * param_mask_2d
    predictor = design_matrix @ posterior_sample  # (num_obs, n_params)
    params = {}
    for j, pname in enumerate(intrinsic_params):
        if pname == "pi" and no_contamination:
            params[pname] = np.zeros(predictor.shape[0], dtype=np.float32)
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
    model_name, variant, n_ppc_samples=None, seed=0, no_contamination=False,
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

    # Default to EVERY stored draw. A thin sample makes the 95% PPC band the extreme
    # order statistics of the draws (at S=30 the endpoints are the 1st and 30th), which
    # is unstable and understates the true predictive spread. Subsample only deliberately.
    n_use = n_samples_available if n_ppc_samples is None else min(n_ppc_samples, n_samples_available)
    sample_indices = rng.choice(n_samples_available, size=n_use, replace=False)
    logging.info(f"Using {n_use} of {n_samples_available} stored posterior draws")

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
    if n_rows != posteriors.shape[2]:
        raise ValueError(
            f"design geometry mismatch: r{MAX_NUM_REGRESSORS}c{MAX_NUM_CATEGORIES} implies "
            f"{n_rows} rows but {post_file.name} stores {posteriors.shape[2]}. Set "
            f"--max_num_regressors/--max_num_categories to the values the checkpoint was "
            f"trained at."
        )
    param_mask_2d = np.asarray(param_mask, dtype=np.float32).reshape(n_rows, n_params)

    obs_rts_by_cond = {}    # condition_key → list of observed RT arrays
    ppc_rts_by_cond = {}    # condition_key → list of (n_subjects * n_trials,) arrays, one per sample
    # Choices are simulated anyway; collecting them costs nothing and is the ONLY way to
    # see response-dimension misfit. For CDM the response IS an angle, so an RT-only PPC
    # is blind to the thing most likely to be wrong.
    obs_ch_by_cond = {}
    ppc_ch_by_cond = {}

    for cond_name, cond_col, label_map in cfg["conditions"]:
        obs_rts_by_cond[cond_name] = {lbl: [] for lbl in label_map.values()}
        ppc_rts_by_cond[cond_name] = {lbl: [[] for _ in sample_indices] for lbl in label_map.values()}
        obs_ch_by_cond[cond_name] = {lbl: [] for lbl in label_map.values()}
        ppc_ch_by_cond[cond_name] = {lbl: [[] for _ in sample_indices] for lbl in label_map.values()}

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
                    obs_ch_by_cond[cond_name][lbl].append(d["choices"][mask])

        # PPC: simulate for each sampled posterior draw
        for k, s_idx in enumerate(sample_indices):
            post_sample = post_subj[s_idx].reshape(n_rows, n_params)
            sim = _simulate_subject(model, link_funs, intrinsic_params, dm, post_sample,
                                    no_contamination=no_contamination,
                                    param_mask_2d=param_mask_2d)

            for cond_name, cond_col, label_map in cfg["conditions"]:
                cond_vals = _condition_labels(d, cond_col, label_map)
                valid_cond = cond_vals[sim["valid_mask"]]
                for val, lbl in label_map.items():
                    mask = valid_cond == val
                    if mask.any():
                        ppc_rts_by_cond[cond_name][lbl][k].append(sim["rts"][mask])
                        ppc_ch_by_cond[cond_name][lbl][k].append(sim["choices"][mask])

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

    obs_ch_pooled = {
        cn: {lbl: np.concatenate(v) for lbl, v in cond.items() if v}
        for cn, cond in obs_ch_by_cond.items()
    }
    ppc_ch_pooled = {
        cn: {
            lbl: [np.concatenate(per_sample) for per_sample in cond[lbl] if any(len(a) > 0 for a in per_sample)]
            for lbl in cond
        }
        for cn, cond in ppc_ch_by_cond.items()
    }

    return obs_pooled, ppc_pooled, cfg["conditions"], obs_ch_pooled, ppc_ch_pooled


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


def plot_choice_ppc(obs_ch, ppc_ch, conditions, model_name, variant, outdir):
    """PPC on the RESPONSE dimension.

    CDM's response is a continuous angle, so its misfit is invisible to an RT-only
    PPC — plot_ppc never touches choices. For CDM this compares the observed angular
    density against the predicted one; for DDM/RDM it compares choice proportions.
    """
    circular = model_name == "cdm"
    outdir.mkdir(parents=True, exist_ok=True)

    for cond_name, _, label_map in conditions:
        labels = [l for l in label_map.values() if l in obs_ch[cond_name]
                  and len(obs_ch[cond_name][l]) and len(ppc_ch[cond_name].get(l, []))]
        if not labels:
            continue
        fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.2), squeeze=False)

        for j, lbl in enumerate(labels):
            ax = axes[0, j]
            o = np.asarray(obs_ch[cond_name][lbl], dtype=float)
            P = ppc_ch[cond_name][lbl]

            if circular:
                xs = np.linspace(-np.pi, np.pi, 400)
                dens = np.array([_kde_eval(np.asarray(p, dtype=float), xs) for p in P])
                lo, hi = np.percentile(dens, [2.5, 97.5], axis=0)
                ax.fill_between(xs, lo, hi, alpha=0.30, color="#c44e52", lw=0, label="PPC 95%")
                ax.plot(xs, np.median(dens, axis=0), color="#c44e52", lw=1.4, label="PPC median")
                ax.plot(xs, _kde_eval(o, xs), color="black", lw=1.8, label="observed")
                for v in (-np.pi / 2, np.pi / 2):
                    ax.axvline(v, color="steelblue", ls=":", lw=1.2)
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xlabel(r"response deviation $\theta$ (rad)", fontsize=9)
                frac_o = float(np.mean(np.abs(o) > np.pi / 2))
                frac_p = float(np.median([np.mean(np.abs(np.asarray(p)) > np.pi / 2) for p in P]))
                ax.set_title(f"{lbl}\n$P(|\\theta|>\\pi/2)$: obs {frac_o:.3f} / ppc {frac_p:.3f}",
                             fontsize=9)
            else:
                levels = np.unique(o)
                obs_p = np.array([np.mean(o == v) for v in levels])
                ppc_p = np.array([[np.mean(np.asarray(p) == v) for v in levels] for p in P])
                lo, hi = np.percentile(ppc_p, [2.5, 97.5], axis=0)
                med = np.median(ppc_p, axis=0)
                x = np.arange(len(levels))
                ax.bar(x, obs_p, width=0.55, color="black", alpha=0.75, label="observed")
                ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", color="#c44e52",
                            capsize=4, lw=1.4, label="PPC median + 95%")
                ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}" for v in levels])
                ax.set_xlabel("choice", fontsize=9)
                ax.set_title(lbl, fontsize=9)

            ax.set_ylabel("density" if circular else "proportion", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.legend(fontsize=7, frameon=False)

        fig.suptitle(f"{model_name.upper()} {variant} — response-dimension PPC ({cond_name})",
                     fontsize=10)
        fig.tight_layout()
        savepath = outdir / f"{model_name}_{variant}_choice_ppc_{cond_name}.pdf"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {savepath}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["ddm", "rdm", "cdm"])
    p.add_argument("--variant", type=str, default="mf", choices=["mf", "mc"])
    p.add_argument("--n_ppc_samples", type=int, default=None,
                   help="Posterior draws to simulate from. Default: all draws stored in "
                        "the npz. Pass a smaller number only for a deliberate quick look — "
                        "a thin sample gives an unstable, too-narrow 95%% band.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--max_num_regressors", type=int, default=MAX_NUM_REGRESSORS)
    p.add_argument("--max_num_categories", type=int, default=MAX_NUM_CATEGORIES)
    p.add_argument("--no_contamination", action="store_true",
                   help="Set pi=0 during simulation to isolate distributional fit from lapse contamination.")
    return p.parse_args()


def main():
    args = parse_args()
    global MAX_NUM_REGRESSORS, MAX_NUM_CATEGORIES
    MAX_NUM_REGRESSORS = args.max_num_regressors
    MAX_NUM_CATEGORIES = args.max_num_categories
    outdir = Path(args.outdir) if args.outdir else FIG_DIR / args.model
    logging.info(f"Running PPC: model={args.model}, variant={args.variant}, n_ppc_samples={args.n_ppc_samples}")

    obs_pooled, ppc_pooled, conditions, obs_ch, ppc_ch = collect_ppc(
        model_name=args.model,
        variant=args.variant,
        n_ppc_samples=args.n_ppc_samples,
        seed=args.seed,
        no_contamination=args.no_contamination,
    )
    plot_ppc(obs_pooled, ppc_pooled, conditions, args.model, args.variant, outdir)
    plot_choice_ppc(obs_ch, ppc_ch, conditions, args.model, args.variant, outdir)
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
