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
import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.link_funs import (ddm_link_fun, rdm_link_fun,
                                                          cdm_link_fun, lba_link_fun)
from cogformer.utils.plot_utils import family_colors, OBSERVED_COLOR, GUIDE_COLOR

from ..configs import MODELS as _BASE_MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from ..utils import add_global_params
from .data_utils import (
    load_ddm_data, load_rdm_data, load_cdm_data, load_lba_data,
    build_real_design_matrix, build_real_param_mask,
)

DATA_DIR = Path("./outputs/data/real_data")
FIG_DIR = Path("./outputs/figures/real_data/ppc")
CACHE_DIR = Path("./outputs/data/real_data/ppc_cache")

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
# Imported, NOT redeclared. The PPC must simulate from the same design the apply
# script fitted; a private copy here would drift silently and the mismatch would
# surface only as a bad-looking predictive.
from .lba_mc_cf_apply import LBA_DESIGN_CONFIG

CDM_CONDITIONS = [
    ("speed",       "u_1", {-0.5: "Unspeeded", 0.5: "Speeded"}),
    ("difficulty",  "u_2", {1: "Easy", 2: "Medium", 3: "Hard"}),
]
# u_2 is a 3-level categorical carried as two dummies, so _condition_labels
# reconstructs levels 1/2/3 exactly as it does for CDM's difficulty.
LBA_CONDITIONS = [
    ("condition",  "u_1", {-0.5: "Accuracy", 0.5: "Speed"}),
    ("frequency",  "u_2", {1: "High", 2: "Low", 3: "Very low"}),
    ("lexicality", "u_3", {-0.5: "Nonword", 0.5: "Word"}),
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
        # Stroop is a THREE-colour task (blue/green/red); responses are accumulator
        # indices, observed at ~1/3 each. Without this the PPC simulates a 2-accumulator
        # race and choice 2 comes out at exactly 0.
        "num_alternatives":    3,
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
    "lba": {
        "loader":        load_lba_data,
        "data_path":     "./data/data_application_lba.csv",
        "design_config": LBA_DESIGN_CONFIG,
        "model_cls":     LBA,
        # fixed_A MUST match the checkpoint the posteriors came from. Without it `a`
        # is the start-point range rather than the mean distance travelled, so the
        # simulator would reinterpret the fitted parameter instead of erroring.
        "model_kwargs":  {"fixed_A": 0.5},
        "link_fun":      lba_link_fun,
        "conditions":    LBA_CONDITIONS,
        "mf_param_key":        "intrinsic_params",
        "mc_local_param_key":  "lba_intrinsic_params",
        "mc_local_to_global":  "lba_local_to_global",
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
                      no_contamination=False, param_mask_2d=None, fixed_natural=None,
                      context=None):
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
    context          : dict or None — forwarded to model.prepare_params. REQUIRED for
                       any family with more than two accumulators: prepare_params does
                       `context or {}` and then `context.get("num_alternatives", 2)`, so
                       omitting it silently simulates a TWO-accumulator race. For RDM
                       that also hardcodes accumulator 1 as favored, making choice 2
                       unreachable and choice 1 absorb ~73% of responses against
                       3-choice data that is observed at ~1/3 each.
    Returns dict with "rts" (num_obs,) and "choices" (num_obs,).
    """
    if param_mask_2d is not None:
        posterior_sample = posterior_sample * param_mask_2d
    predictor = design_matrix @ posterior_sample  # (num_obs, n_params)
    fixed_natural = fixed_natural or {}
    params = {}
    for j, pname in enumerate(intrinsic_params):
        if pname == "pi" and no_contamination:
            params[pname] = np.zeros(predictor.shape[0], dtype=np.float32)
        elif pname in fixed_natural:
            # Override on the NATURAL scale, AFTER the link, because that is where these
            # constants are defined: the LBA literature's s_v = 1 is the SD of the drift
            # distribution, and s_tau = 0 is not reachable through shifted_softplus at
            # all (strictly positive; still 2e-9 at x = -20).
            params[pname] = np.full(predictor.shape[0], float(fixed_natural[pname]),
                                    dtype=np.float32)
        else:
            params[pname] = link_funs[pname](predictor[:, j])

    num_obs = predictor.shape[0]
    prepared = model.prepare_params(params, num_obs=num_obs, context=context)
    result = model.simulate(prepared)
    rts = result["rts"].squeeze()
    choices = result["choices"].squeeze()
    valid = rts > 0
    return {"rts": rts[valid], "choices": choices[valid], "valid_mask": valid}


def collect_ppc(
    model_name, variant, n_ppc_samples=None, seed=0, no_contamination=False,
    fixed_natural=None, posteriors_path=None,
):
    rng = np.random.default_rng(seed)
    cfg = MODEL_REGISTRY[model_name]

    # Load real data
    subject_data, subject_ids = cfg["loader"](cfg["data_path"])
    logging.info(f"Loaded {len(subject_ids)} subjects")

    # Load posteriors
    # posteriors_path lets a CONSTRAINED REFIT be scored without inventing a new
    # --variant: the refit writes its own npz (e.g. .../ddm/mc_sv_fixed/) but is still
    # the `mc` lineage, and the figure filenames should stay `ddm_mc_*` so the
    # finalizer's rename rules keep working.
    post_file = (Path(posteriors_path) if posteriors_path else
                 DATA_DIR / model_name / variant / f"{model_name}_{variant}_posteriors.npz")
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
    model = cfg["model_cls"](**cfg.get("model_kwargs", {}))
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

        # favored_idx / participant are trial metadata, not design columns. They are
        # excluded by NAME rather than by presence in design_config because
        # build_real_design_matrix ignores unknown keys, so a stray one would pass
        # silently here and only surface as a wrong design much further downstream.
        regressors = {k: d[k] for k in d
                      if k not in ("rts", "choices", "favored_idx", "participant")}

        # Per-subject simulation context. Only built for families that declare
        # num_alternatives; DDM/CDM take prepare_params' own defaults.
        sim_context = None
        if cfg.get("num_alternatives") is not None:
            sim_context = {"num_alternatives": int(cfg["num_alternatives"])}
            if "favored_idx" in d:
                sim_context["favored_idx"] = d["favored_idx"]
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
                                    param_mask_2d=param_mask_2d,
                                    fixed_natural=fixed_natural,
                                    context=sim_context)

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


def _cache_key(**kw):
    """Hash of everything that changes the SIMULATION, and nothing that doesn't.

    Colours, labels and figure sizes are deliberately absent: restyling must reuse
    the cache, which is the whole point of having one. Anything that moves a
    simulated number -- family, variant, draw count, seed, contamination, design
    geometry -- is in the key.
    """
    blob = json.dumps(kw, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def collect_ppc_cached(model_name, variant, n_ppc_samples, seed, no_contamination,
                       cache_dir=CACHE_DIR, refresh=False, fixed_natural=None,
                       posteriors_path=None):
    """collect_ppc with an on-disk cache of its simulated draws.

    Simulating 72 subjects x 500 draws takes minutes; re-plotting the same draws
    takes seconds. Without this, every cosmetic change to a PPC figure pays the
    full simulation cost again.
    """
    key = _cache_key(model=model_name, variant=variant, n_ppc_samples=n_ppc_samples,
                     seed=seed, no_contamination=no_contamination,
                     fixed_natural=fixed_natural,
                     # ADDITIVE: only present when set. A constrained REFIT has the
                     # same model, variant and flags as the canonical run but
                     # different posteriors, so it must not share the key -- while
                     # omitting the field entirely when unset keeps every existing
                     # canonical cache valid instead of invalidating all of them.
                     **({"posteriors_path": str(posteriors_path)}
                        if posteriors_path else {}),
                     num_alternatives=MODEL_REGISTRY[model_name].get("num_alternatives"),
                     max_num_regressors=MAX_NUM_REGRESSORS,
                     max_num_categories=MAX_NUM_CATEGORIES,
                     keep_intercept=KEEP_INTERCEPT)
    path = Path(cache_dir) / f"{model_name}_{variant}_{key}.pkl"

    if path.exists() and not refresh:
        logging.info(f"[cache hit] {path}")
        with path.open("rb") as fh:
            return pickle.load(fh)

    result = collect_ppc(
        model_name=model_name, variant=variant, n_ppc_samples=n_ppc_samples,
        seed=seed, no_contamination=no_contamination, fixed_natural=fixed_natural,
        posteriors_path=posteriors_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"[cache write] {path}")
    return result


def _kde_eval(data, xs):
    if len(data) < 5:
        return np.zeros_like(xs)
    try:
        return gaussian_kde(data, bw_method="silverman")(xs)
    except Exception:
        return np.zeros_like(xs)


def _draw_idx(n_draws, max_lines):
    """Evenly spaced subsample of draw indices, so figures are stable across reruns."""
    if n_draws <= max_lines:
        return np.arange(n_draws)
    return np.linspace(0, n_draws - 1, max_lines).round().astype(int)


# Observed data is the shared reference all three families are judged against, so
# it keeps ONE neutral colour in every figure; only the PREDICTED series carries a
# family hue. Giving the observations a colour that moved between families would
# imply the data changed with the model.
OBS_COLOR = OBSERVED_COLOR


def _ppc_colors(model_name):
    """``(line, fill)`` for this family's predictive series.

    The saturated stop for medians and individual draws, the light stop for the
    band beneath them, so band and line read as one family rather than as two
    unrelated series. See ``cogformer.utils.plot_utils.FAMILY_PALETTES``.
    """
    c = family_colors(model_name)
    return c["main_effect"], c["interaction"]

# Ratcliff's conventional RT quantile set. DO NOT change this default: these
# figures are read against published results that use exactly these five levels,
# so the default is fixed by comparability, not by taste.
#
# It is a BODY-focused set, and the tail analysis in
# logs/real_data/ppc_r4c4.log found the DDM misfit concentrated at the 99th and
# 99.9th percentiles, which nothing at or below 0.9 will show. That is a reason
# to produce a SEPARATE tail figure via --rt_quantiles, not to widen this one.
DEFAULT_RT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)

# Grouping variables are named after the data columns, which is right in code and
# wrong on the face of a figure -- "by stim_type" puts a snake_case identifier in
# front of a reader. Titles go through this map; anything absent falls back to the
# raw key, so adding a condition never silently breaks a title.
CONDITION_DISPLAY = {
    "stim_type":  "stimulus type",
    "truth":      "factual truth",
    "congruency": "congruency",
    "session":    "session",
    "speed":      "speed instruction",
    "difficulty": "difficulty",
}


def _cond_label(cond_name):
    return CONDITION_DISPLAY.get(cond_name, cond_name)


def _legend(ax, **kw):
    """Legend whose swatches are legible regardless of how the artist was drawn.

    The individual predictive draws are plotted at alpha 0.15 and lw 0.5, which is
    right on the axes -- a hundred of them overlaid read as a cloud -- and wrong in
    a legend, where a single hairline at 15% opacity is effectively invisible. Rebuild
    the "PPC draws" entry at full opacity and a visible width; pass everything else
    through untouched.
    """
    handles, labels = ax.get_legend_handles_labels()
    fixed = []
    for h, l in zip(handles, labels):
        if l == "PPC draws":
            h = Line2D([], [], color=h.get_color(), lw=1.2, alpha=0.9)
        fixed.append(h)
    return ax.legend(fixed, labels, **kw)


# ONE interval level across every figure in this experiment. The RT density PPC
# previously drew a 90% band while every quantile panel drew 95%, so comparing a
# density against its own quantile plot silently compared two different intervals.
PPC_CI = 95.0
_CI_LO, _CI_HI = (100 - PPC_CI) / 2, 100 - (100 - PPC_CI) / 2


def plot_ppc(obs_pooled, ppc_pooled, conditions, model_name, variant, outdir,
             draw_lines=True, max_lines=100, line_alpha=0.15, line_width=0.5):
    obs_color = OBS_COLOR
    ppc_color, ppc_fill = _ppc_colors(model_name)
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
                lo = np.percentile(ppc_kdes, _CI_LO, axis=0)
                hi = np.percentile(ppc_kdes, _CI_HI, axis=0)
                med = np.median(ppc_kdes, axis=0)
                ax.fill_between(xs, lo, hi, color=ppc_fill, alpha=0.40,
                                label=f"PPC {PPC_CI:g}% CI", zorder=1)

                # Individual posterior-predictive draws. The band summarises the
                # draws POINTWISE, so it hides whether any single draw has the
                # right shape: a set of clearly bimodal draws and a set of
                # uniformly-too-wide ones can produce the same 5-95 envelope.
                # Subsampled evenly rather than randomly so the figure is stable
                # across reruns, and drawn above the band but below the median
                # and observed curves.
                if draw_lines and max_lines > 0:
                    for k, i in enumerate(_draw_idx(len(ppc_kdes), max_lines)):
                        ax.plot(xs, ppc_kdes[i], color=ppc_color, lw=line_width,
                                alpha=line_alpha, zorder=2, solid_capstyle="butt",
                                label="PPC draws" if k == 0 else None)

                ax.plot(xs, med, color=ppc_color, lw=1.5, label="PPC median", zorder=3)

            # Observed
            obs_kde = _kde_eval(obs, xs)
            ax.plot(xs, obs_kde, color=obs_color, lw=1.8, label="Observed", zorder=4)

            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("RT (s)", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        _legend(axes[0], fontsize=7, frameon=False)
        fig.suptitle(
            f"{model_name.upper()} — RT posterior predictive by {_cond_label(cond_name)}",
            fontsize=10,
        )
        fig.tight_layout()

        outdir.mkdir(parents=True, exist_ok=True)
        savepath = outdir / f"{model_name}_{variant}_ppc_{cond_name}.pdf"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {savepath}")


def _quantile_tag(quantiles):
    """Filename suffix marking a non-default quantile set.

    The default figures are compared against published results, so they must keep
    a stable name; an exploratory tail run must not overwrite them. Default set
    -> empty suffix, anything else -> an explicit one.
    """
    if len(quantiles) == len(DEFAULT_RT_QUANTILES) and np.allclose(
            np.asarray(quantiles, dtype=float), np.asarray(DEFAULT_RT_QUANTILES)):
        return ""
    return "_q" + "-".join(f"{q:g}".replace("0.", "") for q in quantiles)


def _quantile_stats(obs, ppc_samples, quantiles):
    """Observed quantiles, plus the posterior-predictive distribution of each quantile.

    Returns (obs_q, ppc_med, lo, hi), each shape (n_quantiles,), or None if there
    are no draws. The PPC interval is over DRAWS of the quantile, not over trials:
    it answers "where does the model think this quantile sits", which is what the
    observed value should be compared against.
    """
    obs_q = np.quantile(obs, quantiles)
    if not ppc_samples:
        return obs_q, None, None, None
    ppc_q = np.stack([np.quantile(s, quantiles) for s in ppc_samples if len(s)], axis=0)
    if ppc_q.size == 0:
        return obs_q, None, None, None
    med = np.median(ppc_q, axis=0)
    lo, hi = np.percentile(ppc_q, [_CI_LO, _CI_HI], axis=0)
    return obs_q, med, lo, hi


def plot_rt_quantiles(obs_pooled, ppc_pooled, conditions, model_name, variant, outdir,
                      quantiles=DEFAULT_RT_QUANTILES, ppc_color=None):
    """RT quantiles by level: observed points against the PPC quantile interval.

    Complements the density PPC. A KDE can look acceptable while specific
    quantiles are systematically displaced, because the eye integrates over the
    curve; here each quantile is its own comparison with its own interval.
    """
    # ppc_color is an override used by ppc_compare_original.py to draw the ORIGINAL
    # authors' series through this same function. An override has no ramp behind it,
    # so line and band share the one colour; the default path takes the family ramp's
    # saturated and light stops.
    if ppc_color is None:
        ppc_color, ppc_fill = _ppc_colors(model_name)
    else:
        ppc_fill = ppc_color
    tag = _quantile_tag(quantiles)
    quantiles = np.asarray(quantiles, dtype=float)
    xs = np.arange(len(quantiles))

    for cond_name, cond_col, label_map in conditions:
        labels = list(label_map.values())
        fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.2),
                                 squeeze=False)

        for ax, lbl in zip(axes[0], labels):
            obs = obs_pooled.get(cond_name, {}).get(lbl, np.array([]))
            if len(obs) == 0:
                ax.set_visible(False)
                continue
            ppc_samples = ppc_pooled.get(cond_name, {}).get(lbl, [])
            obs_q, med, lo, hi = _quantile_stats(obs, ppc_samples, quantiles)

            if med is not None:
                ax.fill_between(xs, lo, hi, color=ppc_fill, alpha=0.40,
                                label=f"PPC {PPC_CI:g}% CI", zorder=1)
                ax.plot(xs, med, "o-", color=ppc_color, lw=1.4, ms=4,
                        label="PPC median", zorder=2)
            ax.plot(xs, obs_q, "s--", color=OBS_COLOR, lw=1.6, ms=4,
                    label="Observed", zorder=3)

            ax.set_xticks(xs)
            ax.set_xticklabels([f"{q:g}" for q in quantiles])
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("RT quantile", fontsize=8)
            ax.set_ylabel("RT (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        _legend(axes[0, 0], fontsize=7, frameon=False)
        fig.suptitle(f"{model_name.upper()} — RT quantiles by {_cond_label(cond_name)}",
                     fontsize=10)
        fig.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        savepath = outdir / f"{model_name}_{variant}_rt_quantiles_{cond_name}{tag}.pdf"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {savepath}")


def plot_rt_qq(obs_pooled, ppc_pooled, conditions, model_name, variant, outdir,
               quantiles=DEFAULT_RT_QUANTILES, ppc_color=None):
    """Q-Q: observed RT quantile against the predicted one, with a 45-degree reference.

    Reads differently from plot_rt_quantiles: deviation from the diagonal is the
    misfit in seconds, so systematic over- or under-prediction shows as a tilt
    rather than as two curves to mentally subtract.
    """
    # Error bars only, no band, so this one needs just the line colour.
    if ppc_color is None:
        ppc_color = _ppc_colors(model_name)[0]
    tag = _quantile_tag(quantiles)
    quantiles = np.asarray(quantiles, dtype=float)

    for cond_name, cond_col, label_map in conditions:
        labels = list(label_map.values())
        fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.2),
                                 squeeze=False)

        for ax, lbl in zip(axes[0], labels):
            obs = obs_pooled.get(cond_name, {}).get(lbl, np.array([]))
            if len(obs) == 0:
                ax.set_visible(False)
                continue
            ppc_samples = ppc_pooled.get(cond_name, {}).get(lbl, [])
            obs_q, med, lo, hi = _quantile_stats(obs, ppc_samples, quantiles)
            if med is None:
                ax.set_visible(False)
                continue

            span = np.concatenate([obs_q, lo, hi])
            pad = 0.05 * (span.max() - span.min() or 1.0)
            ends = [span.min() - pad, span.max() + pad]
            ax.plot(ends, ends, ls="--", color=GUIDE_COLOR, lw=1, zorder=1)
            ax.errorbar(obs_q, med, yerr=[med - lo, hi - med], fmt="o",
                        color=ppc_color, ms=4, lw=1.2, capsize=3,
                        label=f"PPC median + {PPC_CI:g}% CI", zorder=2)
            for x, y, q in zip(obs_q, med, quantiles):
                ax.annotate(f"{q:g}", (x, y), textcoords="offset points",
                            xytext=(4, -8), fontsize=6, color=OBS_COLOR)

            ax.set_xlim(ends); ax.set_ylim(ends)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("Observed RT quantile (s)", fontsize=8)
            ax.set_ylabel("Predicted RT quantile (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        visible = [a for a in axes[0] if a.get_visible()]
        if visible:
            _legend(visible[0], fontsize=7, frameon=False)
        fig.suptitle(f"{model_name.upper()} — RT quantile\u2013quantile by {_cond_label(cond_name)}",
                     fontsize=10)
        fig.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        savepath = outdir / f"{model_name}_{variant}_rt_qq_{cond_name}{tag}.pdf"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {savepath}")


def plot_choice_quantiles(obs_ch, ppc_ch, conditions, model_name, variant, outdir,
                          quantiles=DEFAULT_RT_QUANTILES, ppc_color=None):
    """Quantile plot on the RESPONSE dimension. CDM only.

    CDM's response is a continuous angle, so quantiles of theta are meaningful in
    the same way RT quantiles are. DDM and RDM respond with a binary choice,
    where quantiles carry no information -- for those the response-dimension
    check is the proportion comparison in plot_choice_ppc, not this.
    """
    if model_name != "cdm":
        logging.info(f"[choice quantiles] skipped for {model_name}: response is discrete")
        return

    # ppc_color is an override used by ppc_compare_original.py to draw the ORIGINAL
    # authors' series through this same function. An override has no ramp behind it,
    # so line and band share the one colour; the default path takes the family ramp's
    # saturated and light stops.
    if ppc_color is None:
        ppc_color, ppc_fill = _ppc_colors(model_name)
    else:
        ppc_fill = ppc_color
    quantiles = np.asarray(quantiles, dtype=float)
    tag = _quantile_tag(quantiles)
    xs = np.arange(len(quantiles))
    outdir.mkdir(parents=True, exist_ok=True)

    for cond_name, _, label_map in conditions:
        labels = [l for l in label_map.values()
                  if len(obs_ch.get(cond_name, {}).get(l, []))
                  and len(ppc_ch.get(cond_name, {}).get(l, []))]
        if not labels:
            continue
        fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.2),
                                 squeeze=False)
        for ax, lbl in zip(axes[0], labels):
            o = np.asarray(obs_ch[cond_name][lbl], dtype=float)
            draws = [np.asarray(p, dtype=float) for p in ppc_ch[cond_name][lbl]]
            obs_q, med, lo, hi = _quantile_stats(o, draws, quantiles)
            if med is not None:
                ax.fill_between(xs, lo, hi, color=ppc_fill, alpha=0.40,
                                label=f"PPC {PPC_CI:g}% CI", zorder=1)
                ax.plot(xs, med, "o-", color=ppc_color, lw=1.4, ms=4,
                        label="PPC median", zorder=2)
            ax.plot(xs, obs_q, "s--", color=OBS_COLOR, lw=1.6, ms=4,
                    label="Observed", zorder=3)
            ax.axhline(0.0, color=GUIDE_COLOR, ls=":", lw=0.8, zorder=0)
            ax.set_xticks(xs); ax.set_xticklabels([f"{q:g}" for q in quantiles])
            ax.set_title(lbl, fontsize=9)
            ax.set_xlabel("Response quantile", fontsize=8)
            ax.set_ylabel(r"Response deviation $\theta$ (rad)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        _legend(axes[0, 0], fontsize=7, frameon=False)
        fig.suptitle(f"{model_name.upper()} — response-angle quantiles "
                     f"by {_cond_label(cond_name)}", fontsize=10)
        fig.tight_layout()
        savepath = outdir / f"{model_name}_{variant}_choice_quantiles_{cond_name}{tag}.pdf"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {savepath}")


def plot_choice_ppc(obs_ch, ppc_ch, conditions, model_name, variant, outdir,
                    draw_lines=True, max_lines=100, line_alpha=0.15, line_width=0.5):
    """PPC on the RESPONSE dimension.

    CDM's response is a continuous angle, so its misfit is invisible to an RT-only
    PPC — plot_ppc never touches choices. For CDM this compares the observed angular
    density against the predicted one; for DDM/RDM it compares choice proportions.
    """
    circular = model_name == "cdm"
    ppc_color, ppc_fill = _ppc_colors(model_name)
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
                lo, hi = np.percentile(dens, [_CI_LO, _CI_HI], axis=0)
                ax.fill_between(xs, lo, hi, alpha=0.40, color=ppc_fill, lw=0,
                                label=f"PPC {PPC_CI:g}% CI", zorder=1)

                # Same reasoning as the RT PPC: the band is pointwise, so it
                # cannot show whether individual draws put mass at +-pi/2 or
                # simply spread wider. That distinction is the whole question for
                # CDM, whose misfit lives in the shape of the angular density.
                if draw_lines and max_lines > 0:
                    for k, i in enumerate(_draw_idx(len(dens), max_lines)):
                        ax.plot(xs, dens[i], color=ppc_color, lw=line_width,
                                alpha=line_alpha, zorder=2, solid_capstyle="butt",
                                label="PPC draws" if k == 0 else None)

                ax.plot(xs, np.median(dens, axis=0), color=ppc_color, lw=1.4,
                        label="PPC median", zorder=3)
                ax.plot(xs, _kde_eval(o, xs), color=OBS_COLOR, lw=1.8,
                        label="Observed", zorder=4)
                for v in (-np.pi / 2, np.pi / 2):
                    ax.axvline(v, color=GUIDE_COLOR, ls=":", lw=1.2, zorder=0)
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xlabel(r"Response deviation $\theta$ (rad)", fontsize=9)
                frac_o = float(np.mean(np.abs(o) > np.pi / 2))
                frac_p = float(np.median([np.mean(np.abs(np.asarray(p)) > np.pi / 2) for p in P]))
                ax.set_title(f"{lbl}\n$P(|\\theta|>\\pi/2)$: obs {frac_o:.3f} / ppc {frac_p:.3f}",
                             fontsize=9)
            else:
                levels = np.unique(o)
                obs_p = np.array([np.mean(o == v) for v in levels])
                ppc_p = np.array([[np.mean(np.asarray(p) == v) for v in levels] for p in P])
                lo, hi = np.percentile(ppc_p, [_CI_LO, _CI_HI], axis=0)
                med = np.median(ppc_p, axis=0)
                x = np.arange(len(levels))
                ax.bar(x, obs_p, width=0.55, color=OBS_COLOR, alpha=0.85, label="Observed")
                ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", color=ppc_color,
                            capsize=4, lw=1.4, label=f"PPC median + {PPC_CI:g}% CI")
                ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}" for v in levels])
                ax.set_xlabel("Choice", fontsize=9)
                ax.set_title(lbl, fontsize=9)

            ax.set_ylabel("Density" if circular else "Proportion", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                _legend(ax, fontsize=7, frameon=False)

        what = "response-angle posterior predictive" if circular else "choice proportions"
        fig.suptitle(f"{model_name.upper()} — {what} by {_cond_label(cond_name)}",
                     fontsize=10)
        fig.tight_layout()
        savepath = outdir / f"{model_name}_{variant}_choice_ppc_{cond_name}.pdf"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[saved] {savepath}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   choices=["ddm", "rdm", "cdm", "lba"])
    p.add_argument("--variant", type=str, default="mf", choices=["mf", "mc"])
    p.add_argument("--n_ppc_samples", type=int, default=None,
                   help="Posterior draws to simulate from. Default: all draws stored in "
                        "the npz. Pass a smaller number only for a deliberate quick look — "
                        "a thin sample gives an unstable, too-narrow 95%% band.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--max_num_regressors", type=int, default=MAX_NUM_REGRESSORS)
    p.add_argument("--max_num_categories", type=int, default=MAX_NUM_CATEGORIES)
    p.add_argument("--no_ppc_lines", action="store_true",
                   help="Suppress the individual posterior-predictive draw curves in the RT "
                        "PPC and in CDM's angular PPC, showing only the band, median and "
                        "observed density. No effect on DDM/RDM choice proportions, which "
                        "are bars rather than densities.")
    p.add_argument("--ppc_max_lines", type=int, default=100,
                   help="Cap on individual draw curves per panel (evenly subsampled). "
                        "Bounds PDF size and keeps the band readable underneath.")
    p.add_argument("--ppc_line_alpha", type=float, default=0.15,
                   help="Opacity of the individual draw curves. Raise toward ~0.4 to make "
                        "single draws legible, lower toward ~0.05 for a denser sample.")
    p.add_argument("--ppc_line_width", type=float, default=0.5,
                   help="Line width of the individual draw curves.")
    p.add_argument("--rt_quantiles", type=float, nargs="+", default=list(DEFAULT_RT_QUANTILES),
                   metavar="Q",
                   help="Quantile levels for the RT quantile and Q-Q plots, in (0, 1). "
                        "Default is Ratcliff's 0.1 0.3 0.5 0.7 0.9, which describes the "
                        "body of the distribution; add e.g. 0.99 to probe the tail.")
    p.add_argument("--no_contamination", action="store_true",
                   help="Set pi=0 during simulation to isolate distributional fit from lapse contamination.")
    p.add_argument("--fix_natural", type=str, default=None, metavar="P=V,...",
                   help="Pin parameters to constants on the NATURAL (post-link) scale "
                        "during SIMULATION ONLY, e.g. 's_v=1,s_tau=0'. A DIAGNOSTIC, "
                        "not a refit: the posteriors were obtained with these free, and "
                        "the other parameters were fitted to compensate for whatever "
                        "values they took, so the predictive can get WORSE. It answers "
                        "'what would this posterior predict if these took conventional "
                        "values', not 'what if the model had been fitted that way'.")
    p.add_argument("--no_cache", action="store_true",
                   help="Simulate from scratch and do not read or write the draw cache.")
    p.add_argument("--refresh_cache", action="store_true",
                   help="Re-simulate and overwrite the cached draws for this configuration.")
    p.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
    p.add_argument("--posteriors", type=str, default=None,
                   help="Explicit posteriors .npz, overriding the default "
                        "outputs/data/real_data/<model>/<variant>/ path. Use this to "
                        "score a CONSTRAINED REFIT (e.g. ddm/mc_sv_fixed/) without "
                        "inventing a new --variant. NOTE: if the refit fixes a "
                        "parameter by dropping it from the design, its coefficients "
                        "are masked to 0 and the link still applies -- "
                        "shifted_softplus(0)=0.693, not 0 -- so pass --fix_natural "
                        "for that parameter too.")
    return p.parse_args()


def main():
    args = parse_args()
    bad = [q for q in args.rt_quantiles if not 0.0 < q < 1.0]
    if bad:
        raise SystemExit(f"--rt_quantiles must lie strictly inside (0, 1); got {bad}")
    args.rt_quantiles = sorted(args.rt_quantiles)
    global MAX_NUM_REGRESSORS, MAX_NUM_CATEGORIES
    MAX_NUM_REGRESSORS = args.max_num_regressors
    MAX_NUM_CATEGORIES = args.max_num_categories
    outdir = Path(args.outdir) if args.outdir else FIG_DIR / args.model
    logging.info(f"Running PPC: model={args.model}, variant={args.variant}, n_ppc_samples={args.n_ppc_samples}")

    fixed_natural = None
    if args.fix_natural:
        fixed_natural = {}
        for part in args.fix_natural.split(","):
            k, _, v = part.partition("=")
            fixed_natural[k.strip()] = float(v)
        logging.info(f"[simulation override, natural scale] {fixed_natural}")

    if args.no_cache:
        obs_pooled, ppc_pooled, conditions, obs_ch, ppc_ch = collect_ppc(
            model_name=args.model,
            variant=args.variant,
            n_ppc_samples=args.n_ppc_samples,
            seed=args.seed,
            no_contamination=args.no_contamination,
            fixed_natural=fixed_natural,
            posteriors_path=args.posteriors,
        )
    else:
        obs_pooled, ppc_pooled, conditions, obs_ch, ppc_ch = collect_ppc_cached(
            model_name=args.model,
            variant=args.variant,
            n_ppc_samples=args.n_ppc_samples,
            seed=args.seed,
            no_contamination=args.no_contamination,
            cache_dir=args.cache_dir,
            refresh=args.refresh_cache,
            fixed_natural=fixed_natural,
            posteriors_path=args.posteriors,
        )
    plot_ppc(obs_pooled, ppc_pooled, conditions, args.model, args.variant, outdir,
             draw_lines=not args.no_ppc_lines, max_lines=args.ppc_max_lines,
             line_alpha=args.ppc_line_alpha, line_width=args.ppc_line_width)
    plot_rt_quantiles(obs_pooled, ppc_pooled, conditions, args.model, args.variant,
                      outdir, quantiles=args.rt_quantiles)
    plot_rt_qq(obs_pooled, ppc_pooled, conditions, args.model, args.variant,
               outdir, quantiles=args.rt_quantiles)
    plot_choice_quantiles(obs_ch, ppc_ch, conditions, args.model, args.variant,
                          outdir, quantiles=args.rt_quantiles)
    plot_choice_ppc(obs_ch, ppc_ch, conditions, args.model, args.variant, outdir,
                    draw_lines=not args.no_ppc_lines, max_lines=args.ppc_max_lines,
                    line_alpha=args.ppc_line_alpha, line_width=args.ppc_line_width)
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
