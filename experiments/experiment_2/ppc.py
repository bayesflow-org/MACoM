"""
Global posterior predictive check for real data.

Pools simulated RT distributions across all subjects and compares them to
observed RT distributions, stratified by experimental condition.

Usage:
    python -m experiments.experiment_2.ppc --model ddm --variant mf
    python -m experiments.experiment_2.ppc --model cdm --variant mc
    python -m experiments.experiment_2.ppc --model rdm --variant mf
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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from matplotlib.transforms import Bbox
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
from .lba_mc_cf_apply import LBA_DESIGN_CONFIG, LBA_DESIGNS

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
        # Declared so collect_ppc builds a simulation context at all: that branch is
        # gated on num_alternatives being present, and favored_idx rides along inside
        # it. 2 is what prepare_params would default to, so this adds the per-trial
        # favored_idx pass-through and changes nothing else.
        "num_alternatives":    2,
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


def _prepare_subject(model, link_funs, intrinsic_params, design_matrix, posterior_sample,
                     no_contamination=False, param_mask_2d=None, fixed_natural=None,
                     context=None):
    """
    Per-trial simulator parameters for one set of trials from one posterior sample.
    Simulated by the caller, all draws of a subject in one model.simulate_many call
    (a single launch on the GPU backend; the numba backend loops, in the same order).

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
    Returns the prepared params; _finish_subject turns the simulation into
    {"rts", "choices", "valid_mask"}.
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
    return model.prepare_params(params, num_obs=num_obs, context=context)


def _finish_subject(result):
    rts = result["rts"].squeeze()
    choices = result["choices"].squeeze()
    valid = rts > 0
    return {"rts": rts[valid], "choices": choices[valid], "valid_mask": valid}


def collect_ppc(
    model_name, variant, n_ppc_samples=None, seed=0, no_contamination=False,
    fixed_natural=None, posteriors_path=None, design=None,
):
    rng = np.random.default_rng(seed)
    cfg = MODEL_REGISTRY[model_name]
    if design and design != "full":
        # Copy, never mutate: MODEL_REGISTRY is module state and an in-place edit would
        # leak into any later call in the same process.
        cfg = {**cfg, "design_config": LBA_DESIGNS[design]}

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
    # The npz records the design it was fitted with (design_name, written since
    # 2026-09-16). A mismatch relabels the coefficient rows without changing their
    # shape, so nothing downstream can catch it -- check here rather than trust the
    # caller to pass matching flags to two scripts. Older files lack the field.
    if "design_name" in post_npz.files:
        fitted = str(post_npz["design_name"])
        want = design or "full"
        if fitted != want:
            raise SystemExit(
                f"{post_file} was fitted with design={fitted!r} but the PPC was asked "
                f"for design={want!r}. Pass --design {fitted}.")
        logging.info(f"design={fitted} (matches the fitted posteriors)")
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

        # PPC: simulate every sampled posterior draw of this subject in one call
        requests = [
            (_prepare_subject(model, link_funs, intrinsic_params, dm,
                              post_subj[s_idx].reshape(n_rows, n_params),
                              no_contamination=no_contamination,
                              param_mask_2d=param_mask_2d,
                              fixed_natural=fixed_natural,
                              context=sim_context), sim_context)
            for s_idx in sample_indices
        ]
        for k, result in enumerate(model.simulate_many(requests)):
            sim = _finish_subject(result)

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
                       posteriors_path=None, design=None):
    """collect_ppc with an on-disk cache of its simulated draws.

    Simulating 72 subjects x 500 draws takes minutes; re-plotting the same draws
    takes seconds. Without this, every cosmetic change to a PPC figure pays the
    full simulation cost again.
    """
    # The POSTERIORS' CONTENT is in the key. Before 2026-09-27 the canonical run keyed
    # only on family/variant/flags, so refitting with a new checkpoint and re-running
    # this script silently reloaded draws simulated from the OLD posteriors. Hashing
    # the file (a few MB) ties every cache entry to the exact fit it came from.
    post_file = (Path(posteriors_path) if posteriors_path else
                 DATA_DIR / model_name / variant / f"{model_name}_{variant}_posteriors.npz")
    post_sha = hashlib.sha256(post_file.read_bytes()).hexdigest()[:16]
    key = _cache_key(model=model_name, variant=variant, n_ppc_samples=n_ppc_samples,
                     posteriors_sha=post_sha,
                     seed=seed, no_contamination=no_contamination,
                     fixed_natural=fixed_natural,
                     # ADDITIVE: only present when set. A constrained REFIT has the
                     # same model, variant and flags as the canonical run but
                     # different posteriors, so it must not share the key -- while
                     # omitting the field entirely when unset keeps every existing
                     # canonical cache valid instead of invalidating all of them.
                     **({"posteriors_path": str(posteriors_path)}
                        if posteriors_path else {}),
                     # ADDITIVE, same reason: the design changes what is simulated but
                     # is absent from every cache written before it existed, so adding
                     # it unconditionally would invalidate all of them.
                     **({"design": design} if design and design != "full" else {}),
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
        posteriors_path=posteriors_path, design=design,
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
            ax.set_xlabel("Choice quantile", fontsize=8)
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


# ── combined figures ──────────────────────────────────────────────────────────
# The per-condition figures above stay as they are. These put a family's whole PPC
# on one page: every condition group's quantile panels in a row, then the
# response-dimension check on the right, with the groups kept apart by dividers
# rather than by being separate files.
#
# Group order is the FIGURE's, not the registry's. CDM_CONDITIONS lists speed
# before difficulty; the combined figure reads difficulty first, so the order is
# declared here instead of reordering the registry, which the per-condition
# figures and the cache key both depend on.
COMBINED_COND_ORDER = {
    "ddm": ["stim_type", "truth"],
    "rdm": ["congruency", "session"],
    "cdm": ["difficulty", "speed"],
}

# Dashed separates anything WITHIN one model family -- condition groups, and the
# break between the RT panels and the response-dimension check. Solid is reserved
# for the break between FAMILIES in the all-families figure, which is the only
# boundary across which nothing is comparable.
_DIV_LIGHT = dict(color=GUIDE_COLOR, lw=1.0, ls=(0, (4, 3)))
_DIV_HEAVY = dict(color="#5a5a66", lw=1.5, ls="-")

# Gutter width, as matplotlib's fraction of the mean axes width. Applied uniformly
# after tight_layout so every pair of neighbouring panels is spaced identically,
# whether or not a divider falls between them. Kept to what the widened tick labels
# and a divider actually need -- the dividers are placed against the labels' real
# extent, so the gutter does not have to be padded for them.
WSPACE = 0.26

# Row gap for the two-row (CDM) figure, and the band of figure height each row
# spends on its title, ticks and axis label. The two rows sit closer than the
# default layout puts them: the gap only has to hold the upper row's x label and
# the lower row's titles, and a solid rule marks the break between them.
HSPACE, ROW_BAND = 0.34, 0.46

# Blank space between the lowest x label and the figure legend hung below it, in
# DRAWN points (the page halves it).
LEGEND_GAP_PT = 2.0

# SIZED FOR THE PAGE, NOT FOR THE SCREEN. The target is a LaTeX \textwidth of 5.5 in
# (US Letter, 1.5 in margins). The figure is drawn at FIG_SCALE times that, so
# \includegraphics[width=\textwidth] scales it down by 1/FIG_SCALE and every point
# size below lands at that fraction of its nominal value on the page. Drawing at 2x
# rather than at 1:1 leaves the panels big enough to lay out -- five panels across
# 5.5 in is barely an inch each -- while the text still arrives at a readable size.
#
# A figure drawn at the old 3.1 in per panel was ~17 in wide; at \textwidth that is a
# scale factor near 0.3, which turned 9 pt titles into under 3 pt on the page. That is
# the reason these are absolute rather than per-panel.
TEXT_WIDTH_IN = 5.5
FIG_SCALE = 2.0
FIG_W = TEXT_WIDTH_IN * FIG_SCALE

# Nominal sizes; divide by FIG_SCALE for what the reader gets.
# Title above label, with a real gap between them: at 16/14 the two read as one
# weight. The label drops further than the title, so the panel's name leads and
# the axis name follows.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 14, 11, 10, 8           # 7 / 5.5 / 5 / 4 pt
FS_MINI_TITLE, FS_MINI_LABEL, FS_MINI_TICK, FS_MINI_LEG = 11, 9.5, 7.5, 8

# Gaps inside the 2x2, as fractions of a mini axes. Module level because the
# title-fitting pass has to use the same numbers the layout used.
CHOICE_GW, CHOICE_GH = 0.70, 1.05


def _ordered_conditions(conditions, model_name):
    want = COMBINED_COND_ORDER.get(model_name)
    if not want:
        return list(conditions)
    by_name = {c[0]: c for c in conditions}
    return [by_name[n] for n in want if n in by_name]


# Level order inside a condition group, where the figure wants something other than
# the registry's. CDM's speed map is keyed -0.5/0.5 = Unspeeded/Speeded, so its
# natural order is the reverse of how the figure reads.
COMBINED_LEVEL_ORDER = {"speed": ["Speeded", "Unspeeded"]}


def _levels_with_data(pool_obs, pool_ppc, cond_name, label_map):
    have = [l for l in label_map.values()
            if len(pool_obs.get(cond_name, {}).get(l, []))
            and len(pool_ppc.get(cond_name, {}).get(l, []))]
    want = COMBINED_LEVEL_ORDER.get(cond_name)
    if want:
        have = [l for l in want if l in have] + [l for l in have if l not in want]
    return have


def _draw_quantile_panel(ax, obs, ppc_samples, quantiles, xs, ppc_color, ppc_fill,
                         xlabel, ylabel, zero_guide=False, label_fs=None, tick_fs=None,
                         fill_alpha=0.40):
    """One quantile panel, identical in construction to plot_rt_quantiles'."""
    label_fs = FS_LABEL if label_fs is None else label_fs
    tick_fs = FS_TICK if tick_fs is None else tick_fs
    obs_q, med, lo, hi = _quantile_stats(obs, ppc_samples, quantiles)
    if med is not None:
        ax.fill_between(xs, lo, hi, color=ppc_fill, alpha=fill_alpha,
                        label=f"PPC {PPC_CI:g}% CI", zorder=1)
        ax.plot(xs, med, "o-", color=ppc_color, lw=1.4, ms=4,
                label="PPC median", zorder=2)
    ax.plot(xs, obs_q, "s--", color=OBS_COLOR, lw=1.6, ms=4,
            label="Observed", zorder=3)
    if zero_guide:
        ax.axhline(0.0, color=GUIDE_COLOR, ls=":", lw=0.8, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{q:g}" for q in quantiles])
    ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)


def _choice_grid_axes(fig, pos, nrow=2, ncol=2, gw=CHOICE_GW, gh=CHOICE_GH,
                      square=False):
    """`nrow` x `ncol` axes laid inside the rectangle `pos`, in figure coordinates.

    The whole block has to occupy exactly ONE quantile panel's footprint, so the
    axes are placed by hand inside that panel's rectangle rather than given their
    own gridspec columns -- a nested gridspec is re-sized by tight_layout and would
    not hold the footprint. `pos` is read AFTER layout, so it is the final rectangle.

    THE GAPS ARE LARGER THAN THE USUAL GUTTER ON PURPOSE. The footprint is fixed, so
    the only way to give this block text at the same size as the rest of the figure
    is to spend the footprint on TEXT rather than on drawing area: the axes shrink
    and the gaps grow, and a title or tick label wider than its own axes overhangs
    into the gap instead of into its neighbour. The plots end up small; that is the
    trade the fixed footprint forces, and the alternative is text no one can read.
    """
    out = []
    for row in _choice_grid_rects(pos, nrow, ncol, gw, gh):
        axs = [fig.add_axes(r) for r in row]
        if square:
            # Square each panel rather than letting it fill a cell that may be much
            # wider than tall; the surplus width becomes spacing.
            for ax in axs:
                ax.set_box_aspect(1)
        out.append(axs)
    return out


def _choice_grid_rects(pos, nrow, ncol, gw=CHOICE_GW, gh=CHOICE_GH):
    """The rectangles `_choice_grid_axes` places axes in, as [row][col]."""
    cw = pos.width / (ncol + gw * (ncol - 1))
    ch = pos.height / (nrow + gh * (nrow - 1))
    return [[[pos.x0 + c * (cw + gw * cw),
              pos.y0 + pos.height - (r + 1) * ch - r * gh * ch, cw, ch]
             for c in range(ncol)] for r in range(nrow)]


def _fit_block(fig, axes, pos, nrow, ncol, passes=2):
    """Pull the grid in until everything it draws sits inside `pos`.

    The axes rectangles are only the DRAWING areas; the left column's tick labels
    and y label hang off to their left, and at this size that overhang crosses the
    divider and lands on the quantile panels. Measuring what the block actually
    occupies and re-placing it inside the same footprint is what keeps the block
    within its own column -- an inset guessed up front is wrong as soon as a tick
    label gains a digit.

    LEFT EDGE ONLY. The titles above and the "Choice" label below overhang too, but
    they sit at the same heights as the quantile panels' own title and x label, so
    they collide with nothing; the left overhang is the one that crosses the divider.
    Correcting all four sides instead feeds back through the square aspect -- a
    shorter rectangle gives a smaller square, whose labels are unchanged and so
    overhang proportionally more -- and the block collapses over successive passes.

    Two passes: narrowing the rectangles moves the labels, so the first correction
    slightly overshoots and the second settles it.
    """
    live = [ax for row in axes for ax in row if ax.get_visible()]
    if not live:
        return
    rect = Bbox.from_extents(pos.x0, pos.y0, pos.x1, pos.y1)
    for _ in range(passes):
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        rend = fig.canvas.get_renderer()
        bb = Bbox.union([ax.get_tightbbox(rend).transformed(inv) for ax in live])
        dx0 = max(0.0, pos.x0 - bb.x0)
        if dx0 < 1e-4:
            return
        rect = Bbox.from_extents(rect.x0 + dx0, rect.y0, rect.x1, rect.y1)
        for r, row in enumerate(_choice_grid_rects(rect, nrow, ncol)):
            for c, xywh in enumerate(row):
                if axes[r][c].get_visible():
                    axes[r][c].set_position(xywh)


def _draw_choice_grid(fig, pos, obs_ch, ppc_ch, ordered, ppc_color, bar_color,
                      legend=True, square_panels=False):
    """The response-dimension check as a 2x2 of per-level proportion panels.

    One panel per condition level, as in the standalone `choice_ppc_<cond>.pdf`
    figures, but folded into the footprint a single quantile panel would take so the
    row stays one row. Reading order is row-major and the condition groups are the
    ROWS -- stimulus type above, factual truth below -- with a dashed rule between
    them, the horizontal counterpart of the vertical rules in the quantile block.

    The bars take the family's LIGHT stop and the predictive marker its saturated
    one -- the same pairing the quantile panels use for band and line, so the two
    read as one family while staying apart from each other. This departs from the
    rule in plot_utils that observed data is always neutral; it is deliberate, and
    it means the observed bars here carry a colour that moves between families.

    Returns the axes grid, or None when there is nothing to draw.
    """
    levels = []
    for cond_name, _, label_map in ordered:
        here = _levels_with_data(obs_ch, ppc_ch, cond_name, label_map)
        levels += [(cond_name, l) for l in here]
    if not levels:
        return None

    ncol = 2
    nrow = int(np.ceil(len(levels) / ncol))
    axes = _choice_grid_axes(fig, pos, nrow, ncol, square=square_panels)

    # One y scale across the four, or a level whose proportions happen to sit low
    # would be drawn at the same apparent height as one that sits high.
    top = 0.0
    for cond_name, lbl in levels:
        o = np.asarray(obs_ch[cond_name][lbl], dtype=float)
        P = ppc_ch[cond_name][lbl]
        vals = np.unique(o)
        top = max(top, float(np.max([np.mean(o == v) for v in vals])))
        ppc_p = np.array([[np.mean(np.asarray(p) == v) for v in vals] for p in P])
        top = max(top, float(np.max(np.percentile(ppc_p, _CI_HI, axis=0))))
    # The legend sits under the block now, not in the top-left panel, so the bars
    # get the height back: just enough headroom to keep the tallest CI off the spine.
    top *= 1.12

    for k, (cond_name, lbl) in enumerate(levels):
        r, c = divmod(k, ncol)
        ax = axes[r][c]
        o = np.asarray(obs_ch[cond_name][lbl], dtype=float)
        P = ppc_ch[cond_name][lbl]
        vals = np.unique(o)
        obs_p = np.array([np.mean(o == v) for v in vals])
        ppc_p = np.array([[np.mean(np.asarray(p) == v) for v in vals] for p in P])
        lo, hi = np.percentile(ppc_p, [_CI_LO, _CI_HI], axis=0)
        med = np.median(ppc_p, axis=0)
        x = np.arange(len(vals))
        ax.bar(x, obs_p, width=0.55, color=bar_color, edgecolor=ppc_color,
               linewidth=0.6, label="Observed")
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", color=ppc_color,
                    capsize=2.5, lw=1.1, ms=3.2,
                    label=f"PPC median + {PPC_CI:g}% CI")
        ax.set_xticks(x); ax.set_xticklabels([f"{int(v)}" for v in vals])
        ax.set_ylim(0, top)
        # Three y ticks at most: at this size a default locator puts five labels in
        # a space that holds two, and they collide before they are read.
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_title(lbl, fontsize=FS_MINI_TITLE)
        ax.tick_params(labelsize=FS_MINI_TICK)
        ax.set_box_aspect(1)
        ax.spines[["top", "right"]].set_visible(False)
        # Axis labels only on the outer edges: at quarter size there is no room to
        # repeat them, and repeating them says nothing a shared scale does not.
        if c == 0:
            ax.set_ylabel("Proportion", fontsize=FS_MINI_LABEL)
        else:
            ax.set_yticklabels([])
        if r == nrow - 1:
            ax.set_xlabel("Choice", fontsize=FS_MINI_LABEL)

    # Unused cells, when a family's groups do not fill the grid.
    for k in range(len(levels), nrow * ncol):
        r, c = divmod(k, ncol)
        axes[r][c].set_visible(False)

    _fit_block(fig, axes, pos, nrow, ncol)

    # Long level names -- RDM's "Incongruent" -- do not fit a quarter-size panel at
    # the nominal size and run into the panel beside them. Measure the widest and
    # shrink only as far as it requires, so a family with short names ("Novel") keeps
    # the full size rather than every family paying for the worst case.
    #
    # x0.86 on the budget: two titles centred one pitch apart and each exactly one
    # pitch wide meet edge to edge. The budget is the pitch MINUS a gap, not the pitch.
    #
    # The pitch is MEASURED from the placed axes, not recomputed from `pos`: _fit_block
    # has already narrowed the block by the left overhang, so a budget derived from the
    # original footprint is too generous and the titles come out still touching.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    if ncol > 1:
        pitch = abs(axes[0][1].get_position().x0 - axes[0][0].get_position().x0)
    else:
        pitch = axes[0][0].get_position().width
    avail = pitch * 0.86
    worst = 0.0
    for row in axes:
        for ax in row:
            if ax.title.get_text():
                worst = max(worst, ax.title.get_window_extent(rend).transformed(
                    fig.transFigure.inverted()).width)
    if worst > avail:
        fs = max(7.0, FS_MINI_TITLE * avail / worst)
        for row in axes:
            for ax in row:
                ax.title.set_fontsize(fs)

    # Rows are condition groups, so the group break is between them.
    if nrow > 1:
        b0, b1 = axes[0][0].get_position(), axes[1][0].get_position()
        y = (b1.y1 + b0.y0) / 2
        fig.add_artist(Line2D([pos.x0, pos.x0 + pos.width], [y, y],
                              transform=fig.transFigure, zorder=5, **_DIV_LIGHT))

    # Short labels, not the quantile panels' wording: a quarter-size panel is not
    # wide enough for "PPC median + 95% CI" and the text runs into its neighbour.
    if not legend:
        return axes

    # ONE ROW, UNDER THE BLOCK. Inside a quarter-size panel the legend was wider than
    # its own axes and had to overhang a neighbour; below the grid it has the block's
    # full width and costs the bars no headroom. Placed against the bottom row's
    # measured extent so it clears the "Choice" label rather than a guessed offset.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bot = min(ax.get_tightbbox(rend).transformed(inv).y0
              for ax in axes[nrow - 1] if ax.get_visible())
    fig.legend(
        [Patch(facecolor=bar_color, edgecolor=ppc_color, lw=0.6),
         Line2D([], [], color=ppc_color, marker="o", ms=3.0, lw=1.1)],
        ["Observed", f"PPC {PPC_CI:g}% CI"],
        loc="upper center", bbox_to_anchor=(pos.x0 + pos.width / 2, bot - 0.012),
        ncol=2, fontsize=FS_MINI_LEG, frameon=False,
        handlelength=1.0, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0)
    return axes


def _axes_divider_h(fig, top_row, bottom_row, heavy=True):
    """Horizontal rule in the clear space BETWEEN two rows of axes.

    The counterpart of `_axes_divider` turned ninety degrees, and used for the same
    reason a solid vertical rule separates the quantile panels from the choice
    panel: crossing it, the y axis stops meaning seconds and starts meaning radians.
    Placed against the rows' measured extents, so it falls between the upper row's
    x label and the lower row's titles rather than through either.
    """
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    tb = [ax.get_tightbbox(rend).transformed(inv) for ax in top_row if ax.get_visible()]
    bb = [ax.get_tightbbox(rend).transformed(inv) for ax in bottom_row
          if ax.get_visible()]
    if not tb or not bb:
        return
    lo, hi = min(b.y0 for b in tb), max(b.y1 for b in bb)
    y = (lo + hi) / 2 if lo > hi else (
        min(ax.get_position().y0 for ax in top_row if ax.get_visible())
        + max(ax.get_position().y1 for ax in bottom_row if ax.get_visible())) / 2
    x0 = min(ax.get_position().x0 for ax in top_row if ax.get_visible())
    x1 = max(ax.get_position().x1 for ax in top_row if ax.get_visible())
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, zorder=5,
                          **dict(_DIV_HEAVY if heavy else _DIV_LIGHT)))


def _axes_divider(fig, ax_left, ax_right, heavy=False):
    """Vertical rule in the gutter between two axes, in figure coordinates.

    Drawn on the FIGURE rather than inside an axes so it can span the full height
    of a multi-row grid and sit in the whitespace, where an axvline cannot reach.

    Centred on the CLEAR space rather than on the gap between the two axes
    rectangles: the right panel's tick labels and y label live inside that gap, so
    the rectangles' midpoint lands on top of them. Every gutter is the same width
    now, so measuring the clear space is what keeps the rules consistently placed
    instead of letting one collide.
    """
    bl, br = ax_left.get_position(), ax_right.get_position()
    try:
        tb = ax_right.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted())
        x = (bl.x1 + tb.x0) / 2 if tb.x0 > bl.x1 else (bl.x1 + br.x0) / 2
    except Exception:
        x = (bl.x1 + br.x0) / 2
    y0, y1 = min(bl.y0, br.y0), max(bl.y1, br.y1)
    pad = 0.05 * (y1 - y0)
    style = dict(_DIV_HEAVY if heavy else _DIV_LIGHT)
    fig.add_artist(Line2D([x, x], [y0 - pad, y1 + pad], transform=fig.transFigure,
                          zorder=5, **style))


def _combined_legend(primary, extra_axes=()):
    """One legend, on the first subplot, covering every series in the FIGURE.

    "Only the first panel gets a legend" and "the choice panel uses marks the
    quantile panels never draw" pull against each other; taking the handles from
    the other axes too satisfies both, so nothing in the figure goes unlabelled.
    """
    H, L, seen = [], [], set()
    for ax in (primary,) + tuple(extra_axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen:
                seen.add(l); H.append(h); L.append(l)
    # Lower right: these panels rise left-to-right, so the upper-left corner the
    # default picks is exactly where the data is.
    primary.legend(H, L, fontsize=FS_LEG, frameon=False, loc="lower right")


def plot_combined(obs_pooled, ppc_pooled, obs_ch, ppc_ch, conditions, model_name,
                  variant, outdir, quantiles=DEFAULT_RT_QUANTILES, ppc_color=None):
    """The family's whole PPC as one figure. Written alongside the per-condition set.

    DDM / RDM -- one row: an RT-quantile panel per level of each condition group,
    then a single collapsed choice-proportion panel. Dashed divider between the
    condition groups, solid divider where quantiles give way to proportions.

    CDM -- two rows of one panel per level: RT quantiles on top, response-angle
    quantiles beneath, difficulty then speed across. CDM's response is a continuous
    angle, so its response check is a quantile plot like the RT one rather than a
    bar panel, and it shares the column layout instead of collapsing.

    No titles anywhere: the level name above each panel is the only label, and the
    rest belongs in the caption.
    """
    if ppc_color is None:
        ppc_color, ppc_fill = _ppc_colors(model_name)
    else:
        ppc_fill = ppc_color
    ordered = _ordered_conditions(conditions, model_name)
    if not ordered:
        return
    tag = _quantile_tag(quantiles)
    quantiles = np.asarray(quantiles, dtype=float)
    xs = np.arange(len(quantiles))
    outdir.mkdir(parents=True, exist_ok=True)

    # (cond_name, level) in reading order, with the index of each group boundary.
    cols, breaks = [], []
    for cond_name, _, label_map in ordered:
        levels_here = _levels_with_data(obs_pooled, ppc_pooled, cond_name, label_map)
        if not levels_here:
            continue
        if cols:
            breaks.append(len(cols))
        cols += [(cond_name, l) for l in levels_here]
    if not cols:
        logging.info(f"[combined] skipped for {model_name}: no condition has data")
        return

    two_row = model_name == "cdm"

    # EVERY gutter the same width, dividers included. An earlier version inserted a
    # blank spacer column at each break, which bought room for the rule but left the
    # panels either side of a divider further apart than the rest -- the grouping then
    # read as much from the spacing as from the rule. One uniform wspace instead,
    # widened enough to hold a divider clear of the next panel's labels.
    ratios = [1.0] * len(cols) + ([1.0] if not two_row else [])
    breaks_grid = list(breaks) + ([len(cols)] if not two_row else [])

    n_rows = 2 if two_row else 1
    n_grid = len(ratios)
    # Height from the CELL width, so panels keep their proportions however many
    # condition levels a family has, plus a fixed band per row for the title, the
    # tick labels and the axis label at the sizes set above.
    cell_w = FIG_W / (sum(ratios) + WSPACE * (n_grid - 1))
    # Cell as tall as it is wide, so a square plot area fills it instead of
    # leaving slack, plus a fixed band per row for title, ticks and axis label.
    band = ROW_BAND if n_rows > 1 else 0.55
    height = n_rows * (cell_w + band * FIG_SCALE)
    fig, axes = plt.subplots(n_rows, n_grid, squeeze=False,
                             figsize=(FIG_W, height),
                             gridspec_kw={"width_ratios": ratios})
    panel_cols = list(range(len(cols)))

    for j, (cond_name, lbl) in enumerate(cols):
        ax = axes[0, panel_cols[j]]
        _draw_quantile_panel(
            ax, obs_pooled[cond_name][lbl], ppc_pooled[cond_name].get(lbl, []),
            quantiles, xs, ppc_color, ppc_fill, "RT quantile",
            "RT (s)" if j == 0 else "")
        ax.set_title(lbl, fontsize=FS_TITLE)
        if two_row:
            ax2 = axes[1, panel_cols[j]]
            o = np.asarray(obs_ch.get(cond_name, {}).get(lbl, []), dtype=float)
            draws = [np.asarray(d, dtype=float)
                     for d in ppc_ch.get(cond_name, {}).get(lbl, [])]
            if len(o) and draws:
                # Shorter than the standalone figure's "Response deviation theta
                # (rad)": rotated, that label is taller than a panel at these font
                # sizes and runs into the row above. The x label already says these
                # are response quantiles.
                _draw_quantile_panel(ax2, o, draws, quantiles, xs, ppc_color, ppc_fill,
                                     "Choice quantile",
                                     r"$\theta$ (rad)" if j == 0 else "",
                                     zero_guide=True)
            else:
                ax2.set_visible(False)

    # ONE y SCALE PER ROW, ticks labelled on the first column only -- the same
    # treatment the choice bars already get. Per-panel autoscaling made each panel
    # fill its own box, so Novel running to 4.0 s and Familiar to 3.8 s drew two
    # curves of identical height and the difference between conditions, which is
    # what the figure is for, was scaled away.
    for r in range(n_rows):
        row_axes = [axes[r, panel_cols[j]] for j in range(len(cols))
                    if axes[r, panel_cols[j]].get_visible()]
        if not row_axes:
            continue
        lo = min(a.get_ylim()[0] for a in row_axes)
        hi = max(a.get_ylim()[1] for a in row_axes)
        for j, a in enumerate(row_axes):
            a.set_ylim(lo, hi)
            if j:
                a.tick_params(labelleft=False)

    _combined_legend(axes[0, panel_cols[0]])

    fig.tight_layout()
    # WSPACE LAST: tight_layout computes its own, so setting it up front has no
    # effect. This widens every gutter by the same amount, which is what keeps the
    # panels equally spaced while leaving the rules somewhere to go.
    fig.subplots_adjust(wspace=WSPACE, **({"hspace": HSPACE} if n_rows > 1 else {}))
    fig.canvas.draw()

    # The choice cell is a PLACEHOLDER: it reserves one quantile panel's footprint
    # through layout, then hands its final rectangle to the 2x2 and steps aside.
    grid = None
    if not two_row:
        holder = axes[0, len(cols)]
        # Square too, so the block occupies the same footprint the now-square
        # quantile panels do rather than the full (wider) grid cell.
        holder.set_box_aspect(1)
        fig.canvas.draw()
        pos = holder.get_position()
        holder.remove()
        grid = _draw_choice_grid(fig, pos, obs_ch, ppc_ch, ordered,
                                 ppc_color, ppc_fill)
        fig.canvas.draw()

    # Positions are only final after layout, so the dividers go in last. The last
    # break of a one-row figure is the quantiles/choice one -- a change of MEASURE,
    # drawn solid; every other separates condition groups and is drawn dashed.
    last_row = n_rows - 1
    for k, g in enumerate(breaks_grid):
        # Every within-family break is DASHED. Which axes the rule anchors to is a
        # separate question from how it is drawn: the last break of a one-row figure
        # is the RT/choice one, and the placeholder that reserved the choice column
        # has been removed by now, so it has to anchor to the block itself.
        is_measure_break = (not two_row) and k == len(breaks_grid) - 1
        if two_row:
            # ONE RULE PER ROW, not one spanning both. The rows are different
            # measures on their y axes, so a rule running through both would read
            # as a single shared frame that the condition break cuts across.
            for r in range(n_rows):
                _axes_divider(fig, axes[r, g - 1], axes[r, g], heavy=False)
        else:
            right = grid[0][0] if (is_measure_break and grid) else axes[last_row, g]
            _axes_divider(fig, axes[0, g - 1], right, heavy=False)

    if two_row:
        # Seconds above, radians below: a change of MEASURE, so it takes the solid
        # rule, exactly as the quantile/choice break does in the one-row figures.
        _axes_divider_h(fig, [axes[0, c] for c in panel_cols],
                        [axes[1, c] for c in panel_cols], heavy=False)

    savepath = outdir / f"{model_name}_{variant}_combined_ppc{tag}.pdf"
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {savepath}")


def _legend_below(fig, handles, labels, fontsize):
    """One figure legend in a single row, hung a fixed distance below the lowest x
    label -- MEASURED, rather than at a fixed figure fraction: a fraction is a
    different distance at every figure height, and at one of them it put the entries
    on the bottom row's x labels. Call after the layout is final."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ink_bottom = min(ax.get_tightbbox(rend).y0 for ax in fig.axes if ax.get_visible())
    gap_px = LEGEND_GAP_PT / 72 * fig.dpi
    y = fig.transFigure.inverted().transform((0, ink_bottom - gap_px))[1]
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=len(handles), frameon=False, fontsize=fontsize)


def plot_combined_all(per_family, outpath, quantiles=DEFAULT_RT_QUANTILES):
    """All three families' PPCs in ONE figure, five slots per row.

    The three per-family figures happen to share a column count, which is what makes
    this tile without stretching anything:

        DDM  | 4 RT-quantile panels          | 2x2 choice block
        RDM  | 4 RT-quantile panels          | 2x2 choice block
        CDM  | 5 RT-quantile panels          |
        CDM  | 5 response-angle panels       |

    CDM costs two rows because its response is a continuous angle, so its response
    check is a quantile plot rather than the bar block -- there is no collapsing it
    into one slot the way a discrete choice collapses.

    Dashed rules separate anything WITHIN a family: the condition groups, and the
    break between the RT panels and the response check. Solid rules separate the
    FAMILIES, which is the only boundary in the figure across which nothing is
    comparable -- different data, different subjects, different response space.

    `per_family` maps "ddm"/"rdm"/"cdm" to the five things collect_ppc returns.
    """
    fams = [f for f in ("ddm", "rdm", "cdm") if f in per_family]
    if not fams:
        return
    quantiles = np.asarray(quantiles, dtype=float)
    xs = np.arange(len(quantiles))

    # Plan the rows first: each is (family, kind, [(cond, level), ...], break indices).
    plan = []
    for fam in fams:
        obs_pooled, ppc_pooled, conditions, obs_ch, ppc_ch = per_family[fam]
        ordered = _ordered_conditions(conditions, fam)
        cols, breaks = [], []
        for cond_name, _, label_map in ordered:
            here = _levels_with_data(obs_pooled, ppc_pooled, cond_name, label_map)
            if not here:
                continue
            if cols:
                breaks.append(len(cols))
            cols += [(cond_name, l) for l in here]
        if not cols:
            continue
        plan.append((fam, "rt", cols, breaks))
        if fam == "cdm":
            plan.append((fam, "theta", cols, breaks))

    n_rows = len(plan)
    n_cols = max(len(c) for _, _, c, _ in plan)
    has_choice = any(f != "cdm" for f, _, _, _ in plan)
    # NO extra column for the choice block. It takes the slots a family does not
    # spend on quantile panels -- DDM and RDM have four conditions against CDM's
    # five, so the block occupies what would have been their fifth. An extra column
    # instead left the CDM rows ending short of the others by that column's width.
    n_grid = n_cols

    fig, axes = plt.subplots(n_rows, n_grid, squeeze=False,
                             figsize=(FIG_W, FIG_W * 0.22 * n_rows))
    holders, grids = {}, {}

    for r, (fam, kind, cols, breaks) in enumerate(plan):
        obs_pooled, ppc_pooled, conditions, obs_ch, ppc_ch = per_family[fam]
        ppc_color, ppc_fill = _ppc_colors(fam)
        src_obs, src_ppc = (obs_pooled, ppc_pooled) if kind == "rt" else (obs_ch, ppc_ch)
        xlabel = "RT quantile" if kind == "rt" else "Choice quantile"
        ylabel = "RT (s)" if kind == "rt" else r"$\theta$ (rad)"

        for j, (cond_name, lbl) in enumerate(cols):
            ax = axes[r, j]
            o = np.asarray(src_obs.get(cond_name, {}).get(lbl, []), dtype=float)
            draws = [np.asarray(d, dtype=float)
                     for d in src_ppc.get(cond_name, {}).get(lbl, [])]
            if not len(o) or not draws:
                ax.set_visible(False)
                continue
            _draw_quantile_panel(ax, o, draws, quantiles, xs, ppc_color, ppc_fill,
                                 xlabel, ylabel if j == 0 else "",
                                 zero_guide=(kind == "theta"))
            # The level name only on the FIRST row that shows it: rows 3 and 4 are
            # the same five CDM conditions twice, and repeating the names implies
            # they are different ones.
            if kind == "rt":
                ax.set_title(lbl, fontsize=FS_TITLE)
        for j in range(len(cols), n_cols):
            axes[r, j].set_visible(False)

        if has_choice:
            # The choice block takes every slot this row does not spend on quantile
            # panels, not just the last one. DDM and RDM have four conditions against
            # CDM's five, so pinning the block to the final column would leave a hole
            # in the middle of their rows.
            spare = list(range(len(cols), n_grid))
            for j in spare:
                axes[r, j].set_visible(False)
            if fam != "cdm" and spare:
                holders[r] = spare

    # One shared y scale per ROW, ticks on the first column only -- the same rule the
    # single-family figures use, applied row by row since each row is its own measure.
    for r in range(n_rows):
        live = [axes[r, j] for j in range(n_cols) if axes[r, j].get_visible()]
        if not live:
            continue
        lo = min(a.get_ylim()[0] for a in live)
        hi = max(a.get_ylim()[1] for a in live)
        for j, a in enumerate(live):
            a.set_ylim(lo, hi)
            if j:
                a.tick_params(labelleft=False)

    fig.tight_layout()
    # More row spacing than the single-family figures use. Four rows of five panels
    # leave each row's x labels close to the next row's titles, and the family rule
    # has to sit BETWEEN them -- with the default gap it lands on the labels.
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE * 2.0)
    fig.canvas.draw()

    for r, spare in holders.items():
        fam = plan[r][0]
        obs_pooled, ppc_pooled, conditions, obs_ch, ppc_ch = per_family[fam]
        first, last = axes[r, spare[0]], axes[r, spare[-1]]
        p_first, p_last = first.get_position(), last.get_position()
        # Vertical bounds from a DRAWN quantile panel, not from the grid cell. The
        # panels are square by box_aspect and sit shorter than their cell, so a block
        # sized to the cell starts above them and ends below -- visibly out of line
        # with its own row.
        ref = next((axes[r, j].get_position() for j in range(len(cols))
                    if axes[r, j].get_visible()), p_first)
        # Full spare width. Capping it to a square shrank the block and left a hole
        # where the spare column was; the mini panels are squared individually
        # instead, so the extra width becomes spacing between them rather than
        # stretching each one into a squat rectangle.
        pos = Bbox.from_extents(p_first.x0, ref.y0, p_last.x1, ref.y1)
        for j in spare:
            # Hidden BEFORE removal: a removed axes still reports get_visible() True,
            # and the divider loops below filter on that. Touching a removed axes'
            # bbox makes matplotlib re-apply its aspect against a figure it lost.
            axes[r, j].set_visible(False)
            axes[r, j].remove()
        # No legend inside the block here: under a 2x2 it lands on the next
        # family's panel titles, and one legend already covers the whole figure.
        grids[r] = _draw_choice_grid(fig, pos, obs_ch, ppc_ch,
                                     _ordered_conditions(conditions, fam),
                                     *_ppc_colors(fam), legend=False,
                                     square_panels=True)
    fig.canvas.draw()

    # ONE legend for the figure, BELOW it, covering the bar marks too -- the choice
    # blocks carry none of their own. Not inside a panel: at this size every corner
    # of every panel has data in it, and the first panel's y tick labels were being
    # covered as well.
    # GREY, not any family's colour. The legend describes marks that appear in all
    # three families, each drawn in its own hue; borrowing DDM's artists made it read
    # as a key to the DDM row alone. Built from scratch in neutral grey so it says
    # what the SHAPES mean and leaves colour to mean family.
    grey, dark = "#8a8a94", OBS_COLOR
    H = [Patch(facecolor=grey, alpha=0.40, lw=0),
         Line2D([], [], color=grey, marker="o", ms=4, lw=1.4),
         Line2D([], [], color=dark, marker="s", ms=4, lw=1.6, ls="--"),
         Line2D([], [], color=grey, marker="o", ms=4, lw=1.2)]
    L = [f"PPC {PPC_CI:g}% CI", "PPC median", "Observed",
         f"PPC median + {PPC_CI:g}% CI"]
    if not any(grids.values()):
        H, L = H[:3], L[:3]
    _legend_below(fig, H, L, fontsize=FS_LEG)

    # Dashed: condition-group breaks, and the RT/choice break on the DDM/RDM rows.
    for r, (fam, kind, cols, breaks) in enumerate(plan):
        for b in breaks:
            _axes_divider(fig, axes[r, b - 1], axes[r, b], heavy=False)
        if r in grids and grids[r]:
            _axes_divider(fig, axes[r, len(cols) - 1], grids[r][0][0], heavy=False)

    # Dashed between CDM's two rows; solid between families.
    def row_axes(r):
        """Every axes on a row, INCLUDING the hand-placed choice block -- those are
        not in `axes`, so a divider spanning only `axes` stops short of them."""
        out = [axes[r, j] for j in range(n_grid) if axes[r, j].get_visible()]
        if grids.get(r):
            out += [ax for row in grids[r] for ax in row if ax.get_visible()]
        return out

    for r in range(n_rows - 1):
        same_family = plan[r][0] == plan[r + 1][0]
        top, bot = row_axes(r), row_axes(r + 1)
        if top and bot:
            _axes_divider_h(fig, top, bot, heavy=not same_family)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {outpath}")


# RT-quantile row (plot_rt_quantiles_row). Drawn at FIG_W like the combined figures,
# so \textwidth halves these: 8 / 7 / 5.5 / 6 pt on the page.
ROW_FS_TITLE, ROW_FS_LABEL, ROW_FS_TICK, ROW_FS_LEG = 16, 14, 11, 12
# The band takes the LINE's colour at low opacity rather than the family's light
# stop, matching experiment 3's recovery overlay, which draws fine-tuned LBA in the
# saturated stop alone (its error bars are that colour, translucent).
ROW_FILL_ALPHA = 0.22


def plot_rt_quantiles_row(obs_pooled, ppc_pooled, conditions, model_name, variant, outdir,
                          quantiles=DEFAULT_RT_QUANTILES):
    """Every level of every grouping variable as ONE row of RT-quantile panels.

    The per-variable figures from plot_rt_quantiles, side by side and without their
    suptitles: the level names title the panels, and a dashed rule separates one
    grouping variable from the next, as in the experiment 2 combined figure. One y
    scale across the row, ticks on the first panel only -- every panel is RT in
    seconds, and per-panel scales would hide which levels are slower.
    """
    ppc_color = _ppc_colors(model_name)[0]
    tag = _quantile_tag(quantiles)
    quantiles = np.asarray(quantiles, dtype=float)
    xs = np.arange(len(quantiles))

    groups = [(c, lv) for c, _, label_map in conditions
              if (lv := _levels_with_data(obs_pooled, ppc_pooled, c, label_map))]
    cols = [(c, lbl) for c, lv in groups for lbl in lv]
    if not cols:
        logging.warning(f"{model_name}: no condition level has both observed and PPC RTs; "
                        "skipping the RT-quantile row")
        return
    fig, axes = plt.subplots(1, len(cols), squeeze=False, figsize=(FIG_W, FIG_W * 0.22))
    for j, (c, lbl) in enumerate(cols):
        ax = axes[0, j]
        _draw_quantile_panel(
            ax, np.asarray(obs_pooled[c][lbl], dtype=float),
            [np.asarray(d, dtype=float) for d in ppc_pooled[c][lbl]],
            quantiles, xs, ppc_color, ppc_color, "RT quantile", "RT (s)" if j == 0 else "",
            label_fs=ROW_FS_LABEL, tick_fs=ROW_FS_TICK, fill_alpha=ROW_FILL_ALPHA)
        ax.set_title(lbl, fontsize=ROW_FS_TITLE)
        # Every quantile keeps its tick, every other one its label: seven panels
        # to a row leave no room for five labels at a legible size.
        ax.set_xticklabels([f"{q:g}" if i % 2 == 0 else "" for i, q in enumerate(quantiles)])

    lo = min(a.get_ylim()[0] for a in axes[0])
    hi = max(a.get_ylim()[1] for a in axes[0])
    for j, a in enumerate(axes[0]):
        a.set_ylim(lo, hi)
        if j:
            a.tick_params(labelleft=False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=WSPACE)
    fig.canvas.draw()
    b = 0
    for _, lv in groups[:-1]:
        b += len(lv)
        _axes_divider(fig, axes[0, b - 1], axes[0, b], heavy=False)
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels(), fontsize=ROW_FS_LEG)

    outdir.mkdir(parents=True, exist_ok=True)
    savepath = outdir / f"{model_name}_{variant}_rt_quantiles_row{tag}.pdf"
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[saved] {savepath}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   choices=["ddm", "rdm", "cdm", "lba"])
    p.add_argument("--variant", type=str, default="mf", choices=["mf", "mc"])
    p.add_argument("--design", type=str, default="full", choices=sorted(LBA_DESIGNS),
                   help="LBA only. MUST match the --design the posteriors were fitted "
                        "with: they record it as design_name, which this checks.")
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
            design=args.design,
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
            design=args.design,
        )
    plot_ppc(obs_pooled, ppc_pooled, conditions, args.model, args.variant, outdir,
             draw_lines=not args.no_ppc_lines, max_lines=args.ppc_max_lines,
             line_alpha=args.ppc_line_alpha, line_width=args.ppc_line_width)
    plot_rt_quantiles(obs_pooled, ppc_pooled, conditions, args.model, args.variant,
                      outdir, quantiles=args.rt_quantiles)
    if args.model == "lba":
        # Experiment 3's figure. The other families' RT quantiles are already one
        # figure each (plot_combined, plot_combined_all).
        plot_rt_quantiles_row(obs_pooled, ppc_pooled, conditions, args.model,
                              args.variant, outdir, quantiles=args.rt_quantiles)
    plot_rt_qq(obs_pooled, ppc_pooled, conditions, args.model, args.variant,
               outdir, quantiles=args.rt_quantiles)
    plot_choice_quantiles(obs_ch, ppc_ch, conditions, args.model, args.variant,
                          outdir, quantiles=args.rt_quantiles)
    plot_choice_ppc(obs_ch, ppc_ch, conditions, args.model, args.variant, outdir,
                    draw_lines=not args.no_ppc_lines, max_lines=args.ppc_max_lines,
                    line_alpha=args.ppc_line_alpha, line_width=args.ppc_line_width)
    plot_combined(obs_pooled, ppc_pooled, obs_ch, ppc_ch, conditions, args.model,
                  args.variant, outdir, quantiles=args.rt_quantiles)
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
