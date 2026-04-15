import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import keras
import logging
import numpy as np
import pandas as pd
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path

from cogformer.simulator.model_family import ModelFamily
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, cdm_priors, rdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, cdm_link_fun, rdm_link_fun
from cogformer.diagnostics.plot.adaptive_recovery import adaptive_recovery
from cogformer.diagnostics.plot.adaptive_coverage import adaptive_coverage
from cogformer.diagnostics.plot.adaptive_ecdf import adaptive_ecdf
from cogformer.diagnostics.plot.adaptive_posterior import adaptive_posterior
from cogformer.diagnostics.plot.adaptive_metrics import adaptive_metrics as plot_adaptive_metrics
from cogformer.diagnostics.metric.adaptive_metrics import adaptive_metrics as compute_adaptive_metrics
from cogformer.utils.plot_utils import bf_colors


# Per-model CASE_CONFIGS with param_names for diagnostics ─────────────────────

DDM_CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": False, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "expected_num_active": None,
    },
    "fixed": {
        "free_intrinsics": ["v", "a", "z", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$"],
        "expected_num_active": 4,
    },
    "regressed": {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": ["v", "a", "z"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$",
            r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$",
        ],
        "expected_num_active": 12,
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "a", "z", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v", "a", "z"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$a$", r"$z$", r"$\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$",
            r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$",
        ],
        "expected_num_active": 10,
    },
    "interaction": {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": False,
        "design_config": {
            "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "u_1": ["v", "a", "z", "tau", "s_v"],
            "u_2": ["v", "a", "z", "tau"],
            "u_1:u_2": ["v", "a", "z"],
        },
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$",
            r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$", r"$u_{2, \tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, z}$",
        ],
        "expected_num_active": 18,
    },
    "full": {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True,
        "design_config": {
            "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
            "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
        },
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$", r"$u_{1, z}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{1, s_\tau}$",
            r"$u_{2, v}$", r"$u_{2, a}$", r"$u_{2, z}$", r"$u_{2, \tau}$", r"$u_{2, s_v}$", r"$u_{2, s_\tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, z}$", r"$u_1:u_{2, \tau}$", r"$u_1:u_{2, s_v}$", r"$u_1:u_{2, s_\tau}$",
        ],
        "expected_num_active": 24,
    },
}

CDM_CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": False, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "expected_num_active": None,
    },
    "fixed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$"],
        "expected_num_active": 4,
    },
    "regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$",
            r"$u_{2, v}$", r"$u_{2, a}$",
        ],
        "expected_num_active": 10,
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$",
            r"$u_{1, v}$", r"$u_{1, a}$",
            r"$u_{2, v}$", r"$u_{2, a}$",
        ],
        "expected_num_active": 8,
    },
    "interaction": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": False,
        "design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v"],
            "u_2": ["v", "v_theta", "a", "tau"],
            "u_1:u_2": ["v", "v_theta", "a"],
        },
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, v_\theta}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$",
            r"$u_{2, v}$", r"$u_{2, v_\theta}$", r"$u_{2, a}$", r"$u_{2, \tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_\theta}$", r"$u_1:u_{2, a}$",
        ],
        "expected_num_active": 18,
    },
    "full": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True,
        "design_config": {
            "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
            "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        },
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, v_\theta}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{1, s_\tau}$",
            r"$u_{2, v}$", r"$u_{2, v_\theta}$", r"$u_{2, a}$", r"$u_{2, \tau}$", r"$u_{2, s_v}$", r"$u_{2, s_\tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_\theta}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, \tau}$", r"$u_1:u_{2, s_v}$", r"$u_1:u_{2, s_\tau}$",
        ],
        "expected_num_active": 24,
    },
}

# RDM intercept_only uses fixed_config=True with an explicit design_config (single-row)
RDM_CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True,
        "design_config": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"]},
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "expected_num_active": 6,
    },
    "fixed": {
        "free_intrinsics": ["v", "v_diff", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True,
        "design_config": {"1": ["v", "v_diff", "a", "tau"]},
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "param_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$"],
        "expected_num_active": 4,
    },
    "regressed": {
        "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": ["v_diff", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True,
        "design_config": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$",
            r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$",
        ],
        "expected_num_active": 10,
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "v_diff", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v_diff", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True,
        "design_config": {"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$",
            r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$",
            r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$",
        ],
        "expected_num_active": 8,
    },
    "interaction": {
        "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": False,
        "design_config": {
            "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_diff", "a", "tau", "s_v"],
            "u_2": ["v", "v_diff", "a", "tau"],
            "u_1:u_2": ["v", "v_diff", "a"],
        },
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$",
            r"$u_{2, v}$", r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$", r"$u_{2, \tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_{\mathrm{diff}}}$", r"$u_1:u_{2, a}$",
        ],
        "expected_num_active": 18,
    },
    "full": {
        "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True,
        "design_config": {
            "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
            "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        },
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "param_names": [
            r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$",
            r"$u_{1, v}$", r"$u_{1, v_{\mathrm{diff}}}$", r"$u_{1, a}$", r"$u_{1, \tau}$", r"$u_{1, s_v}$", r"$u_{1, s_\tau}$",
            r"$u_{2, v}$", r"$u_{2, v_{\mathrm{diff}}}$", r"$u_{2, a}$", r"$u_{2, \tau}$", r"$u_{2, s_v}$", r"$u_{2, s_\tau}$",
            r"$u_1:u_{2, v}$", r"$u_1:u_{2, v_{\mathrm{diff}}}$", r"$u_1:u_{2, a}$", r"$u_1:u_{2, \tau}$", r"$u_1:u_{2, s_v}$", r"$u_1:u_{2, s_\tau}$",
        ],
        "expected_num_active": 24,
    },
}

MODELS = {
    "ddm": {
        "model_cls": DDM, "prior_fn": ddm_priors, "link_fn": ddm_link_fun,
        "name": "DDM", "case_configs": DDM_CASE_CONFIGS,
        "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "ckpt_prefix": "ddm_families_bf",
        "figures_dir": "./outputs/figures/model_family/bf/ddm",
        "data_dir": "./outputs/data/model_family",
        "evals_dir": "./outputs/evaluations/model_family/bf",
        "benchmark_configs": {
            "intercept_only": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "fixed": {"1": ["v", "a", "z", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "a", "z", "tau"], "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": []},
            "interaction": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v"], "u_2": ["v", "a", "z", "tau"], "u_1:u_2": ["v", "a", "z"]},
            "full": {"1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"]},
        },
    },
    "cdm": {
        "model_cls": CDM, "prior_fn": cdm_priors, "link_fn": cdm_link_fun,
        "name": "CDM", "case_configs": CDM_CASE_CONFIGS,
        "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "ckpt_prefix": "cdm_families_bf",
        "figures_dir": "./outputs/figures/model_family/bf/cdm",
        "data_dir": "./outputs/data/model_family",
        "evals_dir": "./outputs/evaluations/model_family/bf",
        "benchmark_configs": {
            "intercept_only": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "fixed": {"1": ["v", "v_theta", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "v_theta", "a", "tau"], "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": []},
            "interaction": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v"], "u_2": ["v", "v_theta", "a", "tau"], "u_1:u_2": ["v", "v_theta", "a"]},
            "full": {"1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"]},
        },
    },
    "rdm": {
        "model_cls": RDM, "prior_fn": rdm_priors, "link_fn": rdm_link_fun,
        "name": "RDM", "case_configs": RDM_CASE_CONFIGS,
        "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
        "ckpt_prefix": "rdm_families_bf",
        "figures_dir": "./outputs/figures/model_family/bf/rdm",
        "data_dir": "./outputs/data/model_family",
        "evals_dir": "./outputs/evaluations/model_family/bf",
        "benchmark_configs": {
            "intercept_only": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "regressed": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "fixed": {"1": ["v", "v_diff", "a", "tau"], "u_1": [], "u_2": [], "u_1:u_2": []},
            "fixed_regressed": {"1": ["v", "v_diff", "a", "tau"], "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": []},
            "interaction": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v"], "u_2": ["v", "v_diff", "a", "tau"], "u_1:u_2": ["v", "v_diff", "a"]},
            "full": {"1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"], "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"]},
        },
    },
}


def reshape_bf_to_gpt(bf_samples, design_config, intrinsic_params):
    *leading, num_active = bf_samples.shape
    num_rows = len(design_config)
    num_cols = len(intrinsic_params)
    col_idx = {p: j for j, p in enumerate(intrinsic_params)}
    result = np.zeros((*leading, num_rows, num_cols))
    flat_pos = 0
    for row_i, active_params in enumerate(design_config.values()):
        ordered = [p for p in intrinsic_params if p in active_params]
        for p in ordered:
            result[..., row_i, col_idx[p]] = bf_samples[..., flat_pos]
            flat_pos += 1
    assert flat_pos == num_active, f"Expected {num_active} active params but mapped {flat_pos}."
    return result


class ModelFamilyBF(bf.simulators.Simulator):
    def __init__(self, model_key: str, case: str):
        model_cfg = MODELS[model_key]
        case_configs = model_cfg["case_configs"]
        if case not in case_configs:
            raise ValueError(f"Unknown case: {case}. Must be one of {list(case_configs.keys())}")
        self.case = case
        self.config = case_configs[case]
        self.link_fn = model_cfg["link_fn"]
        self.model_family = ModelFamily(
            model=model_cfg["model_cls"](),
            prior_fun=model_cfg["prior_fn"](),
            regressed_params=self.config["regressed_params"],
            mask_randomizer_kwargs=dict(
                free_intrinsics=self.config["free_intrinsics"],
                fixed_intrinsics=self.config["fixed_intrinsics"],
                fixed_values=self.config["fixed_values"],
            ),
        )

    def sample(self, batch_size, num_obs=500, **kwargs):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]
        sample_kwargs = {
            "min_num_regressors": self.config["min_num_regressors"],
            "max_num_regressors": self.config["max_num_regressors"],
            "max_num_categories": self.config["max_num_categories"],
        }
        if self.config["fixed_config"]:
            sample_kwargs["fixed_config"] = True
        if self.config["design_config"] is not None:
            sample_kwargs["design_config"] = self.config["design_config"]
        samples = self.model_family.batch_sample(
            batch_size=batch_size,
            num_obs=num_obs,
            flatten_param_outputs=self.config["flatten_param_outputs"],
            link_fun=self.link_fn(),
            **sample_kwargs,
            **kwargs,
        )
        param_matrices = samples["param_matrices"]
        param_masks = samples["param_masks"]
        if self.config["squeeze_outputs"]:
            param_matrices = param_matrices.squeeze(axis=1)
            param_masks = param_masks.squeeze(axis=1)
        return {
            "design_matrices": samples["design_matrices"],
            "rts": samples["sim_data"]["rts"],
            "choices": samples["sim_data"]["choices"],
            "params": param_matrices,
            "masks": param_masks,
        }


def test(model_key: str, case: str):
    model_cfg = MODELS[model_key]
    config = model_cfg["case_configs"][case]
    simulator = ModelFamilyBF(model_key=model_key, case=case)
    samples = simulator.sample(4)
    masks = samples["masks"]
    params = samples["params"]

    num_active = int(masks[0].astype(bool).sum())
    expected = config["expected_num_active"]
    if expected is not None:
        status = "OK" if num_active == expected else f"FAIL (expected {expected})"
        print(f"  [Check] Active params: {num_active}  →  {status}")
        assert num_active == expected
    else:
        print(f"  [Check] Active params: {num_active}  →  (variable, no assertion)")

    if config["fixed_config"] or config["design_config"] is not None:
        consistent = np.all(masks == masks[0])
        print(f"  [Check] Mask consistent across batch: {'OK' if consistent else 'FAIL'}")
        assert consistent

    zero_where_masked = np.allclose(params * (1 - masks), 0.0)
    print(f"  [Check] Params zero where masked: {'OK' if zero_where_masked else 'FAIL'}")
    assert zero_where_masked

    checkpoint_path = f"./outputs/checkpoints/model_family/bf/{model_key}/{MODELS[model_key]['ckpt_prefix']}_{case}/model.keras"
    approximator = keras.saving.load_model(checkpoint_path)
    val_sims = simulator.sample(100)
    post_draws = approximator.sample(
        conditions={"rts": val_sims["rts"], "choices": val_sims["choices"], "design_matrices": val_sims["design_matrices"]},
        num_samples=100,
    )
    print(f"Targets: {val_sims['params'].shape}, Estimates: {post_draws['params'].shape}")


def main(model_key: str, case: str, batch_size: int = 200, num_samples: int = 200,
         skip_posteriors: bool = False, skip_log_gamma: bool = True):
    model_cfg = MODELS[model_key]
    model_name = model_key
    case_config = model_cfg["case_configs"][case]
    param_names = case_config["param_names"]
    intrinsic_params = model_cfg["intrinsic_params"]
    variable_names = model_cfg["variable_names"]

    simulator = ModelFamilyBF(model_key=model_key, case=case)
    ckpt_path = f"./outputs/checkpoints/model_family/bf/{model_key}/{model_cfg['ckpt_prefix']}_{case}/model.keras"
    approximator = keras.saving.load_model(ckpt_path)

    data_dir = Path(model_cfg["data_dir"])
    figures_dir = Path(model_cfg["figures_dir"]) / case
    evals_dir = Path(model_cfg["evals_dir"])
    for d in (data_dir, figures_dir, evals_dir):
        d.mkdir(parents=True, exist_ok=True)

    val_sims = simulator.sample(batch_size)
    post_draws = approximator.sample(
        conditions={"rts": val_sims["rts"], "choices": val_sims["choices"], "design_matrices": val_sims["design_matrices"]},
        num_samples=num_samples,
    )
    targets = val_sims["params"]
    estimates = post_draws["params"]

    masks = val_sims["masks"]
    active_idx = masks[0].astype(bool)
    true_params = targets[:, active_idx]
    pred_params = estimates[:, :, active_idx]
    active_param_names = [name for name, active in zip(param_names, active_idx) if active] \
        if len(param_names) == len(active_idx) else param_names

    np.savez(
        data_dir / f"{model_name}_{case}_data.npz",
        rts=val_sims["rts"],
        choices=val_sims["choices"],
        true_set=val_sims["params"],
        pred_set=post_draws["params"],
        design_matrices=val_sims["design_matrices"],
        param_masks=val_sims["masks"],
    )
    logging.info(f"Saved data to {data_dir}")

    rmse = bf.diagnostics.metrics.root_mean_squared_error(
        estimates=pred_params, targets=true_params, variable_names=active_param_names
    )
    calibration_errors = bf.diagnostics.metrics.calibration_error(
        estimates=pred_params, targets=true_params, variable_names=active_param_names
    )
    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=pred_params, targets=true_params, variable_names=active_param_names
    )
    metrics_dict = {
        rmse["metric_name"]: rmse["values"],
        calibration_errors["metric_name"]: calibration_errors["values"],
        contraction["metric_name"]: contraction["values"],
    }
    if not skip_log_gamma:
        log_gamma = bf.diagnostics.metrics.calibration_log_gamma(
            estimates=pred_params, targets=true_params, variable_names=active_param_names
        )
        metrics_dict[log_gamma["metric_name"]] = log_gamma["values"]
    metrics = pd.DataFrame(metrics_dict)
    metrics.to_csv(evals_dir / f"{model_name}_families_bf_{case}_evaluations.csv", sep=";")
    logging.info("Metric evaluation is now finished.")

    design_config = model_cfg["benchmark_configs"][case]
    adaptive_colors = bf_colors()

    true_grid = reshape_bf_to_gpt(true_params, design_config, intrinsic_params)
    pred_grid = reshape_bf_to_gpt(pred_params, design_config, intrinsic_params)
    params_mask = reshape_bf_to_gpt(np.ones((1, true_params.shape[-1])), design_config, intrinsic_params)[0]

    common = dict(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=2,
        parameter_mask=params_mask,
        variable_names=variable_names,
        intercept_color=adaptive_colors["intercept"],
        main_effect_color=adaptive_colors["main_effect"],
        interaction_color=adaptive_colors["interaction"],
    )

    recovery_fig = adaptive_recovery(true=true_grid, pred=pred_grid, **common)
    recovery_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_recovery.pdf", bbox_inches="tight")
    plt.close(recovery_fig)

    coverage_fig = adaptive_coverage(true=true_grid, pred=pred_grid, **common)
    coverage_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_coverage.pdf", bbox_inches="tight")
    plt.close(coverage_fig)

    ecdf_fig = adaptive_ecdf(true=true_grid, pred=pred_grid, difference=True, **common)
    ecdf_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_ecdf.pdf", bbox_inches="tight")
    plt.close(ecdf_fig)

    metrics_fig = plot_adaptive_metrics(true=true_grid, pred=pred_grid, skip_log_gamma=skip_log_gamma, **common)
    metrics_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_metrics.pdf", bbox_inches="tight")
    plt.close(metrics_fig)

    metrics_df = compute_adaptive_metrics(true=true_grid, pred=pred_grid, skip_log_gamma=skip_log_gamma, **common)
    metrics_df.to_csv(figures_dir / f"{model_name}_family_{case}_bf_metrics.csv")
    logging.info(f"Saved diagnostics to {figures_dir}")

    if not skip_posteriors:
        for i in range(10):
            posterior_fig = adaptive_posterior(
                samples=pred_grid[i],
                design_config=design_config,
                intrinsic_params=intrinsic_params,
                max_num_categories=2,
                unfold=False,
                intercept_color=adaptive_colors["intercept"],
                main_effect_color=adaptive_colors["main_effect"],
                interaction_color=adaptive_colors["interaction"],
            )
            posterior_fig.savefig(figures_dir / f"{model_name}_family_{case}_bf_posterior_{i}.pdf", bbox_inches="tight")
            plt.close(posterior_fig.fig)


if __name__ == "__main__":
    all_cases = list(DDM_CASE_CONFIGS.keys())

    parser = argparse.ArgumentParser(description="Model Family BayesFlow Validation")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--case", type=str, default="intercept_only", choices=all_cases)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_posteriors", action="store_true")
    parser.add_argument("--skip_log_gamma", action="store_true", default=True)
    parser.add_argument("--include_log_gamma", dest="skip_log_gamma", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        test(model_key=args.model, case=args.case)
    else:
        main(
            model_key=args.model,
            case=args.case,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            skip_posteriors=args.skip_posteriors,
            skip_log_gamma=args.skip_log_gamma,
        )
