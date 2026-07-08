"""
Shared model definitions for all CogFormer experiments.

Each model config contains:
  model_cls, prior_fn, link_fn, name,
  intrinsic_params, variable_names,
  free_params, fixed_params, global_params, default_fixed_values,
  benchmark_configs, val_scenarios, [train_context (RDM only)]

benchmark_configs and val_scenarios are defined *without* global params (e.g. pi).
Scripts that need global params in their configs should call
utils.add_global_params() after importing.
"""

from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.lba import LBA
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, cdm_priors, lba_priors, rdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, cdm_link_fun, lba_link_fun, rdm_link_fun


# ── Global / nuisance parameters (used only in CF model-family scripts) ──────

GLOBAL_PARAMS = ["pi"]
GLOBAL_VARIABLE_NAMES = [r"$\pi$"]


# ── DDM ───────────────────────────────────────────────────────────────────────

DDM_BENCHMARK_CONFIGS = {
    "intercept_only": {
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "regressed": {
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": [],
    },
    "fixed": {
        "1": ["v", "a", "z", "tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "fixed_regressed": {
        "1": ["v", "a", "z", "tau"],
        "u_1": ["v", "a", "z"], "u_2": ["v", "a", "z"], "u_1:u_2": [],
    },
    "interaction": {
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a", "z", "tau", "s_v"],
        "u_2": ["v", "a", "z", "tau"],
        "u_1:u_2": ["v", "a", "z"],
    },
    "three_reg_interaction": {
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_2": ["v", "a", "z", "tau", "s_v"],
        "u_3": ["v", "a", "z", "tau"],
        "u_1:u_2": ["v", "a", "z"],
        "u_1:u_3": ["v", "a"],
        "u_2:u_3": ["v"],
    },
    "full": {
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
    },
    # ── Holdout configs (excluded from model_family training) ─────────────────
    "two_factor": {  # same structure as "full"; held out with max_num_categories=3
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
    },
    "asymmetric": {  # drift cluster vs. threshold cluster, no interaction
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "z", "s_v"],
        "u_2": ["a", "tau", "s_tau"],
    },
    "multi_factor": {  # 5 regressors, hierarchical parameter shedding, no interaction
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_2": ["v", "a", "z", "tau", "s_v"],
        "u_3": ["v", "a", "z", "tau"],
        "u_4": ["v", "a", "z"],
        "u_5": ["v", "a"],
    },
}

DDM_CONFIG = {
    "model_cls": DDM,
    "prior_fn": ddm_priors,
    "link_fn": ddm_link_fun,
    "name": "DDM",
    "intrinsic_params": ["v", "a", "z", "tau", "s_v", "s_tau"],
    "variable_names": [r"$v$", r"$a$", r"$z$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
    "free_params": ["v", "a", "z", "tau"],
    "fixed_params": ["s_v", "s_tau"],
    "global_params": [],
    "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
    "benchmark_configs": DDM_BENCHMARK_CONFIGS,
    "val_scenarios": [
        ("interaction", DDM_BENCHMARK_CONFIGS["interaction"], {"max_num_categories": 2}),
        # No skip_global: training designs always include global params, so a mask
        # without pi is out-of-distribution and biases a/v recovery downward.
        ("fixed", DDM_BENCHMARK_CONFIGS["fixed"], {"max_num_categories": 2}),
        ("two_factor", DDM_BENCHMARK_CONFIGS["two_factor"], {"max_num_categories": 3}),
    ],
    "holdout_configs": [
        {"design_config": DDM_BENCHMARK_CONFIGS["three_reg_interaction"], "max_num_categories": 3},
        {"design_config": DDM_BENCHMARK_CONFIGS["two_factor"], "max_num_categories": 3},
        {"design_config": DDM_BENCHMARK_CONFIGS["asymmetric"], "max_num_categories": 2},
        {"design_config": DDM_BENCHMARK_CONFIGS["multi_factor"], "max_num_categories": 2},
    ],
}


# ── CDM ───────────────────────────────────────────────────────────────────────

CDM_BENCHMARK_CONFIGS = {
    "intercept_only": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "regressed": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": [],
    },
    "fixed": {
        "1": ["v", "v_theta", "a", "tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "fixed_regressed": {
        "1": ["v", "v_theta", "a", "tau"],
        "u_1": ["v", "a"], "u_2": ["v", "a"], "u_1:u_2": [],
    },
    "interaction": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_theta", "a", "tau", "s_v"],
        "u_2": ["v", "v_theta", "a", "tau"],
        "u_1:u_2": ["v", "v_theta", "a"],
    },
    "three_reg_interaction": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_theta", "a", "tau", "s_v"],
        "u_3": ["v", "v_theta", "a", "tau"],
        "u_1:u_2": ["v", "v_theta", "a"],
        "u_1:u_3": ["v", "v_theta"],
        "u_2:u_3": ["v"],
    },
    "full": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
    },
    # ── Holdout configs (excluded from model_family training) ─────────────────
    "two_factor": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
    },
    "asymmetric": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_theta", "s_v"],
        "u_2": ["a", "tau", "s_tau"],
    },
    "multi_factor": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_theta", "a", "tau", "s_v"],
        "u_3": ["v", "v_theta", "a", "tau"],
        "u_4": ["v", "v_theta", "a"],
        "u_5": ["v", "v_theta"],
    },
}

CDM_CONFIG = {
    "model_cls": CDM,
    "prior_fn": cdm_priors,
    "link_fn": cdm_link_fun,
    "name": "CDM",
    "intrinsic_params": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
    "variable_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
    "free_params": ["v", "v_theta", "a", "tau"],
    "fixed_params": ["s_v", "s_tau"],
    "global_params": [],
    "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
    "benchmark_configs": CDM_BENCHMARK_CONFIGS,
    "val_scenarios": [
        ("interaction", CDM_BENCHMARK_CONFIGS["interaction"], {"max_num_categories": 2}),
        ("fixed", CDM_BENCHMARK_CONFIGS["fixed"], {"max_num_categories": 2}),
        ("two_factor", CDM_BENCHMARK_CONFIGS["two_factor"], {"max_num_categories": 3}),
    ],
    "holdout_configs": [
        {"design_config": CDM_BENCHMARK_CONFIGS["three_reg_interaction"], "max_num_categories": 3},
        {"design_config": CDM_BENCHMARK_CONFIGS["two_factor"], "max_num_categories": 3},
        {"design_config": CDM_BENCHMARK_CONFIGS["asymmetric"], "max_num_categories": 2},
        {"design_config": CDM_BENCHMARK_CONFIGS["multi_factor"], "max_num_categories": 2},
    ],
}


# ── RDM ───────────────────────────────────────────────────────────────────────

RDM_BENCHMARK_CONFIGS = {
    "intercept_only": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "regressed": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": [],
    },
    "fixed": {
        "1": ["v", "v_diff", "a", "tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "fixed_regressed": {
        "1": ["v", "v_diff", "a", "tau"],
        "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": [],
    },
    "interaction": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v"],
        "u_2": ["v", "v_diff", "a", "tau"],
        "u_1:u_2": ["v", "v_diff", "a"],
    },
    "three_reg_interaction": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v"],
        "u_3": ["v", "v_diff", "a", "tau"],
        "u_1:u_2": ["v", "v_diff", "a"],
        "u_1:u_3": ["v", "v_diff"],
        "u_2:u_3": ["v"],
    },
    "full": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
    },
    # ── Holdout configs (excluded from model_family training) ─────────────────
    "two_factor": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
    },
    "asymmetric": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "s_v"],
        "u_2": ["a", "tau", "s_tau"],
    },
    "multi_factor": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v"],
        "u_3": ["v", "v_diff", "a", "tau"],
        "u_4": ["v", "v_diff", "a"],
        "u_5": ["v", "v_diff"],
    },
}

RDM_CONFIG = {
    "model_cls": RDM,
    "prior_fn": rdm_priors,
    "link_fn": rdm_link_fun,
    "name": "RDM",
    "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
    "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
    "free_params": ["v", "v_diff", "a", "tau"],
    "fixed_params": ["s_v", "s_tau"],
    "global_params": [],
    "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
    "benchmark_configs": RDM_BENCHMARK_CONFIGS,
    "val_scenarios": [
        ("interaction", RDM_BENCHMARK_CONFIGS["interaction"], {"max_num_categories": 2}),
        ("fixed", RDM_BENCHMARK_CONFIGS["fixed"], {"max_num_categories": 2}),
        ("two_factor", RDM_BENCHMARK_CONFIGS["two_factor"], {"max_num_categories": 3}),
    ],
    "holdout_configs": [
        {"design_config": RDM_BENCHMARK_CONFIGS["three_reg_interaction"], "max_num_categories": 3},
        {"design_config": RDM_BENCHMARK_CONFIGS["two_factor"], "max_num_categories": 3},
        {"design_config": RDM_BENCHMARK_CONFIGS["asymmetric"], "max_num_categories": 2},
        {"design_config": RDM_BENCHMARK_CONFIGS["multi_factor"], "max_num_categories": 2},
    ],
    "train_context": RDM.build_random_context,
}


# ── LBA ───────────────────────────────────────────────────────────────────────

LBA_BENCHMARK_CONFIGS = {
    "intercept_only": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "regressed": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": [],
    },
    "fixed": {
        "1": ["v", "v_diff", "a", "tau"],
        "u_1": [], "u_2": [], "u_1:u_2": [],
    },
    "fixed_regressed": {
        "1": ["v", "v_diff", "a", "tau"],
        "u_1": ["v_diff", "a"], "u_2": ["v_diff", "a"], "u_1:u_2": [],
    },
    "interaction": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v"],
        "u_2": ["v", "v_diff", "a", "tau"],
        "u_1:u_2": ["v", "v_diff", "a"],
    },
    "three_reg_interaction": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v"],
        "u_3": ["v", "v_diff", "a", "tau"],
        "u_1:u_2": ["v", "v_diff", "a"],
        "u_1:u_3": ["v", "v_diff"],
        "u_2:u_3": ["v"],
    },
    "full": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
    },
    # ── Holdout configs (excluded from model_family training) ─────────────────
    "two_factor": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
    },
    "asymmetric": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "s_v"],
        "u_2": ["a", "tau", "s_tau"],
    },
    "multi_factor": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v"],
        "u_3": ["v", "v_diff", "a", "tau"],
        "u_4": ["v", "v_diff", "a"],
        "u_5": ["v", "v_diff"],
    },
}

LBA_CONFIG = {
    "model_cls": LBA,
    "prior_fn": lba_priors,
    "link_fn": lba_link_fun,
    "name": "LBA",
    "intrinsic_params": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
    "variable_names": [r"$v$", r"$v_{\mathrm{diff}}$", r"$a$", r"$\tau$", r"$s_v$", r"$s_\tau$"],
    "free_params": ["v", "v_diff", "a", "tau"],
    "fixed_params": ["s_v", "s_tau"],
    "global_params": [],
    "default_fixed_values": {"s_v": 0.0, "s_tau": 0.0},
    "benchmark_configs": LBA_BENCHMARK_CONFIGS,
    "val_scenarios": [
        ("interaction", LBA_BENCHMARK_CONFIGS["interaction"], {"max_num_categories": 2}),
        ("fixed", LBA_BENCHMARK_CONFIGS["fixed"], {"max_num_categories": 2}),
        ("two_factor", LBA_BENCHMARK_CONFIGS["two_factor"], {"max_num_categories": 3}),
    ],
    "holdout_configs": [
        {"design_config": LBA_BENCHMARK_CONFIGS["three_reg_interaction"], "max_num_categories": 3},
        {"design_config": LBA_BENCHMARK_CONFIGS["two_factor"], "max_num_categories": 3},
        {"design_config": LBA_BENCHMARK_CONFIGS["asymmetric"], "max_num_categories": 2},
        {"design_config": LBA_BENCHMARK_CONFIGS["multi_factor"], "max_num_categories": 2},
    ],
    "train_context": LBA.build_random_context,
}


# ── Registries ────────────────────────────────────────────────────────────────

MODELS = {
    "ddm": DDM_CONFIG,
    "cdm": CDM_CONFIG,
    "rdm": RDM_CONFIG,
    "lba": LBA_CONFIG,
}

# Union of all parameters across models in ModelClass insertion order
# (DDM → RDM adds v_diff → CDM adds v_theta)
ALL_PARAMS = ["v", "a", "z", "tau", "s_v", "s_tau", "v_diff", "v_theta"]
ALL_VARIABLE_NAMES = [
    r"$v$", r"$a$", r"$z$", r"$\tau$",
    r"$s_v$", r"$s_\tau$",
    r"$v_{\mathrm{diff}}$", r"$v_\theta$",
]


# ── BayesFlow case configs ─────────────────────────────────────────────────────

DDM_BF_CASE_CONFIGS = {
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
        "design_config": DDM_BENCHMARK_CONFIGS["interaction"],
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
        "design_config": DDM_BENCHMARK_CONFIGS["full"],
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

CDM_BF_CASE_CONFIGS = {
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
        "design_config": CDM_BENCHMARK_CONFIGS["interaction"],
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
        "design_config": CDM_BENCHMARK_CONFIGS["full"],
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

RDM_BF_CASE_CONFIGS = {
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
        "design_config": RDM_BENCHMARK_CONFIGS["interaction"],
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
        "design_config": RDM_BENCHMARK_CONFIGS["full"],
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

DDM_CONFIG["bf_case_configs"] = DDM_BF_CASE_CONFIGS
CDM_CONFIG["bf_case_configs"] = CDM_BF_CASE_CONFIGS
RDM_CONFIG["bf_case_configs"] = RDM_BF_CASE_CONFIGS
