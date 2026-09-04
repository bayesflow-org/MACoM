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
    "full": {
        "1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "a", "z", "tau", "s_v", "s_tau"],
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
        # OG benchmark cases + one mixed-predictor case. Continuous cases pin K=2 (category
        # count is cosmetic for continuous regressors) and force an all-continuous mix, so
        # every batch shares one parameter mask (homogeneous -> no padding-cell artifact).
        # No skip_global: training designs always include global params, so a mask without
        # pi is out-of-distribution and biases recovery downward.
        ("intercept_only",  DDM_BENCHMARK_CONFIGS["intercept_only"],  {"max_num_categories": 2}),
        ("fixed",           DDM_BENCHMARK_CONFIGS["fixed"],           {"max_num_categories": 2}),
        ("regressed",       DDM_BENCHMARK_CONFIGS["regressed"],       {"max_num_categories": 2, "categorical_regressors": []}),
        ("fixed_regressed", DDM_BENCHMARK_CONFIGS["fixed_regressed"], {"max_num_categories": 2, "categorical_regressors": []}),
        ("interaction",     DDM_BENCHMARK_CONFIGS["interaction"],     {"max_num_categories": 2, "categorical_regressors": []}),
        # Mixed predictor: interaction design with u_1 categorical (K = training max via
        # exact-K) and u_2 continuous -> a categorical x continuous interaction.
        ("mixed_predictor", DDM_BENCHMARK_CONFIGS["interaction"],
         {"exact_num_categories": True, "categorical_regressors": ["u_1"]}),
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
    "full": {
        "1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
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
        # OG cases (continuous regressors, K=2) + mixed predictor (see DDM notes).
        ("intercept_only",  CDM_BENCHMARK_CONFIGS["intercept_only"],  {"max_num_categories": 2}),
        ("fixed",           CDM_BENCHMARK_CONFIGS["fixed"],           {"max_num_categories": 2}),
        ("regressed",       CDM_BENCHMARK_CONFIGS["regressed"],       {"max_num_categories": 2, "categorical_regressors": []}),
        ("fixed_regressed", CDM_BENCHMARK_CONFIGS["fixed_regressed"], {"max_num_categories": 2, "categorical_regressors": []}),
        ("interaction",     CDM_BENCHMARK_CONFIGS["interaction"],     {"max_num_categories": 2, "categorical_regressors": []}),
        ("mixed_predictor", CDM_BENCHMARK_CONFIGS["interaction"],
         {"exact_num_categories": True, "categorical_regressors": ["u_1"]}),
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
    "full": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
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
        # OG cases (continuous regressors, K=2) + mixed predictor (see DDM notes).
        ("intercept_only",  RDM_BENCHMARK_CONFIGS["intercept_only"],  {"max_num_categories": 2}),
        ("fixed",           RDM_BENCHMARK_CONFIGS["fixed"],           {"max_num_categories": 2}),
        ("regressed",       RDM_BENCHMARK_CONFIGS["regressed"],       {"max_num_categories": 2, "categorical_regressors": []}),
        ("fixed_regressed", RDM_BENCHMARK_CONFIGS["fixed_regressed"], {"max_num_categories": 2, "categorical_regressors": []}),
        ("interaction",     RDM_BENCHMARK_CONFIGS["interaction"],     {"max_num_categories": 2, "categorical_regressors": []}),
        ("mixed_predictor", RDM_BENCHMARK_CONFIGS["interaction"],
         {"exact_num_categories": True, "categorical_regressors": ["u_1"]}),
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
    "full": {
        "1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "u_1:u_2": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
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
        # OG cases (continuous regressors, K=2) + mixed predictor (see DDM notes).
        ("intercept_only",  LBA_BENCHMARK_CONFIGS["intercept_only"],  {"max_num_categories": 2}),
        ("fixed",           LBA_BENCHMARK_CONFIGS["fixed"],           {"max_num_categories": 2}),
        ("regressed",       LBA_BENCHMARK_CONFIGS["regressed"],       {"max_num_categories": 2, "categorical_regressors": []}),
        ("fixed_regressed", LBA_BENCHMARK_CONFIGS["fixed_regressed"], {"max_num_categories": 2, "categorical_regressors": []}),
        ("interaction",     LBA_BENCHMARK_CONFIGS["interaction"],     {"max_num_categories": 2, "categorical_regressors": []}),
        ("mixed_predictor", LBA_BENCHMARK_CONFIGS["interaction"],
         {"exact_num_categories": True, "categorical_regressors": ["u_1"]}),
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
        "expected_num_active": 7,
    },
    "fixed": {
        "free_intrinsics": ["v", "a", "z", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "param_names": [r"$v$", r"$a$", r"$z$", r"$\tau$"],
        "expected_num_active": 5,
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
        "expected_num_active": 13,
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
        "expected_num_active": 11,
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
        "expected_num_active": 19,
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
        "expected_num_active": 25,
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
        "expected_num_active": 7,
    },
    "fixed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "param_names": [r"$v$", r"$v_\theta$", r"$a$", r"$\tau$"],
        "expected_num_active": 5,
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
        "expected_num_active": 11,
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
        "expected_num_active": 9,
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
        "expected_num_active": 19,
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
        "expected_num_active": 25,
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
        "expected_num_active": 7,
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
        "expected_num_active": 5,
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
        "expected_num_active": 11,
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
        "expected_num_active": 9,
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
        "expected_num_active": 19,
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
        "expected_num_active": 25,
    },
}

# ── LBA BF case configs ───────────────────────────────────────────────────────
# model_family_bf_train deliberately had no LBA entry: LBA was reached zero-shot or by
# finetuning from the model-class checkpoint, never by a per-family BayesFlow baseline.
# The amortization-gap matrix needs "LBA: BF vs Finetuned", so a baseline is required.
#
# These are NOT hand-authored. LBA_BENCHMARK_CONFIGS is identical to
# RDM_BENCHMARK_CONFIGS (asserted below), and the two families share intrinsic_params
# (v, v_diff, a, tau, s_v, s_tau), free_params and fixed_params, so every field of the
# RDM BF case configs applies unchanged. Deriving them keeps the two from drifting --
# the field that broke this session (expected_num_active) cannot go stale in one table
# and not the other. Only design_config is repointed at LBA's own table so the configs
# read correctly, and the priors/link/model come from LBA_CONFIG at use sites.
assert LBA_BENCHMARK_CONFIGS == RDM_BENCHMARK_CONFIGS, (
    "LBA BF case configs are derived from RDM's on the assumption that the two "
    "benchmark tables are identical; they have diverged, so LBA needs its own configs."
)

def _lba_cases_from_rdm(rdm_cases: dict) -> dict:
    out = {}
    for case, cfg in rdm_cases.items():
        c = dict(cfg)
        dc = c.get("design_config")
        if dc is not None:
            # Repoint ONLY the entries that reference RDM_BENCHMARK_CONFIGS (interaction,
            # full), matched by identity. The other cases carry inline literals that are
            # deliberately narrower than the benchmark table -- intercept_only is
            # {"1": [...]} with no u_* keys at all, whereas the benchmark entry has empty
            # u_1/u_2/u_1:u_2. Substituting the table version adds phantom design-matrix
            # columns and collate fails broadcasting (500,2) into (500,1). Copy those.
            if any(dc is v for v in RDM_BENCHMARK_CONFIGS.values()):
                c["design_config"] = LBA_BENCHMARK_CONFIGS[case]
            else:
                c["design_config"] = {k: list(v) for k, v in dc.items()}
        out[case] = c
    return out

LBA_BF_CASE_CONFIGS = _lba_cases_from_rdm(RDM_BF_CASE_CONFIGS)


# NOTE: every "expected_num_active" in the three *_BF_CASE_CONFIGS dicts counts the
# global contamination parameter that _bf_cases_with_globals appends below. The values
# predated that function and were all short by exactly one, so `--test` asserted on
# cases that were simulating correctly. Counting pi makes the assertion meaningful
# again -- and it now fails, correctly, on the cases where the simulator does not
# match the config: rdm (pi absent in every case) and interaction (pi absent in all
# three models), plus fixed/fixed_regressed, where the fixed_intrinsics s_v and s_tau
# are still sampled instead of being pinned to fixed_values.
def _bf_cases_with_globals(case_configs: dict, global_params=GLOBAL_PARAMS) -> dict:
    """Give the BayesFlow baselines the same generative model as the CogFormer tiers.

    Both CogFormer trainers call ``add_global_params``, so their simulated data carries
    contamination (pi drawn from its prior). The BF case configs listed pi in NEITHER
    free_intrinsics nor fixed_intrinsics, so ModelFamily fell through to
    ``fixed_values.get("pi", 0.0)`` and the baseline trained on UNCONTAMINATED data --
    measured mean pi = 0.000000 over 64 simulation calls, while
    ``ModelFamily.intrinsic_params`` still *reported* pi because it defaults to
    ``prior.keys()``. The parameter list therefore looked compatible while the data was
    not, which would have confounded any CogFormer-vs-BayesFlow amortization gap with a
    generative-model difference.
    """
    out = {}
    for case, cfg in case_configs.items():
        c = dict(cfg)
        free = list(c["free_intrinsics"])
        fixed = set(c.get("fixed_intrinsics") or ())
        c["free_intrinsics"] = free + [g for g in global_params
                                       if g not in free and g not in fixed]

        # free_intrinsics alone is not enough. A case carrying an explicit
        # design_config takes the branch in ModelFamily._resolve_design that builds the
        # mask straight from that config and never consults free_intrinsics, so pi never
        # entered the parameter grid -- the contaminated-data fix silently missed
        # interaction (all three models) and every rdm case, which are exactly the
        # design_config cases. Add the globals to the intercept row too, the way
        # add_global_params does for the CogFormer configs. Copy rather than mutate:
        # these dicts are shared with the *_BENCHMARK_CONFIGS tables.
        dc = c.get("design_config")
        if dc is not None:
            c["design_config"] = {
                k: (list(v) + [g for g in global_params if g not in v and g not in fixed]
                    if k == "1" else list(v))
                for k, v in dc.items()
            }
        out[case] = c
    return out


DDM_CONFIG["bf_case_configs"] = _bf_cases_with_globals(DDM_BF_CASE_CONFIGS)
CDM_CONFIG["bf_case_configs"] = _bf_cases_with_globals(CDM_BF_CASE_CONFIGS)
RDM_CONFIG["bf_case_configs"] = _bf_cases_with_globals(RDM_BF_CASE_CONFIGS)
LBA_CONFIG["bf_case_configs"] = _bf_cases_with_globals(LBA_BF_CASE_CONFIGS)
