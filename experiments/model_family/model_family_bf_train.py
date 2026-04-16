import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import numpy as np
import bayesflow as bf

from pathlib import Path

from cogformer.simulator.model_family import ModelFamily

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import MODELS as _BASE_MODELS


# Per-model CASE_CONFIGS ─────────────────────────────────────────────────────

DDM_CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": False, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "expected_num_active": None,
    },
    "fixed": {
        "free_intrinsics": ["v", "a", "z", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "expected_num_active": 4,
    },
    "regressed": {
        "free_intrinsics": ["v", "a", "z", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": ["v", "a", "z"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "expected_num_active": 12,
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "a", "z", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v", "a", "z"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
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
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "expected_num_active": None,
    },
    "fixed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "expected_num_active": 4,
    },
    "regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "expected_num_active": 10,
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "v_theta", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
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
        "expected_num_active": 24,
    },
}

RDM_CASE_CONFIGS = {
    "intercept_only": {
        "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": False, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "expected_num_active": None,
    },
    "fixed": {
        "free_intrinsics": ["v", "v_diff", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": None,
        "min_num_regressors": 0, "max_num_regressors": 0, "max_num_categories": 0,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": False, "squeeze_outputs": True,
        "expected_num_active": 4,
    },
    "regressed": {
        "free_intrinsics": ["v", "v_diff", "a", "tau", "s_v", "s_tau"],
        "fixed_intrinsics": [], "fixed_values": {},
        "regressed_params": ["v_diff", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
        "expected_num_active": 10,
    },
    "fixed_regressed": {
        "free_intrinsics": ["v", "v_diff", "a", "tau"],
        "fixed_intrinsics": ["s_v", "s_tau"], "fixed_values": {"s_v": 0, "s_tau": 0},
        "regressed_params": ["v_diff", "a"],
        "min_num_regressors": 2, "max_num_regressors": 2, "max_num_categories": 2,
        "fixed_config": True, "design_config": None,
        "flatten_param_outputs": True, "squeeze_outputs": False,
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
        "expected_num_active": 24,
    },
}

_BF_CASE_CONFIGS = {"ddm": DDM_CASE_CONFIGS, "cdm": CDM_CASE_CONFIGS, "rdm": RDM_CASE_CONFIGS}
_BF_CKPT_PREFIXES = {"ddm": "ddm_families_bf", "cdm": "cdm_families_bf", "rdm": "rdm_families_bf"}

MODELS = {}
for _key, _base in _BASE_MODELS.items():
    _cfg = dict(_base)
    _cfg["case_configs"] = _BF_CASE_CONFIGS[_key]
    _cfg["ckpt_prefix"] = _BF_CKPT_PREFIXES[_key]
    MODELS[_key] = _cfg


class ModelFamilyBF(bf.simulators.Simulator):
    """Generic BayesFlow simulator wrapping ModelFamily for any benchmark model."""

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
            prior=model_cfg["prior_fn"](),
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

    print(f"Model: {model_key.upper()}, Case: {case}")
    for k, v in samples.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")

    masks = samples["masks"]
    params = samples["params"]

    num_active = int(masks[0].astype(bool).sum())
    expected = config["expected_num_active"]
    if expected is not None:
        status = "OK" if num_active == expected else f"FAIL (expected {expected})"
        print(f"\n  [Check] Active params: {num_active}  →  {status}")
        assert num_active == expected
    else:
        print(f"\n  [Check] Active params: {num_active}  →  (variable, no assertion)")

    if config["fixed_config"] or config["design_config"] is not None:
        consistent = np.all(masks == masks[0])
        print(f"  [Check] Mask consistent across batch: {'OK' if consistent else 'FAIL'}")
        assert consistent

    zero_where_masked = np.allclose(params * (1 - masks), 0.0)
    print(f"  [Check] Params zero where masked: {'OK' if zero_where_masked else 'FAIL'}")
    assert zero_where_masked

    adapter = (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )
    adapted = adapter(simulator.sample(4))
    print("\nAdapted samples:")
    for k, v in adapted.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")


def main(model_key: str, case: str, epochs: int = 1000, steps_per_epoch: int = 100, batch_size: int = 64):
    model_cfg = MODELS[model_key]
    simulator = ModelFamilyBF(model_key=model_key, case=case)

    adapter = (
        bf.Adapter()
        .drop(["masks"])
        .convert_dtype("float64", "float32")
        .concatenate(["design_matrices", "rts", "choices"], into="summary_variables")
        .rename("params", "inference_variables")
    )

    summary_net = bf.networks.SetTransformer(
        summary_dim=32, seed_dim=64,
        num_heads=(4, 4, 4, 4), mlp_depths=(1, 1, 1, 1),
        embed_dims=(128, 128, 128, 128), mlp_widths=(128, 128, 128, 128),
        num_seeds=4,
    )
    inference_net = bf.networks.FlowMatching()

    ckpt_prefix = model_cfg["ckpt_prefix"]
    checkpoint_path = f"./outputs/checkpoints/model_family/bf/{model_key}/{ckpt_prefix}_{case}"

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        checkpoint_filepath=checkpoint_path,
    )

    history = workflow.fit_online(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
    )
    return history


if __name__ == "__main__":
    all_cases = list(DDM_CASE_CONFIGS.keys())

    parser = argparse.ArgumentParser(description="Model Family BayesFlow Training")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                        help="Benchmark model (ddm, cdm, rdm)")
    parser.add_argument("--case", type=str, default="intercept_only", choices=all_cases)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        test(model_key=args.model, case=args.case)
    else:
        main(
            model_key=args.model,
            case=args.case,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
        )
