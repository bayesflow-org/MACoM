"""Smoke tests for the simulator sampling path.

Covers the invariants the design generators and coefficient priors must hold whichever
variant is selected -- a bad combination here corrupts training data silently rather
than raising.

Zero-argument test functions with plain asserts, so this runs standalone with no test
dependency and is still collectable by pytest if one is added later:

    python tests/test_simulator_smoke.py
    python -m pytest tests/test_simulator_smoke.py -q     # if pytest is installed
"""

import sys
import traceback

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.simulator.benchmarks.ssms import DDM, RDM
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors
from cogformer.simulator.design_utils import (
    build_design_matrix,
    build_parameter_mask,
    build_random_discrete_mask,
)
from cogformer.simulator.model_class import ModelClass
from cogformer.simulator.model_family import ModelFamily

GENERATORS = ("mixture", "quantile")
COEF_PRIORS = ("budget", "iid")

CONFIG = {"1": ["v", "a"], "u_1": ["v"], "u_2": ["v", "a"], "u_1:u_2": ["v"]}


def make_family():
    return ModelFamily(model=DDM(), prior=ddm_priors(), name="ddm", num_workers=1)


def test_design_matrix_matches_mask_layout():
    """Column count must equal parameter-mask row count, or every downstream reshape
    into (rows, params) silently misaligns."""
    for generator in GENERATORS:
        np.random.seed(0)
        for _ in range(40):
            dm, counts, diag = build_design_matrix(
                CONFIG, num_obs=200, max_num_categories=4, generator=generator,
                return_category_counts=True, return_diagnostics=True,
            )
            pm = build_parameter_mask(
                CONFIG, ["v", "a", "tau"], max_num_categories=4,
                keep_intercept=True, category_counts=counts,
            )
            assert dm.shape[1] == pm.shape[0], (generator, dm.shape, pm.shape)
            keys = set(diag["categorical_keys"]) | set(diag["continuous_keys"])
            assert keys <= set(CONFIG), (generator, keys)


def test_sample_is_finite_and_respects_mask():
    """No coefficient may land on a masked row: those are unidentified, so a nonzero
    value there is an unlearnable target."""
    for generator in GENERATORS:
        for coefficient_prior in COEF_PRIORS:
            np.random.seed(1)
            mf = make_family()
            for _ in range(20):
                out = mf.sample(
                    num_obs=120, num_regressors=2, max_num_categories=4,
                    keep_intercept=True, add_interaction=True,
                    flatten_param_outputs=False, link_fun=ddm_link_fun(),
                    design_generator=generator, coefficient_prior=coefficient_prior,
                )
                mask, matrix = out["param_mask"], out["param_matrix"]
                tag = (generator, coefficient_prior)
                assert mask.shape == matrix.shape, tag
                assert not np.any((matrix != 0) & (mask == 0)), tag
                assert np.all(np.isfinite(matrix)), tag
                for key, arr in out["obs"].items():
                    assert np.all(np.isfinite(arr)), (tag, key)


def test_batch_sample_shapes_and_adapter():
    """A collated batch must survive Adapter.adapt, which is what training consumes."""
    for generator in GENERATORS:
        for coefficient_prior in COEF_PRIORS:
            np.random.seed(2)
            mf = make_family()
            batch = mf.batch_sample(
                batch_size=8, num_obs=100, min_num_regressors=0, max_num_regressors=3,
                max_num_categories=3, keep_intercept=True, add_interaction=True,
                link_fun=ddm_link_fun(), design_generator=generator,
                coefficient_prior=coefficient_prior,
            )
            tag = (generator, coefficient_prior)
            n_rows = batch["param_masks"].shape[-1] // batch["n_params"]
            assert batch["design_matrices"].shape[2] == n_rows, tag
            assert batch["position_codes"].shape[1] == n_rows, tag

            adapted = Adapter.adapt(batch, device="cpu")
            assert adapted["param_masks"].shape == adapted["param_matrices"].shape, tag
            assert adapted["regressor_indices"].shape[-1] == 5, tag
            assert bool(torch.isfinite(adapted["input_data"]).all()), tag


def test_model_class_batch_sample():
    """The multi-family path lifts params into a global space; shapes must line up."""
    for coefficient_prior in COEF_PRIORS:
        np.random.seed(3)
        mc = ModelClass(
            model_families={
                "ddm": ModelFamily(model=DDM(), prior=ddm_priors(), name="ddm", num_workers=1),
                "rdm": ModelFamily(model=RDM(), prior=rdm_priors(), name="rdm", num_workers=1),
            },
            link_funs={"ddm": ddm_link_fun(), "rdm": rdm_link_fun()},
            num_workers=1,
        )
        batch = mc.batch_sample(
            batch_size=8, num_obs=80, max_num_regressors=2, max_num_categories=3,
            keep_intercept=True, coefficient_prior=coefficient_prior,
        )
        n_rows = batch["param_masks"].shape[-1] // batch["n_global_params"]
        assert batch["design_matrices"].shape[2] == n_rows, coefficient_prior
        assert len(batch["model_names"]) == 8, coefficient_prior


def test_budget_makes_predictor_scale_design_invariant():
    """The point of the variance budget: predictor SD must not grow with design size.

    Guards a regression where the budget silently stops being applied. The iid prior
    grows like sqrt(active rows), so the ratio across design sizes separates them.

    Only draws where the parameter actually has an active slope row are counted. Under
    a random design config a parameter is often not regressed at all, and those zeros
    dilute the small-design mean -- which would look like scale growth even if the
    budget were working perfectly.
    """
    np.random.seed(4)
    sds = {}
    for prior_name in COEF_PRIORS:
        by_size = []
        for num_regressors in (1, 3):
            mf = make_family()
            vals = []
            while len(vals) < 50:
                out = mf.sample(
                    num_obs=200, num_regressors=num_regressors, max_num_categories=4,
                    keep_intercept=True, add_interaction=num_regressors > 1,
                    flatten_param_outputs=False, link_fun=ddm_link_fun(),
                    coefficient_prior=prior_name,
                )
                if not out["param_mask"][1:, 0].any():
                    continue
                eta = out["design_matrix"] @ out["param_matrix"]
                vals.append(eta[:, 0].std())
            by_size.append(float(np.mean(vals)))
        sds[prior_name] = by_size

    iid_growth = sds["iid"][1] / max(sds["iid"][0], 1e-9)
    budget_growth = sds["budget"][1] / max(sds["budget"][0], 1e-9)
    assert iid_growth > 1.3, sds
    assert 0.7 < budget_growth < 1.4, sds


def test_hierarchical_discrete_prob_reaches_corners():
    """discrete_prob=None must put real mass on the all-categorical and all-continuous
    corners, which an iid Bernoulli(0.5) reaches only at 0.5**R."""
    np.random.seed(5)
    draws = np.array([build_random_discrete_mask(3, discrete_prob=None) for _ in range(4000)])
    totals = draws.sum(axis=1)
    corners = float(np.mean((totals == 0) | (totals == 3)))
    assert corners > 0.30, corners


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
