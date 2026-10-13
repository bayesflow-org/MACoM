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
from itertools import product

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.simulator.benchmarks.ssms import DDM, RDM
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors
from cogformer.simulator.design_utils import (
    DESIGN_GENERATORS,
    build_design_matrix,
    build_parameter_mask,
    build_random_discrete_mask,
)
from cogformer.simulator.model_class import ModelClass
from cogformer.simulator.model_family import ModelFamily

GENERATORS = DESIGN_GENERATORS
COEF_PRIORS = ("cell_means", "iid")

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
    for generator, coefficient_prior in product(GENERATORS, COEF_PRIORS):
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
            design_generator=generator,
            design_generator_kwargs={"intercept_scale": 0.5} if generator == "gaussian_softmax" else None,
        )
        n_rows = batch["param_masks"].shape[-1] // batch["n_global_params"]
        assert batch["design_matrices"].shape[2] == n_rows, coefficient_prior
        assert len(batch["model_names"]) == 8, coefficient_prior


def test_model_class_carries_per_trial_context():
    """The model-class batch must carry RDM's per-trial favored_idx through to the
    encoder input. Its collate used to drop per_trial_context, so the Adapter fed 0 for
    every training dataset while validation and real data fed the real stimulus --
    v_diff was then unlearnable under stimulus coding, with no error anywhere."""
    np.random.seed(4)
    mc = ModelClass(
        model_families={
            "ddm": ModelFamily(model=DDM(), prior=ddm_priors(), name="ddm", num_workers=1),
            "rdm": ModelFamily(model=RDM(), prior=rdm_priors(), name="rdm", num_workers=1),
        },
        link_funs={"ddm": ddm_link_fun(), "rdm": rdm_link_fun()},
        train_contexts={"rdm": RDM.build_random_context},
        num_workers=1,
    )
    batch = mc.batch_sample(batch_size=16, num_obs=80, max_num_regressors=2,
                            max_num_categories=3, keep_intercept=True)
    fav = batch["per_trial_context"]["favored_idx"]
    assert fav.shape == (16, 80), fav.shape
    is_rdm = np.array([n == "rdm" for n in batch["model_names"]])
    assert is_rdm.any() and (~is_rdm).any(), batch["model_names"]
    k = batch["num_alternatives"]
    for i in np.where(is_rdm)[0]:
        # Uniform over K accumulators: 80 trials all on accumulator 0 has prob <= 2**-80.
        assert fav[i].max() > 0, i
        assert fav[i].max() <= k[i] - 1, (i, fav[i].max(), k[i])
    assert not fav[~is_rdm].any()

    # And it reaches the encoder: the favored_idx input column (second to last).
    x = Adapter.adapt(batch, device="cpu")["input_data"].numpy()
    assert (x[is_rdm, :, -2].max(axis=1) > 0).all()
    assert not x[~is_rdm, :, -2].any()


def test_cell_means_is_exchangeable_across_categories():
    """The property cell_means exists to provide.

    With dummy coding and the reference level pinned at 0, iid contrasts make the
    reference a priori closer to every other level: sd(mu_ref - mu_c) = 1 but
    sd(mu_c - mu_c') = sqrt(2). Relabelling which level is the baseline then changes the
    prior. Drawing one latent per category and taking differences equalizes all pairwise
    spreads and induces rho = +0.5 between contrasts through the shared -mu_0.

    Both priors keep the same N(0, 1) scale, so this is about structure, not width.
    """
    stats = {}
    for prior_name in COEF_PRIORS:
        np.random.seed(6)
        mf = make_family()
        rows = []
        while len(rows) < 800:
            out = mf.sample(
                num_obs=300, num_regressors=1, max_num_categories=4,
                keep_intercept=True, flatten_param_outputs=False,
                link_fun=ddm_link_fun(), categorical_regressors=["u_1"],
                coefficient_prior=prior_name,
            )
            b = out["param_matrix"][1:4, 0]
            if np.count_nonzero(b) == 3:
                rows.append(b)
        b = np.array(rows)
        mu = np.concatenate([np.zeros((len(b), 1)), b], axis=1)
        ref = np.mean([np.std(mu[:, 0] - mu[:, c]) for c in (1, 2, 3)])
        oth = np.mean([np.std(mu[:, i] - mu[:, j]) for i in (1, 2) for j in range(i + 1, 4)])
        rho = np.mean([np.corrcoef(b[:, i], b[:, j])[0, 1] for i in range(3) for j in range(i + 1, 3)])
        stats[prior_name] = (ref, oth, oth / ref, rho)

    _, _, cm_ratio, cm_rho = stats["cell_means"]
    _, _, iid_ratio, iid_rho = stats["iid"]
    assert 0.90 < cm_ratio < 1.10, stats     # exchangeable
    assert cm_rho > 0.35, stats              # shared -mu_0
    assert iid_ratio > 1.25, stats           # incumbent asymmetry, ~sqrt(2)
    assert abs(iid_rho) < 0.15, stats        # incumbent contrasts independent


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
