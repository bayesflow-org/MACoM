"""Statistical and layout checks for the optional Gaussian-softmax design generator.

Run with ``PYTHONPATH=. python tests/test_gaussian_softmax.py`` or collect with pytest.
"""

import sys
import traceback
from unittest.mock import patch

import numpy as np

from cogformer.simulator import design_utils as designs


def build(config, num_obs=200, **kwargs):
    return designs.build_design_matrix(
        config, num_obs, generator="gaussian_softmax",
        return_category_counts=True, return_diagnostics=True, **kwargs,
    )


def original_labels(matrix, start, count, order):
    """Decode dummy columns back to the original category indices in diagnostics."""
    one_hot = np.column_stack([
        1 - matrix[:, start:start + count].sum(axis=1), matrix[:, start:start + count],
    ])
    return order[one_hot.argmax(axis=1)]


def test_fixed_logits_recover_three_category_probabilities():
    np.random.seed(10)
    expected = np.array([0.2, 0.3, 0.5])
    matrix, counts, diag = build(
        {"1": ["v"], "u_1": ["v"]}, num_obs=30000,
        min_num_categories=3, max_num_categories=3, discrete_prob=1,
        logit_scale=0, category_intercepts={"u_1": np.log(expected)},
    )
    labels = original_labels(matrix, 1, counts["u_1"], diag["category_label_orders"]["u_1"])
    observed = np.bincount(labels, minlength=3) / len(labels)
    np.testing.assert_allclose(observed, expected, atol=0.012)


def test_softmax_is_stable_and_invariant_to_common_logit_shift():
    results = []
    with patch.object(designs, "sample_lkj_correlation", side_effect=lambda q, eta: np.eye(q)):
        for shift in (0, 1000):
            np.random.seed(11)
            results.append(build(
                {"u_1": ["v"]}, num_obs=2000, min_num_categories=3, max_num_categories=3,
                discrete_prob=1, category_intercepts={"u_1": np.array([-1, 0, 1]) + shift},
            )[0])
    assert np.isfinite(results).all()
    np.testing.assert_array_equal(*results)


def test_joint_latent_induces_categorical_continuous_dependence():
    np.random.seed(12)
    correlation = np.array([[1, 0.65, -0.65], [0.65, 1, 0], [-0.65, 0, 1]])
    with patch.object(designs, "sample_lkj_correlation", return_value=correlation):
        matrix, counts, diag = build(
            {"u_1": ["v"], "u_2": ["v"]}, num_obs=30000,
            min_num_categories=2, max_num_categories=2, main_discrete_mask=np.array([1, 0]),
            intercept_scale=0, logit_scale=3,
        )
    labels = original_labels(matrix, 0, counts["u_1"], diag["category_label_orders"]["u_1"])
    x = matrix[:, 1]
    assert x[labels == 0].mean() - x[labels == 1].mean() > 0.8
    np.testing.assert_allclose(x.std(), 1, atol=0.025)


def test_joint_latent_induces_dependence_between_categorical_predictors():
    np.random.seed(13)
    correlation = np.array([[1, 0, 0.9, 0], [0, 1, 0, 0.9],
                            [0.9, 0, 1, 0], [0, 0.9, 0, 1]])
    with patch.object(designs, "sample_lkj_correlation", return_value=correlation):
        matrix, counts, diag = build(
            {"u_1": ["v"], "u_2": ["v"]}, num_obs=30000,
            min_num_categories=2, max_num_categories=2, discrete_prob=1,
            intercept_scale=0, logit_scale=4,
        )
    labels = [original_labels(matrix, j, counts[key], diag["category_label_orders"][key])
              for j, key in enumerate(("u_1", "u_2"))]
    assert np.mean(labels[0] == labels[1]) > 0.75


def test_arbitrary_category_counts_mixed_predictors_and_interactions():
    config = designs.build_design_config(["v"], num_regressors=4, keep_intercept=True, add_interaction=True)
    for k in (2, 3, 5):
        np.random.seed(k)
        matrix, counts, diag = build(
            config, min_num_categories=k, max_num_categories=k,
            main_discrete_mask=np.array([1, 0, 1, 0]), intercept_scale=0,
        )
        width = k - 1
        assert matrix.shape == (200, 1 + 4 * width + 6 * width ** 2)
        assert counts == {"u_1": width, "u_2": 1, "u_3": width, "u_4": 1}
        assert diag["latent_correlation"].shape == (2 + 2 * k, 2 + 2 * k)
        # First interaction is u_1:u_2: only the continuous block's first slot is filled.
        inter = matrix[:, 1 + 4 * width:1 + 4 * width + width ** 2].reshape(200, width, width)
        np.testing.assert_allclose(inter[:, :, 0], matrix[:, 1:1 + width] * matrix[:, 1 + width, None])
        assert not inter[:, :, 1:].any()


def test_empty_categories_are_compacted_and_excluded_from_mask():
    config = {"1": ["v"], "u_1": ["v"], "u_2": ["v"], "u_1:u_2": ["v"]}
    matrix, counts, diag = build(
        config, min_num_categories=3, max_num_categories=3,
        main_discrete_mask=np.array([1, 0]), logit_scale=0,
        category_intercepts={"u_1": np.array([-1000, 1000, -1000])},
    )
    assert counts["u_1"] == 0
    assert diag["realized_num_categories"] == [1]
    np.testing.assert_array_equal(diag["category_label_orders"]["u_1"], [1])
    mask = designs.build_parameter_mask(config, ["v"], max_num_categories=3, category_counts=counts)
    assert not mask[1:3].any()
    assert not mask[5:].any()
    assert not matrix[:, 1:3].any()
    assert not matrix[:, 5:].any()


def test_continuous_only_and_no_latent_cases():
    np.random.seed(14)
    correlation = np.array([[1, 0.7], [0.7, 1]])
    with patch.object(designs, "sample_lkj_correlation", return_value=correlation):
        matrix, counts, diag = build({"u_1": ["v"], "u_2": ["v"]}, num_obs=20000,
                                     discrete_prob=0, max_num_categories=3, continuous_scale=2)
    np.testing.assert_allclose(np.cov(matrix[:, [0, 2]].T), correlation * 4, atol=0.12)
    assert counts == {"u_1": 1, "u_2": 1} and not diag["categorical_keys"]

    with patch.object(designs, "sample_lkj_correlation", side_effect=AssertionError("no latent coordinates")):
        matrix, counts, diag = build({"1": ["v"]}, num_obs=10)
        np.testing.assert_array_equal(matrix, np.ones((10, 1)))
        assert counts == {} and diag["latent_correlation"].shape == (0, 0)
        matrix, counts, _ = build({}, num_obs=10)
        assert matrix.shape == (10, 0) and counts == {}
        context = {"u_1": np.arange(10), "u_2": np.arange(10) * 2, "u_1:u_2": -np.ones(10)}
        matrix, counts, _ = build({"u_1": ["v"], "u_2": ["v"], "u_1:u_2": ["v"]},
                                  num_obs=10, context=context, max_num_categories=3)
        np.testing.assert_array_equal(matrix[:, 0], context["u_1"])
        np.testing.assert_array_equal(matrix[:, 2], context["u_2"])
        np.testing.assert_array_equal(matrix[:, 4], context["u_1:u_2"])
        assert counts == {"u_1": 1, "u_2": 1}


def test_invalid_controls_fail_clearly():
    cases = [
        {"eta": 0}, {"eta": np.nan}, {"logit_scale": -1}, {"continuous_scale": np.inf},
        {"intercept_scale": -1}, {"category_concentration": 1},
        {"main_discrete_mask": np.array([1, 0])}, {"main_discrete_mask": np.array([2])},
        {"category_intercepts": {"u_1": [1, 2]}},
        {"category_intercepts": {"u_1": [0, np.nan, 0]}},
        {"category_intercepts": {"absent": [0, 0, 0]}},
        {"num_obs": 0}, {"min_num_categories": 1},
        {"context": {"u_1": np.zeros(3)}},
    ]
    for overrides in cases:
        kwargs = {"num_obs": 20, "discrete_prob": 1, "min_num_categories": 3, "max_num_categories": 3}
        kwargs.update(overrides)
        try:
            build({"u_1": ["v"]}, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid controls: {overrides}")


def test_category_bounds_ignored_without_categoricals():
    """The BayesFlow intercept-only / fixed cases pass max_num_categories=0: no categorical
    regressor is drawn, so the bounds must not be validated (the mixture generator never
    did). Still enforced whenever a categorical regressor exists (test above)."""
    for gen in ("mixture", "gaussian_softmax"):
        m = designs.build_design_matrix({"1": ["v", "a"]}, 50, min_num_categories=2,
                                        max_num_categories=0, generator=gen)
        assert m.shape == (50, 1) and np.all(m == 1.0)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception:
                failures += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    sys.exit(bool(failures))
