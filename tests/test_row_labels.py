"""Regression tests for design-matrix ROW labels (the y-axis of every diagnostic).

These exist because the failure mode is SILENT: a mislabeled row produces a
perfectly normal-looking figure whose y-axis attributes coefficients to the wrong
predictor or the wrong category. Nothing raises, and it is not visible without
knowing the design that produced the plot.

The specific trap: ``build_row_labels`` can infer which regressors are categorical
from the parameter mask, but that inference is unreliable in BOTH directions --

  * at K=2 a categorical occupies only contrast slot 1, indistinguishable from a
    continuous regressor, so it is labeled ``u_1`` when it should be ``u_1:c_1``;
  * at K>=3 a mask built without realized category counts activates every slot of
    a block, so a CONTINUOUS regressor is labeled ``u_2:c_1``/``u_2:c_2``.

So every plotting call site must pass ``categorical_regressors`` explicitly. These
tests pin the correct behaviour, and pin the inference limitation too, so that if
someone "fixes" the fallback the change is deliberate rather than silent.

    python tests/test_row_labels.py
"""

import sys
import traceback

import numpy as np

from cogformer.simulator.design_utils import (
    build_ensemble_row_labels,
    build_parameter_mask,
    build_row_labels,
    sample_covering_design_configs,
)

DC = {"1": ["v", "a"], "u_1": ["v"], "u_2": ["v"], "u_1:u_2": ["v"]}
PARAMS = ["v", "a"]


def test_k2_categorical_carries_contrast_suffix():
    """The headline case: dummy coding drops c_0, so a K=2 categorical is u_1:c_1."""
    got = build_row_labels(DC, 2, math=False, categorical_regressors=["u_1"])
    assert got == ["1", "u_1:c_1", "u_2", "u_1:c_1 × u_2"], got


def test_k2_continuous_has_no_suffix():
    got = build_row_labels(DC, 2, math=False, categorical_regressors=[])
    assert got == ["1", "u_1", "u_2", "u_1 × u_2"], got


def test_mask_inference_cannot_see_a_k2_categorical():
    """Pins the KNOWN limitation, so a change to it has to be deliberate."""
    mask = build_parameter_mask(design_config=DC, intrinsic_params=PARAMS,
                                max_num_categories=2, keep_intercept=True)
    got = build_row_labels(DC, 2, math=False, parameter_mask=mask)
    assert got == ["1", "u_1", "u_2", "u_1 × u_2"], got


def test_k3_categorical_enumerates_contrasts():
    got = build_row_labels(DC, 3, math=False, categorical_regressors=["u_1"])
    assert got[:3] == ["1", "u_1:c_1", "u_1:c_2"], got[:3]
    assert "u_1:c_2 × u_2" in got, got


def test_label_count_matches_mask_rows():
    for k in (2, 3, 4):
        labels = build_row_labels(DC, k, math=False)
        mask = build_parameter_mask(design_config=DC, intrinsic_params=PARAMS,
                                    max_num_categories=k, keep_intercept=True)
        assert len(labels) == mask.shape[0], (k, len(labels), mask.shape[0])


def test_ensemble_parenthesises_only_genuine_disagreement():
    mask = build_parameter_mask(design_config=DC, intrinsic_params=PARAMS,
                                max_num_categories=2, keep_intercept=True)
    disagree = build_ensemble_row_labels([DC, DC], [mask, mask], 2, max_num_regressors=2,
                                         categorical_regressors_list=[["u_1"], []],
                                         math=False)
    assert disagree == ["1", "u_1(:c_1)", "u_2", "u_1(:c_1) × u_2"], disagree
    agree = build_ensemble_row_labels([DC, DC], [mask, mask], 2, max_num_regressors=2,
                                      categorical_regressors_list=[["u_1"], ["u_1"]],
                                      math=False)
    assert agree == ["1", "u_1:c_1", "u_2", "u_1:c_1 × u_2"], agree


def test_labels_from_the_real_sampler_path():
    """End-to-end: labels derived the way the ensemble script derives them.

    The script reads ``discrete_masks`` from the sampler rather than counting mask
    slots; this pins that the resulting label marks the categorical regressor.
    """
    from cogformer.simulator import ModelFamily
    from cogformer.simulator.benchmarks.ssms import DDM
    from cogformer.simulator.benchmarks.ssms.priors import ddm_priors
    from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun

    params = ["v", "a", "z", "tau", "s_v", "s_tau"]
    mf = ModelFamily(model=DDM(), prior=ddm_priors(), intrinsic_params=params)
    dc = {"1": params, "u_1": ["v"], "u_2": ["v"]}
    s = mf.batch_sample(
        design_config=dc, batch_size=2, num_obs=60,
        max_num_regressors=2, max_num_categories=2, min_num_categories=2,
        categorical_regressors=["u_1"], keep_intercept=True,
        flatten_param_outputs=True, fixed_config=True,
        # build_link_functions requires link_fun keys to be a subset of
        # intrinsic_params; ddm_link_fun carries pi, which this design omits.
        link_fun={k: v for k, v in ddm_link_fun().items() if k in params},
    )
    disc = np.asarray(s["discrete_masks"][0][:2])
    cats = [f"u_{i + 1}" for i, v in enumerate(disc) if v == 1.0]
    assert cats == ["u_1"], f"sampler reported {cats}, expected ['u_1']"
    labels = build_row_labels(dc, 2, math=False, categorical_regressors=cats)
    assert labels[1] == "u_1:c_1", labels
    assert labels[2] == "u_2", labels


def test_covering_sampler_leaves_no_uncovered_cell():
    """The rejection sampler must actually guarantee what it claims."""
    params = ["v", "a", "z", "tau", "s_v", "s_tau", "pi"]
    fixed = ["s_v", "s_tau"]
    free = [p for p in params if p not in fixed]
    np.random.seed(0)
    configs = sample_covering_design_configs(
        n_configs=4, intrinsic_params=params, num_regressors=2,
        free_intrinsics=free, fixed_intrinsics=fixed, keep_intercept=True,
        add_interaction=True, free_prob=0.5, global_params=["pi"],
    )
    assert len(configs) == 4, len(configs)
    slope_legal = [p for p in params if p != "pi"]
    for key in {k for c in configs for k in c}:
        legal = params if key == "1" else slope_legal
        union = {p for c in configs for p in c.get(key, [])}
        missing = [p for p in legal if p not in union]
        assert not missing, f"row {key} leaves {missing} uncovered -> N/A cells"


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
    print(f"\n{'all passed' if not failures else f'{failures} failure(s)'}")
    sys.exit(1 if failures else 0)
