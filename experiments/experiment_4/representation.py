"""Representational analysis of CogFormer's encoder: RSA and CKA.

Experiment 4 asks whether CogFormer's encoder learns a UNIVERSAL representation.
The posterior-level evidence (C2ST, CRPS, log-density in addm_arm_compare.py)
compares what the two arms DECODE. This module compares what they REPRESENT, which
posterior metrics cannot reach: the frozen CogFormer summary is 128-dimensional and
the learned SetTransformer summary is 16-dimensional, so they share no coordinate
system and cannot be compared directly at all.

RSA and CKA both solve exactly that -- they compare the GEOMETRY two systems impose
on the same stimuli, invariant to basis and dimensionality.

REPEATS, AND WHY THEY MATTER
----------------------------
Plain RDM correlations are inflated by measurement noise: two noisy estimates of the
same geometry look less similar than the geometries actually are, and the bias depends
on how noisy each system is, so it does not cancel between systems of different
dimensionality. rsatoolbox's crossnobis distance is unbiased, but it needs REPEATED
measurements of each condition.

Here a "condition" is a parameter vector theta and a "measurement" is one simulated
dataset encoded to a summary. Repeats are therefore free and exact: hold theta fixed
and re-simulate with fresh noise. `simulate_repeats` does this by capturing the
prepared parameters from one ModelFamily.sample call and re-invoking model.simulate,
so the theta draw goes through the real pipeline and only the noise differs.
"""
from __future__ import annotations

import numpy as np


def simulate_repeats(model_family, model, n_conditions, n_repeats, num_obs,
                     design_config, mask_kwargs, link_fun, max_num_regressors,
                     max_num_categories, extra_sample_kwargs=None, seed=0):
    """Simulate ``n_repeats`` independent datasets per parameter vector.

    Returns
    -------
    data       : (n_conditions, n_repeats, num_obs, n_channels) raw per-trial tensors
    thetas     : (n_conditions, n_active) the parameter vector of each condition
    masks      : (n_conditions, n_cells) parameter masks
    designs    : (n_conditions, num_obs, n_cols) design matrices, shared across repeats
    """
    np.random.seed(seed)
    captured: list[tuple] = []
    # On the GPU backend batch_sample defers simulation and resolves the whole batch in
    # ONE model.simulate_many call, so simulate() is never invoked per dataset: spy on
    # simulate_many there. Spying on both would double-capture, since each delegates to
    # the other depending on the backend.
    gpu = model.uses_gpu
    name = "simulate_many" if gpu else "simulate"
    orig = getattr(type(model), name)

    if gpu:
        def spy(self, requests):
            captured.extend(requests)
            return orig(self, requests)
    else:
        def spy(self, params, context=None):
            captured.append((params, context))
            return orig(self, params, context)

    setattr(type(model), name, spy)
    try:
        samples = model_family.batch_sample(
            batch_size=n_conditions, num_obs=num_obs, design_config=design_config,
            mask_randomizer_kwargs=mask_kwargs, max_num_regressors=max_num_regressors,
            max_num_categories=max_num_categories, keep_intercept=True,
            fixed_config=True, flatten_param_outputs=True, link_fun=link_fun,
            **(extra_sample_kwargs or {}),
        )
        if len(captured) != n_conditions:
            raise RuntimeError(f"captured {len(captured)} simulate requests for "
                               f"{n_conditions} conditions (backend: {name})")
        # One capture per condition, in order. Re-simulate each with fresh noise: same
        # theta, same design, independent data. On the GPU, all repeats of all
        # conditions go out in a single launch.
        if gpu:
            flat = orig(model, [req for req in captured for _ in range(n_repeats)])
            reps = [flat[i * n_repeats:(i + 1) * n_repeats] for i in range(n_conditions)]
        else:
            reps = [[orig(model, params, context) for _ in range(n_repeats)]
                    for params, context in captured]
    finally:
        setattr(type(model), name, orig)

    return samples, reps


def cka(x: np.ndarray, y: np.ndarray, unbiased: bool = True) -> float:
    """Linear Centered Kernel Alignment between two representations.

    x : (n, p) and y : (n, q) -- same n stimuli, ANY p and q. Invariant to orthogonal
    transforms and isotropic scaling, so it compares geometry rather than coordinates.

    ``unbiased`` uses the Song et al. HSIC estimator, which removes the positive bias
    that makes small-sample CKA look artificially high. Biased CKA on n=500 with p=128
    against q=16 would read high almost regardless of the data.

    NOTE the known failure mode: linear CKA is dominated by a few high-variance
    directions, so a representation can score high while differing on everything the
    dominant components do not span. That is why RSA is the primary metric here and
    CKA the cross-check, not the other way round.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    K = x @ x.T
    L = y @ y.T
    if not unbiased:
        hsic = np.sum(K * L)
        return float(hsic / np.sqrt(np.sum(K * K) * np.sum(L * L)))

    def hsic1(A, B):
        n = A.shape[0]
        A = A.copy(); B = B.copy()
        np.fill_diagonal(A, 0.0); np.fill_diagonal(B, 0.0)
        t1 = np.sum(A * B)
        t2 = A.sum() * B.sum() / ((n - 1) * (n - 2))
        t3 = 2.0 / (n - 2) * (A.sum(0) @ B.sum(0))
        return (t1 + t2 - t3) / (n * (n - 3))

    num = hsic1(K, L)
    den = np.sqrt(hsic1(K, K) * hsic1(L, L))
    return float(num / den) if den > 0 else float("nan")


def assemble_input(samples, obs, max_alternatives=4):
    """Per-trial encoder input for ANY family, mirroring Adapter.adapt.

    Kept in step with cogformer/adapter/adapter.py (the `input_data` concatenation).
    Duplicated rather than imported because Adapter.adapt also builds position codes,
    masks and torch tensors this analysis does not need -- but the CHANNEL LAYOUT must
    match exactly, so `test_assemble_matches_adapter` pins it. A divergence here would
    surface as a representational difference and be indistinguishable from a real one.

    Handles the family-specific channels: favored_idx is RDM's boosted accumulator,
    theta is CDM's per-trial heading; both are zero for families that lack them.
    """
    design = samples["design_matrices"]
    rts, choices = obs["rts"], obs["choices"]
    b, t = design.shape[0], design.shape[1]

    num_alt = np.asarray(samples.get("num_alternatives", np.full(b, 2)),
                         dtype=np.float32).reshape(b)
    denom = np.maximum(num_alt - 1.0, 1.0).reshape(b, 1, 1)
    choices_norm = choices / denom
    alt_col = np.repeat(((num_alt - 1.0) / max(max_alternatives - 1, 1)).reshape(b, 1, 1),
                        t, axis=1).astype(np.float32)

    ptc = samples.get("per_trial_context", {}) or {}
    zeros = np.zeros((b, t), dtype=np.float32)
    fav = np.asarray(ptc.get("favored_idx", zeros), dtype=np.float32)
    fav_col = (fav / max(max_alternatives - 1, 1))[..., None]
    theta = np.asarray(ptc.get("theta", zeros), dtype=np.float32)
    theta_col = (theta / np.pi)[..., None]

    return np.concatenate([design, rts, choices_norm, alt_col, fav_col, theta_col],
                          axis=-1).astype(np.float32)
