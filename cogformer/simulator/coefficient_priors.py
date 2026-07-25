"""Block-structured coefficient priors: R^2 variance budget + cell-means contrasts.

Drop-in alternative to :func:`cogformer.simulator.design_utils.sample_parameter_matrix`,
kept separate so it can be A/B'd against the incumbent before a retrain.

Two defects in the incumbent, both stemming from sampling coefficients one row at a
time from a fixed ``N(0, 1)``:

1. **The induced prior on the intrinsic parameters depends on design complexity.**
   Every active row contributes unit variance to the linear predictor, so predictor
   SD grows like ``sqrt(R)`` in the number of active rows: for ``tau`` (intercept
   prior ``N(-1.2, 0.5)``) the 99th percentile runs from 0.66 s at zero slope rows to
   6.3 s at nine. The slope scale also ignores the parameter's own prior scale --
   ``build_intrinsic_priors`` computes ``spec.std()`` and then discards it -- and
   ignores each column's realized variance, so a dummy at p = 0.05 (column SD 0.22)
   and a continuous column (SD 1.0) get the same coefficient prior.

2. **The contrast prior is not exchangeable across categories.** With ``beta_c ~
   N(0,1)`` iid and the reference pinned at 0, ``sd(mu_ref - mu_c) = 1`` but
   ``sd(mu_c - mu_c') = 1.41``. The reference category is a priori closer to every
   other category, for the arbitrary reason that it got index 0 -- so relabelling
   which level is the baseline changes the prior, and hence the posterior.

The fixes here:

**Variance budget.** Per intrinsic parameter, sample the share ``w`` of linear-predictor
variance that is design-driven, set ``tau^2 = sigma_prior^2 * w / (1 - w)``, split it
across active blocks with a Dirichlet, draw each block's raw coefficients, and rescale
so the realized contribution variance hits its share. A final global rescale absorbs
between-block covariance (correlated regressors), so ``Var_i(eta_slopes) == tau^2``
exactly. Predictor scale is then invariant to the number of active rows and tied to the
parameter's own prior scale.

**Cell means.** A categorical block's contrasts are derived from ``K`` exchangeable
category means rather than drawn iid: ``mu_k ~ N(0,1)``, ``beta_c = mu_c - mu_0``. All
pairwise category spreads become equal and the contrasts pick up the ``rho = +0.5``
correlation that encodes "measured against a common, itself-uncertain baseline".
Two-way interactions use the corresponding two-way contrast
``M[i,j] - M[i,0] - M[0,j] + M[0,0]``.

The network's targets are unchanged -- still the contrasts, same rows, same mask, same
collate. Only the distribution they are drawn from changes.

Note that the per-parameter ``"slope"`` samplers in the intrinsic prior are bypassed:
scale is set by the budget. ``"intercept"`` samplers are used as before.
"""

import numpy as np


def _var(x: np.ndarray) -> float:
    """Population variance of a 1-D array.

    ``ndarray.var()`` carries ~4.4 us of dispatch overhead against ~1.9 us here, and the
    budget calls it once per block plus once per parameter on every simulated dataset.
    """
    n = x.shape[0]
    if n == 0:
        return 0.0
    m = x.sum() / n
    return float(x.dot(x) / n - m * m)


def estimate_prior_scales(
    prior: dict[str, dict],
    num_draws: int = 2048,
) -> dict[str, float]:
    """Per-parameter SD of the intercept prior, used as the reference scale for the
    variance budget.

    Call once at setup: drawing these inside the per-dataset sampling loop would cost
    ``num_draws`` extra samples per parameter per dataset.
    """
    scales: dict[str, float] = {}
    for name, spec in prior.items():
        sampler = spec["intercept"] if isinstance(spec, dict) else spec
        if hasattr(sampler, "std"):
            scales[name] = float(sampler.std())
            continue
        draws = np.array([float(sampler()) for _ in range(num_draws)])
        sd = float(draws.std())
        scales[name] = sd if sd > 0 else 1.0
    return scales


def iter_blocks(
    design_config: dict[str, list[str]],
    max_num_categories: int,
) -> list[dict]:
    """Row blocks of the design layout, in the canonical order used everywhere else:
    intercept, then main-effect blocks (``block_width`` rows), then interaction blocks
    (``block_width ** 2`` rows, row-major ``(i, j)``).

    Returns one dict per non-intercept block with its key, start row, width, and -- for
    interactions -- the two side keys.
    """
    block_width = max_num_categories - 1
    keys = [k for k in design_config if k != "1"]
    main_keys = [k for k in keys if ":" not in k]
    inter_keys = [k for k in keys if ":" in k]

    blocks: list[dict] = []
    row = 1 if "1" in design_config else 0
    for key in main_keys:
        blocks.append({"key": key, "kind": "main", "start": row, "width": block_width})
        row += block_width
    for key in inter_keys:
        a, b = key.split(":")
        blocks.append({
            "key": key, "kind": "interaction", "start": row,
            "width": block_width ** 2, "side_a": a, "side_b": b,
        })
        row += block_width ** 2
    return blocks


def _num_categories(key: str, category_counts: dict[str, int] | None,
                    categorical_keys: set[str] | None) -> int | None:
    """Realized category count for a main-effect key, or None if it is continuous.

    With ``categorical_keys`` unset, a width-1 block is read as continuous. That is
    ambiguous only for K = 2 categoricals, and harmlessly so: a K = 2 cell-means draw
    is ``mu_1 - mu_0``, which differs from an iid draw by a scale factor alone -- and
    the budget rescales the block regardless, so the two are identical in distribution.
    """
    if categorical_keys is not None:
        if key not in categorical_keys:
            return None
        width = category_counts.get(key, 1) if category_counts else 1
        return int(width) + 1
    width = category_counts.get(key, 1) if category_counts else 1
    return int(width) + 1 if width > 1 else None


def _cell_mean_contrasts(k: int) -> np.ndarray:
    """``beta_c = mu_c - mu_0`` for ``c = 1..k-1`` with ``mu ~ N(0, 1)^k``.

    Exchangeable across categories: every pairwise spread ``sd(mu_i - mu_j)`` is equal,
    and no category is privileged by being the dropped reference.
    """
    mu = np.random.randn(k)
    return mu[1:] - mu[0]


def _draw_block_raw(
    block: dict,
    block_width: int,
    cat_k: dict[str, int | None],
    cell_means: bool,
) -> np.ndarray:
    """Unscaled coefficients for one block, laid out over its rows.

    Main blocks: slot 0 only for a continuous regressor, ``K-1`` contrasts for a
    categorical one. Interaction blocks: row-major ``(i, j)`` over the two sides'
    contrast slots, so a cat x cat block carries the two-way contrast and a cat x cont
    block carries per-category slope differences.
    """
    raw = np.zeros(block["width"])

    if block["kind"] == "main":
        k = cat_k.get(block["key"])
        if k is None:
            raw[0] = np.random.randn()
        elif cell_means:
            contrasts = _cell_mean_contrasts(k)
            raw[:min(k - 1, block_width)] = contrasts[:block_width]
        else:
            raw[:min(k - 1, block_width)] = np.random.randn(min(k - 1, block_width))
        return raw

    ka = cat_k.get(block["side_a"])
    kb = cat_k.get(block["side_b"])
    grid = np.zeros((block_width, block_width))

    if ka is not None and kb is not None:
        na, nb = min(ka - 1, block_width), min(kb - 1, block_width)
        if cell_means:
            # Two-way cell means: the (i, j) interaction contrast against both margins.
            M = np.random.randn(ka, kb)
            for i in range(na):
                for j in range(nb):
                    grid[i, j] = M[i + 1, j + 1] - M[i + 1, 0] - M[0, j + 1] + M[0, 0]
        else:
            grid[:na, :nb] = np.random.randn(na, nb)
    elif ka is not None or kb is not None:
        # One categorical side: per-category differences in the continuous slope.
        k = ka if ka is not None else kb
        n = min(k - 1, block_width)
        vals = _cell_mean_contrasts(k)[:n] if cell_means else np.random.randn(n)
        if ka is not None:
            grid[:n, 0] = vals
        else:
            grid[0, :n] = vals
    else:
        grid[0, 0] = np.random.randn()

    return grid.reshape(-1)


def sample_parameter_matrix_budget(
    parameter_mask: np.ndarray,
    design_matrix: np.ndarray,
    design_config: dict[str, list[str]],
    intrinsic_params: list[str],
    prior: dict[str, dict],
    max_num_categories: int,
    prior_scales: dict[str, float] | None = None,
    category_counts: dict[str, int] | None = None,
    categorical_keys: set[str] | list[str] | None = None,
    keep_intercept: bool = False,
    design_share: float | None = None,
    design_share_prior: tuple[float, float] = (1.5, 3.0),
    max_design_share: float = 0.9,
    block_concentration: float = 1.0,
    cell_means: bool = True,
) -> np.ndarray:
    """Sample a parameter matrix under a per-parameter variance budget with cell-means
    contrasts.

    Mirrors :func:`design_utils.sample_parameter_matrix` in shape -- returns
    ``(num_rows, num_intrinsic_params)`` -- but needs the design matrix and config to
    work block-wise.

    design_share : ``w``, the fraction of linear-predictor variance that is
        design-driven, so total predictor SD is ``sigma_prior / sqrt(1 - w)``
        regardless of how many rows are active. None -> ``Beta(*design_share_prior)``,
        drawn once per parameter and capped at ``max_design_share``.
    block_concentration : Dirichlet concentration for splitting the budget across
        active blocks. 1.0 is uniform over the simplex.
    cell_means : draw categorical contrasts from exchangeable category means rather
        than iid. Set False to isolate the budget in an A/B.
    prior_scales : per-parameter reference SD; see :func:`estimate_prior_scales`.
        Falls back to estimating on the fly, which is slow in a sampling loop.
    """
    num_rows, num_params = parameter_mask.shape
    out = np.zeros((num_rows, num_params))
    if num_rows == 0:
        return out

    if prior_scales is None:
        prior_scales = estimate_prior_scales(prior, num_draws=256)
    if categorical_keys is not None:
        categorical_keys = set(categorical_keys)

    block_width = max_num_categories - 1
    has_intercept = keep_intercept and "1" in design_config and parameter_mask[0].any()
    blocks = iter_blocks(design_config, max_num_categories)

    cat_k = {
        b["key"]: _num_categories(b["key"], category_counts, categorical_keys)
        for b in blocks if b["kind"] == "main"
    }

    for j, name in enumerate(intrinsic_params):
        col = parameter_mask[:, j]

        if has_intercept and col[0] == 1.0:
            out[0, j] = prior[name]["intercept"]()

        active_blocks = [
            b for b in blocks
            if b["start"] < num_rows and col[b["start"]:b["start"] + b["width"]].any()
        ]
        if not active_blocks:
            continue

        w = design_share
        if w is None:
            w = float(np.random.beta(*design_share_prior))
        w = min(float(w), max_design_share)
        sigma = float(prior_scales.get(name, 1.0))
        tau2 = (sigma ** 2) * w / (1.0 - w) if w > 0 else 0.0
        if tau2 <= 0.0:
            continue

        phi = np.random.dirichlet(block_concentration * np.ones(len(active_blocks)))

        # Stage 1: draw each block and scale it to its share of the budget.
        for b, share in zip(active_blocks, phi):
            rows = np.arange(b["start"], min(b["start"] + b["width"], num_rows))
            raw = _draw_block_raw(b, block_width, cat_k, cell_means)[:rows.shape[0]]
            raw = np.where(col[rows] == 1.0, raw, 0.0)

            contribution = design_matrix[:, rows] @ raw
            var = _var(contribution)
            if var <= 1e-12:
                continue
            out[rows, j] = raw * np.sqrt(share * tau2 / var)

        # Stage 2: renormalize the total. Blocks are correlated whenever the regressors
        # are, so the per-block variances do not sum to tau^2; this makes the realized
        # predictor variance exactly tau^2 while phi keeps setting the relative split.
        slope_rows = np.arange(1 if has_intercept else 0, num_rows)
        total = design_matrix[:, slope_rows] @ out[slope_rows, j]
        total_var = _var(total)
        if total_var > 1e-12:
            out[slope_rows, j] *= np.sqrt(tau2 / total_var)

    return out
