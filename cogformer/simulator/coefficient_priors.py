"""Coefficient priors with one latent Gaussian per category.

Alternative to :func:`cogformer.simulator.design_utils.sample_parameter_matrix`, kept
separate so it can be A/B'd against the incumbent.

The incumbent draws every coefficient row iid from ``N(0, 1)``. With dummy coding and the
reference level pinned at 0 that prior is **not exchangeable across categories**:
``sd(mu_ref - mu_c) = 1`` but ``sd(mu_c - mu_c') = 1.41``. The reference level is a priori
closer to every other level, for the arbitrary reason that it got index 0 -- so
relabelling which level is the baseline changes the prior, and hence the posterior.

**The construction.** A categorical regressor carries one latent value per category,
``mu_k ~ N(0, 1)``, on the same scale as a continuous regressor's single ``N(0, 1)``
coefficient. Dummy coding is the differencing operator on that latent:

    beta_c = mu_c - mu_0

This is the random-utility / latent-variable representation of a factor, and everything
follows from it rather than being chosen separately:

* only differences are identified (K latents, K-1 contrasts), so the reference level is
  a normalization, not a parameter;
* the prior is invariant to relabelling -- all pairwise category spreads are equal;
* contrasts are correlated at ``rho = +0.5`` through the shared ``-mu_0``, which is what
  "measured against a common, itself-uncertain baseline" means;
* a continuous regressor is the same object with a linear latent ``mu(x) = beta * x``,
  so one scale governs both regressor types.

The latent distribution is the probit/logit choice: Gaussian latents give Gaussian
contrasts (implemented here); Gumbel latents would give logistic contrasts, i.e. heavier
tails. Only the Gaussian variant exists at present.

**Nothing is rescaled.** Coefficient scale is independent of category frequency, of the
number of active blocks, and of any realized quantity, so predictor variance grows with
the design exactly as it did under the incumbent. An earlier version of this module
imposed an R^2 variance budget on top -- normalizing blocks by their realized
contribution variance and splitting a fixed total across blocks with a Dirichlet. Both
were mistakes with the same shape: bookkeeping constraints with no generative referent.
The first sent categorical coefficients to ``|beta| = 33`` because ``Var(dummy) = p(1-p)``
goes to zero for a rare category; the second made adding a regressor shrink the effects
of the others, piling half of all slope coefficients below 0.2. Do not reintroduce either
without a generative story that motivates it.

The network's targets are unchanged -- still the contrasts, same rows, same mask, same
collate. Only the distribution they are drawn from changes.
"""

import numpy as np


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
    ambiguous only for K = 2 categoricals, where a cell-means draw ``mu_1 - mu_0`` has
    SD ``sqrt(2)`` against ``1`` for a continuous draw. Nothing rescales that away, so
    pass ``categorical_keys`` whenever the assignment is known -- ``ModelFamily.sample``
    does, via the generator's diagnostics.
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


def sample_parameter_matrix_cell_means(
    parameter_mask: np.ndarray,
    design_config: dict[str, list[str]],
    intrinsic_params: list[str],
    prior: dict[str, dict],
    max_num_categories: int,
    category_counts: dict[str, int] | None = None,
    categorical_keys: set[str] | list[str] | None = None,
    keep_intercept: bool = False,
    cell_means: bool = True,
    order_scale: float = 1.0,
) -> np.ndarray:
    """Sample a parameter matrix with one latent Gaussian per category.

    Same shape contract as :func:`design_utils.sample_parameter_matrix` --
    ``(num_rows, num_intrinsic_params)`` -- and the same ``N(0, 1)`` coefficient scale.
    The only change is the *structure* of a categorical block's draw.

    A categorical regressor carries one latent value per category, ``mu_k ~ N(0, 1)``,
    on the same scale as a continuous regressor's single ``N(0, 1)`` coefficient. Dummy
    coding is the differencing operator on that latent: ``beta_c = mu_c - mu_0``. Only
    differences are identified (K latents, K-1 contrasts), the reference level is an
    arbitrary normalization, and the contrasts inherit ``rho = +0.5`` from the shared
    ``-mu_0``. Two-way interactions use the corresponding two-way contrast
    ``M[i,j] - M[i,0] - M[0,j] + M[0,0]``.

    Nothing is rescaled. Coefficient scale does not depend on category frequency, on the
    number of active blocks, or on any realized quantity, so predictor variance grows
    with the design exactly as it did under the incumbent per-row ``N(0, 1)``.

    cell_means : set False to draw contrasts iid instead, recovering the incumbent's
        non-exchangeable prior (reference level a priori closer to every other level).
    order_scale : effect-hierarchy multiplier ``lambda``. A block of interaction order
        ``d`` (1 for main effects and continuous slopes, 2 for two-way interactions) is
        scaled by ``lambda ** (d - 1)``. Default 1.0 imposes no hierarchy.

        Hierarchy does NOT fall out of "one Gaussian per category" -- it is an extra
        assumption and has to be asked for. With iid cell means each differencing step
        adds variance (continuous sd 1, main-effect contrast sd sqrt(2), two-way contrast
        sd 2), and the interaction subspace has ``(K-1)^2`` dimensions against ``K-1``
        for a margin, so at r4c4 interactions carry ~69% of design-driven predictor
        variance against ~31% for main effects. That inverts the usual effect-heredity
        expectation. Reference points at K = 4:

          lambda = 1.000  interactions 69% of design variance (no hierarchy, default)
          lambda = 0.816  interaction and main-effect contributions equal
          lambda = 0.577  interaction contribution half of main effects
          lambda = 0.500  interaction contrast sd 1.0, below the main-effect sd 1.41
    """
    num_rows, num_params = parameter_mask.shape
    out = np.zeros((num_rows, num_params))
    if num_rows == 0:
        return out

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

        for b in blocks:
            if b["start"] >= num_rows:
                continue
            rows = np.arange(b["start"], min(b["start"] + b["width"], num_rows))
            if not col[rows].any():
                continue
            raw = _draw_block_raw(b, block_width, cat_k, cell_means)[:rows.shape[0]]
            if order_scale != 1.0 and b["kind"] == "interaction":
                raw = raw * order_scale
            out[rows, j] = np.where(col[rows] == 1.0, raw, 0.0)

    return out
