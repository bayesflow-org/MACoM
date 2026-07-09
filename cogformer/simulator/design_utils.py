import numpy as np
from itertools import combinations
from typing import Callable
from scipy.stats import invwishart

from cogformer.utils.simulator_utils import shifted_softplus


def design_config_signature(design_config: dict) -> frozenset:
    """Category-agnostic structural hash of a design config.

    Two configs with the same parameter assignments per regressor key produce
    the same signature, regardless of max_num_categories.  Used to reject
    holdout configs during random design sampling.
    """
    return frozenset((k, frozenset(v)) for k, v in design_config.items())


def split_design_space(
    free_intrinsics: list[str],
    split_ratio: float = 0.5,
    seed: int | None = None,
    include_empty: bool = True,
) -> tuple[list[frozenset[str]], list[frozenset[str]]]:
    """
    Partition the space of per-regressor slope patterns into two disjoint sets.

    Every design config produced with ``allowed_slope_patterns=set_a`` will
    use only slope patterns drawn from set A, and vice-versa for set B.
    The two sets are guaranteed to be disjoint, enabling held-out design
    evaluation (train on A, test on B).

    Parameters
    ----------
    free_intrinsics : list[str]
        Parameters eligible to appear as slopes.
    split_ratio : float
        Fraction of patterns assigned to set A. Default 0.5.
    seed : int or None
        Random seed for reproducible splits.
    include_empty : bool
        Whether to include the empty pattern (a regressor with no slopes).
        Default True.

    Returns
    -------
    set_a, set_b : tuple of list[frozenset[str]]
        Two disjoint lists of slope patterns. Pass either as
        ``allowed_slope_patterns`` to ``build_random_design_config`` or
        ``ModelFamily.batch_sample``.

    Examples
    --------
    set_a, set_b = split_design_space(["v", "a", "z", "tau"], seed=0)
    train_batch = mf.batch_sample(..., allowed_slope_patterns=set_a)
    test_batch  = mf.batch_sample(..., allowed_slope_patterns=set_b)
    """
    rng = np.random.default_rng(seed)
    n = len(free_intrinsics)
    start = 0 if include_empty else 1

    all_patterns = [
        frozenset(combo)
        for r in range(start, n + 1)
        for combo in combinations(free_intrinsics, r)
    ]

    order = rng.permutation(len(all_patterns))
    n_a = round(len(all_patterns) * split_ratio)

    set_a = [all_patterns[i] for i in order[:n_a]]
    set_b = [all_patterns[i] for i in order[n_a:]]
    return set_a, set_b


def build_design_config(
    intrinsic_params: list[str],
    regressed_params: list[str] = None,
    num_regressors: int = 2,
    keep_intercept: bool = False,
    add_interaction: bool = False,
) -> dict[str, list[str]]:
    config: dict[str, list[str]] = {}

    if regressed_params is None:
        regressed = list(intrinsic_params)
    else:
        regressed_set = set(regressed_params)
        regressed = [p for p in intrinsic_params if p in regressed_set]

    if keep_intercept:
        config["1"] = list(intrinsic_params)

    main_keys: list[str] = []
    for r in range(num_regressors):
        key = f"u_{r + 1}"
        config[key] = list(regressed)
        main_keys.append(key)

    if add_interaction:
        for i in range(len(main_keys)):
            for j in range(i + 1, len(main_keys)):
                a, b = main_keys[i], main_keys[j]
                shared_set = set(config[a]) & set(config[b])
                if shared_set:
                    shared = [p for p in intrinsic_params if p in shared_set]
                    config[f"{a}:{b}"] = shared

    return config


def build_random_design_config(
    intrinsic_params: list[str],
    num_regressors: int,
    free_intrinsics: list[str] | set[str] | None = None,
    fixed_intrinsics: list[str] | set[str] | None = None,
    keep_intercept: bool = False,
    free_prob: float | None = None,
    intercept_prob: float | None = None,
    add_interaction: bool = False,
    allowed_slope_patterns: list[frozenset[str]] | None = None,
    global_params: list[str] | set[str] | None = None,
) -> dict[str, list[str]]:
    """
    Randomly build a design_config mapping regressors and optional intercept
    to subsets of intrinsic parameters.

    Notes
    -----
    - The intercept row (key "1") is randomized when `keep_intercept=True`.
    - Slopes are only allowed for intrinsic parameters which have an intercept entry.
    - Fixed intrinsics are always included in the intercept and never appear in slope terms.
    - Global params are always included in the intercept and never appear in slope terms.
      Unlike fixed intrinsics, their intercept values are sampled from their prior.
    - When `free_prob` or `intercept_prob` is None, each is sampled independently
      from Beta(2, 2), covering the full [0, 1] range while concentrating probability
      near the centre to expose the model to the full design space during training.
    """
    if free_prob is None:
        free_prob = float(np.random.beta(2, 2))
    if intercept_prob is None:
        intercept_prob = float(np.random.beta(2, 2))

    free_set = None if free_intrinsics is None else set(free_intrinsics)
    fixed_set: set[str] = set() if fixed_intrinsics is None else set(fixed_intrinsics)
    global_set: set[str] = set() if global_params is None else set(global_params)

    config = {}

    intercept_names: list[str] = []
    if keep_intercept or global_set:
        for p in intrinsic_params:
            if p in global_set:
                intercept_names.append(p)
                continue
            if free_set is not None and p in free_set:
                intercept_names.append(p)
                continue
            if p in fixed_set:
                if np.random.rand() < intercept_prob:
                    intercept_names.append(p)
                continue
            if free_set is not None:
                continue
            if np.random.rand() < intercept_prob:
                intercept_names.append(p)

        config["1"] = intercept_names

    allowed_slope_params = set(intercept_names) if keep_intercept else set(intrinsic_params)
    allowed_slope_params -= global_set

    main_keys = []
    for r in range(num_regressors):
        key = f"u_{r + 1}"
        if allowed_slope_patterns is not None:
            pattern = allowed_slope_patterns[np.random.randint(len(allowed_slope_patterns))]
            names = [p for p in intrinsic_params if p in pattern and p in allowed_slope_params]
        else:
            names = [
                param for param in intrinsic_params
                if param in allowed_slope_params and np.random.rand() < free_prob
            ]
        config[key] = names
        main_keys.append(key)

    if add_interaction:
        for i in range(len(main_keys)):
            for j in range(i + 1, len(main_keys)):
                a, b = main_keys[i], main_keys[j]
                shared_set = set(config[a]) & set(config[b])
                if shared_set:
                    shared = [p for p in intrinsic_params if p in shared_set]
                    config[f"{a}:{b}"] = shared

    return config


def build_parameter_mask(
    design_config: dict[str, list[str]],
    intrinsic_params: list[str],
    fixed_params: list[str] = None,
    max_num_categories: int = 3,
    keep_intercept: bool = False,
    category_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Convert a design_config dict into a binary parameter mask mapping
    regressors (rows) to intrinsic parameters (columns).

    Main-effect blocks occupy block_width rows each.
    Interaction blocks occupy block_width² rows each (one per cross-product of dummy columns).

    category_counts : optional dict mapping each main-effect key to the number of
        design-matrix columns that were actually filled (1 for continuous, K-1 for
        a K-category discrete regressor).  When provided, only those rows are activated;
        the remaining padding rows (for unused category levels) stay zero.  This prevents
        the model from being trained to recover unidentifiable parameters.
    """
    regressor_keys = list(design_config.keys())
    has_intercept = "1" in regressor_keys
    regressor_keys = [k for k in regressor_keys if k != "1"]

    main_keys = [k for k in regressor_keys if ":" not in k]
    inter_keys = [k for k in regressor_keys if ":" in k]

    block_width = max_num_categories - 1
    inter_block_width = block_width ** 2
    num_intrinsic_params = len(intrinsic_params)
    num_cols = (
        len(main_keys) * block_width
        + len(inter_keys) * inter_block_width
        + (1 if has_intercept else 0)
    )

    mask = np.zeros((num_cols, num_intrinsic_params), dtype=np.float32)
    fixed_set = set(fixed_params) if fixed_params is not None else set()

    row_idx = 0
    if has_intercept:
        for name in design_config.get("1", []):
            if name in intrinsic_params and name not in fixed_set:
                mask[row_idx, intrinsic_params.index(name)] = 1.0
        row_idx = 1

    for key in main_keys:
        active = min(category_counts[key], block_width) if category_counts else block_width
        for intrinsic in design_config[key]:
            if intrinsic in intrinsic_params and intrinsic not in fixed_set:
                param_index = intrinsic_params.index(intrinsic)
                for r in range(row_idx, row_idx + active):
                    mask[r, param_index] = 1.0
        row_idx += block_width

    for key in inter_keys:
        if category_counts:
            a, b = key.split(":")
            count_a = min(category_counts.get(a, 1), block_width)
            count_b = min(category_counts.get(b, 1), block_width)
        else:
            count_a = count_b = block_width
        # Identifiable interaction rows form the sub-grid {(i, j): i < count_a, j < count_b}
        # of the row-major (i * block_width + j) flattening — non-contiguous whenever
        # count_b < block_width, so a contiguous range would activate zero-column rows
        # while masking real cross-products.
        for intrinsic in design_config[key]:
            if intrinsic in intrinsic_params and intrinsic not in fixed_set:
                param_index = intrinsic_params.index(intrinsic)
                for i in range(count_a):
                    for j in range(count_b):
                        mask[row_idx + i * block_width + j, param_index] = 1.0
        row_idx += inter_block_width

    return mask


def build_position_codes(
    design_config: dict[str, list[str]],
    max_num_categories: int = 3,
) -> np.ndarray:
    """
    Build factorized identity codes for each design row, aligned with the row layout
    of build_design_matrix / build_parameter_mask (intercept, then main-effect blocks
    in config order, then interaction blocks in config order).

    Each row gets an int64 code (reg_a, reg_b, slot_a, slot_b):
      intercept                     : (0, 0, 0, 0)
      main effect u_r, block col c  : (r, 0, c + 1, 0)      — slot 1 = first design column
                                      (category-2 contrast for discrete, the value column
                                      for continuous regressors)
      interaction u_a:u_b at (i, j) : (a, b, i + 1, j + 1)  — row-major i * block_width + j
    Regressor ids are 1-based positions in main-key order; 0 everywhere means "none",
    so padding rows can safely carry the all-zero code.

    Returns shape (num_rows, 4).
    """
    block_width = max_num_categories - 1
    regressor_keys = [k for k in design_config if k != "1"]
    main_keys = [k for k in regressor_keys if ":" not in k]
    inter_keys = [k for k in regressor_keys if ":" in k]
    main_id = {k: i + 1 for i, k in enumerate(main_keys)}

    rows: list[tuple[int, int, int, int]] = []
    if "1" in design_config:
        rows.append((0, 0, 0, 0))
    for key in main_keys:
        r = main_id[key]
        rows.extend((r, 0, c, 0) for c in range(1, block_width + 1))
    for key in inter_keys:
        a, b = key.split(":")
        reg_a, reg_b = main_id.get(a, 0), main_id.get(b, 0)
        rows.extend(
            (reg_a, reg_b, i, j)
            for i in range(1, block_width + 1)
            for j in range(1, block_width + 1)
        )

    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


def normalize_design_config(
    design_config: dict,
    max_num_regressors: int,
) -> dict:
    """
    Expand a sparse design_config to canonical form by filling in missing main-effect
    regressor keys (u_1 … u_max_num_regressors) with empty lists.

    This ensures correct positional alignment in the design matrix: the model always
    expects column block i to correspond to u_i, so absent regressors must be present
    as empty entries rather than omitted entirely.  Existing keys (including interactions)
    are preserved as-is.
    """
    out = {}
    if "1" in design_config:
        out["1"] = design_config["1"]
    for r in range(1, max_num_regressors + 1):
        out[f"u_{r}"] = design_config.get(f"u_{r}", [])
    for key, val in design_config.items():
        if ":" in key:
            out[key] = val
    return out


def expand_design_config_to_mask(
    design_config: dict,
    parameter_mask: np.ndarray,
    max_num_categories: int,
) -> dict:
    """
    Infer the number of main-effect regressor blocks from parameter_mask.shape[0] and
    expand a sparse design_config to include all of them as empty keys where absent.

    Used in diagnostic functions that receive the original (possibly sparse) config but
    need label generation to match the mask that was built from the normalized config.
    """
    block_width = max_num_categories - 1
    inter_block_width = block_width ** 2
    has_intercept = "1" in design_config
    inter_keys = [k for k in design_config if ":" in k]
    main_rows = parameter_mask.shape[0] - (1 if has_intercept else 0) - len(inter_keys) * inter_block_width
    num_main_blocks = max(main_rows // block_width, 0)
    out = {}
    if has_intercept:
        out["1"] = design_config.get("1", [])
    for r in range(1, num_main_blocks + 1):
        out[f"u_{r}"] = design_config.get(f"u_{r}", [])
    for key in inter_keys:
        out[key] = design_config[key]
    return out


def build_random_parameter_mask(
    intrinsic_params: list[str],
    num_regressors: int = 2,
    max_num_categories: int = 4,
    free_intrinsics: list[str] | set[str] | None = None,
    fixed_intrinsics: list[str] | set[str] | None = None,
    free_prob: float | None = None,
    intercept_prob: float | None = None,
    keep_intercept: bool = False,
    add_interaction: bool = False,
    allowed_slope_patterns: list[frozenset[str]] | None = None,
    global_params: list[str] | set[str] | None = None,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    design_config = build_random_design_config(
        intrinsic_params=intrinsic_params,
        num_regressors=num_regressors,
        free_prob=free_prob,
        intercept_prob=intercept_prob,
        keep_intercept=keep_intercept,
        free_intrinsics=free_intrinsics,
        fixed_intrinsics=fixed_intrinsics,
        add_interaction=add_interaction,
        allowed_slope_patterns=allowed_slope_patterns,
        global_params=global_params,
    )

    parameter_mask = build_parameter_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        max_num_categories=max_num_categories,
        keep_intercept=keep_intercept,
    )

    return parameter_mask, design_config


def build_intrinsic_priors(
    prior: Callable | dict,
    design_config: dict[str, list[str]],
    fixed_values: dict[str, float] | None = None,
) -> dict[str, dict[str, Callable]]:
    if fixed_values is None:
        fixed_values = {}

    free_intrinsics = design_config.get("1", [])
    fixed_intrinsics = list(set(prior.keys()) - set(free_intrinsics))

    priors = {}
    for k, spec in prior.items():
        if k in free_intrinsics:
            if hasattr(spec, "rvs"):
                scale = spec.std()
                sampler = lambda rv=spec: rv.rvs()
                priors[k] = {
                    "intercept": sampler,
                    "slope": lambda std=scale: np.random.normal(0.0, 1.0),
                }
            else:
                priors[k] = {
                    "intercept": spec,
                    "slope": lambda: np.random.normal(0.0, 1.0),
                }
        elif k in fixed_intrinsics:
            fixed_value = fixed_values.get(k, 0.0)
            priors[k] = {
                "intercept": lambda v=fixed_value: v,
                "slope": lambda: 0.0,
            }

    return priors


def build_discrete_mask(
    design_config: dict[str, list[str]],
    discrete_prob: float = 0.5,
    keep_intercept: bool = False,
    add_interaction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a discrete mask for main regressors only and expand it to a total-length
    mask aligned with design_config key order (excluding intercept).
    """
    regressor_keys = [k for k in design_config.keys() if k != "1"]
    main_keys = [k for k in regressor_keys if ":" not in k]

    main_discrete_mask = build_random_discrete_mask(
        num_regressors=len(main_keys),
        discrete_prob=discrete_prob,
    ).astype(np.float32, copy=False)

    discrete_mask = np.full((len(regressor_keys),), -1.0, dtype=np.float32)
    ptr = 0
    for i, k in enumerate(regressor_keys):
        if ":" in k:
            continue
        discrete_mask[i] = float(main_discrete_mask[ptr])
        ptr += 1

    return main_discrete_mask, discrete_mask


def build_random_discrete_mask(
    num_regressors: int,
    discrete_prob: float = 0.5,
) -> np.ndarray:
    """Bernoulli mask over regressors: 1 -> discrete {0,1}, 0 -> continuous U[0,1]."""
    if num_regressors == 0:
        return np.array([])
    return np.array(1 * (np.random.rand(num_regressors) < discrete_prob))


def build_regressor_mask(
    design_config: dict[str, list[str]],
    max_num_categories: int = 4,
    keep_intercept: bool = False,
) -> np.ndarray:
    block_width = max_num_categories - 1
    inter_block_width = block_width ** 2
    regressor_keys = [k for k in design_config if k != "1"]
    n_main = sum(1 for k in regressor_keys if ":" not in k)
    n_inter = sum(1 for k in regressor_keys if ":" in k)
    num_cols = n_main * block_width + n_inter * inter_block_width + (1 if keep_intercept else 0)
    return np.ones(num_cols, dtype=np.float32)


def build_design_matrix(
    design_config: dict[str, list[str]],
    num_obs: int,
    context: dict[str, np.ndarray] | None = None,
    discrete_prob: float = 0.5,
    min_num_categories: int = 2,
    max_num_categories: int = 4,
    main_discrete_mask: np.ndarray = None,
    collinearity_df: float | None = None,
    category_concentration: float | None = None,
    return_category_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    """
    Build design matrix based on the respective configuration of free intrinsic parameters.

    All regressors (not context vars) are drawn jointly from X ~ MVN(0, Sigma),
    Sigma ~ Inv-Wishart(collinearity_df, I_k), giving a shared latent covariance structure
    across both continuous and discrete regressors.

    Continuous columns are used directly (standardized to unit variance).
    Discrete columns are discretized via Dirichlet-sampled quantile cuts:
      p ~ Dirichlet(category_concentration * 1_k), then z is cut at quantiles implied by cumsum(p).
    """
    context = context or {}
    regressor_keys = list(design_config.keys())
    has_intercept = "1" in regressor_keys
    regressor_keys = [k for k in regressor_keys if k != "1"]

    main_effect_keys = [k for k in regressor_keys if ":" not in k]
    interaction_keys = [k for k in regressor_keys if ":" in k]

    block_width = max_num_categories - 1
    inter_block_width = block_width ** 2
    n_main = len(main_effect_keys)
    n_inter = len(interaction_keys)
    num_cols = n_main * block_width + n_inter * inter_block_width + (1 if has_intercept else 0)
    design_matrix = np.zeros((num_obs, num_cols))

    if main_discrete_mask is None:
        main_discrete_mask = build_random_discrete_mask(
            num_regressors=len(main_effect_keys),
            discrete_prob=discrete_prob,
        )

    non_context_indices = [
        j for j, key in enumerate(main_effect_keys) if key not in context
    ]
    k = len(non_context_indices)
    if k > 0:
        collinearity_df_val = float(collinearity_df) if collinearity_df is not None else np.random.uniform(k + 1, 4 * k + 1)
        Sigma = invwishart.rvs(df=collinearity_df_val, scale=np.eye(k))
        Sigma = np.atleast_2d(Sigma).reshape(k, k)
        X_mvn = np.random.multivariate_normal(np.zeros(k), Sigma, size=num_obs)
        X_mvn /= np.sqrt(np.diag(Sigma))[np.newaxis, :]
        mvn_col_map = {j: X_mvn[:, i].astype(np.float32) for i, j in enumerate(non_context_indices)}
    else:
        mvn_col_map = {}

    col_idx = 0
    actual_col_counts: dict[str, int] = {}

    if has_intercept:
        design_matrix[:, 0] = 1.0
        col_idx = 1

    # Cumulative start positions: main effects get block_width cols, interactions get block_width² cols.
    start_col = {}
    cur = col_idx
    for k in main_effect_keys:
        start_col[k] = cur
        cur += block_width
    for k in interaction_keys:
        start_col[k] = cur
        cur += inter_block_width

    for j, key in enumerate(main_effect_keys):
        start = start_col[key]

        if key in context:
            col = np.asarray(context[key], dtype=np.float32).reshape(-1)
            if col.shape[0] != num_obs:
                raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")
            design_matrix[:, start] = col
            actual_col_counts[key] = 1
            continue

        if main_discrete_mask[j] == 1:
            num_categories = np.random.randint(min_num_categories, max_num_categories + 1)
            dummies = _discretize_column(mvn_col_map[j], num_categories, category_concentration)
            design_matrix[:, start:(start + dummies.shape[1])] = dummies
            actual_col_counts[key] = dummies.shape[1]  # = num_categories - 1
        else:
            design_matrix[:, start] = mvn_col_map[j]
            actual_col_counts[key] = 1

    for key in interaction_keys:
        start = start_col[key]

        if key in context:
            col = np.asarray(context[key], dtype=np.float32).reshape(-1)
            if col.shape[0] != num_obs:
                raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")
            design_matrix[:, start] = col
            continue

        try:
            a, b = key.split(":")
        except ValueError:
            continue

        a0 = start_col.get(a)
        b0 = start_col.get(b)
        if a0 is None or b0 is None:
            continue

        # Full outer product of dummy-code blocks: (num_obs, bw) x (num_obs, bw) → (num_obs, bw²)
        a_cols = design_matrix[:, a0:a0 + block_width]
        b_cols = design_matrix[:, b0:b0 + block_width]
        interaction = (a_cols[:, :, None] * b_cols[:, None, :]).reshape(num_obs, inter_block_width)
        design_matrix[:, start:start + inter_block_width] = interaction

    if return_category_counts:
        return design_matrix, actual_col_counts
    return design_matrix


def build_link_functions(
    intrinsic_params: list[str],
    link_fun: Callable | dict | None,
    default_link_fun: Callable | None = None,
) -> dict[str, Callable]:
    """Build a per-intrinsic-parameter link-function mapping."""
    if link_fun is None:
        return {k: shifted_softplus for k in intrinsic_params}

    if callable(link_fun):
        return {k: link_fun for k in intrinsic_params}

    if not isinstance(link_fun, dict):
        raise TypeError(
            f"link_fun must be a callable, a dict[str, callable], or None; got {type(link_fun)}"
        )

    unknown = [k for k in link_fun if k not in intrinsic_params]
    if unknown:
        raise KeyError("Unknown link_fun keys not in intrinsic_params: " + ", ".join(map(str, unknown)))

    missing = [k for k in intrinsic_params if k not in link_fun]
    if missing:
        if default_link_fun is None:
            raise KeyError("Missing link_fun for intrinsic_params: " + ", ".join(map(str, missing)))
        return {k: link_fun.get(k, default_link_fun) for k in intrinsic_params}

    return {k: link_fun[k] for k in intrinsic_params}


def mask_to_design_config(
    parameter_mask: np.ndarray,
    intrinsic_params: list[str],
    keep_intercept: bool = False,
    ignore_padding: bool = True,
) -> dict[str, list[str]]:
    """Convert a (num_regressors × num_intrinsics) binary mask into a design_config dict."""
    num_rows, _ = parameter_mask.shape
    config: dict[str, list[str]] = {}

    for i in range(num_rows):
        if ignore_padding and not parameter_mask[i].any():
            continue
        key = "1" if (i == 0 and keep_intercept) else f"u_{i + 1}"
        config[key] = [
            intrinsic_params[j]
            for j in range(len(intrinsic_params))
            if parameter_mask[i, j] == 1.0
        ]

    return config


def sample_parameter_matrix(
    parameter_mask: np.ndarray,
    intrinsic_params: list[str],
    prior: dict[str, Callable | dict[str, Callable]],
    keep_intercept: bool = False,
) -> np.ndarray:
    """
    Sample parameter matrix based on the given parameter mask and the associated priors.
    Returns shape (num_regressors, num_intrinsic_params).
    """
    num_regressors, num_intrinsic_params = parameter_mask.shape
    parameter_matrix = np.zeros((num_regressors, num_intrinsic_params))

    if num_regressors == 0:
        return parameter_matrix

    has_intercept = keep_intercept and parameter_mask[0].any()

    for design_index in range(num_regressors):
        for param_index, intrinsic_param in enumerate(intrinsic_params):
            if parameter_mask[design_index, param_index] == 1.0:
                if design_index == 0 and has_intercept:
                    sampler = prior[intrinsic_param]["intercept"]
                else:
                    sampler = prior[intrinsic_param]["slope"]
                parameter_matrix[design_index, param_index] = sampler()

    return parameter_matrix


def build_column_labels(
    design_config: dict[str, list[str]],
    max_num_categories: int = 4,
) -> list[str]:
    """Build human-readable column labels aligned with build_design_matrix() output."""
    regressor_keys = list(design_config.keys())
    has_intercept = "1" in regressor_keys
    regressor_keys = [k for k in regressor_keys if k != "1"]
    main_keys = [k for k in regressor_keys if ":" not in k]
    inter_keys = [k for k in regressor_keys if ":" in k]

    block_width = max_num_categories - 1
    labels: list[str] = []

    if has_intercept:
        labels.append("1")

    for key in main_keys:
        labels.append(key)
        for j in range(1, block_width):
            labels.append(f"{key}[{j+1}]")

    for key in inter_keys:
        for i in range(block_width):
            for j in range(block_width):
                idx = i * block_width + j
                labels.append(key if idx == 0 else f"{key}[{i+1},{j+1}]")

    return labels


def _discretize_column(
    z: np.ndarray,
    k: int,
    category_concentration: float | None = None,
) -> np.ndarray:
    """
    Discretize a latent continuous column z into k dummy-coded categories.

    Category probabilities are sampled from Dirichlet(category_concentration * 1_k),
    then used as quantile boundaries on z.
    Returns an (n, k-1) float32 array of dummy variables (reference category dropped).
    """
    concentration = float(category_concentration) if category_concentration is not None else np.random.uniform(0.1, 2.0)
    p = np.random.dirichlet(concentration * np.ones(k))
    cuts = np.clip(np.cumsum(p)[:-1], 0.0, 1.0)
    boundaries = np.quantile(z, cuts)
    bin_ids = np.digitize(z, boundaries)
    one_hot = (bin_ids[:, None] == np.arange(k)[None, :]).astype(np.float32)
    return one_hot[:, 1:]
