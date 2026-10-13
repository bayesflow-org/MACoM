import numpy as np
from itertools import combinations
from typing import Callable

from cogformer.utils.simulator_utils import shifted_softplus, sample_lkj_correlation


DESIGN_GENERATORS = ("mixture", "quantile", "gaussian_softmax")


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
    - Fixed intrinsics enter the intercept with probability `intercept_prob`, and once
      there they are slope-eligible like any other non-global parameter (slopes are
      drawn from the intercept set minus globals).
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


def sample_covering_design_configs(
    n_configs: int,
    intrinsic_params: list[str],
    num_regressors: int,
    free_intrinsics: list[str] | None = None,
    fixed_intrinsics: list[str] | None = None,
    keep_intercept: bool = True,
    add_interaction: bool = False,
    free_prob: float | None = None,
    intercept_prob: float | None = None,
    global_params: list[str] | set[str] | None = None,
    max_attempts: int = 20000,
    rng_seed: int | None = None,
) -> list[dict[str, list[str]]]:
    """``n_configs`` random design configs whose UNION covers every estimable cell.

    The ensemble grid keeps a row if ANY config activates it, so a cell renders N/A
    whenever one config opens a row no other config covers. The previous fix made the
    last config of each family a deterministic saturated config, which left the
    ensemble a mixture of random and constructed designs -- awkward to describe and
    not a random sample of anything.

    This draws the whole SET and rejects it unless the union covers, so every config
    stays a genuine random draw and the set is exactly "a uniform sample from the
    covering sets". Rejection is nearly free: config construction is pure
    combinatorics (~0.1 ms) and only the accepted set is ever simulated.

    Measured at the ensemble's settings (4 configs/family, 2 regressors, interactions
    on, K=2), acceptance is ~1.2% per family, i.e. ~80 draws, a few milliseconds.

    Raising ``free_prob`` instead does NOT work: coverage caps near 71% however high
    it goes, because a slope is only legal for a parameter that has an intercept
    entry, so ``intercept_prob`` becomes the binding constraint. Raising both to 1.0
    reaches 100% only by making every config the saturated one.

    ESTIMABLE CELLS
    ---------------
    The intercept row can carry any parameter. Slope rows can carry only free,
    non-global parameters: ``build_random_design_config`` never places a fixed
    intrinsic or a global (e.g. ``pi``) in a slope term, so those slope cells are
    structurally unreachable and are excluded from the coverage requirement.
    Demanding them would make coverage impossible at any setting.

    Coverage is evaluated per ``(design-config key, parameter)`` rather than per mask
    row. A config's parameter set is identical across every contrast slot of a block,
    so for the K=2 case the ensemble runs these are equivalent.

    Raises
    ------
    RuntimeError
        If no covering set is found within ``max_attempts``. Silently returning a
        non-covering set would reintroduce N/A cells with no signal that it happened.
    """
    global_set = set(global_params or ())

    # Slope-legal is every NON-GLOBAL parameter, fixed intrinsics included. The
    # docstring of build_random_design_config claims fixed intrinsics never take
    # slopes, but the code computes
    #     allowed_slope_params = set(intercept_names) - global_set
    # so a fixed intrinsic that lands in the intercept (probability intercept_prob)
    # IS slope-eligible. Excluding them here would under-demand coverage and leave
    # exactly the N/A cells this function exists to prevent.
    slope_legal = [p for p in intrinsic_params if p not in global_set]
    intercept_legal = list(intrinsic_params)

    if rng_seed is not None:
        np.random.seed(rng_seed)

    def covers(configs: list[dict]) -> bool:
        keys = set()
        for c in configs:
            keys.update(c.keys())
        for key in keys:
            legal = intercept_legal if key == "1" else slope_legal
            union = set()
            for c in configs:
                union.update(c.get(key, []))
            if any(p not in union for p in legal):
                return False
        return True

    for attempt in range(1, max_attempts + 1):
        configs = [
            build_random_design_config(
                intrinsic_params=intrinsic_params,
                num_regressors=num_regressors,
                free_intrinsics=free_intrinsics,
                fixed_intrinsics=fixed_intrinsics,
                keep_intercept=keep_intercept,
                add_interaction=add_interaction,
                free_prob=free_prob,
                intercept_prob=intercept_prob,
                global_params=global_params,
            )
            for _ in range(n_configs)
        ]
        if covers(configs):
            return configs

    raise RuntimeError(
        f"No covering set of {n_configs} design configs found in {max_attempts} "
        f"attempts (free_prob={free_prob}, intercept_prob={intercept_prob}). Either "
        "raise max_attempts, raise n_configs, or loosen the coverage requirement."
    )


def build_row_labels(
    design_config: dict[str, list[str]],
    max_num_categories: int = 3,
    math: bool = True,
    return_kinds: bool = False,
    categorical_regressors: list[str] | set[str] | None = None,
    parameter_mask: np.ndarray | None = None,
) -> list[str] | tuple[list[str], list[str]]:
    """Human-readable labels for each design-matrix ROW (coefficient), aligned with
    the row order of :func:`build_parameter_mask` / :func:`build_position_codes`:
    intercept, then main-effect blocks (``block_width`` rows each) in config order,
    then interaction blocks (``block_width ** 2`` rows, row-major ``(i, j)``) in
    config order.

    Label format (contrast indices are 1-based, matching :func:`build_column_labels`):

      - intercept                                   -> ``1``
      - continuous main effect ``u_r``              -> ``u_r``
      - categorical main effect ``u_r``, contrast c -> ``u_r:c_c``
      - interaction ``u_a:u_b``, slot ``(i, j)``    -> ``u_a × u_b`` where each part
        carries its own contrast index when that regressor is categorical, e.g.
        ``u_a × u_b:c_j`` (cont × cat) or ``u_a:c_i × u_b:c_j`` (cat × cat).

    A continuous regressor only ever occupies the first slot of its block; the
    remaining (padding) slots repeat the same label so indices stay row-aligned.

    Args:
        math         : wrap labels in ``$…$`` for LaTeX rendering (matplotlib mathtext).
        return_kinds : also return a parallel list of row kinds, each one of
                       ``"intercept"``, ``"main"``, ``"interaction"`` — convenient for
                       mapping rows to per-kind colors.
        categorical_regressors : main-effect keys (e.g. ``["u_1"]``) that are
                       contrast-coded. Takes precedence over mask inference. Dummy
                       coding drops the base category ``c_0``, so a categorical
                       regressor ALWAYS carries a contrast suffix — a K=2 categorical
                       is labeled ``u_r:c_1``, never bare ``u_r``.
        parameter_mask : row-aligned mask used to infer categorical regressors when
                       ``categorical_regressors`` is None — a block with an active row
                       beyond its first slot must be categorical. A K=2 categorical
                       occupies only slot 1 and is indistinguishable from a continuous
                       regressor this way, so pass ``categorical_regressors`` (e.g.
                       from scenario opts or the sampler's ``discrete_masks``) whenever
                       the assignment is known.
    """
    block_width = max_num_categories - 1
    keys = [k for k in design_config if k != "1"]
    main_keys = [k for k in keys if ":" not in k]
    inter_keys = [k for k in keys if ":" in k]

    if categorical_regressors is not None:
        cat_set = set(categorical_regressors)
    elif parameter_mask is not None and block_width > 1:
        cat_set = set()
        row0 = 1 if "1" in design_config else 0
        for m_i, key in enumerate(main_keys):
            block = parameter_mask[row0 + m_i * block_width: row0 + (m_i + 1) * block_width]
            if block.shape[0] > 1 and np.any(block[1:] == 1.0):
                cat_set.add(key)
    else:
        cat_set = set()

    def wrap(body: str) -> str:
        return f"${body}$" if math else body

    def part(key: str, contrast: int) -> str:
        return f"{key}:c_{contrast}" if key in cat_set else key

    labels: list[str] = []
    kinds: list[str] = []
    if "1" in design_config:
        labels.append(wrap("1"))
        kinds.append("intercept")
    for key in main_keys:
        for c in range(1, block_width + 1):
            labels.append(wrap(part(key, c)))
            kinds.append("main")
    for key in inter_keys:
        a, b = key.split(":")
        times = r" \times " if math else " × "
        for i in range(1, block_width + 1):
            for j in range(1, block_width + 1):
                labels.append(wrap(part(a, i) + times + part(b, j)))
                kinds.append("interaction")

    return (labels, kinds) if return_kinds else labels


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
    max_num_regressors: int | None = None,
) -> dict:
    """
    Infer the number of main-effect regressor blocks from parameter_mask.shape[0] and
    expand a sparse design_config to include all of them as empty keys where absent.

    Used in diagnostic functions that receive the original (possibly sparse) config but
    need label generation to match the mask that was built from the normalized config.

    max_num_regressors : the number of main-effect blocks the mask layout was padded
        to (collate pads to ``max_num_regressors`` main blocks plus
        ``C(max_num_regressors, 2)`` interaction slots). Without it, padded interaction
        slots are indistinguishable from main-effect rows and the row-count inference
        invents phantom regressors (u_5, u_6, …) — always pass it for collated masks.
    """
    block_width = max_num_categories - 1
    inter_block_width = block_width ** 2
    has_intercept = "1" in design_config
    inter_keys = [k for k in design_config if ":" in k]
    main_rows = parameter_mask.shape[0] - (1 if has_intercept else 0) - len(inter_keys) * inter_block_width
    num_main_blocks = max(main_rows // block_width, 0)
    if max_num_regressors is not None:
        num_main_blocks = min(num_main_blocks, max_num_regressors)
    out = {}
    if has_intercept:
        out["1"] = design_config.get("1", [])
    for r in range(1, num_main_blocks + 1):
        out[f"u_{r}"] = design_config.get(f"u_{r}", [])
    for key in inter_keys:
        out[key] = design_config[key]
    return out


def build_ensemble_row_labels(
    design_configs: list[dict],
    parameter_masks: list[np.ndarray],
    max_num_categories: int,
    max_num_regressors: int | None = None,
    categorical_regressors_list: list[list[str] | None] | None = None,
    math: bool = True,
    return_kinds: bool = False,
) -> list[str] | tuple[list[str], list[str]]:
    """Row labels for an ensemble grid that overlays several design configs on the
    same (padded) row layout.

    Builds ONE unified label per row instead of joining each config's whole label
    with `` / `` (which compounds badly for interactions, e.g. the old
    ``u_1 × u_2 / u_1:c_1 × u_2 / u_1 × u_2:c_1``). A regressor's category suffix is
    parenthesized (``u_1(:c_1)``) only when the configs active at that row disagree
    about whether it's continuous or categorical *at the first contrast slot* — the
    only slot where that ambiguity can exist, since any slot beyond the first is
    unambiguously categorical whenever active (a continuous regressor never populates
    it). Interaction labels combine each side's (possibly parenthesized) part with
    `` × ``, e.g. ``u_1(:c_1) × u_2:c_2``. Rows no config touches fall back to
    ``Row r``. Row kinds (for tinting) come directly from row structure
    (intercept/main/interaction); length equals ``max(mask rows)``.

    Interaction keys are unioned across all configs (first-seen order) since
    :func:`expand_design_config_to_mask` does not pad declared interactions the way it
    pads main-effect blocks; main-effect keys are taken from the first config, which is
    safe whenever every config shares the same ``max_num_regressors`` (true at every
    current call site).
    """
    block_width = max_num_categories - 1
    num_rows = max(m.shape[0] for m in parameter_masks)

    exp_configs: list[dict] = []
    cat_sets: list[set[str]] = []
    for k, (dc, mask) in enumerate(zip(design_configs, parameter_masks)):
        dc_exp = expand_design_config_to_mask(dc, mask, max_num_categories, max_num_regressors)
        exp_configs.append(dc_exp)

        cat_regs = categorical_regressors_list[k] if categorical_regressors_list else None
        if cat_regs is not None:
            cat_sets.append(set(cat_regs))
        elif block_width > 1:
            cs: set[str] = set()
            main_keys_k = [key for key in dc_exp if key != "1" and ":" not in key]
            row0 = 1 if "1" in dc_exp else 0
            for m_i, key in enumerate(main_keys_k):
                block = mask[row0 + m_i * block_width: row0 + (m_i + 1) * block_width]
                if block.shape[0] > 1 and np.any(block[1:] == 1.0):
                    cs.add(key)
            cat_sets.append(cs)
        else:
            cat_sets.append(set())

    main_keys = [k for k in exp_configs[0] if k != "1" and ":" not in k]
    inter_keys: list[str] = []
    seen_inter: set[str] = set()
    for dc_exp in exp_configs:
        for k in dc_exp:
            if ":" in k and k not in seen_inter:
                seen_inter.add(k)
                inter_keys.append(k)

    def wrap(body: str) -> str:
        return f"${body}$" if math else body

    def active_at(r: int) -> list[int]:
        return [
            k for k, mask in enumerate(parameter_masks)
            if r < mask.shape[0] and np.any(mask[r] == 1.0)
        ]

    def part(key: str, slot: int, active_configs: list[int]) -> str:
        if slot > 1:
            return f"{key}:c_{slot}"
        is_cat = {key in cat_sets[k] for k in active_configs}
        if is_cat == {True}:
            return f"{key}:c_1"
        if is_cat == {False}:
            return key
        return f"{key}(:c_1)"

    labels: list[str] = []
    kinds: list[str] = []
    row = 0

    if "1" in exp_configs[0]:
        labels.append(wrap("1"))
        kinds.append("intercept")
        row += 1

    for key in main_keys:
        for slot in range(1, block_width + 1):
            active = active_at(row)
            labels.append(wrap(part(key, slot, active)) if active else f"Row {row}")
            kinds.append("main")
            row += 1

    times = r" \times " if math else " × "
    for key in inter_keys:
        a, b = key.split(":")
        for i in range(1, block_width + 1):
            for j in range(1, block_width + 1):
                active = active_at(row)
                if active:
                    labels.append(wrap(part(a, i, active) + times + part(b, j, active)))
                else:
                    labels.append(f"Row {row}")
                kinds.append("interaction")
                row += 1

    # Defensive: keep the contract that length always equals max(mask rows), even if
    # heterogeneous configs (different max_num_regressors/interaction sets) make the
    # structural layout above land short or long.
    if len(labels) < num_rows:
        labels.extend(f"Row {r}" for r in range(len(labels), num_rows))
        kinds.extend("main" for _ in range(len(kinds), num_rows))
    elif len(labels) > num_rows:
        labels = labels[:num_rows]
        kinds = kinds[:num_rows]

    return (labels, kinds) if return_kinds else labels


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


def draw_correlated_normals(R: np.ndarray, num_obs: int) -> np.ndarray:
    """Draw ``num_obs`` rows from ``MVN(0, R)`` via a Cholesky factor.

    ``np.random.multivariate_normal`` runs an SVD plus ``isclose`` symmetry checks on
    every call (~26 us at k=3, n=128) against ~8 us here, and it is called once per
    simulated dataset. ``R`` comes from :func:`sample_lkj_correlation`, which builds it
    as ``L @ L.T`` and so is PSD by construction; the jitter fallback only guards the
    near-singular tail at small ``eta``.

    Draws are distributionally identical but not bit-identical to the SVD route, so a
    fixed seed no longer reproduces pre-optimization design matrices.
    """
    k = R.shape[0]
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(R + 1e-10 * np.eye(k))
    return np.random.randn(num_obs, k) @ L.T


def _standard_normal_slope() -> float:
    return np.random.normal(0.0, 1.0)


def _zero_slope() -> float:
    return 0.0


def _fast_scalar_sampler(spec) -> Callable:
    """Wrap a prior spec in the cheapest correct scalar sampler.

    ``scipy``'s frozen ``.rvs()`` costs ~24 us per scalar draw -- it broadcasts, validates
    and reduces arguments on every call -- against ~1 us for the equivalent numpy call.
    At one draw per free parameter per simulated dataset that dominated the non-numba
    portion of the sampling loop, so normal priors (which is all of the SSM priors) are
    rerouted to ``np.random.normal`` with the loc/scale read out once.
    """
    dist = getattr(spec, "dist", None)
    if dist is not None and getattr(dist, "name", None) == "norm":
        args, kwds = getattr(spec, "args", ()), getattr(spec, "kwds", {})
        loc = float(kwds.get("loc", args[0] if len(args) > 0 else 0.0))
        scale = float(kwds.get("scale", args[1] if len(args) > 1 else 1.0))
        return lambda: float(np.random.normal(loc, scale))
    if hasattr(spec, "rvs"):
        return lambda rv=spec: float(rv.rvs())
    return spec


# Per-prior sampler cache. Keyed by id() but the prior object is stored alongside, both
# to pin the id (so it cannot be reused after a GC) and to verify identity on lookup.
_INTRINSIC_SAMPLER_CACHE: dict[int, tuple[dict, dict[str, Callable]]] = {}


def _cached_intrinsic_samplers(prior: dict) -> dict[str, Callable]:
    key = id(prior)
    hit = _INTRINSIC_SAMPLER_CACHE.get(key)
    if hit is not None and hit[0] is prior:
        return hit[1]
    samplers = {k: _fast_scalar_sampler(v) for k, v in prior.items()}
    _INTRINSIC_SAMPLER_CACHE[key] = (prior, samplers)
    return samplers


def build_intrinsic_priors(
    prior: Callable | dict,
    design_config: dict[str, list[str]],
    fixed_values: dict[str, float] | None = None,
) -> dict[str, dict[str, Callable]]:
    if fixed_values is None:
        fixed_values = {}

    free_intrinsics = design_config.get("1", [])
    free_set = set(free_intrinsics)
    samplers = _cached_intrinsic_samplers(prior)

    priors = {}
    for k in prior:
        if k in free_set:
            priors[k] = {"intercept": samplers[k], "slope": _standard_normal_slope}
        else:
            fixed_value = fixed_values.get(k, 0.0)
            priors[k] = {
                "intercept": lambda v=fixed_value: v,
                "slope": _zero_slope,
            }

    return priors


def build_discrete_mask(
    design_config: dict[str, list[str]],
    discrete_prob: float = 0.5,
    keep_intercept: bool = False,
    add_interaction: bool = False,
    main_discrete_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a discrete mask for main regressors only and expand it to a total-length
    mask aligned with design_config key order (excluding intercept).

    ``main_discrete_mask`` may be supplied to fix the continuous (0) / categorical (1)
    assignment deterministically (length = number of main-effect keys); otherwise it is
    drawn at random with probability ``discrete_prob`` per regressor.
    """
    regressor_keys = [k for k in design_config.keys() if k != "1"]
    main_keys = [k for k in regressor_keys if ":" not in k]

    if main_discrete_mask is None:
        main_discrete_mask = build_random_discrete_mask(
            num_regressors=len(main_keys),
            discrete_prob=discrete_prob,
        ).astype(np.float32, copy=False)
    else:
        main_discrete_mask = np.asarray(main_discrete_mask, dtype=np.float32)

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
    discrete_prob: float | None = 0.5,
) -> np.ndarray:
    """Bernoulli mask over regressors: 1 -> categorical, 0 -> continuous.

    ``discrete_prob=None`` draws the rate per dataset from ``Beta(2, 2)`` instead of
    fixing it, mirroring the hierarchical treatment ``free_prob`` and ``intercept_prob``
    already get in :func:`build_random_design_config`. That puts real mass on the
    all-categorical and all-continuous corners -- the two regimes actual experiments
    live in -- which an iid Bernoulli(0.5) reaches only with probability 0.5**R.
    """
    if num_regressors == 0:
        return np.array([])
    if discrete_prob is None:
        discrete_prob = float(np.random.beta(2, 2))
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
    eta: float | None = None,
    category_concentration: float | None = None,
    return_category_counts: bool = False,
    return_diagnostics: bool = False,
    generator: str = "mixture",
    **generator_kwargs,
) -> np.ndarray | tuple:
    """
    Build a design matrix from a configuration of free intrinsic parameters.

    generator : which construction to use.

      ``"mixture"`` (default) -- latent-class: category assignment is drawn first from a
        joint contingency table over all categorical regressors, then each continuous
        regressor is drawn from one Gaussian per category.  Categories are nominal, the
        categorical/continuous association is a directly dialable correlation ratio, and
        balanced factorial designs are a first-class case.  See
        :func:`_build_design_matrix_mixture` for the extra keyword arguments.

      ``"quantile"`` -- the pre-2026-07 construction, kept selectable so the old
        training distribution stays reproducible.  Cuts a single shared LKJ latent at
        Dirichlet-sampled quantiles, which makes categories ordinal by construction
        (group means of any correlated continuous regressor are monotone in category
        index ~92% of the time) and caps the reachable association strength.

      ``"gaussian_softmax"`` -- one joint LKJ-correlated Gaussian draw per trial,
        with one coordinate per continuous regressor and one score per category.
        Categorical blocks are sampled through softmax. See
        :func:`_build_design_matrix_gaussian_softmax` for its scale/intercept controls.

    return_diagnostics : also return the sampled/realized design hyperparameters. All
        generators report at least ``categorical_keys`` / ``continuous_keys``, which
        downstream coefficient priors need in order to identify categorical blocks.
    """
    if generator == "mixture":
        design_matrix, counts, diagnostics = _build_design_matrix_mixture(
            design_config=design_config, num_obs=num_obs, context=context,
            discrete_prob=discrete_prob, min_num_categories=min_num_categories,
            max_num_categories=max_num_categories, main_discrete_mask=main_discrete_mask,
            eta=eta, category_concentration=category_concentration, **generator_kwargs,
        )
    elif generator == "quantile":
        design_matrix, counts, diagnostics = _build_design_matrix_quantile(
            design_config=design_config, num_obs=num_obs, context=context,
            discrete_prob=discrete_prob, min_num_categories=min_num_categories,
            max_num_categories=max_num_categories, main_discrete_mask=main_discrete_mask,
            eta=eta, category_concentration=category_concentration, **generator_kwargs,
        )
    elif generator == "gaussian_softmax":
        design_matrix, counts, diagnostics = _build_design_matrix_gaussian_softmax(
            design_config=design_config, num_obs=num_obs, context=context,
            discrete_prob=discrete_prob, min_num_categories=min_num_categories,
            max_num_categories=max_num_categories, main_discrete_mask=main_discrete_mask,
            eta=eta, category_concentration=category_concentration, **generator_kwargs,
        )
    else:
        raise ValueError(f"generator must be one of {DESIGN_GENERATORS}; got {generator!r}")

    if not (return_category_counts or return_diagnostics):
        return design_matrix
    out: tuple = (design_matrix,)
    if return_category_counts:
        out = out + (counts,)
    if return_diagnostics:
        out = out + (diagnostics,)
    return out


def _build_design_matrix_quantile(
    design_config: dict[str, list[str]],
    num_obs: int,
    context: dict[str, np.ndarray] | None = None,
    discrete_prob: float = 0.5,
    min_num_categories: int = 2,
    max_num_categories: int = 4,
    main_discrete_mask: np.ndarray = None,
    eta: float | None = None,
    category_concentration: float | None = None,
) -> tuple[np.ndarray, dict[str, int], dict]:
    """
    Legacy generator. Reachable via ``build_design_matrix(generator="quantile")``.

    All regressors (not context vars) are drawn jointly from X ~ MVN(0, R),
    R ~ LKJ(eta), giving a shared latent correlation structure across both
    continuous and discrete regressors. eta = 1 is uniform over correlation
    matrices; eta > 1 concentrates toward the identity (weaker collinearity);
    eta < 1 spreads mass toward near-singular/extreme collinearity.

    Continuous columns are used directly (already unit variance since R has
    unit diagonal). Discrete columns are discretized via Dirichlet-sampled
    quantile cuts:
      p ~ Dirichlet(category_concentration * 1_k), then z is cut at quantiles implied by cumsum(p).

    Because every category is a contiguous interval of the same latent axis, categories
    carry an implicit ordering: see the module docstring notes on ``"mixture"``.
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
    eta_val = float(eta) if eta is not None else None
    if k > 0:
        eta_val = eta_val if eta_val is not None else float(np.random.uniform(0.5, 5.0))
        R = sample_lkj_correlation(k, eta_val)
        X_mvn = draw_correlated_normals(R, num_obs)
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

    diagnostics = {
        "generator": "quantile",
        "categorical_keys": [
            key for j, key in enumerate(main_effect_keys)
            if key not in context and main_discrete_mask[j] == 1
        ],
        "continuous_keys": [
            key for j, key in enumerate(main_effect_keys)
            if key not in context and main_discrete_mask[j] != 1
        ],
        "eta": eta_val,
    }
    return design_matrix, actual_col_counts, diagnostics


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


# ---------------------------------------------------------------------------
# Latent-class ("one Gaussian per category") generator -- the default.
#
# Category assignment is drawn first, from a joint contingency table over all
# categorical regressors; each continuous regressor is then drawn from one Gaussian
# per category. This replaces cutting a single shared latent at quantiles, which made
# categories ordinal by construction, capped the reachable association strength, and
# put almost no mass on balanced factorial designs.
# ---------------------------------------------------------------------------

def sample_category_marginal(
    num_categories: int,
    num_obs: int,
    balanced: bool = False,
    concentration: float | None = None,
    min_cell_count: int = 2,
    min_cell_frac: float = 0.0,
) -> np.ndarray:
    """Category probabilities for one categorical regressor, with a cell-size floor.

    ``balanced=True`` returns exact balance (the factorial-experiment case, which the
    incumbent quantile-cut sampler reaches ~7% of the time).  Otherwise the
    concentration defaults to ``LogUniform(0.5, 50)``: the upper end gives
    near-uniform proportions, the lower end genuine imbalance, without the mass on
    degenerate spikes that ``U(0.1, 2.0)`` carries.

    The floor is applied as a convex shift ``floor + (1 - K*floor) * p`` rather than
    clip-then-renormalize, which can push entries back below the floor.
    """
    if balanced:
        p = np.full(num_categories, 1.0 / num_categories)
    else:
        if concentration is None:
            concentration = float(np.exp(np.random.uniform(np.log(0.5), np.log(50.0))))
        p = np.random.dirichlet(concentration * np.ones(num_categories))

    floor = max(min_cell_count / max(num_obs, 1), min_cell_frac)
    if num_categories * floor >= 1.0:
        # Design too small to honour the requested floor; keep it feasible.
        floor = 0.5 / num_categories
    return floor + (1.0 - num_categories * floor) * p


def ipf(table: np.ndarray, marginals: list[np.ndarray], num_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Iterative proportional fitting: rescale ``table`` to the given marginals.

    Alternates a multiplicative rescale along each axis until every axis sum matches
    its target.  Converges in a handful of sweeps for the small tables here (at most
    ``max_num_regressors`` axes of ``max_num_categories`` levels).
    """
    p = np.asarray(table, dtype=np.float64)
    p = p / p.sum()
    ndim = p.ndim
    for _ in range(num_iter):
        err = 0.0
        for axis, target in enumerate(marginals):
            axes = tuple(i for i in range(ndim) if i != axis)
            current = p.sum(axis=axes)
            err = max(err, float(np.abs(current - target).max()))
            scale = np.divide(target, current, out=np.ones_like(current), where=current > 0)
            shape = [1] * ndim
            shape[axis] = -1
            p = p * scale.reshape(shape)
        if err < tol:
            break
    return p / p.sum()


def build_joint_category_table(
    marginals: list[np.ndarray],
    assoc_lambda: float,
    coupling_concentration: float = 1.0,
) -> np.ndarray:
    """Joint distribution over the categories of every categorical regressor, holding
    each regressor's marginal EXACTLY at the value already drawn for its main effect.

    ``P = (1 - lam) * (p_1 x ... x p_R) + lam * IPF(Q; p_1, ..., p_R)``

    Reusing the main-effect marginals matters: drawing a free Dirichlet over the joint
    cells would leave the marginals as whatever the axis sums happen to be, so a
    regressor's category balance would depend on whether an interaction was sampled.
    Both terms above share the marginals and the set of tables with fixed marginals is
    convex, so the mixture has them for any ``lam``.  ``lam = 0`` is the orthogonal
    factorial case; ``lam = 1`` is a maximally re-coupled table.  The raw coupling ``Q``
    is exchangeable, so the induced association is non-monotone in category index --
    unlike a comonotone (northwest-corner) coupling, which would reintroduce exactly
    the ordinality this module exists to remove.
    """
    independent = marginals[0]
    for m in marginals[1:]:
        independent = np.multiply.outer(independent, m)

    if len(marginals) < 2 or assoc_lambda <= 0.0:
        return independent

    shape = tuple(len(m) for m in marginals)
    raw = np.random.dirichlet(
        coupling_concentration * np.ones(int(np.prod(shape)))
    ).reshape(shape)
    coupled = ipf(raw, marginals)
    return (1.0 - assoc_lambda) * independent + assoc_lambda * coupled


def _compact_labels(assign: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, list[int]]:
    """Drop unpopulated categories and relabel survivors ``0..K'-1`` in random order.

    Compaction keeps each dummy block contiguous (no all-zero column wedged between
    populated ones, which would otherwise rely on the downstream ``realized_cols``
    prune).  The random relabel makes the dropped reference category an arbitrary one,
    which is what "nominal" requires.
    """
    out = np.empty_like(assign)
    realized: list[int] = []
    for r in range(assign.shape[1]):
        counts = np.bincount(assign[:, r], minlength=shape[r])
        present = np.flatnonzero(counts > 0)
        lookup = np.full(shape[r], -1, dtype=np.int64)
        lookup[present] = np.random.permutation(present.shape[0])
        out[:, r] = lookup[assign[:, r]]
        realized.append(int(present.shape[0]))
    return out, realized


def assign_categories(
    table: np.ndarray,
    num_obs: int,
    min_cell_count: int = 2,
    max_tries: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """Draw per-trial category assignments from the joint table.

    Unlike the incumbent empirical-quantile cut -- which pins realized proportions to
    ``p`` exactly, with no sampling noise -- this draws assignments, so cell counts
    carry proper multinomial variability.  Retries a bounded number of times if any
    category lands below ``min_cell_count``; the marginal floor makes that rare.

    Returns ``(assignments (num_obs, n_categorical), realized_num_categories)``.
    """
    shape = table.shape
    flat = np.asarray(table, dtype=np.float64).ravel()
    flat = flat / flat.sum()

    assign = None
    for _ in range(max_tries):
        cells = np.random.choice(flat.shape[0], size=num_obs, p=flat)
        assign = np.stack(np.unravel_index(cells, shape), axis=-1)
        if all(
            np.bincount(assign[:, r], minlength=shape[r]).min() >= min_cell_count
            for r in range(len(shape))
        ):
            break
    return _compact_labels(assign, shape)


def sample_continuous_columns(
    assign: np.ndarray,
    num_continuous: int,
    eta: float,
    eta_squared: float,
) -> np.ndarray:
    """One Gaussian per category over the continuous regressors.

    ``x_j = sqrt(eta2) * s_j + sqrt(1 - eta2) * eps_j`` with ``s_j`` the standardized
    sum of per-category offsets and ``eps ~ MVN(0, LKJ(eta))``.  ``eta_squared`` is the
    *joint* correlation ratio of all categorical regressors on each continuous column.
    Per-column standardization is affine, so it preserves the LKJ correlation among the
    ``eps`` columns while leaving every output column at unit variance.
    """
    num_obs = assign.shape[0]
    n_cat = assign.shape[1]

    R = sample_lkj_correlation(num_continuous, float(eta))
    eps = draw_correlated_normals(R, num_obs)
    eps_sd = eps.std(axis=0)
    eps = (eps - eps.mean(axis=0)) / np.where(eps_sd == 0.0, 1.0, eps_sd)

    if n_cat == 0 or eta_squared <= 0.0:
        return eps.astype(np.float32)

    signal = np.zeros((num_obs, num_continuous))
    for r in range(n_cat):
        k = int(assign[:, r].max()) + 1
        signal += np.random.randn(k, num_continuous)[assign[:, r]]

    sig_sd = signal.std(axis=0)
    signal = (signal - signal.mean(axis=0)) / np.where(sig_sd == 0.0, 1.0, sig_sd)

    x = np.sqrt(eta_squared) * signal + np.sqrt(1.0 - eta_squared) * eps
    return x.astype(np.float32)


def _assemble_design_columns(
    design_config: dict[str, list[str]],
    num_obs: int,
    max_num_categories: int,
    main_columns: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    """Pack sampled main effects, context overrides and interactions into the shared layout."""
    main_keys = [k for k in design_config if k != "1" and ":" not in k]
    inter_keys = [k for k in design_config if ":" in k]
    block_width = max_num_categories - 1
    inter_width = block_width ** 2
    has_intercept = "1" in design_config
    num_cols = int(has_intercept) + len(main_keys) * block_width + len(inter_keys) * inter_width
    matrix = np.zeros((num_obs, num_cols))
    if has_intercept:
        matrix[:, 0] = 1.0

    starts = {key: int(has_intercept) + j * block_width for j, key in enumerate(main_keys)}
    counts: dict[str, int] = {}

    def context_column(key: str) -> np.ndarray:
        col = np.asarray(context[key], dtype=np.float32).reshape(-1)
        if col.shape[0] != num_obs:
            raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")
        return col

    for key in main_keys:
        start = starts[key]
        if key in context:
            matrix[:, start] = context_column(key)
            counts[key] = 1
        else:
            columns = main_columns[key]
            width = min(columns.shape[1], block_width)
            matrix[:, start:start + width] = columns[:, :width]
            counts[key] = width

    inter_start = int(has_intercept) + len(main_keys) * block_width
    for j, key in enumerate(inter_keys):
        start = inter_start + j * inter_width
        if key in context:
            matrix[:, start] = context_column(key)
            continue
        parts = key.split(":")
        if len(parts) != 2 or any(k not in starts for k in parts):
            continue
        a, b = (starts[k] for k in parts)
        left = matrix[:, a:a + block_width]
        right = matrix[:, b:b + block_width]
        matrix[:, start:start + inter_width] = (
            left[:, :, None] * right[:, None, :]
        ).reshape(num_obs, inter_width)
    return matrix, counts


def _build_design_matrix_gaussian_softmax(
    design_config: dict[str, list[str]],
    num_obs: int,
    context: dict[str, np.ndarray] | None = None,
    discrete_prob: float = 0.5,
    min_num_categories: int = 2,
    max_num_categories: int = 4,
    main_discrete_mask: np.ndarray | None = None,
    eta: float | None = None,
    category_concentration: float | None = None,
    logit_scale: float = 1.0,
    intercept_scale: float = 1.0,
    continuous_scale: float = 1.0,
    category_intercepts: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, int], dict]:
    """Joint Gaussian predictors with logistic-normal categorical probability blocks.

    Per dataset, ``Omega ~ LKJ(eta)`` (eta defaults to U(0.5, 5)). Per trial,
    ``h ~ N(0, D Omega D)``: continuous coordinates come first, then one full
    K-score block per categorical regressor. Continuous coordinates have scale
    ``continuous_scale``; all coordinates in a categorical block share ``logit_scale``.
    No empirical standardization or quantile cutting is applied.

    Category intercepts default to iid N(0, intercept_scale**2), drawn once per
    dataset. ``category_intercepts`` may override named blocks with finite vectors
    of their sampled K lengths (use min_num_categories=max_num_categories to fix K).
    ``softmax(alpha + h)`` gives per-trial probabilities; assignments are conditionally
    independent given h. Full score blocks and symmetric intercept priors avoid
    privileging a reference category. The reference is chosen only during dummy coding.

    This generator has no Dirichlet concentration, marginal calibration, count floor,
    or assignment retries. Empty levels are compacted and randomly relabelled just as
    in the mixture generator. Diagnostics retain the original intercepts and the
    original category index for each compacted label in ``category_label_orders``.
    Equal intercepts do not guarantee uniform marginals for an arbitrary Omega, and
    latent correlations do not directly specify observed predictor associations.
    """
    if num_obs < 1:
        raise ValueError("gaussian_softmax requires num_obs >= 1")
    if not 2 <= min_num_categories <= max_num_categories:
        raise ValueError("require 2 <= min_num_categories <= max_num_categories")
    if category_concentration is not None:
        raise ValueError("gaussian_softmax does not use category_concentration; use intercept_scale instead")
    for name, value in (
        ("logit_scale", logit_scale), ("intercept_scale", intercept_scale),
        ("continuous_scale", continuous_scale),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    eta = float(np.random.uniform(0.5, 5.0)) if eta is None else float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")

    context = context or {}
    main_keys = [k for k in design_config if k != "1" and ":" not in k]
    if main_discrete_mask is None:
        main_discrete_mask = build_random_discrete_mask(len(main_keys), discrete_prob=discrete_prob)
    mask = np.asarray(main_discrete_mask)
    if mask.shape != (len(main_keys),) or not np.isin(mask, [0, 1]).all():
        raise ValueError("main_discrete_mask must contain one 0/1 value per main regressor")
    cat_keys = [key for j, key in enumerate(main_keys) if key not in context and mask[j] == 1]
    cont_keys = [key for j, key in enumerate(main_keys) if key not in context and mask[j] == 0]
    nominal_k = [int(np.random.randint(min_num_categories, max_num_categories + 1)) for _ in cat_keys]

    overrides = {} if category_intercepts is None else category_intercepts
    if set(overrides) - set(cat_keys):
        raise ValueError("category_intercepts keys must name sampled categorical regressors")
    intercepts: dict[str, np.ndarray] = {}
    for key, k in zip(cat_keys, nominal_k):
        alpha = (
            np.asarray(overrides[key], dtype=np.float64) if key in overrides
            else np.random.normal(0.0, intercept_scale, size=k)
        )
        if alpha.shape != (k,) or not np.isfinite(alpha).all():
            raise ValueError(f"category_intercepts['{key}'] must be a finite vector of length {k}")
        intercepts[key] = alpha

    n_latent = len(cont_keys) + sum(nominal_k)
    if n_latent:
        correlation = sample_lkj_correlation(n_latent, eta)
        scales = np.concatenate([
            np.full(len(cont_keys), continuous_scale), np.full(sum(nominal_k), logit_scale),
        ])
        latent = draw_correlated_normals(correlation, num_obs) * scales
    else:
        correlation = np.empty((0, 0))
        scales = np.empty(0)
        latent = np.empty((num_obs, 0))

    main_columns = {key: latent[:, j:j + 1] for j, key in enumerate(cont_keys)}
    raw_assign = np.empty((num_obs, len(cat_keys)), dtype=np.int64)
    start = len(cont_keys)
    for r, (key, k) in enumerate(zip(cat_keys, nominal_k)):
        logits = latent[:, start:start + k] + intercepts[key]
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        cumulative = probabilities.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        raw_assign[:, r] = (np.random.rand(num_obs, 1) >= cumulative).sum(axis=1)
        start += k

    if cat_keys:
        assign, realized_k = _compact_labels(raw_assign, tuple(nominal_k))
    else:
        assign, realized_k = raw_assign, []
    label_orders: dict[str, np.ndarray] = {}
    for r, (key, k) in enumerate(zip(cat_keys, realized_k)):
        main_columns[key] = (assign[:, r, None] == np.arange(1, k)[None, :]).astype(np.float32)
        order = np.empty(k, dtype=np.int64)
        order[assign[:, r]] = raw_assign[:, r]
        label_orders[key] = order

    matrix, counts = _assemble_design_columns(
        design_config, num_obs, max_num_categories, main_columns, context,
    )
    diagnostics = {
        "generator": "gaussian_softmax", "eta": eta,
        "logit_scale": logit_scale, "intercept_scale": intercept_scale,
        "continuous_scale": continuous_scale,
        "categorical_keys": cat_keys, "continuous_keys": cont_keys,
        "nominal_num_categories": nominal_k, "realized_num_categories": realized_k,
        "latent_correlation": correlation, "latent_scales": scales,
        "category_intercepts": intercepts, "category_label_orders": label_orders,
        "min_cell_count": (
            min(int(np.bincount(assign[:, r]).min()) for r in range(len(cat_keys)))
            if cat_keys else None
        ),
    }
    return matrix, counts, diagnostics


def _build_design_matrix_mixture(
    design_config: dict[str, list[str]],
    num_obs: int,
    context: dict[str, np.ndarray] | None = None,
    discrete_prob: float = 0.5,
    min_num_categories: int = 2,
    max_num_categories: int = 4,
    main_discrete_mask: np.ndarray | None = None,
    eta: float | None = None,
    category_concentration: float | None = None,
    balance_prob: float = 0.3,
    factorial_prob: float = 0.5,
    assoc_lambda: float | None = None,
    eta_squared: float | None = None,
    coupling_concentration: float | None = None,
    min_cell_count: int = 2,
    min_cell_frac: float = 0.0,
) -> tuple[np.ndarray, dict[str, int], dict]:
    """Build a design matrix with categories drawn first and continuous regressors
    drawn conditionally, one Gaussian per category.

    Signature mirrors :func:`design_utils.build_design_matrix` for the shared arguments;
    the extras control the new sampling regime:

    balance_prob      : probability a categorical regressor gets exactly balanced
                        categories (the factorial-experiment case).
    factorial_prob    : probability the categorical regressors are drawn mutually
                        orthogonal (``assoc_lambda = 0``).
    assoc_lambda      : categorical/categorical association strength in [0, 1].
                        None -> 0 with probability ``factorial_prob``, else U(0, 1).
    eta_squared       : joint correlation ratio of the categorical regressors on each
                        continuous column. None -> Beta(1.2, 3.0) capped at 0.85.
    coupling_concentration : Dirichlet concentration for the raw joint coupling.
                        None -> LogUniform(0.3, 10).
    min_cell_count    : floor on observations per category, enforced through the
                        marginal floor plus bounded resampling of the assignments.
    """
    context = context or {}
    main_effect_keys = [k for k in design_config if k != "1" and ":" not in k]

    if main_discrete_mask is None:
        main_discrete_mask = build_random_discrete_mask(
            num_regressors=len(main_effect_keys), discrete_prob=discrete_prob,
        )

    # Partition the non-context main effects into categorical and continuous.
    cat_indices = [
        j for j, key in enumerate(main_effect_keys)
        if key not in context and main_discrete_mask[j] == 1
    ]
    cont_indices = [
        j for j, key in enumerate(main_effect_keys)
        if key not in context and main_discrete_mask[j] != 1
    ]

    if eta is None:
        eta = float(np.random.uniform(0.5, 5.0))
    if eta_squared is None:
        eta_squared = float(min(np.random.beta(1.2, 3.0), 0.85))
    if assoc_lambda is None:
        assoc_lambda = 0.0 if np.random.rand() < factorial_prob else float(np.random.rand())
    if coupling_concentration is None:
        coupling_concentration = float(np.exp(np.random.uniform(np.log(0.3), np.log(10.0))))

    # 1. Category marginals, one per categorical regressor.
    nominal_k: list[int] = []
    marginals: list[np.ndarray] = []
    balanced_flags: list[bool] = []
    for _ in cat_indices:
        k = int(np.random.randint(min_num_categories, max_num_categories + 1))
        balanced = bool(np.random.rand() < balance_prob)
        nominal_k.append(k)
        balanced_flags.append(balanced)
        marginals.append(
            sample_category_marginal(
                num_categories=k, num_obs=num_obs, balanced=balanced,
                concentration=category_concentration,
                min_cell_count=min_cell_count, min_cell_frac=min_cell_frac,
            )
        )

    # 2. Joint table over all categorical regressors, then per-trial assignment.
    if cat_indices:
        table = build_joint_category_table(marginals, assoc_lambda, coupling_concentration)
        assign, realized_k = assign_categories(table, num_obs, min_cell_count=min_cell_count)
    else:
        assign = np.zeros((num_obs, 0), dtype=np.int64)
        realized_k = []

    # 3. Continuous columns, conditional on the assignment.
    if cont_indices:
        cont_cols = sample_continuous_columns(
            assign=assign, num_continuous=len(cont_indices),
            eta=eta, eta_squared=eta_squared,
        )
    else:
        cont_cols = np.zeros((num_obs, 0), dtype=np.float32)

    main_columns = {
        main_effect_keys[j]: cont_cols[:, i:i + 1]
        for i, j in enumerate(cont_indices)
    }
    for r, j in enumerate(cat_indices):
        main_columns[main_effect_keys[j]] = (
            assign[:, r, None] == np.arange(1, realized_k[r])[None, :]
        ).astype(np.float32)
    design_matrix, actual_col_counts = _assemble_design_columns(
        design_config, num_obs, max_num_categories, main_columns, context,
    )

    diagnostics = {
        "generator": "mixture",
        "eta": eta,
        "eta_squared": eta_squared,
        "assoc_lambda": assoc_lambda,
        "coupling_concentration": coupling_concentration,
        "categorical_keys": [main_effect_keys[j] for j in cat_indices],
        "continuous_keys": [main_effect_keys[j] for j in cont_indices],
        "nominal_num_categories": nominal_k,
        "realized_num_categories": realized_k,
        "balanced": balanced_flags,
        "min_cell_count": (
            min(int(np.bincount(assign[:, r]).min()) for r in range(assign.shape[1]))
            if assign.shape[1] else None
        ),
    }
    return design_matrix, actual_col_counts, diagnostics
