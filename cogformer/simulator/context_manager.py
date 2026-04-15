import numpy as np
from itertools import combinations
from typing import Callable
from scipy.stats import invwishart

from cogformer.utils.simulator_utils import shifted_softplus


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
    >>> set_a, set_b = split_design_space(["v", "a", "z", "tau"], seed=0)
    >>> train_batch = mf.batch_sample(..., allowed_slope_patterns=set_a)
    >>> test_batch  = mf.batch_sample(..., allowed_slope_patterns=set_b)
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


class ContextManager:
    """
    Utility class for constructing design configurations, parameter masks,
    design matrices, and sampled parameter matrices for NestedModelFamily.
    """
    def __init__(self):
        pass

    def build_design_config(
            self,
            intrinsic_params: list[str],
            regressed_params: list[str] = None,
            num_regressors: int = 2,
            keep_intercept: bool = False,
            add_interaction: bool = False,
    ):

        config: dict[str, list[str]] = {}

        # Choose which intrinsics are eligible to be regressed
        if regressed_params is None:
            regressed = list(intrinsic_params)
        else:
            regressed_set = set(regressed_params)
            regressed = [p for p in intrinsic_params if p in regressed_set]

        # Intercept first
        if keep_intercept:
            # All intrinsic params get an intercept; fixedness is expressed via the mask/prior
            config["1"] = list(intrinsic_params)

        # Then main effect
        main_keys: list[str] = []
        for r in range(num_regressors):
            key = f"u_{r + 1}"
            config[key] = list(regressed)  # slopes only for regressed params
            main_keys.append(key)

        # Optionally, interaction effect
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
            self,
            intrinsic_params: list[str],
            num_regressors: int,
            free_intrinsics: list[str] | set[str] | None = None,
            fixed_intrinsics: list[str] | set[str] | None = None,
            keep_intercept: bool = False,
            free_prob: float = 0.5,
            intercept_prob: float = 0.5,
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
        - Slopes (main effects and interactions) are only allowed for intrinsic
          parameters which have an intercept entry.
        - Fixed intrinsics (if provided) are always included in the intercept and
          never appear in slope terms.
        - Global params (if provided) are always included in the intercept and
          never appear in slope terms. Unlike fixed intrinsics, their intercept
          values are sampled from their prior (not set to a constant).
        """
        # Normalize inputs
        free_set = None if free_intrinsics is None else set(free_intrinsics)
        fixed_set: set[str] = set() if fixed_intrinsics is None else set(fixed_intrinsics)
        global_set: set[str] = set() if global_params is None else set(global_params)

        config = {}

        # --- Intercept ---
        # Randomize intercept membership for free intrinsics, but always include fixed/global intrinsics.
        # Always build the intercept row when keep_intercept=True OR global params are present,
        # so that global params are never silently absent from the parameter matrix.
        intercept_names: list[str] = []
        if keep_intercept or global_set:
            for p in intrinsic_params:
                if p in global_set:
                    # global params always get an intercept, never a slope
                    intercept_names.append(p)
                    continue

                if free_set is not None and p in free_set:
                    # (1) free intrinsics always get intercepts
                    intercept_names.append(p)
                    continue

                if p in fixed_set:
                    # (2) fixed intrinsics sometimes get intercepts
                    if np.random.rand() < intercept_prob:
                        intercept_names.append(p)
                    continue

                # If free_set is provided, exclude non-free / non-fixed intrinsics entirely
                if free_set is not None:
                    continue

                # Otherwise treat as "free-ish": random intercept
                if np.random.rand() < intercept_prob:
                    intercept_names.append(p)

            config["1"] = intercept_names

        # Helper: slopes only allowed when intercept exists for the intrinsic
        allowed_slope_params = set(intercept_names) if keep_intercept else set(intrinsic_params)
        # Global params never appear in slope terms
        allowed_slope_params -= global_set

        # Main effect
        main_keys = []
        for r in range(num_regressors):
            key = f"u_{r + 1}"
            if allowed_slope_patterns is not None:
                pattern = allowed_slope_patterns[np.random.randint(len(allowed_slope_patterns))]
                names = [p for p in intrinsic_params if p in pattern and p in allowed_slope_params]
            else:
                names = []
                for param in intrinsic_params:
                    if (param in allowed_slope_params) and (np.random.rand() < free_prob):
                        names.append(param)
            config[key] = names
            main_keys.append(key)

        # Interaction effect
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
            self,
            design_config: dict[str, list[str]],
            intrinsic_params: list[str],
            fixed_params: list[str] = None,
            max_num_categories: int = 3,
            keep_intercept: bool = False
    ) -> np.ndarray:
        """
        Convert a design_config dict into a binary parameter mask mapping
        regressors (rows) to intrinsic parameters (columns).
        """
        regressor_keys = list(design_config.keys())
        # Design config is the source of truth: "1" is present only when it was
        # intentionally added (keep_intercept=True or global params exist).
        has_intercept = "1" in regressor_keys
        regressor_keys = [k for k in regressor_keys if k != "1"]

        # Figure out block width and total number of columns
        num_regressors = len(regressor_keys)
        block_width = max_num_categories - 1
        num_intrinsic_params = len(intrinsic_params)
        num_cols = num_regressors * block_width + (1 if has_intercept else 0)

        mask = np.zeros((num_cols, num_intrinsic_params), dtype=np.float32)

        # Building the matrix
        # Start with intercept
        row_idx = 0
        if has_intercept:
            fixed_set = set(fixed_params) if fixed_params is not None else set()
            for name in design_config.get("1", []):
                if name in intrinsic_params and name not in fixed_set:
                    j = intrinsic_params.index(name)
                    mask[row_idx, j] = 1.0

            row_idx = 1

        # Then to each regressor
        for design_index, key in enumerate(regressor_keys):
            start = row_idx + design_index * block_width
            rows = list(range(start, start + block_width))
            for intrinsic in design_config[key]:
                if intrinsic in intrinsic_params:
                    param_index = intrinsic_params.index(intrinsic)
                    if fixed_params is not None and intrinsic in fixed_params:
                        continue
                    for r in rows:
                        mask[r, param_index] = 1.0

        return mask

    def build_random_parameter_mask(
        self,
        intrinsic_params: list[str],
        num_regressors: int = 2,
        # max_num_regressors: int = 5,
        max_num_categories: int = 4,
        free_intrinsics: list[str] | set[str] | None = None,
        fixed_intrinsics: list[str] | set[str] | None = None,
        free_prob: float = 0.5,     # Probability of a param being free
        intercept_prob: float = 0.5,
        keep_intercept: bool = False,
        add_interaction: bool = False,
        allowed_slope_patterns: list[frozenset[str]] | None = None,
        global_params: list[str] | set[str] | None = None,
    ) -> tuple:
        """
        Randomly generate both a design_config and its corresponding parameter mask.

        free_intrinsics: both intercept and slopes are sampled
        fixed_intrinsics: only intercept is sampled, slopes are not
        frozen_intrinsics: neither intercept nor slope are sampled
        """
        design_config = self.build_random_design_config(
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

        parameter_mask = self.build_parameter_mask(
            design_config=design_config,
            intrinsic_params=intrinsic_params,
            # max_num_regressors=max_num_regressors,
            max_num_categories=max_num_categories,
            keep_intercept=keep_intercept,
        )

        return parameter_mask, design_config

    def build_intrinsic_priors(
        self,
        prior_fun: Callable | dict = None,
        design_config: dict[str, list[str]] = None,
        free_intrinsics: list[str] | set[str] | None = None,
        fixed_intrinsics: list[str] | set[str] | None = None,
        fixed_values: dict[str, float] | None = None
    ):
        if fixed_values is None:
            fixed_values = {}

        if design_config is not None:
            free_intrinsics = design_config.get("1", [])
            fixed_intrinsics = list(set(list(prior_fun.keys())) - set(free_intrinsics))

        priors = {}

        for k, spec in prior_fun.items():

            if k in free_intrinsics:
                # SciPy priors
                if hasattr(spec, "rvs"):
                    # print("Identified SciPy priors")
                    scale = spec.std()
                    # scale = spec.mean()
                    sampler = lambda rv=spec: rv.rvs()
                    priors[k] = {
                        "intercept": sampler,
                        "slope": lambda std=scale: np.random.normal(0.0, 1.0)
                    }
                else:
                    priors[k] = {
                        "intercept": spec,
                        "slope": lambda: np.random.normal(0.0, 1.0)
                    }
            elif k in fixed_intrinsics:
                # If the user provides a fixed value for a fixed intrinsic parameter,
                # use that value. Otherwise, default to zero.
                if k in fixed_values:
                    fixed_value = fixed_values[k]
                else:
                    fixed_value = 0.0
                priors[k] = {
                    "intercept": lambda v=fixed_value: v,
                    "slope": lambda: 0.0
                }

        return priors

    def build_discrete_mask(
            self,
            design_config: dict[str, list[str]],
            discrete_prob: float = 0.5,
            keep_intercept: bool = False,
            add_interaction: bool = False,
    ):
        """
        Sample a discrete mask for main regressors only and expand it to a total-length
        mask aligned with design_config key order (excluding intercept).
        """
        regressor_keys = list(design_config.keys())
        regressor_keys = [k for k in regressor_keys if k != "1"]

        main_keys = [k for k in regressor_keys if ":" not in k]
        num_main = len(main_keys)

        main_discrete_mask = self.build_random_discrete_mask(
            num_regressors=num_main,
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
        self,
        num_regressors: int,
        discrete_prob: float = 0.5
    ) -> np.ndarray:
        """
        Bernoulli mask over regressors: 1->discrete {0,1}, 0->continuous U[0,1].
        By default, keeps the intercept (col 0) continuous.
        """
        # No intercept, no discrete mask
        if num_regressors == 0:
            return np.array([])

        discrete_mask = np.array(1 * (np.random.rand(num_regressors) < discrete_prob))
        return discrete_mask

    def build_regressor_mask(
        self,
        num_regressors: int,
        max_num_categories: int = 4,
        keep_intercept: bool = False,
    ) -> np.ndarray:
        block_width = max_num_categories - 1
        num_cols = num_regressors * block_width + (1 if keep_intercept else 0)
        mask = np.ones(num_cols, dtype=np.float32)

        return mask

    def build_design_matrix(
        self,
        design_config: dict[str, list[str]],
        num_obs: int,
        # max_num_regressors: int = 5,
        context: dict[str, np.ndarray] | None = None,
        discrete_prob: float = 0.5,
        keep_intercept: bool = False,
        min_num_categories: int = 2,
        max_num_categories: int = 4,
        main_discrete_mask: np.ndarray = None,
        collinearity_df: float | None = None,
        category_concentration: float | None = None,
    ) -> np.ndarray:
        """
        Build design matrix based on the respective configuration of free intrinsic parameters.

        All non-context main effect regressors are drawn jointly from X ~ MVN(0, Sigma),
        Sigma ~ Inv-Wishart(collinearity_df, I_k), giving a shared latent covariance structure across
        both continuous and discrete regressors.

        Conticollinearity_dfous columns are used directly (standardized to unit variance).
        Discrete columns are discretized via Dirichlet-sampled quantile cuts:
          p ~ Dirichlet(category_concentration * 1_k), then z is cut at quantiles implied by cumsum(p).
          category_concentration=None samples category_concentration ~ Uniform(0.1, 2.0) each call, covering the range
          from heavily skewed to near-uniform category distributions.
        """
        # Provide context
        context = context or {}
        regressor_keys = list(design_config.keys())
        has_intercept = "1" in regressor_keys

        # Exclude intercept from regressor keys
        regressor_keys = [k for k in regressor_keys if k != "1"]
        num_regressors = len(regressor_keys)

        # Split regressor keys further into effect keys: main vs. interaction
        main_effect_keys = [k for k in regressor_keys if ":" not in k]
        interaction_keys = [k for k in regressor_keys if ":" in k]

        # Construct per-parameter column blocks
        block_width = max_num_categories - 1
        num_cols = num_regressors * block_width + (1 if has_intercept else 0)

        # Design matrix always includes intercept column if present in design_config
        design_matrix = np.zeros((num_obs, num_cols))

        # Generate discrete mask for the non-intercept regressors if none provided
        if main_discrete_mask is None:
            main_discrete_mask = self.build_random_discrete_mask(
                num_regressors=len(main_effect_keys),
                discrete_prob=discrete_prob,
            )

        # Pre-sample ALL non-context main effect regressors jointly via MVN(0, Sigma),
        # Sigma ~ Inv-Wishart(collinearity_df, I_k), so continuous and discrete regressors share
        # the same latent covariance structure.  Conticollinearity_dfous columns are used directly;
        # discrete columns are subsequently discretized via Dirichlet-sampled quantile cuts.
        non_context_indices = [
            j for j, key in enumerate(main_effect_keys)
            if key not in context
        ]
        k = len(non_context_indices)
        if k > 0:
            collinearity_df_val = float(collinearity_df) if collinearity_df is not None else np.random.uniform(k + 1, 4 * k + 1)
            Sigma = invwishart.rvs(df=collinearity_df_val, scale=np.eye(k))
            Sigma = np.atleast_2d(Sigma).reshape(k, k)
            X_mvn = np.random.multivariate_normal(np.zeros(k), Sigma, size=num_obs)
            X_mvn /= X_mvn.std(axis=0).clip(1e-8)   # standardize to unit variance
            mvn_col_map = {j: X_mvn[:, i].astype(np.float32) for i, j in enumerate(non_context_indices)}
        else:
            mvn_col_map = {}

        # Fill in the matrix
        col_idx = 0
        if has_intercept:
            design_matrix[:, 0] = 1.0
            col_idx = 1

        # Map regressor keys to the start of its associated block (first column)
        start_col = {k: col_idx + j * block_width for j, k in enumerate(regressor_keys)}

        # Main effect
        for j, key in enumerate(main_effect_keys):
            start = start_col[key]

            if key in context:
                col = np.asarray(context[key], dtype=np.float32).reshape(-1)
                if col.shape[0] != num_obs:
                    raise ValueError(f"context['{key}'] length {col.shape[0]} != num_obs {num_obs}")

                design_matrix[:, start] = col
                continue

            if main_discrete_mask[j] == 1:
                num_categories = np.random.randint(min_num_categories, max_num_categories + 1)
                dummies = self._discretize_column(mvn_col_map[j], num_categories, category_concentration)
                design_matrix[:, start:(start + dummies.shape[1])] = dummies
            else:
                design_matrix[:, start] = mvn_col_map[j]

        # Interaction effect
        for key in interaction_keys:
            start = start_col[key]

            # Allow explicit override via context if desired
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

            # first-column indices of parent blocks
            a0 = start_col.get(a)
            b0 = start_col.get(b)
            if (a0 is None) or (b0 is None):
                continue

            design_matrix[:, start] = design_matrix[:, a0] * design_matrix[:, b0]
        return design_matrix

    def build_link_functions(
            self,
            intrinsic_params: list[str],
            link_fun: Callable | dict | None,
            default_link_fun: Callable | None = None,
    ) -> dict[str, Callable]:
        """Build a per-intrinsic-parameter link-function mapping.

        Parameters
        ----------
        intrinsic_params
            List of intrinsic parameter names.
        link_fun
            Either a single callable (applied to all parameters), a dictionary
            mapping parameter name -> callable, or None.
        default_link_fun
            Optional default callable used to fill missing keys when `link_fun`
            is a dict. If None and keys are missing, raises.

        Returns
        -------
        link_funs
            Dictionary mapping each intrinsic parameter name to a callable.
        """
        # None -> default to softplus
        if link_fun is None:
            return {k: shifted_softplus for k in intrinsic_params}

        # Single callable -> broadcast to all params
        if callable(link_fun):
            return {k: link_fun for k in intrinsic_params}

        # Dict -> validate and (optionally) fill missing
        if not isinstance(link_fun, dict):
            raise TypeError(
                "link_fun must be a callable, a dict[str, callable], or None; "
                f"got {type(link_fun)}"
            )

        unknown = [k for k in link_fun.keys() if k not in intrinsic_params]
        if unknown:
            raise KeyError(
                "Unknown link_fun keys not in intrinsic_params: " + ", ".join(map(str, unknown))
            )

        missing = [k for k in intrinsic_params if k not in link_fun]
        if missing:
            if default_link_fun is None:
                raise KeyError(
                    "Missing link_fun for intrinsic_params: " + ", ".join(map(str, missing))
                )
            return {k: link_fun.get(k, default_link_fun) for k in intrinsic_params}

        return {k: link_fun[k] for k in intrinsic_params}

    def build_parameter_indices(
        self,
        intrinsic_params: list[str],
        num_regressors: int,
        num_categories: int,
    ) -> np.ndarray:
        """
        Build an array of positional indices for the intrinsic parameters.
        """
        # Get number of intrinsic params
        num_intrinsic_params = len(intrinsic_params)
        indices = np.tile(
            np.repeat(np.linspace(0.0, 0.5, num_intrinsic_params), num_categories - 1),
            num_regressors
        )
        return indices

    def mask_to_design_config(
        self,
        parameter_mask: np.ndarray,
        intrinsic_params: list[str],
        keep_intercept: bool = False,
        ignore_padding: bool = True
    ) -> dict[str, list[str]]:
        """
        Convert a (num_regressors × num_intrinsics) binary mask into a design_config dict.
        Row 0 -> key "1" (intercept), rows 1 -> "u_1", "u_2", ...
        """
        num_rows, num_intrinsic_params = parameter_mask.shape
        config: dict[str, list[str]] = {}

        for i in range(num_rows):
            # Skip padded rows if requested
            if ignore_padding and not parameter_mask[i].any():
                continue

            if i == 0 and keep_intercept:
                key = "1"
            else:
                key = f"u_{i + 1}"

            config[key] = []
            for j in range(num_intrinsic_params):
                if parameter_mask[i, j] == 1.0:
                    config[key].append(intrinsic_params[j])

        return config

    def sample_parameter_matrix(
        self,
        parameter_mask: np.ndarray,
        intrinsic_params: list[str],
        prior_fun: dict[str, Callable | dict[str, Callable]],
        keep_intercept: bool = False,
    ) -> np.ndarray:
        """
        Sample parameter matrix based on the given parameter mask and the associated priors.
        This has shape (num_regressors, num_intrinsic_params).
        """
        num_regressors, num_intrinsic_params = parameter_mask.shape
        parameter_matrix = np.zeros((num_regressors, num_intrinsic_params))

        # Handle empty parameter_mask
        if num_regressors == 0:
            return parameter_matrix

        # Heuristic: treat the very first row as intercept if any nonzero exists there.
        has_intercept = keep_intercept and parameter_mask[0].any()

        for design_index in range(num_regressors):
            is_intercept_row = has_intercept and (design_index == 0)

            for param_index, intrinsic_param in enumerate(intrinsic_params):
                # Sample prior if parameter or regressor is not masked
                if parameter_mask[design_index, param_index] == 1.0:
                    if (design_index == 0) and has_intercept:
                        sampler = prior_fun[intrinsic_param]["intercept"]
                    else:
                        sampler = prior_fun[intrinsic_param]["slope"]

                    parameter_matrix[design_index, param_index] = sampler()

        return parameter_matrix


    def _discretize_column(
        self,
        z: np.ndarray,
        k: int,
        category_concentration: float | None = None,
    ) -> np.ndarray:
        """
        Discretize a latent continuous column z into k dummy-coded categories.

        Category probabilities are sampled from Dirichlet(category_concentration * 1_k), then used
        as quantile boundaries on z.  category_concentration controls skewness:
          category_concentration << 1  → one category dominates (heavy imbalance)
          category_concentration = 1   → uniform over the simplex (random imbalance)
          category_concentration >> 1  → nearly equal-sized categories
        When category_concentration is None it is sampled from Uniform(0.1, 2.0).

        Returns an (n, k-1) float32 array of dummy variables (reference category dropped).
        """
        category_concentration_val = float(category_concentration) if category_concentration is not None else np.random.uniform(0.1, 2.0)
        p = np.random.dirichlet(category_concentration_val * np.ones(k))
        cuts = np.cumsum(p)[:-1]          # k-1 quantile boundaries in (0, 1)
        boundaries = np.quantile(z, cuts)
        bin_ids = np.digitize(z, boundaries)   # values in {0, ..., k-1}
        one_hot = (bin_ids[:, None] == np.arange(k)[None, :]).astype(np.float32)
        return one_hot[:, 1:]             # drop reference category

    def sample_dummies(
        self,
        num_obs: int,
        min_num_categories: int = 2,
        max_num_categories: int = 4,
    ):
        """
        Generate categorical dummy encoding for the design matrix.
        """
        # Randomly generate a num_categories
        num_categories = np.random.randint(min_num_categories, max_num_categories + 1)

        p = float(1.0 / num_categories)
        one_hot = np.random.multinomial(1, [p] * num_categories, size=num_obs)
        dummies = one_hot[:, :num_categories - 1]
        return dummies

    def build_column_labels(
        self,
        design_config: dict[str, list[str]],
        max_num_categories: int = 4,
        keep_intercept: bool = False,
    ) -> list[str]:
        """
        Build human-readable column labels aligned with build_design_matrix() output.

        Only for prior predictive checks.
        """
        regressor_keys = list(design_config.keys())
        has_intercept = "1" in regressor_keys
        regressor_keys = [k for k in regressor_keys if k != "1"]

        block_width = max_num_categories - 1
        labels: list[str] = []

        if has_intercept:
            labels.append("1")

        for key in regressor_keys:
            # Reserve a whole block; first col usually holds continuous regressor
            labels.append(key)
            for j in range(1, block_width):
                labels.append(f"{key}[{j+1}]")  # 2..block_width as dummy-like slots

        return labels
