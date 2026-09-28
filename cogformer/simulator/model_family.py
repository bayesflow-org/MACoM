import os
import numpy as np
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .model import Model
from .design_utils import (
    build_design_config,
    build_random_design_config,
    build_parameter_mask,
    build_random_parameter_mask,
    build_intrinsic_priors,
    build_discrete_mask,
    build_design_matrix,
    build_link_functions,
    build_regressor_mask,
    sample_parameter_matrix,
    normalize_design_config,
    design_config_signature,
    build_position_codes,
)
from .coefficient_priors import sample_parameter_matrix_cell_means

from ..utils.simulator_utils import softplus, inspect


def _init_numba_threads(n: int) -> None:
    try:
        import numba
        numba.set_num_threads(n)
    except ImportError:
        pass


class ModelFamily:
    def __init__(
        self,
        model: Model,
        prior: dict[str, dict[str, Callable]],
        intrinsic_params: list[str] | None = None,
        regressed_params: list[str] | None = None,
        mask_randomizer_kwargs: dict | None = None,
        name: str | None = None,
        num_workers: int | None = None,
    ):
        self.name = name
        self.model = model
        self.prior = prior

        if intrinsic_params is None:
            intrinsic_params = list(prior.keys())
        self.intrinsic_params = intrinsic_params
        self.regressed_params = regressed_params

        self.mask_randomizer_kwargs = mask_randomizer_kwargs
        # Optimal split: 6 numba threads per Python worker saturates simulation
        # without over-fragmenting the thread pool. E.g. 24 cores → 4 workers × 6 numba threads.
        default_workers = max(1, (os.cpu_count() or 1) // 6)
        self._num_workers: int = num_workers if num_workers is not None else default_workers
        self._executor: ThreadPoolExecutor | None = None
        self._holdout_signatures: set = set()

    @property
    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            try:
                import numba
                threads_per_worker = max(1, numba.get_num_threads() // self._num_workers)
            except ImportError:
                threads_per_worker = 1
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers,
                initializer=_init_numba_threads,
                initargs=(threads_per_worker,),
            )
        return self._executor

    def __del__(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def set_holdout_configs(self, holdout_configs: list[dict]) -> None:
        """Pre-compute holdout signatures so _resolve_design can reject matching designs."""
        self._holdout_signatures = {
            design_config_signature(hc["design_config"]) for hc in holdout_configs
        }

    @property
    def parameter_names(self) -> list[str]:
        return self.intrinsic_params

    def _resolve_design(
        self,
        design_config: dict[str, list[str]] | None,
        prior: dict,
        num_regressors: int,
        max_num_categories: int,
        keep_intercept: bool,
        free_prob: float | None,
        fixed_config: bool,
        add_interaction: bool,
        allowed_slope_patterns: list[frozenset[str]] | None,
        kwargs: dict,
    ) -> tuple[dict, np.ndarray, dict]:
        """
        Resolve design_config, parameter_mask, and intrinsic prior from the
        three possible specification modes:
          1. Explicit design_config provided — build mask directly from it.
          2. fixed_config=True — build a deterministic config from regressed_params
             (or a random-but-fixed config from mask_randomizer_kwargs), then mask.
          3. Fully random — sample both config and mask jointly.
        Returns (design_config, parameter_mask, intrinsic_prior).
        """
        is_intrinsic_priors = all(isinstance(v, dict) for v in prior.values())

        if design_config is not None:
            parameter_mask = build_parameter_mask(
                design_config=design_config,
                intrinsic_params=self.intrinsic_params,
                fixed_params=None,
                max_num_categories=max_num_categories,
                keep_intercept=keep_intercept,
            )
        elif fixed_config:
            if self.regressed_params is not None:
                design_config = build_design_config(
                    intrinsic_params=self.intrinsic_params,
                    regressed_params=self.regressed_params,
                    num_regressors=num_regressors,
                    keep_intercept=keep_intercept,
                    add_interaction=add_interaction,
                )
            else:
                while True:
                    design_config = build_random_design_config(
                        intrinsic_params=self.intrinsic_params,
                        free_intrinsics=kwargs.get("free_intrinsics"),
                        fixed_intrinsics=kwargs.get("fixed_intrinsics"),
                        num_regressors=num_regressors,
                        keep_intercept=keep_intercept,
                        add_interaction=add_interaction,
                        global_params=kwargs.get("global_params"),
                    )
                    if not self._holdout_signatures or design_config_signature(design_config) not in self._holdout_signatures:
                        break
            parameter_mask = build_parameter_mask(
                design_config=design_config,
                intrinsic_params=self.intrinsic_params,
                fixed_params=kwargs.get("fixed_intrinsics"),
                max_num_categories=max_num_categories,
                keep_intercept=keep_intercept,
            )
        else:
            while True:
                parameter_mask, design_config = build_random_parameter_mask(
                    intrinsic_params=self.intrinsic_params,
                    num_regressors=num_regressors,
                    max_num_categories=max_num_categories,
                    keep_intercept=keep_intercept,
                    free_prob=free_prob,
                    intercept_prob=kwargs.get("intercept_prob"),
                    free_intrinsics=kwargs.get("free_intrinsics"),
                    fixed_intrinsics=kwargs.get("fixed_intrinsics"),
                    add_interaction=add_interaction,
                    allowed_slope_patterns=allowed_slope_patterns,
                    global_params=kwargs.get("global_params"),
                )
                if not self._holdout_signatures or design_config_signature(design_config) not in self._holdout_signatures:
                    break

        if is_intrinsic_priors:
            intrinsic_prior = prior
        else:
            intrinsic_prior = build_intrinsic_priors(
                prior=prior,
                design_config=design_config,
                fixed_values=kwargs.get("fixed_values"),
            )

        return design_config, parameter_mask, intrinsic_prior

    def sample(
        self,
        design_config: dict[str, list[str]] = None,
        prior: dict[str, dict[str, Callable]] = None,
        num_obs: int = 10,
        num_regressors: int = 2,
        max_num_regressors: int = 5,
        max_num_categories: int = 4,
        min_num_categories: int = 2,
        link_fun: dict | Callable = softplus,
        context: dict[str, np.ndarray] | None = None,
        mask_randomizer_kwargs: dict | None = None,
        discrete_prob: float = 0.5,
        categorical_regressors: list[str] | None = None,
        free_prob: float | None = None,
        keep_intercept: bool = True,
        flatten_param_outputs: bool = True,
        debug: bool = False,
        fixed_config: bool = False,
        add_interaction: bool = False,
        eta: float | None = None,
        category_concentration: float | None = None,
        allowed_slope_patterns: list[frozenset[str]] | None = None,
        design_generator: str = "mixture",
        coefficient_prior: str = "cell_means",
        design_generator_kwargs: dict | None = None,
        coefficient_prior_kwargs: dict | None = None,
        design_matrix: np.ndarray | None = None,
        category_counts: dict[str, int] | None = None,
    ):
        """
        design_matrix : optional FIXED design, ``(num_obs, n_cols)`` in exactly the layout
            ``build_design_matrix`` produces for this (design_config, max_num_regressors,
            max_num_categories). Replaces ONLY the design draw: the coefficient prior,
            realized-support mask, fixed-value injection, context and simulation are the
            training code path unchanged. Used to validate on an empirical design (the
            real subjects' regressors), which the generators cannot reproduce.
            Requires ``category_counts`` ({regressor: realized dummy columns}, as
            build_design_matrix returns) and ``categorical_regressors``.
        design_generator : ``"mixture"`` (default, latent-class / nominal categories) or
            ``"quantile"`` (the pre-2026-07 shared-latent construction). See
            :func:`design_utils.build_design_matrix`.
        coefficient_prior : ``"cell_means"`` (default, one latent Gaussian per category,
            contrasts as differences) or ``"iid"`` (the original per-row ``N(0, 1)``,
            which is not exchangeable across categories). Both use the same ``N(0, 1)``
            coefficient scale. See :mod:`coefficient_priors`.
        """
        if prior is None:
            prior = self.prior
        kwargs = mask_randomizer_kwargs if mask_randomizer_kwargs is not None else (self.mask_randomizer_kwargs or {})

        if design_config is not None:
            design_config = normalize_design_config(design_config, max_num_regressors)

        design_config, parameter_mask, intrinsic_prior = self._resolve_design(
            design_config=design_config,
            prior=prior,
            num_regressors=num_regressors,
            max_num_categories=max_num_categories,
            keep_intercept=keep_intercept,
            free_prob=free_prob,
            fixed_config=fixed_config,
            add_interaction=add_interaction,
            allowed_slope_patterns=allowed_slope_patterns,
            kwargs=kwargs,
        )

        # Optional deterministic continuous(0)/categorical(1) assignment of the main
        # regressors (e.g. a validation case that needs a fixed mix). When given, the
        # same mask drives BOTH the reported discrete_mask and the design matrix so they
        # stay consistent; otherwise the mix is drawn at random per regressor as before.
        explicit_main_mask = None
        if categorical_regressors is not None:
            main_keys = [k for k in design_config if k != "1" and ":" not in k]
            explicit_main_mask = np.array(
                [1.0 if k in categorical_regressors else 0.0 for k in main_keys], dtype=np.float32
            )
        _, discrete_mask = build_discrete_mask(
            design_config=design_config, discrete_prob=discrete_prob,
            main_discrete_mask=explicit_main_mask,
        )

        if context is None and hasattr(self.model, 'build_default_context'):
            context = self.model.build_default_context(num_obs=num_obs)

        if design_matrix is not None:
            if category_counts is None or categorical_regressors is None:
                raise ValueError("a fixed design_matrix needs category_counts and "
                                 "categorical_regressors (they drive the prior and mask)")
            design_matrix = np.asarray(design_matrix, dtype=np.float32)
            if design_matrix.shape[0] != num_obs:
                raise ValueError(f"design_matrix has {design_matrix.shape[0]} rows, "
                                 f"num_obs={num_obs}")
            actual_col_counts = dict(category_counts)
            main_keys = [k for k in design_config if k != "1" and ":" not in k]
            design_diagnostics = {
                "categorical_keys": [k for k in main_keys if k in categorical_regressors],
                "continuous_keys": [k for k in main_keys if k not in categorical_regressors],
            }
        else:
            design_matrix, actual_col_counts, design_diagnostics = build_design_matrix(
                design_config=design_config,
                num_obs=num_obs,
                context=context,
                discrete_prob=discrete_prob,
                min_num_categories=min_num_categories,
                max_num_categories=max_num_categories,
                main_discrete_mask=explicit_main_mask,
                eta=eta,
                category_concentration=category_concentration,
                return_category_counts=True,
                return_diagnostics=True,
                generator=design_generator,
                **(design_generator_kwargs or {}),
            )

        # Rebuild the mask with actual category counts so that rows corresponding to
        # unused category levels (zero-padded design matrix columns) are excluded.
        # No fixed_params override here: for a randomized design the config is
        # authoritative, since fixed_intrinsics are probabilistically included via
        # build_random_design_config (intercept_prob ~ Beta(2,2)) and are estimation
        # targets when they appear.
        rebuilt_mask = build_parameter_mask(
            design_config=design_config,
            intrinsic_params=self.intrinsic_params,
            max_num_categories=max_num_categories,
            keep_intercept="1" in design_config,
            category_counts=actual_col_counts,
        )

        # ...but the rebuild must only ever REMOVE entries, never resurrect one that
        # _resolve_design deliberately excluded. On the deterministic build_design_config
        # path the intercept row carries every intrinsic, and fixed_params was the only
        # thing holding fixed_intrinsics out; dropping it silently made "fixed" params
        # estimation targets again (s_v/s_tau in the ddm/cdm fixed* cases). Intersecting
        # keeps the category-count refinement while preserving that exclusion.
        if rebuilt_mask.shape != parameter_mask.shape:
            raise ValueError(
                f"mask shape changed during category-count rebuild: "
                f"{parameter_mask.shape} -> {rebuilt_mask.shape}"
            )
        parameter_mask = rebuilt_mask * parameter_mask

        # Refine by realized support: a category level (or interaction cell) that
        # received zero observations leaves its design column all-zero, so the data
        # carries no information about that coefficient — it must not be an
        # estimation target. The category-count rebuild above only excludes the
        # structurally unused columns; this also drops columns that are empty by
        # chance (sparse Dirichlet category draws, unpopulated interaction cells).
        realized_cols = (np.abs(design_matrix) > 0).any(axis=0)
        parameter_mask[~realized_cols] = 0.0

        if coefficient_prior == "cell_means":
            parameter_matrix = sample_parameter_matrix_cell_means(
                parameter_mask=parameter_mask,
                design_config=design_config,
                intrinsic_params=self.intrinsic_params,
                prior=intrinsic_prior,
                max_num_categories=max_num_categories,
                category_counts=actual_col_counts,
                categorical_keys=design_diagnostics.get("categorical_keys"),
                keep_intercept=keep_intercept,
                **(coefficient_prior_kwargs or {}),
            )
        elif coefficient_prior == "iid":
            parameter_matrix = sample_parameter_matrix(
                parameter_mask=parameter_mask,
                prior=intrinsic_prior,
                intrinsic_params=self.intrinsic_params,
                keep_intercept=keep_intercept,
            )
        else:
            raise ValueError(
                f"coefficient_prior must be 'cell_means' or 'iid'; got {coefficient_prior!r}"
            )

        link_funs = build_link_functions(
            intrinsic_params=self.intrinsic_params,
            link_fun=link_fun,
            default_link_fun=None,
        )

        predictor = design_matrix @ parameter_matrix
        fixed_vals = kwargs.get("fixed_values", {})
        regressed_parameters = np.empty_like(predictor, dtype=np.float32)
        for j, name in enumerate(self.intrinsic_params):
            if name in fixed_vals and parameter_mask[:, j].sum() == 0:
                regressed_parameters[:, j] = fixed_vals[name]
            else:
                regressed_parameters[:, j] = link_funs[name](predictor[:, j]).astype(np.float32, copy=False)

        params = {
            name: regressed_parameters[:, j].astype(np.float32, copy=False)
            for j, name in enumerate(self.intrinsic_params)
        }
        params = self.model.prepare_params(params=params, num_obs=num_obs, context=context)
        obs = self.model.simulate(params, context=context)

        has_intercept_col = "1" in design_config
        regressor_mask = build_regressor_mask(
            design_config=design_config,
            max_num_categories=max_num_categories,
            keep_intercept=has_intercept_col,
        )

        if flatten_param_outputs:
            parameter_mask = parameter_mask.flatten()
            parameter_matrix = parameter_matrix.flatten()

        ctx = context or {}
        out = {
            "model_name": f"{self.name}",
            "design_config": design_config,
            "design_matrix": design_matrix,
            "param_mask": parameter_mask,
            "param_matrix": parameter_matrix,
            "obs": obs,
            "discrete_mask": discrete_mask,
            "regressor_mask": regressor_mask,
            "position_codes": build_position_codes(
                design_config=design_config,
                max_num_categories=max_num_categories,
            ),
            "max_num_regressors": max_num_regressors,
            "keep_intercept": has_intercept_col,
            "num_alternatives": int(ctx.get("num_alternatives", 2)),
            "per_trial_context": {
                k: np.asarray(v, dtype=np.float32)
                for k, v in ctx.items()
                if isinstance(v, np.ndarray) and v.shape == (num_obs,)
            },
        }

        if debug:
            inspect(out)

        return out

    def batch_sample(
        self,
        batch_size: int,
        num_obs: int | None = None,
        design_config: dict[str, list[str]] | None = None,
        prior: dict[str, dict[str, Callable]] = None,
        num_regressors: int | None = None,
        context: dict[str, np.ndarray] | None = None,
        link_fun: dict | Callable = softplus,
        min_num_obs: int = 10,
        max_num_obs: int = 600,
        num_obs_bucket: int = 1,
        min_num_regressors: int = 0,
        max_num_regressors: int = 3,
        max_num_categories: int = 3,
        min_num_categories: int = 2,
        discrete_prob: float = 0.5,
        categorical_regressors: list[str] | None = None,
        free_prob: float | None = None,
        keep_intercept: bool = True,
        remove_intercept_on_collate: bool = False,
        pad_to_max_num_categories: int | None = None,
        mask_randomizer_kwargs: dict | None = None,
        flatten_param_outputs: bool = True,
        fixed_config: bool = False,
        add_interaction: bool = False,
        eta: float | None = None,
        category_concentration: float | None = None,
        allowed_slope_patterns: list[frozenset[str]] | None = None,
        design_generator: str = "mixture",
        coefficient_prior: str = "cell_means",
        design_generator_kwargs: dict | None = None,
        coefficient_prior_kwargs: dict | None = None,
    ) -> list[dict] | dict:
        if prior is None:
            prior = self.prior

        if num_obs is None:
            num_obs = np.random.randint(min_num_obs, max_num_obs + 1)
            if num_obs_bucket > 1:
                # Quantize the observation count so batch shapes repeat. Drawn uniformly
                # over [200, 500] it takes ~300 distinct values, and torch.compile with
                # max-autotune re-runs its kernel autotuning sweep per new shape -- spikes
                # of up to 25 s against an 88 ms steady-state step. Bucketing collapses
                # that to a handful of shapes that are all compiled within the first epoch.
                num_obs = int(min(
                    max_num_obs,
                    -(-num_obs // num_obs_bucket) * num_obs_bucket,
                ))

        if num_regressors is None:
            num_regressors_array = np.random.randint(min_num_regressors, max_num_regressors + 1, size=batch_size)
        else:
            num_regressors_array = np.full(batch_size, num_regressors)

        if mask_randomizer_kwargs is None:
            mask_randomizer_kwargs = self.mask_randomizer_kwargs

        def _sample_one(i: int) -> dict:
            if context is None:
                ctx_i = self.model.build_default_context(num_obs=num_obs) if hasattr(self.model, 'build_default_context') else None
            elif callable(context):
                ctx_i = context(num_obs)
            else:
                ctx_i = {k: v(num_obs) if callable(v) else v for k, v in context.items()}

            sim_instance = self.sample(
                design_config=design_config,
                prior=prior,
                num_obs=num_obs,
                context=ctx_i,
                mask_randomizer_kwargs=mask_randomizer_kwargs,
                num_regressors=int(num_regressors_array[i]),
                max_num_regressors=max_num_regressors,
                discrete_prob=discrete_prob,
                free_prob=free_prob,
                keep_intercept=keep_intercept,
                max_num_categories=max_num_categories,
                min_num_categories=min_num_categories,
                categorical_regressors=categorical_regressors,
                flatten_param_outputs=flatten_param_outputs,
                fixed_config=fixed_config,
                add_interaction=add_interaction,
                link_fun=link_fun,
                eta=eta,
                category_concentration=category_concentration,
                allowed_slope_patterns=allowed_slope_patterns,
                design_generator=design_generator,
                coefficient_prior=coefficient_prior,
                design_generator_kwargs=design_generator_kwargs,
                coefficient_prior_kwargs=coefficient_prior_kwargs,
            )
            sim_instance["num_obs"] = num_obs
            sim_instance["num_regressors"] = num_regressors_array[i]
            sim_instance["max_num_regressors"] = max_num_regressors
            sim_instance["max_num_categories"] = max_num_categories
            return sim_instance

        if self._num_workers == 1:
            list_batch = [_sample_one(i) for i in range(batch_size)]
        else:
            list_batch = list(self._pool.map(_sample_one, range(batch_size)))

        return self.collate(
            list_batch,
            flatten_param_outputs=flatten_param_outputs,
            remove_intercept=remove_intercept_on_collate,
            pad_to_max_num_categories=pad_to_max_num_categories,
        )

    def collate(
        self,
        list_batch: list[dict],
        flatten_param_outputs: bool = True,
        remove_intercept: bool = True,
        pad_to_max_num_categories: int | None = None,
    ) -> dict[str, np.ndarray]:

        batch_size = len(list_batch)
        num_params = len(self.intrinsic_params)
        max_num_obs = max(b["num_obs"] for b in list_batch)
        max_num_regressors = list_batch[0]["max_num_regressors"]
        max_num_categories = list_batch[0]["max_num_categories"]

        block_width = max_num_categories - 1
        max_total_regressors = max_num_regressors * (max_num_regressors + 1) // 2
        n_main = max_num_regressors
        n_inter = max_num_regressors * (max_num_regressors - 1) // 2
        max_num_cols = (
            n_main * block_width
            + n_inter * block_width ** 2
            + (1 if list_batch[0]["keep_intercept"] else 0)
        )

        if flatten_param_outputs:
            param_outputs_shape = (batch_size, max_num_cols * num_params)
        else:
            param_outputs_shape = (batch_size, max_num_cols, num_params)

        design_matrices = np.zeros((batch_size, max_num_obs, max_num_cols), dtype=np.float32)
        param_masks = np.zeros(param_outputs_shape, dtype=np.float32)
        param_matrices = np.zeros(param_outputs_shape, dtype=np.float32)
        regressor_masks = np.zeros((batch_size, max_num_cols), dtype=np.float32)
        position_codes = np.zeros((batch_size, max_num_cols, 4), dtype=np.int64)
        discrete_masks = -np.ones((batch_size, max_total_regressors))
        num_obs_array = np.zeros((batch_size, 1))
        num_regressors_array = np.zeros((batch_size, 1))
        num_alternatives_array = np.zeros(batch_size, dtype=np.int32)

        model_names, design_configs = [], []

        sim_keys = list_batch[0]["obs"].keys()
        sim_data = {k: np.empty((batch_size, max_num_obs, 1), dtype=np.float32) for k in sim_keys}
        ptc_keys = list(list_batch[0].get("per_trial_context", {}).keys())
        per_trial_context = {k: np.zeros((batch_size, max_num_obs), dtype=np.float32) for k in ptc_keys}

        for i, b in enumerate(list_batch):
            model_names.append(b["model_name"])
            design_configs.append(b["design_config"])

            num_obs = b["num_obs"]
            num_cols = b["design_matrix"].shape[1]

            design_matrices[i, :num_obs, :num_cols] = b["design_matrix"]

            if flatten_param_outputs:
                param_masks[i, :num_cols * num_params] = b["param_mask"]
                param_matrices[i, :num_cols * num_params] = b["param_matrix"]
            else:
                param_masks[i, :num_cols, :num_params] = b["param_mask"]
                param_matrices[i, :num_cols, :num_params] = b["param_matrix"]

            regressor_masks[i, :num_cols] = b["regressor_mask"]
            position_codes[i, :num_cols] = b["position_codes"]
            discrete_masks[i, :b["discrete_mask"].shape[0]] = b["discrete_mask"]
            num_obs_array[i] = b["num_obs"]
            num_regressors_array[i] = b["num_regressors"]
            num_alternatives_array[i] = b.get("num_alternatives", 2)
            for k, arr in b.get("per_trial_context", {}).items():
                if k in per_trial_context:
                    per_trial_context[k][i, :num_obs] = arr

            for k in sim_keys:
                v = b["obs"][k]
                sim_data[k][i, :v.shape[0]] = v

        if pad_to_max_num_categories is not None and pad_to_max_num_categories > max_num_categories:
            wide_bw = pad_to_max_num_categories - 1
            wide_inter_bw = wide_bw ** 2
            inter_bw = block_width ** 2
            has_intercept = bool(list_batch[0]["keep_intercept"])
            wide_cols = n_main * wide_bw + n_inter * wide_inter_bw + (1 if has_intercept else 0)

            col_map: dict[int, int] = {}
            nc, wc = 0, 0
            if has_intercept:
                col_map[0] = 0
                nc, wc = 1, 1
            for _ in range(n_main):
                for d in range(block_width):
                    col_map[nc + d] = wc + d
                nc += block_width
                wc += wide_bw
            for _ in range(n_inter):
                for d in range(inter_bw):
                    col_map[nc + d] = wc + d
                nc += inter_bw
                wc += wide_inter_bw

            wide_dm = np.zeros((batch_size, max_num_obs, wide_cols))
            for nc_idx, wc_idx in col_map.items():
                wide_dm[:, :, wc_idx] = design_matrices[:, :, nc_idx]
            design_matrices = wide_dm

            wide_pc = np.zeros((batch_size, wide_cols, 4), dtype=np.int64)
            for nc_idx, wc_idx in col_map.items():
                wide_pc[:, wc_idx] = position_codes[:, nc_idx]
            position_codes = wide_pc

            src_pm  = param_masks.reshape(batch_size, max_num_cols, num_params)
            src_pmx = param_matrices.reshape(batch_size, max_num_cols, num_params)
            dst_pm  = np.zeros((batch_size, wide_cols, num_params))
            dst_pmx = np.zeros((batch_size, wide_cols, num_params))
            for nc_idx, wc_idx in col_map.items():
                dst_pm[:, wc_idx, :]  = src_pm[:, nc_idx, :]
                dst_pmx[:, wc_idx, :] = src_pmx[:, nc_idx, :]
            param_masks    = dst_pm.reshape(batch_size, wide_cols * num_params) if flatten_param_outputs else dst_pm
            param_matrices = dst_pmx.reshape(batch_size, wide_cols * num_params) if flatten_param_outputs else dst_pmx

            max_num_categories = pad_to_max_num_categories

        if remove_intercept:
            design_matrices = design_matrices[:, :, 1:]

        return {
            "model_names": model_names,
            "design_configs": design_configs,
            "design_matrices": design_matrices,
            "param_masks": param_masks,
            "param_matrices": param_matrices,
            "sim_data": sim_data,
            "regressor_masks": regressor_masks,
            "position_codes": position_codes,
            "discrete_masks": discrete_masks,
            "num_obs": num_obs_array,
            "num_regressors": num_regressors_array,
            "num_alternatives": num_alternatives_array,
            "per_trial_context": per_trial_context,
            "max_num_regressors": max_num_regressors,
            "max_num_categories": max_num_categories,
            "n_params": len(self.intrinsic_params),
        }
