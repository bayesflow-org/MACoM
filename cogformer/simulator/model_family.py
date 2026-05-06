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
)

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
                design_config = build_random_design_config(
                    intrinsic_params=self.intrinsic_params,
                    free_intrinsics=kwargs.get("free_intrinsics"),
                    fixed_intrinsics=kwargs.get("fixed_intrinsics"),
                    num_regressors=num_regressors,
                    keep_intercept=keep_intercept,
                    add_interaction=add_interaction,
                    global_params=kwargs.get("global_params"),
                )
            parameter_mask = build_parameter_mask(
                design_config=design_config,
                intrinsic_params=self.intrinsic_params,
                fixed_params=kwargs.get("fixed_intrinsics"),
                max_num_categories=max_num_categories,
                keep_intercept=keep_intercept,
            )
        else:
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
        link_fun: dict | Callable = softplus,
        context: dict[str, np.ndarray] | None = None,
        mask_randomizer_kwargs: dict | None = None,
        discrete_prob: float = 0.5,
        free_prob: float | None = None,
        keep_intercept: bool = True,
        flatten_param_outputs: bool = True,
        debug: bool = False,
        fixed_config: bool = False,
        add_interaction: bool = False,
        collinearity_df: float | None = None,
        category_concentration: float | None = None,
        allowed_slope_patterns: list[frozenset[str]] | None = None,
    ):
        if prior is None:
            prior = self.prior
        kwargs = mask_randomizer_kwargs if mask_randomizer_kwargs is not None else (self.mask_randomizer_kwargs or {})

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

        _, discrete_mask = build_discrete_mask(design_config=design_config, discrete_prob=discrete_prob)

        if context is None and hasattr(self.model, 'build_default_context'):
            context = self.model.build_default_context(num_obs=num_obs)

        design_matrix, actual_col_counts = build_design_matrix(
            design_config=design_config,
            num_obs=num_obs,
            context=context,
            discrete_prob=discrete_prob,
            max_num_categories=max_num_categories,
            collinearity_df=collinearity_df,
            category_concentration=category_concentration,
            return_category_counts=True,
        )

        # Rebuild the mask with actual category counts so that rows corresponding to
        # unused category levels (zero-padded design matrix columns) are excluded.
        parameter_mask = build_parameter_mask(
            design_config=design_config,
            intrinsic_params=self.intrinsic_params,
            max_num_categories=max_num_categories,
            keep_intercept="1" in design_config,
            category_counts=actual_col_counts,
        )

        parameter_matrix = sample_parameter_matrix(
            parameter_mask=parameter_mask,
            prior=intrinsic_prior,
            intrinsic_params=self.intrinsic_params,
            keep_intercept=keep_intercept,
        )

        link_funs = build_link_functions(
            intrinsic_params=self.intrinsic_params,
            link_fun=link_fun,
            default_link_fun=None,
        )

        predictor = design_matrix @ parameter_matrix
        regressed_parameters = np.empty_like(predictor, dtype=np.float32)
        for j, name in enumerate(self.intrinsic_params):
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

        out = {
            "model_name": f"{self.name}",
            "design_config": design_config,
            "design_matrix": design_matrix,
            "param_mask": parameter_mask,
            "param_matrix": parameter_matrix,
            "obs": obs,
            "discrete_mask": discrete_mask,
            "regressor_mask": regressor_mask,
            "max_num_regressors": max_num_regressors,
            "keep_intercept": has_intercept_col,
            "num_alternatives": int((context or {}).get("num_alternatives", 2)),
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
        min_num_regressors: int = 0,
        max_num_regressors: int = 3,
        max_num_categories: int = 3,
        discrete_prob: float = 0.5,
        free_prob: float | None = None,
        keep_intercept: bool = True,
        remove_intercept_on_collate: bool = False,
        mask_randomizer_kwargs: dict | None = None,
        flatten_param_outputs: bool = True,
        fixed_config: bool = False,
        add_interaction: bool = False,
        collinearity_df: float | None = None,
        category_concentration: float | None = None,
        allowed_slope_patterns: list[frozenset[str]] | None = None,
    ) -> list[dict] | dict:
        if prior is None:
            prior = self.prior

        if num_obs is None:
            num_obs = np.random.randint(min_num_obs, max_num_obs + 1)

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
                discrete_prob=discrete_prob,
                free_prob=free_prob,
                keep_intercept=keep_intercept,
                max_num_categories=max_num_categories,
                flatten_param_outputs=flatten_param_outputs,
                fixed_config=fixed_config,
                add_interaction=add_interaction,
                link_fun=link_fun,
                collinearity_df=collinearity_df,
                category_concentration=category_concentration,
                allowed_slope_patterns=allowed_slope_patterns,
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
        )

    def collate(
        self,
        list_batch: list[dict],
        flatten_param_outputs: bool = True,
        remove_intercept: bool = True,
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

        design_matrices = np.zeros((batch_size, max_num_obs, max_num_cols))
        param_masks = np.zeros(param_outputs_shape)
        param_matrices = np.zeros(param_outputs_shape)
        regressor_masks = np.zeros((batch_size, max_num_cols))
        discrete_masks = -np.ones((batch_size, max_total_regressors))
        num_obs_array = np.zeros((batch_size, 1))
        num_regressors_array = np.zeros((batch_size, 1))
        num_alternatives_array = np.zeros(batch_size, dtype=np.int32)

        model_names, design_configs = [], []

        sim_keys = list_batch[0]["obs"].keys()
        sim_data = {k: np.empty((batch_size, max_num_obs, 1)) for k in sim_keys}

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
            discrete_masks[i, :b["discrete_mask"].shape[0]] = b["discrete_mask"]
            num_obs_array[i] = b["num_obs"]
            num_regressors_array[i] = b["num_regressors"]
            num_alternatives_array[i] = b.get("num_alternatives", 2)

            for k in sim_keys:
                v = b["obs"][k]
                sim_data[k][i, :v.shape[0]] = v

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
            "discrete_masks": discrete_masks,
            "num_obs": num_obs_array,
            "num_regressors": num_regressors_array,
            "num_alternatives": num_alternatives_array,
            "max_num_regressors": max_num_regressors,
            "max_num_categories": max_num_categories,
            "n_params": len(self.intrinsic_params),
        }
