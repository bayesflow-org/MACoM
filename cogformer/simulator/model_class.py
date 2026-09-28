import os
import numpy as np
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .model_family import ModelFamily, _init_numba_threads, resolve_deferred


class ModelClass:
    """
    A collection of NestedModelFamily instances representing multiple cognitive
    model families. During simulation, a model is sampled per batch element,
    enabling a single GPT to perform inference across all families and design
    configurations.

    Parameters
    ----------
    model_families : dict[str, NestedModelFamily]
        Mapping from model name to its NestedModelFamily instance.
    link_funs : dict[str, dict | Callable]
        Per-model link functions, keyed by model name.
    weights : list[float] | None
        Sampling weights over model families. Uniform if None.
    train_contexts : dict[str, Callable] | None
        Optional per-model context builders, keyed by model name. Each is called
        as fn(num_obs) per batch element in batch_sample, overriding the model's
        build_default_context (e.g. RDM.build_random_context to train over 2-4
        alternatives, matching the model_family training distribution).
    """

    def __init__(
        self,
        model_families: dict[str, ModelFamily],
        link_funs: dict[str, dict | Callable],
        weights: list[float] | None = None,
        num_workers: int | None = None,
        train_contexts: dict[str, Callable] | None = None,
    ):
        if set(model_families.keys()) != set(link_funs.keys()):
            raise ValueError("model_families and link_funs must have the same keys.")

        self.model_families = model_families
        self.link_funs = link_funs
        self.model_names = list(model_families.keys())
        self.model_index = {name: i for i, name in enumerate(self.model_names)}
        self.num_models = len(self.model_names)

        # Build global parameter space as ordered union across all model families
        seen_order: dict[str, int] = {}
        for mf in model_families.values():
            for p in mf.intrinsic_params:
                if p not in seen_order:
                    seen_order[p] = len(seen_order)
        self.global_param_names: list[str] = list(seen_order.keys())
        self.n_global_params: int = len(self.global_param_names)

        # Per-model local→global index mappings and binary model masks
        self.local_to_global: dict[str, list[int]] = {}
        self.model_masks: dict[str, np.ndarray] = {}
        for name, mf in model_families.items():
            g_indices = [self.global_param_names.index(p) for p in mf.intrinsic_params]
            self.local_to_global[name] = g_indices
            mask = np.zeros(self.n_global_params, dtype=np.float32)
            mask[g_indices] = 1.0
            self.model_masks[name] = mask

        if weights is not None:
            weights = np.array(weights, dtype=np.float64)
            weights /= weights.sum()
        self.weights = weights
        self.train_contexts = train_contexts or {}
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

    def batch_sample(
        self,
        batch_size: int,
        num_obs: int | None = None,
        design_config: dict[str, list[str]] | None = None,
        num_regressors: int | None = None,
        min_num_obs: int = 10,
        max_num_obs: int = 600,
        num_obs_bucket: int = 1,
        min_num_regressors: int = 0,
        max_num_regressors: int = 3,
        max_num_categories: int = 3,
        discrete_prob: float = 0.5,
        keep_intercept: bool = True,
        remove_intercept_on_collate: bool = False,
        mask_randomizer_kwargs: dict | None = None,
        flatten_param_outputs: bool = True,
        fixed_config: bool = False,
        add_interaction: bool = False,
        design_generator: str = "mixture",
        coefficient_prior: str = "cell_means",
        design_generator_kwargs: dict | None = None,
        coefficient_prior_kwargs: dict | None = None,
    ) -> dict:
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
            num_regressors_array = np.random.randint(
                min_num_regressors, max_num_regressors + 1, size=batch_size
            )
        else:
            num_regressors_array = np.full(batch_size, num_regressors)

        sampled_names = np.random.choice(
            self.model_names, size=batch_size, p=self.weights
        )

        def _sample_one(i: int) -> dict:
            model_name = sampled_names[i]
            mf = self.model_families[model_name]

            ctx_fn = self.train_contexts.get(model_name)
            if ctx_fn is not None:
                ctx_i = ctx_fn(num_obs)
            elif hasattr(mf.model, "build_default_context"):
                ctx_i = mf.model.build_default_context(num_obs=num_obs)
            else:
                ctx_i = None

            sim_instance = mf.sample(
                design_config=design_config,
                num_obs=num_obs,
                num_regressors=int(num_regressors_array[i]),
                context=ctx_i,
                mask_randomizer_kwargs=mask_randomizer_kwargs,
                max_num_categories=max_num_categories,
                discrete_prob=discrete_prob,
                keep_intercept=keep_intercept,
                flatten_param_outputs=False,
                fixed_config=fixed_config,
                add_interaction=add_interaction,
                link_fun=self.link_funs[model_name],
                design_generator=design_generator,
                coefficient_prior=coefficient_prior,
                design_generator_kwargs=design_generator_kwargs,
                coefficient_prior_kwargs=coefficient_prior_kwargs,
                defer_simulation=mf.model.uses_gpu,
            )
            sim_instance["model_id"] = self.model_index[model_name]
            sim_instance["num_obs"] = num_obs
            sim_instance["num_regressors"] = int(num_regressors_array[i])
            sim_instance["max_num_regressors"] = max_num_regressors
            sim_instance["max_num_categories"] = max_num_categories
            return sim_instance

        if self._num_workers == 1:
            list_batch = [_sample_one(i) for i in range(batch_size)]
        else:
            list_batch = list(self._pool.map(_sample_one, range(batch_size)))

        # GPU-backed families were sampled with defer_simulation: one launch per family.
        for name, mf in self.model_families.items():
            if mf.model.uses_gpu:
                resolve_deferred(mf.model, [b for b in list_batch if b["model_name"] == name])

        return self.collate(
            list_batch,
            flatten_param_outputs=flatten_param_outputs,
            remove_intercept=remove_intercept_on_collate,
        )

    def collate(
        self,
        list_batch: list[dict],
        flatten_param_outputs: bool = True,
        remove_intercept: bool = False,
    ) -> dict:
        from collections import defaultdict

        batch_size = len(list_batch)
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
        n_global = self.n_global_params

        # Pre-allocate global output arrays (params always 3D internally; flattened at end if needed)
        design_matrices = np.zeros((batch_size, max_num_obs, max_num_cols))
        param_masks = np.zeros((batch_size, max_num_cols, n_global))
        param_matrices = np.zeros((batch_size, max_num_cols, n_global))
        regressor_masks = np.zeros((batch_size, max_num_cols))
        position_codes = np.zeros((batch_size, max_num_cols, 4), dtype=np.int64)
        discrete_masks = -np.ones((batch_size, max_total_regressors))
        num_obs_array = np.zeros((batch_size, 1))
        num_regressors_array = np.zeros((batch_size, 1))
        num_alternatives_array = np.zeros(batch_size, dtype=np.int32)
        model_ids = np.zeros(batch_size, dtype=np.int64)

        all_sim_keys = set().union(*(b["obs"].keys() for b in list_batch))
        sim_data = {k: np.zeros((batch_size, max_num_obs, 1)) for k in all_sim_keys}

        model_names_out = [None] * batch_size
        design_configs = [None] * batch_size

        # Group by model_name, preserving original indices
        groups: dict[str, list[dict]] = defaultdict(list)
        group_indices: dict[str, list[int]] = defaultdict(list)
        for i, b in enumerate(list_batch):
            groups[b["model_name"]].append(b)
            group_indices[b["model_name"]].append(i)

        for model_name, items in groups.items():
            mf = self.model_families[model_name]
            indices = group_indices[model_name]

            # Delegate common collation to ModelFamily (3D params, no intercept removal)
            local = mf.collate(items, flatten_param_outputs=False, remove_intercept=False)

            local_n_obs = local["design_matrices"].shape[1]

            # Copy common fields into global arrays at the correct batch positions
            design_matrices[indices, :local_n_obs, :] = local["design_matrices"]
            regressor_masks[indices] = local["regressor_masks"]
            position_codes[indices] = local["position_codes"]
            discrete_masks[indices] = local["discrete_masks"]
            num_obs_array[indices] = local["num_obs"]
            num_regressors_array[indices] = local["num_regressors"]
            num_alternatives_array[indices] = local["num_alternatives"]
            model_ids[indices] = self.model_index[model_name]

            for k in all_sim_keys:
                if k in local["sim_data"]:
                    v = local["sim_data"][k]
                    sim_data[k][indices, :v.shape[1]] = v

            for local_i, global_i in enumerate(indices):
                model_names_out[global_i] = model_name
                design_configs[global_i] = items[local_i]["design_config"]

            # Lift params from model-local to global parameter space
            for local_idx, global_idx in enumerate(self.local_to_global[model_name]):
                param_matrices[indices, :, global_idx] = local["param_matrices"][:, :, local_idx]
                param_masks[indices, :, global_idx] = local["param_masks"][:, :, local_idx]

        if remove_intercept:
            design_matrices = design_matrices[:, :, 1:]

        if flatten_param_outputs:
            param_matrices = param_matrices.reshape(batch_size, max_num_cols * n_global)
            param_masks = param_masks.reshape(batch_size, max_num_cols * n_global)

        return {
            "model_names": model_names_out,
            "model_ids": model_ids,
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
            "max_num_regressors": max_num_regressors,
            "max_num_categories": max_num_categories,
            "n_global_params": self.n_global_params,
            "n_params": self.n_global_params,
            "num_models": self.num_models,
        }

    def lift_to_global_space(
        self,
        model_name: str,
        param_matrices: np.ndarray,
        param_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reindex flattened param_matrices/param_masks from model-local to global param positions.

        Parameters
        ----------
        model_name : str
        param_matrices : ndarray, shape (batch_size, num_cols * n_local)
        param_masks    : ndarray, shape (batch_size, num_cols * n_local)

        Returns
        -------
        (param_matrices_global, param_masks_global), both shape (batch_size, num_cols * n_global_params)
        """
        n_local = len(self.model_families[model_name].intrinsic_params)
        n_global = self.n_global_params
        batch_size = param_matrices.shape[0]
        n_rows = param_matrices.shape[1] // n_local
        global_indices = self.local_to_global[model_name]

        pm_local = param_matrices.reshape(batch_size, n_rows, n_local)
        pmask_local = param_masks.reshape(batch_size, n_rows, n_local)

        pm_global = np.zeros((batch_size, n_rows, n_global), dtype=param_matrices.dtype)
        pmask_global = np.zeros((batch_size, n_rows, n_global), dtype=param_masks.dtype)

        for local_idx, global_idx in enumerate(global_indices):
            pm_global[:, :, global_idx] = pm_local[:, :, local_idx]
            pmask_global[:, :, global_idx] = pmask_local[:, :, local_idx]

        return (
            pm_global.reshape(batch_size, n_rows * n_global),
            pmask_global.reshape(batch_size, n_rows * n_global),
        )
