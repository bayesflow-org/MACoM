import os

import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod
from collections.abc import Mapping


_DEFAULT_BACKEND: str | None = None


def _default_backend() -> str:
    """ "gpu" if CUDA is usable in this process, else "numba" (checked once)."""
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        try:
            import torch
            _DEFAULT_BACKEND = "gpu" if torch.cuda.is_available() else "numba"
        except ImportError:
            _DEFAULT_BACKEND = "numba"
    return _DEFAULT_BACKEND


class Model(ABC):
    """
    Abstract base class for simulation-based inference models.

    Defines the interface for simulating data with num_obs trials per simulation,
    with batching handled by NestedModelFamily.
    """

    # Simulation backend. "gpu" runs the Triton kernels (gpu_kernels.py) and lets
    # ModelFamily/ModelClass batch every dataset of a batch into one launch; "numba" runs
    # the CPU kernels per dataset. Resolved per instance: an explicit ``sim_backend``
    # wins, else the COGFORMER_SIM_BACKEND environment variable (which spawned
    # simulation workers inherit), else "gpu" when CUDA is available and "numba"
    # otherwise. The two are distributionally equivalent (experiments/
    # check_gpu_simulators.py) but NOT draw-for-draw identical: a seeded run reproduces
    # only on the backend it was made with.
    #
    # Never simulate on the GPU inside a process FORKED after CUDA was initialised
    # (CUDA does not survive fork): use spawn (make_sim_loader) or threads
    # (model_family_bf_train) -- or set COGFORMER_SIM_BACKEND=numba there.
    sim_backend: str | None = None
    supports_gpu: bool = False

    @property
    def uses_gpu(self) -> bool:
        backend = self.sim_backend or os.environ.get("COGFORMER_SIM_BACKEND") or _default_backend()
        if backend not in ("numba", "gpu"):
            raise ValueError(f"unknown simulation backend {backend!r} (numba | gpu)")
        return backend == "gpu" and self.supports_gpu

    def simulate_many(self, requests: list[tuple[dict, dict | None]]) -> list:
        """Simulate several datasets, given as (prepared params, context) pairs.

        Default: one simulate() call each. GPU-capable models override this to run all
        trials of all datasets in a single kernel launch.
        """
        return [self.simulate(params, context=context) for params, context in requests]

    def prepare_params(
        self,
        params: dict[str, float],
        num_obs: int,
        context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float | np.ndarray]:
        """Optional: models can normalize/broadcast shapes here. Default: passthrough."""
        return params

    @abstractmethod
    def simulate(
        self,
        params: dict[str, np.ndarray],
        context: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Mapping[str, np.ndarray]]:

        pass
