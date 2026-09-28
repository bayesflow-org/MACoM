import os

import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod
from collections.abc import Mapping


class Model(ABC):
    """
    Abstract base class for simulation-based inference models.

    Defines the interface for simulating data with num_obs trials per simulation,
    with batching handled by NestedModelFamily.
    """

    # Simulation backend. "numba" (default) runs the CPU kernels per dataset; "gpu" runs
    # the Triton kernels (gpu_kernels.py) and lets ModelFamily/ModelClass batch every
    # dataset of a batch into one launch. Resolved per instance: an explicit
    # ``sim_backend`` wins, else the COGFORMER_SIM_BACKEND environment variable (which
    # spawned simulation workers inherit), else "numba". Models without GPU kernels
    # (supports_gpu = False) always run on the CPU.
    sim_backend: str | None = None
    supports_gpu: bool = False

    @property
    def uses_gpu(self) -> bool:
        backend = self.sim_backend or os.environ.get("COGFORMER_SIM_BACKEND", "numba")
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
