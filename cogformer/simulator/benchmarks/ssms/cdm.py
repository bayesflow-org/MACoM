import numpy as np
from numba import njit, prange

from cogformer.simulator import Model
from cogformer.utils.simulator_utils import as_1d
from cogformer.utils.simulator_utils import contaminate


@njit
def simulate_cdm_trial(
    v: np.ndarray,
    a: float,
    tau: float,
    s_v: float,
    s_tau: float,
    decay: float = 0.0,
    dt: float = 0.001,
    sigma: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    v_i = v + s_v * np.random.randn(2)
    tau_i = tau + np.random.uniform(0, s_tau)

    c = np.sqrt(dt) * sigma
    x = np.zeros(2)
    for i_iter in range(max_steps):
        x[0] += v_i[0] * dt + c * np.random.randn()
        x[1] += v_i[1] * dt + c * np.random.randn()
        threshold_sq = a * a if decay == 0.0 else (a * np.exp(-decay * i_iter * dt)) ** 2
        if x[0] * x[0] + x[1] * x[1] >= threshold_sq:
            return np.array([tau_i + i_iter * dt, np.arctan2(x[1], x[0])])
    return np.array([-4.0, -4.0])


@njit(fastmath=True, cache=True)
def _simulate_cdm_serial(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    # Fast path for the constant-threshold (decay == 0) case: squared threshold
    # and per-trial drift increments hoisted out of the step loop, and per-step
    # scalar normal draws replaced by block-pregenerated noise (see
    # _simulate_ddm_serial in ddm.py). Measured 2.3x vs the trial-loop kernel
    # with an identical sampling distribution; serial only — the gain is zero
    # under prange. Draws unused after a boundary hit are discarded (iid).
    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)
    block = 128
    c = np.sqrt(dt) * sigma

    for i in range(n):
        v_i0 = v[i, 0] + s_v[i] * np.random.randn()
        v_i1 = v[i, 1] + s_v[i] * np.random.randn()
        tau_i = tau[i] + np.random.uniform(0, s_tau[i])
        threshold_sq = a[i] * a[i]
        d0 = v_i0 * dt
        d1 = v_i1 * dt
        x0 = 0.0
        x1 = 0.0
        rt = -4.0
        theta = -4.0
        i_iter = 0
        while i_iter < max_steps and rt == -4.0:
            k = min(block, max_steps - i_iter)
            noise = np.random.standard_normal(2 * k)
            for j in range(k):
                x0 += d0 + c * noise[2 * j]
                x1 += d1 + c * noise[2 * j + 1]
                if x0 * x0 + x1 * x1 >= threshold_sq:
                    rt = tau_i + i_iter * dt
                    theta = np.arctan2(x1, x0)
                    break
                i_iter += 1
        sim_data[i, 0] = rt
        sim_data[i, 1] = theta

    return sim_data


@njit(parallel=True)
def _simulate_cdm_parallel(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    decay: float = 0.0,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000
):
    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)

    for i in prange(n):
        sim_trial = simulate_cdm_trial(
            v=v[i],
            a=a[i],
            tau=tau[i],
            s_v=s_v[i],
            s_tau=s_tau[i],
            decay=decay,
            dt=dt,
            sigma=sigma,
            max_steps=max_steps,
        )
        sim_data[i] = sim_trial

    return sim_data


def simulate_cdm(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    decay: float = 0.0,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000
):
    """Dispatch to the fast serial kernel when numba has a single thread
    (the training sim-worker configuration) and the threshold is constant
    (decay == 0), else the prange kernel."""
    import numba
    if numba.get_num_threads() == 1 and decay == 0.0:
        return _simulate_cdm_serial(v, a, tau, s_v, s_tau, sigma, dt, max_steps)
    return _simulate_cdm_parallel(v, a, tau, s_v, s_tau, decay, sigma, dt, max_steps)


class CDM(Model):

    def __init__(self, dt: float = 5e-4, max_steps: int = 20000, decay: float = 0.0, pi: float = 0.0):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.decay = decay
        self.pi = pi

    def prepare_params(
            self,
            params: dict[str, np.ndarray],
            num_obs: int,
            context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        CDM expects:
          - v: (num_obs, 2) Cartesian drift per trial
          - a, tau, decay: (num_obs,)
        Polar form: v (magnitude) and v_theta (angle in radians).
        """
        context = context or {}

        v_theta  = as_1d(params["v_theta"], num_obs, "v_theta")
        v        = as_1d(params["v"],       num_obs, "v")
        a        = as_1d(params["a"],       num_obs, "a")
        tau      = as_1d(params["tau"],     num_obs, "tau")
        s_v      = as_1d(params["s_v"],     num_obs, "s_v")
        s_tau    = as_1d(params["s_tau"],   num_obs, "s_tau")
        pi       = as_1d(params.get("pi", self.pi), num_obs)

        # Polar -> Cartesian
        v_x = v * np.cos(v_theta)
        v_y = v * np.sin(v_theta)
        v   = np.stack([v_x, v_y], axis=1)

        return {"v": v, "a": a, "tau": tau, "s_v": s_v, "s_tau": s_tau, "pi": pi}

    @staticmethod
    def build_context(num_obs: int, theta_mode: str = "random_uniform") -> dict[str, np.ndarray]:
        """
        Optional helper to generate per-trial headings when using (v, v_theta).
        theta_mode:
          - 'random_uniform': theta ~ U[-pi, pi)
          - 'zeros': theta = 0 for all trials
        """
        if theta_mode == "random_uniform":
            theta = np.random.uniform(-np.pi, np.pi, size=num_obs)
        elif theta_mode == "zeros":
            theta = np.zeros(num_obs)
        else:
            raise ValueError("Unsupported theta_mode.")
        return {"theta": theta}

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        theta_mode = np.random.choice(["random_uniform", "zeros"])
        return CDM.build_context(num_obs, theta_mode)

    supports_gpu = True

    def _finish(self, results, pi):
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        pi_val = pi if pi is not None else self.pi
        rts, choices = contaminate(rts, choices, pi=pi_val, is_circular=True)
        return {"rts": rts, "choices": choices}

    def simulate(self, params: dict[str, np.ndarray], context=None):
        if self.uses_gpu:
            return self.simulate_many([(params, context)])[0]
        cdm_params = {k: v for k, v in params.items() if k != "pi"}
        results = simulate_cdm(**cdm_params, decay=self.decay, dt=self.dt, max_steps=self.max_steps)
        return self._finish(results, params.get("pi", None))

    def simulate_many(self, requests):
        if not self.uses_gpu:
            return super().simulate_many(requests)
        if self.decay != 0.0:
            raise NotImplementedError("GPU CDM kernel supports constant thresholds only (decay == 0)")
        from .gpu_kernels import simulate_cdm_gpu
        lens = [p["v"].shape[0] for p, _ in requests]
        v = np.concatenate([np.asarray(p["v"]) for p, _ in requests])
        cat = {k: np.concatenate([np.asarray(p[k]).reshape(-1) for p, _ in requests])
               for k in ("a", "tau", "s_v", "s_tau")}
        results = simulate_cdm_gpu(v, **cat, sigma=1.0, dt=self.dt, max_steps=self.max_steps)
        return [self._finish(r, p.get("pi", None))
                for r, (p, _) in zip(np.split(results, np.cumsum(lens)[:-1]), requests)]
