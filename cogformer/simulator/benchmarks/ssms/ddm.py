import numpy as np
from numba import njit, prange
from scipy.stats import halfnorm

from .priors import ddm_priors

from cogformer.simulator import Model
from cogformer.utils.simulator_utils import as_1d, contaminate


@njit
def simulate_ddm_trial(
    v: float,
    a: float,
    tau: float,
    s_v: float,
    s_tau: float,
    decay: float = 0.0,
    z: float = 0.0,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
):
    v_i = np.random.normal(v, s_v)
    tau_i = tau + np.random.uniform(0, s_tau)

    a0 = a if a > 1e-6 else 1e-6
    d  = decay if decay > 0.0 else 0.0

    x = z * a0
    t = tau_i
    c = sigma * np.sqrt(dt)

    for _ in range(max_steps):
        t += dt
        bound = a0 if d == 0.0 else a0 * np.exp(-d * t)
        if bound < 1e-3:
            bound = 1e-3
        x += v_i * dt + c * np.random.normal()
        if x >= bound:
            return np.array([t, 1.0], dtype=np.float32)
        if x <= -bound:
            return np.array([t, 0.0], dtype=np.float32)
    # No decision within max_steps
    return np.array([-1.0, -1.0], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _simulate_ddm_serial(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    # Same Euler-Maruyama update as simulate_ddm_trial (constant bounds), but the
    # per-step scalar np.random.normal() is replaced by block-pregenerated noise:
    # scalar draws dominated the loop cost, and drawing standard normals 128 at a
    # time measured 2.2x faster with an identical sampling distribution. Draws
    # left unused when a trial terminates mid-block are discarded (iid, so this
    # does not bias anything). NOTE: the speedup only materializes in a serial
    # njit kernel — under parallel=True, prange routes the array RNG through
    # numba's thread-safe parallel machinery and the gain measured zero, hence
    # the serial/parallel split with dispatch in simulate_ddm().

    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)
    block = 128

    for i in range(n):
        v_i = np.random.normal(v[i], s_v[i])
        t = tau[i] + np.random.uniform(0.0, s_tau[i])
        a0 = a[i] if a[i] > 1e-6 else 1e-6
        x = z[i] * a0
        drift = v_i * dt
        c = sigma * np.sqrt(dt)
        rt = -1.0
        choice = -1.0
        steps = 0
        while steps < max_steps and rt < 0.0:
            k = min(block, max_steps - steps)
            noise = np.random.standard_normal(k)
            for j in range(k):
                t += dt
                x += drift + c * noise[j]
                if x >= a0:
                    rt = t
                    choice = 1.0
                    break
                if x <= -a0:
                    rt = t
                    choice = 0.0
                    break
            steps += k
        sim_data[i, 0] = rt
        sim_data[i, 1] = choice

    return sim_data


@njit(parallel=True)
def _simulate_ddm_parallel(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:

    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)

    for i in prange(n):
        sim_trial = simulate_ddm_trial(
            v=v[i],
            a=a[i],
            tau=tau[i],
            s_tau=s_tau[i],
            s_v=s_v[i],
            z=z[i],
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        sim_data[i] = sim_trial

    return sim_data


def simulate_ddm(
    v: np.ndarray,
    a: np.ndarray,
    z: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    """Dispatch to the fast serial kernel when numba has a single thread
    (the training sim-worker configuration), else the prange kernel."""
    import numba
    if numba.get_num_threads() == 1:
        return _simulate_ddm_serial(v, a, z, tau, s_v, s_tau, sigma, dt, max_steps)
    return _simulate_ddm_parallel(v, a, z, tau, s_v, s_tau, sigma, dt, max_steps)


class DDM(Model):

    def __init__(self, dt: float = 5e-4, max_steps: int = 20000, pi: float = 0.0):
        self.dt = dt
        self.max_steps = max_steps
        self.pi = pi

    def prepare_params(
        self,
        params: dict[str, np.ndarray],
        num_obs: int,
        context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        v     = as_1d(params["v"],     num_obs)
        a     = as_1d(params["a"],     num_obs)
        z     = as_1d(params["z"],     num_obs)
        tau   = as_1d(params["tau"],   num_obs)
        s_v   = as_1d(params["s_v"],   num_obs)
        s_tau = as_1d(params["s_tau"], num_obs)
        pi    = as_1d(params.get("pi", self.pi), num_obs)
        return {"v": v, "a": a, "z": z, "tau": tau, "s_v": s_v, "s_tau": s_tau, "pi": pi}

    supports_gpu = True

    def _finish(self, results, pi):
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        pi_val = pi if pi is not None else self.pi
        rts, choices = contaminate(rts, choices, pi=pi_val, num_alternatives=2)
        return {"rts": rts, "choices": choices}

    def simulate(self, params: dict[str, np.ndarray], context=None):
        if self.uses_gpu:
            return self.simulate_many([(params, context)])[0]
        ddm_params = {k: v for k, v in params.items() if k != "pi"}
        results = simulate_ddm(**ddm_params, sigma=1.0, dt=self.dt, max_steps=self.max_steps)
        return self._finish(results, params.get("pi", None))

    def simulate_many(self, requests):
        if not self.uses_gpu:
            return super().simulate_many(requests)
        from .gpu_kernels import simulate_ddm_gpu
        lens = [len(p["v"]) for p, _ in requests]
        cat = {k: np.concatenate([np.asarray(p[k]).reshape(-1) for p, _ in requests])
               for k in ("v", "a", "z", "tau", "s_v", "s_tau")}
        results = simulate_ddm_gpu(**cat, sigma=1.0, dt=self.dt, max_steps=self.max_steps)
        return [self._finish(r, p.get("pi", None))
                for r, (p, _) in zip(np.split(results, np.cumsum(lens)[:-1]), requests)]

    def sample(self, batch_size: int | tuple, num_obs: int = 500, context=None):
        if isinstance(batch_size, tuple):
            batch_size = batch_size[0]

        params = {k: [] for k in ddm_priors().keys()}
        rts = []
        choices = []

        for i in range(batch_size):
            prior_draw = ddm_priors()
            for k, v in prior_draw.items():
                params[k].append(v)
            priors = {k: np.full(num_obs, v) for k, v in prior_draw.items()}
            results = self.simulate(params=priors, context=context)
            rts.append(results["rts"])
            choices.append(results["choices"])

        prior_draws = {k: np.array(v)[:, None] for k, v in params.items()}
        sim_data = {"rts": np.array(rts), "choices": np.array(choices)}
        return prior_draws | sim_data
