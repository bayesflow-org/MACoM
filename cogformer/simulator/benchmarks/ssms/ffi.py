import numpy as np
from numba import njit, prange

from cogformer.simulator import Model
from cogformer.utils.simulator_utils import as_1d, contaminate


@njit
def sample_ffi_trial(
    v: np.ndarray,
    k: float,
    beta: float,
    a: float,
    s_v: float,
    s_tau: float,
    tau: float,
    dt: float = 0.001,
    sigma: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    """Single FFI trial: feed-forward inhibition applied to inputs, then leaky accumulation."""
    n_c = v.shape[0]

    # Pre-compute effective drifts: each unit's input reduced by beta * sum of other inputs
    v_total = 0.0
    for c in range(n_c):
        v_total += v[c]

    v_eff = np.empty(n_c, dtype=np.float64)
    for c in range(n_c):
        v_eff[c] = v[c] - beta * (v_total - v[c])

    # Sample trial-level variability on effective drifts
    v_i = np.empty(n_c, dtype=np.float64)
    for c in range(n_c):
        v_i[c] = np.random.normal(v_eff[c], s_v)
    tau_i = tau + np.random.uniform(0.0, s_tau)

    x = np.zeros(n_c, dtype=np.float64)
    c_noise = sigma * np.sqrt(dt)

    for step in range(max_steps):
        for i in range(n_c):
            x[i] = x[i] + (v_i[i] - k * x[i]) * dt + c_noise * np.random.normal()
            if x[i] < 0.0:
                x[i] = 0.0
            if x[i] >= a:
                return np.array([tau_i + (step + 1) * dt, float(i)], dtype=np.float32)

    return np.array([-1.0, -1.0], dtype=np.float32)


@njit(parallel=True)
def simulate_ffi(
    v: np.ndarray,
    k: np.ndarray,
    beta: np.ndarray,
    a: np.ndarray,
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
        sim_data[i] = sample_ffi_trial(
            v=v[i],
            k=k[i],
            beta=beta[i],
            a=a[i],
            s_v=s_v[i],
            s_tau=s_tau[i],
            tau=tau[i],
            dt=dt,
            sigma=sigma,
            max_steps=max_steps,
        )
    return sim_data


class FFI(Model):

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
        context = context or {}
        num_alternatives = int(context.get("num_alternatives", 2))
        if num_alternatives < 2:
            raise ValueError("num_alternatives must be >= 2.")

        v_base = as_1d(params["v"],      num_obs)
        v_diff = as_1d(params["v_diff"], num_obs)
        k      = as_1d(params["k"],      num_obs)
        beta   = as_1d(params["beta"],   num_obs)
        a      = as_1d(params["a"],      num_obs)
        tau    = as_1d(params["tau"],    num_obs)
        s_v    = as_1d(params["s_v"],    num_obs)
        s_tau  = as_1d(params["s_tau"],  num_obs)
        pi     = as_1d(params.get("pi", self.pi), num_obs)

        if num_alternatives == 2:
            v = np.zeros((num_obs, 2), dtype=np.float64)
            v[:, 0] = v_base
            v[:, 1] = v_base + v_diff
        else:
            if "favored_idx" not in context:
                raise ValueError(
                    "For K > 2 alternatives, context['favored_idx'] is required."
                )
            favored_idx = np.asarray(context["favored_idx"]).reshape(-1)
            if favored_idx.shape[0] != num_obs:
                raise ValueError(
                    f"favored_idx length {favored_idx.shape[0]} != num_obs {num_obs}"
                )
            v = np.full((num_obs, num_alternatives), 0.0, dtype=np.float64)
            for i in range(num_obs):
                v[i, :] = v_base[i]
                v[i, int(favored_idx[i])] = v_base[i] + v_diff[i]

        return {
            "v": v, "k": k, "beta": beta, "a": a,
            "tau": tau, "s_v": s_v, "s_tau": s_tau, "pi": pi,
        }

    @staticmethod
    def build_context(num_obs: int, num_alternatives: int = 2) -> dict[str, np.ndarray]:
        context = {"num_alternatives": num_alternatives}
        if num_alternatives > 2:
            context["favored_idx"] = np.random.randint(0, num_alternatives, size=num_obs)
        return context

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        return {"num_alternatives": 2}

    @staticmethod
    def build_random_context(num_obs: int) -> dict[str, np.ndarray]:
        num_alternatives = np.random.randint(2, 5)
        return FFI.build_context(num_obs, num_alternatives)

    def simulate(self, params: dict[str, np.ndarray], context=None):
        pi = params.get("pi", None)
        results = simulate_ffi(
            v=params["v"],
            k=params["k"],
            beta=params["beta"],
            a=params["a"],
            tau=params["tau"],
            s_v=params["s_v"],
            s_tau=params["s_tau"],
            sigma=1.0,
            dt=self.dt,
            max_steps=self.max_steps,
        )
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        num_alternatives = int((context or {}).get("num_alternatives", 2))
        pi_val = pi if pi is not None else self.pi
        rts, choices = contaminate(rts, choices, pi=pi_val, num_alternatives=num_alternatives)
        return {"rts": rts, "choices": choices}

    def sample(self):
        raise NotImplementedError
