import numpy as np
from numba import njit, prange

from cogformer.simulator import Model
from cogformer.utils.simulator_utils import as_1d
from cogformer.utils.simulator_utils import contaminate


@njit
def sample_lba_trial(
    v: np.ndarray,
    A: float,
    b: float,
    tau: float,
    s_v: float,
    s_tau: float,
    max_time: int = 10,
) -> np.ndarray:
    n_choices = v.shape[0]
    threshold = A + b

    while True:
        min_t = 1e25
        winner = -1
        for c in range(n_choices):
            start_point = np.random.random() * A
            drift_rate = np.random.normal(v[c], s_v)
            if drift_rate > 0:
                t_accum = (threshold - start_point) / drift_rate
                if t_accum < min_t:
                    min_t = t_accum
                    winner = c
        if winner != -1:
            break

    tau_i = tau + np.random.uniform(0.0, s_tau)
    if min_t + tau_i < max_time:
        return np.array([min_t + tau_i, float(winner)], dtype=np.float32)
    else:
        return np.array([-1.0, -1.0], dtype=np.float32)


@njit(parallel=True)
def simulate_lba(
    v: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    max_time: int = 10,
):
    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)
    for i in prange(n):
        sim_data[i] = sample_lba_trial(
            v=v[i],
            A=A[i],
            b=b[i],
            tau=tau[i],
            s_v=s_v[i],
            s_tau=s_tau[i],
            max_time=max_time,
        )
    return sim_data


class LBA(Model):
    """
    Linear Ballistic Accumulator with response coding.

    Uses the same response-coding convention as RDM:
      - v:      base drift rate
      - v_diff: drift advantage for the favored accumulator
      - a:      start-point upper bound (A in classical LBA notation)
      - tau:    non-decision time
      - s_v:    drift SD (sigma in classical notation)
      - s_tau:  non-decision time variability (uniform half-width)

    The response threshold is a + b_offset where b_offset is a fixed
    class-level hyperparameter (default 0.5).
    """

    def __init__(self, b_offset: float = 0.5, max_time: int = 10):
        self.b_offset = b_offset
        self.max_time = max_time

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

        v_base = as_1d(params["v"], num_obs)
        v_diff = as_1d(params["v_diff"], num_obs)
        a      = as_1d(params["a"], num_obs)
        tau    = as_1d(params["tau"], num_obs)
        s_v    = as_1d(params["s_v"], num_obs)
        s_tau  = as_1d(params["s_tau"], num_obs)
        pi     = as_1d(params.get("pi", 0.0), num_obs)

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

        b = np.full(num_obs, self.b_offset, dtype=np.float64)

        return {"v": v, "A": a, "b": b, "tau": tau, "s_v": s_v, "s_tau": s_tau, "pi": pi}

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
        return LBA.build_context(num_obs, num_alternatives)

    def simulate(self, params: dict[str, np.ndarray], context=None):
        pi = params.get("pi", None)
        lba_params = {k: v for k, v in params.items() if k != "pi"}
        results = simulate_lba(**lba_params, max_time=self.max_time)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        num_alternatives = int((context or {}).get("num_alternatives", 2))
        pi_val = pi if pi is not None else 0.0
        rts, choices = contaminate(rts, choices, pi=pi_val, num_alternatives=num_alternatives)
        return {"rts": rts, "choices": choices}

    def sample(self):
        raise NotImplementedError
