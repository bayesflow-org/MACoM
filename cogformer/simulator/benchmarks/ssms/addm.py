import numpy as np
from numba import njit, prange

from cogformer.simulator import Model
from cogformer.utils.simulator_utils import as_1d, contaminate

# first_fixation modes, encoded as ints so the njit kernel can branch on them.
FIRST_FIX_RANDOM = 0        # standard aDDM: first fixation lands on either item, 50/50
FIRST_FIX_ITEM0 = 1         # deterministic: always start on item 0
FIRST_FIX_ITEM1 = 2         # deterministic: always start on item 1

FIRST_FIX_MODES = {
    "random": FIRST_FIX_RANDOM,
    "item0": FIRST_FIX_ITEM0,
    "item1": FIRST_FIX_ITEM1,
}


@njit
def sample_addm_trial(
    v_0: float,
    v_1: float,
    omega: float,
    a: float,
    s_v: float,
    s_tau: float,
    tau: float,
    mean_fix_dur: float,
    first_fix_mode: int,
    dt: float = 0.001,
    sigma: float = 1.0,
    max_steps: int = 10000,
) -> np.ndarray:
    """Single aDDM trial (Krajbich et al., 2010).

    One relative accumulator on [-a, a]. The drift depends on which item is
    currently fixated: the attended item enters at full weight, the unattended
    one discounted by ``omega``.

        fixating item 0:  drift =  v_0 - omega * v_1
        fixating item 1:  drift =  omega * v_0 - v_1

    omega = 1 makes attention irrelevant (collapses to a plain DDM on the value
    difference); omega = 0 means only the attended item drives accumulation.

    Fixations alternate, with exponential durations of mean ``mean_fix_dur``.
    ``first_fix_mode`` selects where gaze starts — see FIRST_FIX_* above. This
    matters for identifiability, not just realism: under FIRST_FIX_RANDOM the
    first-fixation advantage averages out across trials, so with no fixation
    covariate omega reaches the data only through extra within-trial drift
    variance. Pinning the first fixation turns that advantage into a systematic
    choice bias, which is recoverable from marginal choice proportions alone.

    Returns [rt, choice]; [-1, -1] if the trial fails to terminate.
    """
    v_0_i = np.random.normal(v_0, s_v)
    v_1_i = np.random.normal(v_1, s_v)
    tau_i = tau + np.random.uniform(0.0, s_tau)

    if first_fix_mode == FIRST_FIX_ITEM0:
        fix = 0
    elif first_fix_mode == FIRST_FIX_ITEM1:
        fix = 1
    else:
        fix = 0 if np.random.random() < 0.5 else 1

    steps_left = int(np.random.exponential(mean_fix_dur) / dt)
    if steps_left < 1:
        steps_left = 1

    x = 0.0
    c_noise = sigma * np.sqrt(dt)

    for step in range(max_steps):
        if steps_left <= 0:
            fix = 1 - fix
            steps_left = int(np.random.exponential(mean_fix_dur) / dt)
            if steps_left < 1:
                steps_left = 1

        if fix == 0:
            drift = v_0_i - omega * v_1_i
        else:
            drift = omega * v_0_i - v_1_i

        x += drift * dt + c_noise * np.random.normal()
        steps_left -= 1

        if x >= a:
            return np.array([tau_i + (step + 1) * dt, 0.0], dtype=np.float32)
        if x <= -a:
            return np.array([tau_i + (step + 1) * dt, 1.0], dtype=np.float32)

    return np.array([-1.0, -1.0], dtype=np.float32)


@njit(parallel=True)
def simulate_addm(
    v_0: np.ndarray,
    v_1: np.ndarray,
    omega: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    s_v: np.ndarray,
    s_tau: np.ndarray,
    mean_fix_dur: np.ndarray,
    first_fix_mode: int,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    n = v_0.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)
    for i in prange(n):
        sim_data[i] = sample_addm_trial(
            v_0=v_0[i],
            v_1=v_1[i],
            omega=omega[i],
            a=a[i],
            s_v=s_v[i],
            s_tau=s_tau[i],
            tau=tau[i],
            mean_fix_dur=mean_fix_dur[i],
            first_fix_mode=first_fix_mode,
            dt=dt,
            sigma=sigma,
            max_steps=max_steps,
        )
    return sim_data


class ADDM(Model):
    """Attentional drift-diffusion model (Krajbich, Armel & Rangel, 2010).

    Two-alternative only. ``first_fixation`` selects the gaze-onset rule and can
    be set here or overridden per call via ``context["first_fixation"]``:

        "random" — 50/50 (standard aDDM)
        "item0"  — always item 0
        "item1"  — always item 1

    ``mean_fix_dur`` is a model constant rather than a free parameter, so the
    inferred vector stays comparable to the other SSM families.
    """

    def __init__(
        self,
        dt: float = 1e-3,
        max_steps: int = 20000,
        pi: float = 0.0,
        mean_fix_dur: float = 0.3,
        first_fixation: str = "random",
    ):
        if first_fixation not in FIRST_FIX_MODES:
            raise ValueError(
                f"first_fixation must be one of {sorted(FIRST_FIX_MODES)}, got {first_fixation!r}"
            )
        self.dt = dt
        self.max_steps = max_steps
        self.pi = pi
        self.mean_fix_dur = mean_fix_dur
        self.first_fixation = first_fixation

    def prepare_params(
        self,
        params: dict[str, np.ndarray],
        num_obs: int,
        context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        context = context or {}

        v_base = as_1d(params["v"],      num_obs)
        v_diff = as_1d(params["v_diff"], num_obs)
        omega  = as_1d(params["omega"],  num_obs)
        a      = as_1d(params["a"],      num_obs)
        tau    = as_1d(params["tau"],    num_obs)
        s_v    = as_1d(params["s_v"],    num_obs)
        s_tau  = as_1d(params["s_tau"],  num_obs)
        pi     = as_1d(params.get("pi", self.pi), num_obs)

        mean_fix_dur = as_1d(
            context.get("mean_fix_dur", self.mean_fix_dur), num_obs
        )

        # item 1 carries the value advantage, matching the RDM/LCA v/v_diff convention
        return {
            "v_0": v_base,
            "v_1": v_base + v_diff,
            "omega": omega,
            "a": a,
            "tau": tau,
            "s_v": s_v,
            "s_tau": s_tau,
            "mean_fix_dur": mean_fix_dur,
            "pi": pi,
        }

    @staticmethod
    def build_context(num_obs: int, first_fixation: str = "random") -> dict:
        return {"first_fixation": first_fixation}

    @staticmethod
    def build_default_context(num_obs: int) -> dict:
        return {"first_fixation": "random"}

    @staticmethod
    def build_random_context(num_obs: int) -> dict:
        return {"first_fixation": "random"}

    supports_gpu = True

    def _mode(self, context):
        mode_name = (context or {}).get("first_fixation", self.first_fixation)
        if mode_name not in FIRST_FIX_MODES:
            raise ValueError(
                f"first_fixation must be one of {sorted(FIRST_FIX_MODES)}, got {mode_name!r}"
            )
        return FIRST_FIX_MODES[mode_name]

    def _finish(self, results, pi):
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        pi_val = pi if pi is not None else self.pi
        rts, choices = contaminate(rts, choices, pi=pi_val, num_alternatives=2)
        return {"rts": rts, "choices": choices}

    def simulate(self, params: dict[str, np.ndarray], context=None):
        if self.uses_gpu:
            return self.simulate_many([(params, context)])[0]
        results = simulate_addm(
            v_0=params["v_0"],
            v_1=params["v_1"],
            omega=params["omega"],
            a=params["a"],
            tau=params["tau"],
            s_v=params["s_v"],
            s_tau=params["s_tau"],
            mean_fix_dur=params["mean_fix_dur"],
            first_fix_mode=self._mode(context),
            sigma=1.0,
            dt=self.dt,
            max_steps=self.max_steps,
        )
        return self._finish(results, params.get("pi", None))

    def simulate_many(self, requests):
        if not self.uses_gpu:
            return super().simulate_many(requests)
        from .gpu_kernels import simulate_addm_gpu
        lens = [len(np.asarray(p["v_0"]).reshape(-1)) for p, _ in requests]
        cat = {k: np.concatenate([np.asarray(p[k]).reshape(-1) for p, _ in requests])
               for k in ("v_0", "v_1", "omega", "a", "tau", "s_v", "s_tau", "mean_fix_dur")}
        mode = np.concatenate([np.full(n, self._mode(c), dtype=np.int32)
                               for n, (_, c) in zip(lens, requests)])
        results = simulate_addm_gpu(**cat, first_fix_mode=mode, sigma=1.0, dt=self.dt,
                                    max_steps=self.max_steps)
        return [self._finish(r, p.get("pi", None))
                for r, (p, _) in zip(np.split(results, np.cumsum(lens)[:-1]), requests)]

    def sample(self):
        raise NotImplementedError
