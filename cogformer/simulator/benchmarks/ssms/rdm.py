import numpy as np
from numba import njit, prange

from cogformer.simulator import Model
from cogformer.utils.simulator_utils import as_1d
from cogformer.utils.simulator_utils import contaminate


@njit
def sample_rdm_trial(
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
    num_alternatives = v.shape[0]
    c = sigma * np.sqrt(dt)

    v_i = np.empty(num_alternatives, dtype=np.float64)
    for k in range(num_alternatives):
        v_i[k] = np.random.normal(v[k], s_v)
    tau_i = tau + np.random.uniform(0, s_tau)

    X = np.zeros(num_alternatives)
    for i_iter in range(max_steps):
        threshold = a if decay == 0.0 else a * np.exp(-decay * i_iter * dt)
        for i in range(num_alternatives):
            X[i] += v_i[i] * dt + np.random.normal(0.0, c)
            if X[i] >= threshold:
                return np.array([tau_i + i_iter * dt, i], dtype=np.float32)
    return np.array([-1.0, -1.0], dtype=np.float32)


# Serial fast-path kernel, generated per accumulator count.
#
# The optimization that matters is keeping every accumulator in a scalar local: measured
# per-trial cost at m=2 is 9.6 us with scalars against 21 us for any array-backed state.
# That gap is per-iteration, not extra work (an array version was measured doing 1.05x
# the accumulator-steps for 2.2x the time), and it survives hoisting the array element
# into a local, reordering the loops, and making m a compile-time literal. So the kernel
# has to be specialized per m -- but generating it keeps that from imposing a maximum
# number of alternatives or duplicating a kernel per m by hand.
#
# Semantics match the reference `sample_rdm_trial` exactly: drifts are drawn per
# accumulator in order, the threshold is hoisted out of the step loop (valid only for
# decay == 0), per-step scalar normal draws are replaced by block-pregenerated noise, and
# accumulators keep in-order crossing semantics -- if accumulator q crosses, accumulators
# after it are not updated that step and their draws are discarded (iid-safe).
# Serial only: the gain is zero under prange. KS-validated against the prange kernel.
_RDM_SERIAL_TEMPLATE = '''
def _rdm_serial(v, a, tau, s_v, s_tau, sigma, dt, max_steps):
    n = v.shape[0]
    sim_data = np.zeros((n, 2), dtype=np.float32)
    block = 128
    c = sigma * np.sqrt(dt)
    for i in range(n):
{init}
        tau_i = tau[i] + np.random.uniform(0, s_tau[i])
        threshold = a[i]
        rt = -1.0
        choice = -1.0
        i_iter = 0
        while i_iter < max_steps and rt < 0.0:
            k = min(block, max_steps - i_iter)
            noise = np.random.standard_normal({m} * k)
            for j in range(k):
                b = {m} * j
{steps}
                i_iter += 1
        sim_data[i, 0] = rt
        sim_data[i, 1] = choice
    return sim_data
'''

_RDM_SERIAL_STEP = '''                X{q} += d{q} + c * noise[b + {q}]
                if X{q} >= threshold:
                    rt = tau_i + i_iter * dt
                    choice = {q}.0
                    break'''

_RDM_SERIAL_KERNELS: dict[int, object] = {}


def _rdm_serial_kernel(m: int):
    """Compile (once per process, per m) the scalarized serial kernel for m accumulators.

    Costs ~0.2-0.6 s the first time a given m is seen. Numba's on-disk ``cache=True``
    cannot be used for generated source, so this is paid per worker process rather than
    once per machine; at the 2-4 alternatives RDM trains over that is well under a second
    of startup against a 2.2x steady-state gain.
    """
    kernel = _RDM_SERIAL_KERNELS.get(m)
    if kernel is None:
        init = "\n".join(
            f"        d{q} = np.random.normal(v[i, {q}], s_v[i]) * dt\n        X{q} = 0.0"
            for q in range(m)
        )
        steps = "\n".join(_RDM_SERIAL_STEP.format(q=q) for q in range(m))
        source = _RDM_SERIAL_TEMPLATE.format(m=m, init=init, steps=steps)
        namespace = {"np": np}
        exec(compile(source, f"<rdm_serial_m{m}>", "exec"), namespace)
        kernel = njit(fastmath=True)(namespace["_rdm_serial"])
        _RDM_SERIAL_KERNELS[m] = kernel
    return kernel


@njit(parallel=True)
def _simulate_rdm_parallel(
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
        sim_trial = sample_rdm_trial(
            v=v[i],
            a=a[i],
            tau=tau[i],
            s_v=s_v[i],
            s_tau=s_tau[i],
            decay=decay,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        sim_data[i] = sim_trial

    return sim_data


def simulate_rdm(
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
    (decay == 0), else the prange kernel. Any number of accumulators is
    supported: the serial kernel is generated per accumulator count."""
    import numba
    if numba.get_num_threads() == 1 and decay == 0.0:
        kernel = _rdm_serial_kernel(v.shape[1])
        return kernel(v, a, tau, s_v, s_tau, sigma, dt, max_steps)
    return _simulate_rdm_parallel(v, a, tau, s_v, s_tau, decay, sigma, dt, max_steps)


class RDM(Model):
    """
    Race Diffusion Model with response coding.

    This implementation uses response coding where drift rates are specified
    directly for each accumulator without reference to stimulus correctness.

    For any K >= 2 alternatives (stimulus coding):
        - v: base drift rate applied to all accumulators
        - v_diff: added to accumulator specified by context['favored_idx']
          (required for K > 2; defaults to accumulator 1 for K = 2)
    """

    def __init__(self, dt: float = 5e-4, max_steps: int = 20000, decay: float = 0.0, pi: float = 0.0):
        self.dt = dt
        self.max_steps = max_steps
        self.decay = decay  # Fixed decay parameter (0.0 = no collapsing bounds)
        self.pi = pi

    def prepare_params(
        self,
        params: dict[str, np.ndarray],
        num_obs: int,
        context: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Build per-trial K-way drift rates using response coding.

        Parameters
        ----------
        params : dict
            v : base drift rate
            v_diff : drift difference between accumulators
            a : threshold
            tau : non-decision time
            s_v : inter-trial drift variability (std dev)
            s_tau : inter-trial non-decision time variability (uniform half-width)

        context : dict, optional
            num_alternatives : number of accumulators (default: 2)
            favored_idx : which accumulator receives v + v_diff on each trial while
                the others receive v. Required for K > 2; defaults to 1 for 2-alt.
        """
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
        pi     = as_1d(params.get("pi", self.pi), num_obs)

        if num_alternatives == 2:
            # STIMULUS CODING, matching LBA: v_diff goes to the accumulator named per
            # trial by context['favored_idx']. This branch used to hardcode
            # `v[:, 1] = v_base + v_diff` (accumulator 1 favoured on every trial), i.e.
            # accuracy coding. Accuracy coding is still reachable by passing
            # favored_idx = ones, which is also the default when none is given.
            favored = np.asarray(
                context.get("favored_idx", np.ones(num_obs, dtype=np.int32))
            ).reshape(-1)
            if favored.shape[0] != num_obs:
                raise ValueError(
                    f"favored_idx length {favored.shape[0]} != num_obs {num_obs}"
                )
            v = np.zeros((num_obs, 2), dtype=np.float64)
            v[:, :] = v_base[:, None]
            v[np.arange(num_obs), favored.astype(np.int64)] += v_diff
        else:
            if "favored_idx" not in context:
                raise ValueError(
                    "For K > 2 alternatives, context['favored_idx'] is required "
                    "to specify which accumulator receives the drift boost."
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

        return {"v": v, "a": a, "tau": tau, "s_v": s_v, "s_tau": s_tau, "pi": pi}

    @staticmethod
    def build_context(num_obs: int, num_alternatives: int = 2) -> dict[str, np.ndarray]:
        """Build context for RDM simulation."""
        context = {"num_alternatives": num_alternatives}
        # Drawn per trial for EVERY K including 2 (stimulus coding), as in LBA. A
        # constant favoured accumulator would be accuracy coding.
        context["favored_idx"] = np.random.randint(0, num_alternatives, size=num_obs)
        return context

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        """Build default context (2 alternatives, favoured accumulator drawn per trial)."""
        return RDM.build_context(num_obs, num_alternatives=2)

    @staticmethod
    def build_random_context(num_obs: int) -> dict[str, np.ndarray]:
        num_alternatives = np.random.randint(2, 5)
        return RDM.build_context(num_obs, num_alternatives)

    supports_gpu = True

    def _finish(self, results, pi, context):
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        num_alternatives = int((context or {}).get("num_alternatives", 2))
        pi_val = pi if pi is not None else self.pi
        rts, choices = contaminate(rts, choices, pi=pi_val, num_alternatives=num_alternatives)
        return {"rts": rts, "choices": choices}

    def simulate(self, params: dict[str, np.ndarray], context=None, randomize=False):
        if self.uses_gpu:
            return self.simulate_many([(params, context)])[0]
        rdm_params = {k: v for k, v in params.items() if k != "pi"}
        results = simulate_rdm(**rdm_params, decay=self.decay, dt=self.dt, max_steps=self.max_steps)
        return self._finish(results, params.get("pi", None), context)

    def simulate_many(self, requests):
        if not self.uses_gpu:
            return super().simulate_many(requests)
        if self.decay != 0.0:
            raise NotImplementedError("GPU RDM kernel supports constant thresholds only (decay == 0)")
        from .gpu_kernels import simulate_rdm_gpu
        lens = [p["v"].shape[0] for p, _ in requests]
        # Datasets may differ in their number of accumulators: pad drifts to the
        # largest and pass the per-trial count (padding columns are ignored).
        k_max = max(p["v"].shape[1] for p, _ in requests)
        v = np.concatenate([np.pad(np.asarray(p["v"]), ((0, 0), (0, k_max - p["v"].shape[1])))
                            for p, _ in requests])
        m = np.concatenate([np.full(n, p["v"].shape[1], dtype=np.int32)
                            for n, (p, _) in zip(lens, requests)])
        cat = {k: np.concatenate([np.asarray(p[k]).reshape(-1) for p, _ in requests])
               for k in ("a", "tau", "s_v", "s_tau")}
        results = simulate_rdm_gpu(v, **cat, sigma=1.0, dt=self.dt, max_steps=self.max_steps,
                                   num_alternatives=m)
        return [self._finish(r, p.get("pi", None), c)
                for r, (p, c) in zip(np.split(results, np.cumsum(lens)[:-1]), requests)]

    def sample(self):
        raise NotImplementedError
