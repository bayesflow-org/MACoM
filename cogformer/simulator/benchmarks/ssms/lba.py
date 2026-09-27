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
    max_iter: int = 1000,
) -> np.ndarray:
    # A trial is only defined when at least one accumulator draws a positive drift, so
    # the draw is rejection-sampled. This loop was previously `while True`, which is
    # unbounded: a non-finite v/s_v makes `drift_rate > 0` false forever, and a
    # non-finite threshold makes `t_accum < min_t` false forever, so the trial never
    # resolves. Because this runs under @njit (no GIL, no interrupt point) inside a
    # prange body, one such trial wedges the whole batch, its worker, and -- via the
    # DataLoader's round-robin ordering -- the entire training loop, with no traceback
    # and no exit code. Capping the retries matches the max_steps bound the DDM/RDM/CDM
    # kernels already use, and reuses their convention of reporting an unresolved trial
    # as [-1, -1].
    #
    # The cap is not a throughput concern: v is strictly positive (shifted_softplus), so
    # P(accept) = 1 - prod_c Phi(-v_c/s_v) >= 0.75 and exhausting 1000 retries has
    # probability <= 0.25**1000 for any well-formed parameter set. It fires only on the
    # degenerate draws that used to hang.
    n_choices = v.shape[0]
    threshold = A + b

    min_t = np.inf
    winner = -1
    for _ in range(max_iter):
        min_t = np.inf
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

    if winner == -1:
        return np.array([-1.0, -1.0], dtype=np.float32)

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
    max_iter: int = 1000,
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
            max_iter=max_iter,
        )
    return sim_data


class LBA(Model):
    """
    Linear Ballistic Accumulator with response coding.

    Uses the same response-coding convention as RDM:
      - v:      base drift rate
      - v_diff: drift advantage for the favored accumulator
      - a:      MEAN DISTANCE TRAVELLED to the boundary (see ``fixed_A``)
      - tau:    non-decision time
      - s_v:    drift SD (sigma in classical notation)
      - s_tau:  non-decision time variability (uniform half-width)

    With ``fixed_A`` set, ``a`` is defined as the mean distance an accumulator
    travels, E[threshold - start], which is the quantity RDM's ``a`` already
    measures (its accumulators start at 0 and travel exactly ``a``). Defining it
    as the absolute threshold instead would leave the two families differing by
    A/2, so their values would not be comparable on an absolute scale.

    With ``fixed_A=None`` the legacy parameterization applies and the response
    threshold is a + b_offset, where b_offset is a fixed class-level
    hyperparameter (default 0.5).
    """

    def __init__(self, b_offset: float = 0.5, max_time: int = 10,
                 fixed_A: float | None = None, max_iter: int = 1000):
        """
        fixed_A : selects what the intrinsic parameter ``a`` means.

            None (default) -- the original parameterization: ``a`` is ``A``, the
            start-point range, and the threshold is ``A + b_offset``. Raising ``a`` then
            raises the threshold AND widens the start-point distribution, so its net
            effect on mean RT is only half that of a true threshold and it is entangled
            with RT variability. That is why ``a`` recovers poorly and why it does not
            transfer to model families where ``a`` is a threshold.

            A float -- ``A`` is pinned at this value and ``a`` is the MEAN DISTANCE
            TRAVELLED. The start point is uniform on [0, A] with mean A/2, so the
            threshold is set to ``a + A/2``: ``b = a - A/2``, giving
            ``E[threshold - start] == a`` exactly. This is what makes ``a``
            numerically comparable to RDM's threshold, whose accumulators start at 0
            and therefore also travel ``a`` on average.

            ``A`` is capped at ``2a`` so ``b >= 0``. The cap preserves the definition
            exactly rather than breaking it: at ``A = 2a`` we get ``b = 0`` and the
            mean distance is still ``2a - a == a``. It binds on only 0.02% of prior
            draws, against 3% for the previous threshold parameterization, because
            the binding condition moved from ``a < fixed_A`` to ``a < fixed_A/2``.

            ``fixed_A=0`` removes start-point variability entirely, leaving a pure
            threshold model directly comparable to RDM.

        max_iter : cap on the drift rejection-sampling retries per trial. A trial that
            exhausts it is reported as [-1, -1], the same convention the DDM/RDM/CDM
            kernels use when they hit ``max_steps``. See ``sample_lba_trial``: without a
            cap, a degenerate parameter set hangs a worker permanently and silently.
        """
        self.b_offset = b_offset
        self.max_time = max_time
        self.fixed_A = fixed_A
        self.max_iter = max_iter

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
            # STIMULUS CODING. v_diff is the drift advantage of the accumulator the
            # STIMULUS favours, named per trial by context['favored_idx'] -- the same
            # rule the K > 2 branch has always used. This branch used to hardcode
            # `v[:, 1] = v_base + v_diff`, which made accumulator 1 favoured on every
            # trial; the two accumulators were then "correct" and "error" rather than
            # the two responses, i.e. ACCURACY coding, and load_lba_data documented
            # that as forced-by-the-simulator and not comparable with load_ddm_data's
            # response coding on v_diff.
            #
            # Accuracy coding is still reachable -- pass favored_idx = ones, which is
            # exactly what the old hardcoding did -- so this is a change of DEFAULT,
            # not a loss of capability. Coding is a property of the context, which is
            # where it belongs: it describes how the data is labelled, not the model.
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

        if self.fixed_A is None:
            A = a
            b = np.full(num_obs, self.b_offset, dtype=np.float64)
        else:
            # `a` is the mean distance travelled: start ~ U(0, A) has mean A/2, so
            # a threshold of a + A/2 gives E[threshold - start] == a. See the
            # fixed_A docstring for why this, and not the absolute threshold, is
            # the definition that makes `a` comparable with RDM's.
            A = np.minimum(float(self.fixed_A), 2.0 * a)
            b = a - A / 2.0    # threshold = A + b = a + A/2  ->  E[distance] = a

        return {"v": v, "A": A, "b": b, "tau": tau, "s_v": s_v, "s_tau": s_tau, "pi": pi}

    @staticmethod
    def build_context(num_obs: int, num_alternatives: int = 2) -> dict[str, np.ndarray]:
        context = {"num_alternatives": num_alternatives}
        # Drawn per trial for EVERY K including 2. A constant favoured accumulator is
        # accuracy coding; a varying one is stimulus coding, and the encoder already
        # sees per-trial favored_idx from RDM (which varies it for every K, including
        # 2), which is what makes this in-distribution for the model-class checkpoint
        # rather than a new kind of input.
        context["favored_idx"] = np.random.randint(0, num_alternatives, size=num_obs)
        return context

    @staticmethod
    def build_default_context(num_obs: int) -> dict[str, np.ndarray]:
        # Delegates to build_context rather than returning a bare dict. It used to
        # return {"num_alternatives": 2} with NO favored_idx -- and since
        # ModelFamily.sample reaches for build_default_context (not build_context),
        # that was the path essentially every LBA simulation took. So the favored_idx
        # channel build_context documents as deliberately populated "so the encoder can
        # see it" was in fact empty for LBA, and the Adapter filled it with its zero
        # default while the simulator boosted accumulator 1 -- the channel disagreed
        # with the simulation. Delegating fixes that and makes the default stimulus-coded.
        return LBA.build_context(num_obs, num_alternatives=2)

    @staticmethod
    def build_random_context(num_obs: int) -> dict[str, np.ndarray]:
        num_alternatives = np.random.randint(2, 5)
        return LBA.build_context(num_obs, num_alternatives)

    def simulate(self, params: dict[str, np.ndarray], context=None):
        pi = params.get("pi", None)
        lba_params = {k: v for k, v in params.items() if k != "pi"}
        results = simulate_lba(**lba_params, max_time=self.max_time,
                               max_iter=self.max_iter)
        rts = results[:, 0][..., None]
        choices = results[:, 1][..., None]
        num_alternatives = int((context or {}).get("num_alternatives", 2))
        pi_val = pi if pi is not None else 0.0
        rts, choices = contaminate(rts, choices, pi=pi_val, num_alternatives=num_alternatives)
        return {"rts": rts, "choices": choices}

    def sample(self):
        raise NotImplementedError
