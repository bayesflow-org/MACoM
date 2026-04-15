import numpy as np


def contaminate(
    rts: np.ndarray,
    choices: np.ndarray,
    pi: float | np.ndarray,
    num_alternatives: int = 2,
    is_circular: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply uniform mixture contamination to simulated RT and choice data.

    Implements the mixture model from Radev et al. (2024):
        F^{pi,G} = (1 - pi) * F + pi * G

    With probability `pi`, each trial is replaced by a contaminated draw:
      - RT     ~ Uniform(0, 20)
      - Choice ~ Uniform over alternatives  (or Uniform(-pi, pi) for circular models)

    Parameters
    ----------
    rts : np.ndarray, shape (n, 1)
    choices : np.ndarray, shape (n, 1)
    pi : float or np.ndarray
        Contamination rate(s) in [0, 1]. A scalar applies one rate to all
        trials; a 1-D array of length n applies per-trial rates.
        Zero / all-zero disables contamination entirely.
    num_alternatives : int
        Number of discrete choice options (DDM: 2, RDM: K). Ignored when
        ``is_circular=True``.
    is_circular : bool
        When True (CDM), contaminated choices are drawn from Uniform(-pi, pi)
        instead of a discrete uniform distribution.

    Returns
    -------
    rts, choices : np.ndarray
        Copies with contaminated trials replaced.
    """
    pi_arr = np.asarray(pi, dtype=np.float32).reshape(-1)
    if (pi_arr <= 0.0).all():
        return rts, choices

    n = rts.shape[0]
    u = np.random.uniform(size=n)
    if pi_arr.size == 1:
        mask = u < pi_arr[0]
    else:
        if pi_arr.size != n:
            raise ValueError(f"pi array length {pi_arr.size} != num_obs {n}")
        mask = u < pi_arr
    if not mask.any():
        return rts, choices

    n_c = int(mask.sum())

    rt_c = np.random.uniform(0.0, 20.0, size=n_c)

    if is_circular:
        choice_c = np.random.uniform(-np.pi, np.pi, size=n_c)
    else:
        choice_c = np.random.randint(0, num_alternatives, size=n_c).astype(np.float32)

    rts = rts.copy()
    choices = choices.copy()
    rts[mask, 0]     = rt_c.astype(np.float32)
    choices[mask, 0] = choice_c.astype(np.float32)

    return rts, choices

def exponential(x: np.ndarray):
    return np.exp(x)

def sigmoid(x: np.ndarray):
    return 1. / (1. + np.exp(-x))

def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x))

def softplus(x: np.ndarray):
    sp = np.log1p(np.exp(x))
    return sp

def shifted_softplus(x: np.ndarray):
    """
    Vectorized shifted softplus: f(x) = log(1 + exp(x)) - log(2)
    This ensures f(0) = 0 and is stable for large |x|.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).

    Returns
    -------
    np.ndarray
        Same shape as x.
    """
    # Stable softplus: max(x,0) + log1p(exp(-|x|))
    sp = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    return sp

def scaled_sigmoid(
    x: float | np.ndarray,
    lower_bound: float | np.ndarray = -1.0,
    upper_bound: float | np.ndarray = 1.0,
) -> float:
    """
    Apply a sigmoid transformation and rescale to a bounded interval.
    """
    return lower_bound + (upper_bound - lower_bound) / (1.0 + np.exp(-x))



def as_1d(x, n: int, allow_scalar: bool = True) -> np.ndarray:
    """Return a 1D array of length n. Broadcast scalars if allowed; else validate."""
    a = np.asarray(x)
    if a.ndim == 0:
        if allow_scalar:
            return np.full((n,), a.item())
    a = a.reshape(-1)
    if a.size == 1 and allow_scalar:
        return np.full((n,), a.item())
    if a.size != n:
        raise ValueError(f"array has length {a.size}; expected {n}.")
    return a

def inspect(out: dict, verbose=False):
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        elif isinstance(v, list):
            print(k, len(v) if not verbose else (v[i] for i in v))
        elif isinstance(v, dict):
            print(v.keys())
        else:
            print(k, v)
