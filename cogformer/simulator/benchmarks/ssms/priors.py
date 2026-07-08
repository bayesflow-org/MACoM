import numpy as np
from numba import njit
from scipy.stats import norm


def _pi_sampler() -> float:
    """Sample pi directly from Beta(1, 10) — no link function applied."""
    return float(np.random.beta(0.5, 50))


def cdm_priors():
    return {
        "v": norm(loc=1.0, scale=0.8),
        "v_theta": norm(loc=0.0, scale=0.5),
        "a": norm(loc=0.5, scale=0.5),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.0, scale=0.6),
        "s_tau": norm(loc=-1.5, scale=0.7),
        "pi": _pi_sampler,
    }

def ddm_priors():
    return {
        "v": norm(loc=1.0, scale=0.8),
        "a": norm(loc=0.5, scale=0.5),
        "z": norm(loc=0.0, scale=0.8),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.0, scale=0.6),
        "s_tau": norm(loc=-1.5, scale=0.7),
        "pi": _pi_sampler,
    }

def rdm_priors():
    return {
        "v": norm(loc=0.6, scale=0.5),
        "v_diff": norm(loc=0.6, scale=0.5),
        "a": norm(loc=0.25, scale=0.5),
        "tau": norm(loc=-1.2, scale=0.5),
        "s_v": norm(loc=-1.0, scale=0.6),
        "s_tau": norm(loc=-1.5, scale=0.7),
        "pi": _pi_sampler,
    }

def lba_priors():
    return {
        "v":     norm(loc=0.6, scale=0.5),
        "v_diff": norm(loc=0.6, scale=0.5),
        "a":     norm(loc=0.25, scale=0.5),
        "tau":   norm(loc=-1.2, scale=0.5),
        "s_v":   norm(loc=-1.0, scale=0.6),
        "s_tau": norm(loc=-1.5, scale=0.7),
        "pi":    _pi_sampler,
    }

def lca_priors():
    return {
        "v":      norm(loc=0.6,  scale=0.5),
        "v_diff": norm(loc=0.6,  scale=0.5),
        "k":      norm(loc=-0.5, scale=0.5),
        "beta":   norm(loc=-1.0, scale=0.5),
        "a":      norm(loc=0.25, scale=0.5),
        "tau":    norm(loc=-1.2, scale=0.5),
        "s_v":    norm(loc=-1.0, scale=0.6),
        "s_tau":  norm(loc=-1.5, scale=0.7),
        "pi":     _pi_sampler,
    }

def ffi_priors():
    return {
        "v":      norm(loc=0.6,  scale=0.5),
        "v_diff": norm(loc=0.6,  scale=0.5),
        "k":      norm(loc=-0.5, scale=0.5),
        "beta":   norm(loc=-1.0, scale=0.5),
        "a":      norm(loc=0.25, scale=0.5),
        "tau":    norm(loc=-1.2, scale=0.5),
        "s_v":    norm(loc=-1.0, scale=0.6),
        "s_tau":  norm(loc=-1.5, scale=0.7),
        "pi":     _pi_sampler,
    }
