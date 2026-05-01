import numpy as np

from cogformer.utils.simulator_utils import exponential, sigmoid, scaled_sigmoid, shifted_softplus


def cdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_theta": lambda x: scaled_sigmoid(x, lower_bound=-np.pi, upper_bound=np.pi),
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus,
        "pi": lambda x: x,  # identity — pi is sampled directly in [0, 1]
    }

def ddm_link_fun():
    return {
        "v": shifted_softplus,
        "a": shifted_softplus,
        "z": scaled_sigmoid,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus,
        "pi": lambda x: x,  # identity — pi is sampled directly in [0, 1]
    }



def rdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_diff": shifted_softplus,
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus,
        "pi": lambda x: x,  # identity — pi is sampled directly in [0, 1]
    }

def lba_link_fun():
    return {
        "v":     shifted_softplus,
        "v_diff": shifted_softplus,
        "a":     shifted_softplus,
        "tau":   shifted_softplus,
        "s_v":   shifted_softplus,
        "s_tau": shifted_softplus,
        "pi":    lambda x: x,  # identity — pi is sampled directly in [0, 1]
    }

def lca_link_fun():
    return {
        "v":      shifted_softplus,
        "v_diff": shifted_softplus,
        "k":      shifted_softplus,
        "beta":   shifted_softplus,
        "a":      shifted_softplus,
        "tau":    shifted_softplus,
        "s_v":    shifted_softplus,
        "s_tau":  shifted_softplus,
        "pi":     lambda x: x,
    }

def ffi_link_fun():
    return {
        "v":      shifted_softplus,
        "v_diff": shifted_softplus,
        "k":      shifted_softplus,
        "beta":   shifted_softplus,
        "a":      shifted_softplus,
        "tau":    shifted_softplus,
        "s_v":    shifted_softplus,
        "s_tau":  shifted_softplus,
        "pi":     lambda x: x,
    }
