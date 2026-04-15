import numpy as np

from cogformer.utils.simulator_utils import exponential, sigmoid, scaled_sigmoid, shifted_softplus


def cdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_theta": lambda x: scaled_sigmoid(x, lower_bound=-np.pi, upper_bound=np.pi),
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus
    }
#
# def cdm_link_fun2():
#     return {
#         "v": exponential,
#         "v_theta": scaled_sigmoid,
#         "a": exponential,
#         "tau": exponential,
#         "s_v": exponential,
#         "s_tau": exponential
#     }


# def ddm_link_fun():  # 5 params, no z
#     return {
#         "v": shifted_softplus,
#         "a": shifted_softplus,
#         "tau": shifted_softplus,
#         "s_v": shifted_softplus,
#         "s_tau": shifted_softplus
#     }


def ddm_link_fun():
    return {
        "v": shifted_softplus,
        "a": shifted_softplus,
        "z": scaled_sigmoid,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus
    }

def ddm_link_fun2():
    return {
        "v": exponential,
        "a": exponential,
        "tau": exponential,
        "s_v": exponential,
        "s_tau": sigmoid
    }


def rdm_link_fun():
    return {
        "v": shifted_softplus,
        "v_diff": shifted_softplus,
        "a": shifted_softplus,
        "tau": shifted_softplus,
        "s_v": shifted_softplus,
        "s_tau": shifted_softplus,
    }

def rdm_link_fun2():
    return {
        "v": exponential,
        "v_diff": shifted_softplus,
        "a": exponential,
        "tau": exponential,
        "s_v": exponential,
        "s_tau": sigmoid
    }
