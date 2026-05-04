import torch
from typing import Callable

SOLVERS = ("euler", "midpoint", "rk4")


def ode_step(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    solver: str,
) -> torch.Tensor:
    """One ODE step for flow-matching inference.

    v_fn  : callable(theta, t) → velocity tensor, same shape as theta
    solver: "euler" | "midpoint" | "rk4"
    """
    if solver == "euler":
        return theta + v_fn(theta, t) * dt
    elif solver == "midpoint":
        v1 = v_fn(theta, t)
        return theta + v_fn(theta + 0.5 * dt * v1, t + 0.5 * dt) * dt
    elif solver == "rk4":
        v1 = v_fn(theta, t)
        v2 = v_fn(theta + 0.5 * dt * v1, t + 0.5 * dt)
        v3 = v_fn(theta + 0.5 * dt * v2, t + 0.5 * dt)
        v4 = v_fn(theta + dt * v3, t + dt)
        return theta + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
    else:
        raise ValueError(f"Unknown solver {solver!r}. Choose one of {SOLVERS}.")
