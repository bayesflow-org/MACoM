import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeMLP(nn.Module):
    """Time-conditioned output MLP.

    Replaces a plain linear output projection with a small MLP that conditions on
    the time embedding, letting the velocity field condition on where in the flow
    trajectory (t) it is at each step.

    Two conditioning modes are available:

    - ``"concat"`` (default): the hidden state and time embedding are concatenated
      and passed through a two-layer MLP to a scalar output.
    - ``"film"``: FiLM (feature-wise affine) modulation applies a time-derived
      per-feature scale (γ) and shift (β) to the hidden state before the MLP.
      FiLM reference: Perez et al. (2018), FiLM: Visual Reasoning with a General
      Conditioning Layer.

    Ablations showed the ``"concat"`` MLP outperforms FiLM, so it is the default;
    FiLM is retained for comparison.

    Args:
        input_dim         : dimensionality of the incoming hidden state (proj_dim)
        time_embedding_dim: dimensionality of the time embedding
        hidden_dim        : width of the intermediate layer; defaults to input_dim
        conditioning      : "concat" (default) or "film"
    """

    def __init__(
            self,
            input_dim: int,
            time_embedding_dim: int,
            hidden_dim: int = None,
            conditioning: str = "concat",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        assert conditioning in ("concat", "film"), \
            f"conditioning must be 'concat' or 'film', got {conditioning!r}"
        self.conditioning = conditioning

        if conditioning == "film":
            # FiLM: maps time embedding → per-feature scale (γ) and shift (β)
            self.film_projection = nn.Linear(time_embedding_dim, 2 * input_dim)
            self.hidden = nn.Linear(input_dim, hidden_dim)
        else:
            # concat: [hidden state ; time embedding] → hidden
            self.hidden = nn.Linear(input_dim + time_embedding_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "film":
            # FiLM modulation: scale and shift from time embedding
            gamma, beta = self.film_projection(time_embedding).chunk(2, dim=-1)
            x = (1 + gamma) * x + beta
        else:
            x = torch.cat([x, time_embedding], dim=-1)

        return self.output(F.mish(self.hidden(x)))
