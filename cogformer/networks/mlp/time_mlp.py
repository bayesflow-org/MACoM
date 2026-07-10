import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeMLP(nn.Module):
    """Time-conditioned output MLP.

    Replaces a plain linear output projection with a small MLP that conditions on
    the time embedding, letting the velocity field condition on where in the flow
    trajectory (t) it is at each step.

    Three conditioning modes are available:

    - ``"linear"``: a single linear layer to the scalar output; the time embedding
      is ignored here. The head carries no time-dependence of its own, so all time
      conditioning must be injected upstream (input projection + per-block FiLM).
      This is the modernized default: an ablation on CogFormer's predecessor found a
      plain linear head — paired with per-block FiLM for late-stage time
      conditioning — outperforms a time-conditioned output MLP.
    - ``"concat"``: the hidden state and time embedding are concatenated and passed
      through a two-layer MLP to a scalar output.
    - ``"film"``: FiLM (feature-wise affine) modulation applies a time-derived
      per-feature scale (γ) and shift (β) to the hidden state before the MLP.
      FiLM reference: Perez et al. (2018), FiLM: Visual Reasoning with a General
      Conditioning Layer.

    Args:
        input_dim         : dimensionality of the incoming hidden state (proj_dim)
        time_embedding_dim: dimensionality of the time embedding
        hidden_dim        : width of the intermediate layer; defaults to input_dim
        conditioning      : "linear" (default), "concat", or "film"
    """

    def __init__(
            self,
            input_dim: int,
            time_embedding_dim: int,
            hidden_dim: int = None,
            conditioning: str = "linear",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        assert conditioning in ("linear", "concat", "film"), \
            f"conditioning must be 'linear', 'concat' or 'film', got {conditioning!r}"
        self.conditioning = conditioning

        if conditioning == "linear":
            # Single linear projection to the scalar velocity; time enters upstream.
            self.output = nn.Linear(input_dim, 1)
            return

        if conditioning == "film":
            # FiLM: maps time embedding → per-feature scale (γ) and shift (β)
            self.film_projection = nn.Linear(time_embedding_dim, 2 * input_dim)
            self.hidden = nn.Linear(input_dim, hidden_dim)
        else:
            # concat: [hidden state ; time embedding] → hidden
            self.hidden = nn.Linear(input_dim + time_embedding_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "linear":
            return self.output(x)

        if self.conditioning == "film":
            # FiLM modulation: scale and shift from time embedding
            gamma, beta = self.film_projection(time_embedding).chunk(2, dim=-1)
            x = (1 + gamma) * x + beta
        else:
            x = torch.cat([x, time_embedding], dim=-1)

        return self.output(F.mish(self.hidden(x)))
