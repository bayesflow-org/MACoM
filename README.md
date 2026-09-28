# MACoM

**Meta-amortized cognitive modeling**

Jerry M. Huang, Lukas Schumacher, Niek Stevenson, Mischa von Krause, and Stefan T. Radev

Repository for the Nature Communications paper on meta-amortized cognitive models.

# Install

**Prerequisites**

- Python >= 3.12
- [PyTorch](https://pytorch.org/get-started/locally/) >= 2.10.0 — install separately following the official instructions for your platform/CUDA version before running either install command below

**Option A — reproducible install with [uv](https://docs.astral.sh/uv/) (recommended)**

```bash
uv sync --locked                    # add --extra tracking for wandb logging
```

`--locked` installs exactly `uv.lock`, and fails rather than re-resolving if the lock
and `pyproject.toml` disagree. The lock reproduces the stack every reported result was
validated on (BayesFlow 2.0.14 at a pinned commit, numpy 2.5.3, numba 0.67, keras 3.15,
jax 0.11.1, torch 2.11). Keep every machine on it: a numpy/numba change alters the
simulated datasets, so results from different stacks are not directly comparable.

**Option B — standard pip**

```bash
pip install -e .
```

# Citation
[TBD]
