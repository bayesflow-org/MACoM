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

# Alternative design generator

The default `mixture` generator remains unchanged. Select `gaussian_softmax` for a
joint Gaussian construction with one latent coordinate per continuous predictor and
one score per categorical level. Each dataset draws an LKJ correlation matrix;
categorical score blocks pass through softmax before category assignments are sampled.
Dummy coding, interactions, coefficient priors, and downstream masks use the existing
layout, including compaction of unobserved levels.

```python
samples = model_family.batch_sample(
    batch_size=8,
    num_obs=200,
    max_num_regressors=4,
    max_num_categories=4,
    design_generator="gaussian_softmax",
    design_generator_kwargs={
        "logit_scale": 1.0,
        "intercept_scale": 1.0,
        "continuous_scale": 1.0,
    },
)
```

These scales default to 1. Category intercepts are drawn once per dataset from
`Normal(0, intercept_scale**2)`; trial-level category scores have scale `logit_scale`.
`intercept_scale=0` sets all intercepts to zero. A `category_intercepts` mapping can
override named categorical predictors with vectors of their category counts; fix the
counts with `min_num_categories=max_num_categories` when using these overrides.
Also specify `categorical_regressors` so those predictors are always categorical.
The existing `eta` argument controls the LKJ shape (default `Uniform(0.5, 5)`).
`category_concentration` is a Dirichlet control and is rejected for this generator.

Latent correlations do not equal observed predictor associations, and specified
intercepts do not guarantee marginal category proportions or exact balance. There
is no count floor or assignment resampling. Continuous predictors retain their
Gaussian scale without empirical standardization.

Training scripts that expose `--design_generator` accept `gaussian_softmax`.
Experiment 1's standalone validation and ensemble scripts also accept this option,
so existing checkpoints can be evaluated on alternative designs without retraining.
Saved validation datasets are reused as stored; training-time benchmark validation
continues to use its original generator. Adding this option does not change existing
checkpoints or results.

# Citation
[TBD]
