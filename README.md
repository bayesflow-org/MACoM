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

# Parameter-specific interaction structure

The decoder identifies each active coefficient by its parameter, regressor pair,
category slots and mask. Random training designs can now sample optional interactions
separately for each parameter:

```python
samples = model_class.batch_sample(
    batch_size=8,
    num_obs=200,
    max_num_regressors=4,
    max_num_categories=4,
    add_interaction=True,
    interaction_policy="optional",
)
```

Both parents must affect a parameter for their interaction to be eligible. Each
eligible `(regressor pair, parameter)` is retained independently given an inclusion
probability. By default, that probability is drawn once per design from `Beta(2,2)`;
set `interaction_prob=0.5` for a fixed probability. This allows two regressors to
interact for `v` while having only main effects for `tau`. Explicit design configs
continue to specify the exact parameter assignments.

An experimental encoder mode also conditions observation encoding on the same
joint parameter/term identities and active coefficient masks. Enable both features
for a new training run with:

```bash
python -m experiments.experiment_1.model_class_cf_train \
    --interaction_policy optional \
    --encoder_structure_conditioning
```

The encoder mode uses no coefficient values. Training and posterior sampling share
the conditioning path, including compacted coefficient layouts. Pass
`--encoder_structure_conditioning` when loading that checkpoint in validation,
ensemble, fine-tuning or real-data application tools. Direct construction uses
`CogFormer(..., encoder_structure_conditioning=True)`.

Both features are opt-in: the defaults remain `interaction_policy="all"` (legacy
forced eligible interactions) and an unconditioned encoder, preserving existing
checkpoint behavior. New training checkpoints record the structural settings and
use distinct filenames for the enabled options. Encoder conditioning adds weights
and requires training; its calibration benefit has not been established. See the
[implementation and checks](experiments/diagnostics/PARAMETER_STRUCTURE_IMPLEMENTATION.md).

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

To rerun the whole finals pipeline under `gaussian_softmax`, run it in its own study
directory, which keeps every output separate from the main study:

```bash
bash experiments/setup_study.sh outputs/gaussian-softmax gaussian_softmax
STUDY_ROOT=outputs/gaussian-softmax bash run_experiments.sh --list
```

The study regenerates its own shared test set and retrains the BayesFlow baselines
under its generator; the main study's files are never modified. See
`.claude/skills/experiment-orchestration/SKILL.md` (section 5a) for details and the
`--reuse-main-bench-bf` alternative.

# 1,000-configuration evaluation with BayesFlow metrics

`experiments.experiment_1.ensemble_metrics_sweep` calls BayesFlow's public NRMSE,
posterior contraction, and calibration error functions directly. NRMSE explicitly
uses range normalization. Cells are evaluated only on datasets supporting them;
configuration weighting and separate global pi summaries are retained. Run metadata
records the BayesFlow version and commit, and prevents resuming with legacy metric
records.

The prepared replay uses the original checkpoint, exact 1,000-design plan, and
sampling settings. It generates fresh datasets, since the old run discarded draws
and did not reset the separate Numba RNG. The new run seeds NumPy, PyTorch, and
Numba per configuration and records new and original input hashes.
Launch or resume its frozen source with:

```bash
.venv/bin/python outputs/10-26-2026/continuation/ensemble_metrics_1000_bayesflow_20261105/launch.py
```

Results are written to
`outputs/10-26-2026/outputs/evaluations/ensemble_1000_bayesflow/`, with reports every
200 configurations. The original evaluation discarded its posterior draws, so this
replay repeats inference. See the README in that output directory for details.

# BayesFlow benchmark ensemble plots

Overlay the five saved continuous benchmark cases for DDM, RDM, CDM, and LBA
using the same recovery, ECDF, coverage, and metric grids and case colors as
CogFormer. Global pi is plotted separately. No checkpoints or new inference are
needed:

```bash
.venv/bin/python -m experiments.experiment_1a.model_family_bf_benchmark_ensemble \
    --data_dir outputs/10-26-2026/outputs/data/model_family \
    --outdir outputs/10-26-2026/outputs/figures/model_family/bf/ensemble_benchmark
```

Inputs are `<family>_<case>_data.npz` files from `model_family_bf_validate`.
Each family produces eight PDFs, plus a shared `manifest.json` recording input
paths, dataset/draw counts, outputs, and the log-gamma seed. Metric tiles report
the mean, SD, and number of cases in which that coefficient is active; NRMSE
uses range normalization. Use `--families ddm rdm` to select families or
`--no-metrics` to omit metric heatmaps.

# Citation
[TBD]
