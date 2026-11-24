# MACoM

**Meta-amortized cognitive modeling**

Jerry M. Huang, Lukas Schumacher, Niek Stevenson, Mischa von Krause, and Stefan T. Radev

Repository for the Nature Communications paper on meta-amortized cognitive models.

## Reproducing the study

The code and locked environment are tracked here. Empirical inputs and saved
study artifacts are supplied separately. Place `data_application_ddm.csv`,
`data_application_rdm.csv`, `data_application_cdm.csv`, and
`data_application_lba.csv` in `data/`. The original-prediction comparison scripts
also use `data/pred_data_exp_2.csv`.

For the reported run, restore the original artifacts under `outputs/` with
their directory layout intact. Use the trained checkpoints and their completion
markers for rescoring, `outputs/data/benchmark/seed0/` for the shared benchmark,
and the saved predictions and predictive caches when redrawing the reported
figures. Final figures and tables are assembled in `outputs/finals/`.

To initialize a fresh run with the paper's settings, install the environment
below, then run:

```bash
USE_WANDB=0 bash run_experiments.sh all
```

The experiments read `study.env` from the selected study directory when present.
The default `outputs/` directory uses `experiments/paper_study.env` as its
fallback. Set `STUDY_ROOT=/path/to/study` to run in a different directory with
its own `study.env`. Paths resolve independently of the working directory.

With the saved arrays in place, regenerate figures with the corresponding replot
module. Run `bash fetch_fonts.sh` to install the pinned figure typeface. For example:

```bash
.venv/bin/python -m experiments.experiment_4.replot_crossfamily
.venv/bin/python -m experiments.experiment_4.replot_representation
.venv/bin/python -m experiments.experiment_1.replot_per_case
USE_WANDB=0 bash run_experiments.sh finalize
```

The last command copies regenerated source figures into `outputs/finals/`.
Run each experiment's `appendix_captions` module and
`experiments.experiment_1a.gap_tables` to refresh its text and tables.
Inspect the full command sequence with
`DRY_RUN=1 FORCE=1 USE_WANDB=0 bash run_experiments.sh all`.

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
and requires training. Its calibration benefit remains to be established. See the
[parameter-specific structure tests](tests/test_parameter_specific_structure.py).

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

With the saved study artifacts restored, prepare an additional study with its own
datasets and checkpoints:

```bash
bash experiments/setup_study.sh /path/to/new-study gaussian_softmax
STUDY_ROOT=/path/to/new-study bash run_experiments.sh --list
```

The study regenerates its own shared test set and retrains the BayesFlow baselines
under its generator. `--reuse-main-bench-bf` instead copies the retained study's
benchmark files and BayesFlow baselines into the new study.

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
With the original continuation files restored under `outputs/continuation/`,
launch or resume its frozen source with:

```bash
.venv/bin/python outputs/continuation/ensemble_metrics_1000_bayesflow_20261105/launch.py
```

Results are written to
`outputs/evaluations/ensemble_1000_bayesflow/`, with reports every
200 configurations. The original evaluation discarded its posterior draws, so this
replay repeats inference. See the README in that output directory for details.

# BayesFlow benchmark ensemble plots

Overlay the five saved continuous benchmark cases for DDM, RDM, CDM, and LBA
using the same recovery, ECDF, coverage, and metric grids and case colors as
CogFormer. Global pi is plotted separately. No checkpoints or new inference are
needed:

```bash
.venv/bin/python -m experiments.experiment_1a.model_family_bf_benchmark_ensemble \
    --data_dir outputs/data/model_family \
    --outdir outputs/figures/model_family/bf/ensemble_benchmark
```

Inputs are `<family>_<case>_data.npz` files from `model_family_bf_validate`.
Each family produces eight PDFs, plus a shared `manifest.json` recording input
paths, dataset/draw counts, outputs, and the log-gamma seed. Metric tiles report
the mean, SD, and number of cases in which that coefficient is active; NRMSE
uses range normalization. Use `--families ddm rdm` to select families or
`--no-metrics` to omit metric heatmaps.

# Citation
[TBD]
