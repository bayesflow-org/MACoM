# Experiment 1 — generalising to design space

Frozen snapshot. Every figure here comes from ONE checkpoint; this tree supersedes
the scattered per-run directories under `outputs/figures/model_class/cf/`.

Assembled 2026-09-02 19:05:11Z by
`experiments/finalize_experiment_1.py` — 259 files.

## Provenance

- checkpoint: `outputs/checkpoints/model_class/cf/r4c4_b256_e10000/cogformer_model_class_fm_l8_h8_p256_s32_d128_b256_e10000_t100.pt
  sha256: 56e997f37e1c945c4ae6d022ebe546b2010e6f257ad537cee1a9b966d7409811`
- git commit: `c9d6f9704b88c743d6b301c82b16f83bd7d28e97`
- per-file sources, sizes and original mtimes: `MANIFEST.csv`

## Contents

| folder | what |
|---|---|
| `random_ensemble/` | 12 randomly drawn design configs, 4 per family (one combined set) |
| `benchmark_ensemble/` | `bench5` (5 OG cases) and `bench6` (+ `mixed_predictor`), per family |
| `per_case/` | the 18: 6 benchmark cases × 3 families, plotted separately |
| `information_gain/` | expected information gain vs trial count, four continuous/binary designs |
| `amortization_gap/` | reserved — BayesFlow comparison, not yet run |
| `predictions/` | raw posterior draws backing every figure, for re-plotting |

## What is included

An allowlist of plot stems -- `recovery`, `coverage`, `ecdf`, `metrics` and their
`_pi` companions -- not "whatever was in the source directory". The validate script
emits posterior pairplots by default, so a sweep-everything copier would pull in 180
extra files whenever a run forgets `--skip_posteriors`.

## Naming

Redundant prefixes are stripped: the directory already says which family, case and
plot family a file belongs to, so `ddm_interaction_fm_recovery.pdf` becomes
`per_case/ddm/interaction/recovery.pdf`.

`_global` is renamed `_pi`. That panel shows the global-parameter estimates, which
today means exactly pi (contamination probability). **If a second global parameter is
ever added, this rename becomes wrong** — the panel would no longer be pi-specific.
`MANIFEST.csv` flags every renamed file so the mapping is recoverable.

## Information gain

`information_gain/` sweeps four designs (CC/CB/BC/BB — each of two predictors either
continuous or binary) against trial count, both predictors loading on `v` and `a`.
The metric is a Gaussian expected information gain, a log-volume RATIO, chosen because
a ratio cancels the per-SD rescaling that puts continuous and binary regressors on a
common estimand. Two caveats travel with the numbers:

- The designs carry DIFFERENT priors over that estimand (1.704 nats between CC and BB,
  from the `cell_means` contrast having twice the variance of a continuous slope). That
  offset is constant in n, so it shifts the level of every gap but cancels from the
  slope — only the TREND and the derived trial multipliers are prior-free.
- Category proportions are Dirichlet-drawn, so "binary" means an unbalanced
  dichotomisation (realised balance 0.38–0.45), not a median split. These curves answer
  factor-vs-parametric, not median-split-vs-continuous.

## How the design configs were drawn

The random ensemble uses rejection sampling: the whole SET of 12 is redrawn until its
union covers every estimable cell, so the grid has no N/A cells and every config is
still a genuine random draw. This replaced a scheme that made the last config of each
family a deterministic saturated design. The configs actually drawn are recorded in
each `random_ensemble/*/design_configs.csv`.

Note the configs are not marginally independent — they are a uniform sample from the
*covering* sets. Measured density cost of that conditioning is about +10% relative to
unconditional sampling.
