# Experiment 1 — generalising to design space

Frozen snapshot. Every figure here comes from ONE checkpoint; this tree supersedes
the scattered per-run directories under `outputs/figures/model_class/cf/`.

Assembled 2026-09-11 21:40:06Z by
`experiments/finalize_experiment_1.py` — 271 files.

## Provenance

- checkpoint: `outputs/checkpoints/model_class/cf/r4c4_b256_e10000/cogformer_model_class_fm_l8_h8_p256_s32_d128_b256_e10000_t100.pt
  sha256: 56e997f37e1c945c4ae6d022ebe546b2010e6f257ad537cee1a9b966d7409811`
- git commit: `88b0fe7782e724ad5410ea8b19cc7fae9dced6aa`
- per-file sources, sizes and original mtimes: `MANIFEST.csv`

## Contents

| folder | what |
|---|---|
| `random_ensemble/` | 12 randomly drawn design configs, 4 per family (one combined set) |
| `benchmark_ensemble/` | `bench5` (5 OG cases) and `bench6` (+ `mixed_predictor`), per family |
| `per_case/` | the 18: 6 benchmark cases × 3 families, plotted separately |
| `information_gain/` | expected information gain and NRMSE vs trial count, four continuous/binary designs |
| `amortization_gap/` | C2ST against the per-case BayesFlow baselines — CDM, DDM, 52 CSVs; RDM pending |
| `predictions/` | raw posterior draws backing every figure, for re-plotting |

## Amortization gap — how to read it

`amortization_gap/<family>/summary.csv` has one row per case with `mean_accuracy`
(mean over per-parameter cells) and `joint_c2st` (all active cells at once).
Each `<family>/<case>/` holds the comparison and both arms that produced it:

| file | what |
|---|---|
| `orig_vs_bf.csv` | per-cell C2ST, plus `_mean_accuracy_` and `_joint_c2st_` rows |
| `bf_*.pdf` / `bf_metrics.csv` | the per-case BayesFlow baseline's own diagnostics |
| `bf_evaluations.csv` | BF NRMSE, calibration error, posterior contraction |
| `cf_*.pdf` / `cf_metrics*.csv` | the CogFormer arm, same datasets |

The `cf_*` files come from the PAIRED validate run, not from `per_case/`. `per_case/`
simulated its own datasets; the C2ST re-ran validation against the BF simulations so
both arms score identical data. Compare `bf_*` with `cf_*` here — comparing `bf_*`
against `per_case/` would be comparing across different data.

**Read `joint_c2st`, not `mean_accuracy`.** Both metrics pool draws across all 200
datasets, and the pooled 1-D marginal of any single parameter is dominated by
between-dataset spread, which both approximators reproduce — so `mean_accuracy`
sits near 0.55 while `joint_c2st` runs 0.96–0.998. That is not a contradiction and
not seven weak signals accumulating: seven independent cells at 0.55 would combine
to about 0.63, not 0.99. Within a single dataset the posteriors genuinely differ —
centres by 1–2.5 posterior SDs, widths by 2.3–10x — and only the joint classifier
can localise a draw well enough to see it. Same pooling effect that
`model_class_paired_c2st.py` documents for its per-dataset arm.

**These numbers have no null floor.** `model_family_c2st.py` has none of the
null-calibration machinery in `model_class_paired_c2st.py`, so nothing here is
baseline-corrected. The joint values are far past any plausible floor (~0.49), but
a per-cell 0.545 must not be quoted as "0.045 above chance".

**C2ST is direction-blind, and here the direction favours CogFormer.** On identical
datasets, the amortized model beats the per-case BayesFlow baseline on every
parameter of both families — DDM `v` correlation 0.965 vs 0.817, RMSE 0.209 vs
0.460 — while producing *wider* posteriors. The BF baselines are badly
overconfident: their own diagnostics in
`outputs/evaluations/model_family/bf/` report calibration error 0.39–0.47 and
posterior contraction 0.993–0.999, and their posterior SDs are ~10x smaller than
their own RMSE. So a large gap here is mostly a statement about BF, not a cost of
amortization. Budgets differ and are not matched: ~51M simulations per BF case
against ~256M for the one checkpoint covering every case and family.

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
The left panel is a Gaussian expected information gain, a log-volume RATIO, chosen
because a ratio cancels the per-SD rescaling that puts continuous and binary regressors
on a common estimand.

The right panel is mean NRMSE over the 11 active cells (draw-based RMSE against the
truth, normalised by the SD of the targets -- the repo's canonical `_rmse`). It
REPLACED a posterior-contraction panel, which said nothing the EIG panel did not: EIG
is the log-volume contraction of the same posterior, so the two are near-monotone
transforms of one another and BOTH are pure posterior-WIDTH measures. Neither can see a
design that sharpens the posterior around the wrong value. Since draw-based MSE
decomposes as

    NRMSE^2 = bias^2 / var(true) + (1 - contraction)

the old panel reported only the second term. The recovered bias/sd(true) runs 0.15-0.44
and is non-zero at every design and trial count, so that omission was real rather than
theoretical. It also separates two readings the old panel confounded: BB's NRMSE
flattens at 0.443/0.444 between 800 and 1000 trials while CC keeps improving to 0.305,
so the all-binary design stops converting extra trials into ACCURACY, not merely into
posterior width.

`posterior_draws.npz` holds the draws and targets behind both panels for all 20 cells
(`<design>_<n>_pred` (200, 200, 11), `<design>_<n>_true` (200, 11)). The first version
of this sweep persisted only summary scalars, so adding NRMSE later cost a second
four-hour GPU run -- posterior draws against truth cannot be reconstructed from a
contraction value. Any further metric is now a CPU replot.

Two caveats travel with the numbers:

- The designs carry DIFFERENT priors over that estimand (1.704 nats between CC and BB,
  from the `cell_means` contrast having twice the variance of a continuous slope). That
  offset is constant in n, so it shifts the level of every gap but cancels from the
  slope. Sigma_prior is itself estimated from 1500 draws, so re-running the sweep moves
  EIG by ~0.2 nats (max 0.78 observed) without changing the design ordering at any
  trial count. Only — only the TREND and the derived trial multipliers are prior-free.
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
