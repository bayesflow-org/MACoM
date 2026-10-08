# Experiment 1a — amortization gap

Assembled 2026-10-07 16:30:02Z by `experiments/finalize_experiment_1a.py` at commit `9394c50c99b753b5d4e855735f40a4a81265af3b`.
348 files copied this run, 0 carried from earlier runs.

## What this measures

A paired C2ST per bench5 case: how distinguishable are the posteriors of a
case-specific BayesFlow baseline from those of the single amortized ModelClass
CogFormer? **0.5 = indistinguishable = no gap.** Both arms score the *same* simulated
datasets: the shared benchmark files written once by the `e1_data` stage.

## Layout

`<family>/summary.csv` — the C2ST summary for that family.
`<family>/<case>/` — one folder per bench5 case, holding both arms:

| file | arm |
|---|---|
| `bf_recovery.pdf`, `bf_coverage.pdf`, `bf_ecdf.pdf`, `bf_metrics.pdf` | BayesFlow baseline |
| `bf_evaluations.csv`, `bf_metrics.csv` | NRMSE / calibration error / contraction |
| `cf_*.pdf`, `cf_*.csv` | paired CogFormer (`_pi` = global-parameter panel) |
| `orig_vs_bf.csv` | the C2ST itself |

| family | cases | contents |
|---|---|---|
| DDM | 5 | 116 files: fixed, fixed_regressed, interaction, intercept_only, regressed |
| RDM | 5 | 116 files: fixed, fixed_regressed, interaction, intercept_only, regressed |
| CDM | 5 | 116 files: fixed, fixed_regressed, interaction, intercept_only, regressed |

`amortization_gap_tables.tex` and `captions.tex` have no manifest row: they are generated
from the same sources by `python -m experiments.experiment_1a.gap_tables` and
`python -m experiments.experiment_1a.appendix_captions` (re-run both after this script).
`captions.tex` holds the appendix figures: one per case and arm (CogFormer, then the
BayesFlow baseline) stacking recovery, ECDF and metrics, with the full caption on the first.

## How to read it

`<family>/summary.csv` has one row per case. Pooled arms (one classifier over draws
from all 200 datasets): `mean_accuracy` (mean over per-parameter cells) and `joint_c2st`
(all active cells at once). Per-dataset arm (one classifier per dataset, averaged over
datasets): `perds_joint_c2st` and `perds_mean_accuracy`, each with its draw-split
`*_null_floor`. `<case>/orig_vs_bf.csv` has the per-cell values of both arms, and
`<case>/orig_vs_bf_per_dataset.csv` one row per dataset.

The `cf_*` files are experiment 1's per-case validation of the same cases
(`experiment-1/per_case/`), which scored those same shared files. So `bf_*`, `cf_*` and
`experiment-1/per_case/` describe identical datasets and compare directly. (The
seed alone would NOT make that true: it pins the parameters but not the simulated RTs
or regressor designs. Loading the saved files is what pairs the arms.)

**Read the per-dataset arm.** Pooled over 200 datasets, one parameter's draws trace
the aggregate posterior, which is roughly the prior for any calibrated approximator, so
the pooled `mean_accuracy` sits near 0.52 however far apart the per-dataset posteriors
are. The pooled `joint_c2st` sees per-dataset differences only once the datasets
separate in parameter space (9+ active cells), so it undershoots on intercept_only and
fixed and saturates near 0.95 on the rest -- it is not comparable across cases with
different numbers of parameters. Measured 2026-10-01: within a dataset, CogFormer's
posterior means sit a median 0.3–0.8 baseline posterior SDs from the baseline's, and
its posteriors are 1.2–1.6x wider. Same pooling effect `model_class_paired_c2st.py`
documents for experiment 3.

**Per-dataset numbers have a null floor; read them against it.** Real = first halves of
both arms' draws (~100 per side); floor = the larger of each arm's first-half vs
second-half C2ST on the same dataset. The joint uses the MLP; the per-parameter test
uses a quadratic discriminant, because at ~100 draws per side the 1-D MLP barely learns
(0.53 vs 0.67 with QDA on ddm/interaction, null 0.50). The pooled arms have no null.

**C2ST is direction-blind.** A large value says the two posteriors differ, not which
one is better. Read it next to `bf_metrics.csv` and `cf_metrics.csv` (NRMSE,
calibration error, contraction) on the same datasets: both come from the same metric
code. `bf_evaluations.csv` is BayesFlow's own implementation, whose NRMSE aggregates
differently (per-dataset RMSE, then the median over datasets) and is not comparable to
`cf_metrics.csv`; contraction and calibration error agree between the two.

## Budget

Matched by construction (decided 2026-09-27): the model-class CogFormer trains on
15000 epochs x 100 steps x 128 = 192M simulated datasets
across three families, and each of the 15 BayesFlow baselines on
1000 x 100 x 128 = 12.8M, i.e. 64M per family on both sides, at the same batch size.

## Excluded

`full` (dropped by `model_family_c2st`) and `mixed_predictor` (no BF baseline was
ever trained on it) have no BF arm to compare against, so they are absent by
construction rather than by omission.
