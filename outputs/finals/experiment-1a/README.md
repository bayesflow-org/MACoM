# Experiment 1a — amortization gap

Assembled 2026-10-01 04:55:12Z by `experiments/finalize_experiment_1a.py` at commit `2ec3724dd7178aae536c28ee372b3ece1d72d9bc`.
333 files copied this run, 0 carried from earlier runs.

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
| DDM | 5 | 111 files: fixed, fixed_regressed, interaction, intercept_only, regressed |
| RDM | 5 | 111 files: fixed, fixed_regressed, interaction, intercept_only, regressed |
| CDM | 5 | 111 files: fixed, fixed_regressed, interaction, intercept_only, regressed |

`amortization_gap.tex` is hand-written (the paper table) and has no manifest row —
it is authored, not copied.

## How to read it

`<family>/summary.csv` has one row per case with `mean_accuracy` (mean over
per-parameter cells) and `joint_c2st` (all active cells at once).

The `cf_*` files are experiment 1's per-case validation of the same cases
(`experiment-1/per_case/`), which scored those same shared files. So `bf_*`, `cf_*` and
`experiment-1/per_case/` describe identical datasets and compare directly. (The
seed alone would NOT make that true: it pins the parameters but not the simulated RTs
or regressor designs. Loading the saved files is what pairs the arms.)

**Read `joint_c2st`, not `mean_accuracy`.** Both metrics pool draws across all 200
datasets, and the pooled 1-D marginal of any single parameter is dominated by
between-dataset spread, which both approximators reproduce — so `mean_accuracy` sits
near 0.55 while `joint_c2st` runs 0.96–0.998. That is not a contradiction and not
seven weak signals accumulating: seven independent cells at 0.55 would combine to
about 0.63, not 0.99. Within a single dataset the posteriors genuinely differ —
centres by 1–2.5 posterior SDs, widths by 2.3–10x — and only the joint classifier can
localise a draw well enough to see it. Same pooling effect that
`model_class_paired_c2st.py` documents for its per-dataset arm.

**These numbers have no null floor.** `model_family_c2st.py` has none of the
null-calibration machinery in `model_class_paired_c2st.py`, so nothing here is
baseline-corrected. The joint values are far past any plausible floor (~0.49), but a
per-cell 0.545 must not be quoted as "0.045 above chance".

**C2ST is direction-blind.** A large value says the two posteriors differ, not which
one is better. Read it next to `bf_evaluations.csv` and `cf_metrics.csv` (NRMSE,
calibration error, contraction) on the same datasets.

## Budget

Matched by construction (decided 2026-09-27): the model-class CogFormer trains on
15000 epochs x 100 steps x 128 = 192M simulated datasets
across three families, and each of the 15 BayesFlow baselines on
1000 x 100 x 128 = 12.8M, i.e. 64M per family on both sides, at the same batch size.

## Excluded

`full` (dropped by `model_family_c2st`) and `mixed_predictor` (no BF baseline was
ever trained on it) have no BF arm to compare against, so they are absent by
construction rather than by omission.
