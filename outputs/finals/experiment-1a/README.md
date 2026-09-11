# Experiment 1a — amortization gap

Assembled 2026-09-11 21:51:06Z by `experiments/finalize_experiment_1a.py` at commit `675154b89ed6bfb949ab32b9755ebed45a93e3fd`.
10 files copied this run, 162 carried from earlier runs.

## What this measures

A paired C2ST per bench5 case: how distinguishable are the posteriors of a
case-specific BayesFlow baseline from those of the single amortized ModelClass
CogFormer? **0.5 = indistinguishable = no gap.** Both arms score the *same* simulated
datasets — the CogFormer arm is the PAIRED validation run, not the per-case run
behind `experiment-1/per_case/`.

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
| DDM | 5 | 86 files: fixed, fixed_regressed, interaction, intercept_only, regressed |
| RDM | — | absent — never run |
| CDM | 5 | 86 files: fixed, fixed_regressed, interaction, intercept_only, regressed |

`amortization_gap.tex` is hand-written (the paper table) and has no manifest row —
it is authored, not copied.

## How to read it

`<family>/summary.csv` has one row per case with `mean_accuracy` (mean over
per-parameter cells) and `joint_c2st` (all active cells at once).

The `cf_*` files come from the PAIRED validate run, not from
`experiment-1/per_case/`. `per_case/` simulated its own datasets; the C2ST re-ran
validation against the BF simulations so both arms score identical data. Compare
`bf_*` with `cf_*` here — comparing `bf_*` against `per_case/` would be comparing
across different data.

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

**C2ST is direction-blind, and here the direction favours CogFormer.** On identical
datasets the amortized model beats the per-case BayesFlow baseline on every parameter
of both families — DDM `v` correlation 0.965 vs 0.817, RMSE 0.209 vs 0.460 — while
producing *wider* posteriors. The BF baselines are badly overconfident: their own
diagnostics report calibration error 0.39–0.47 and posterior contraction 0.993–0.999,
and their posterior SDs are ~10x smaller than their own RMSE. So a large gap here is
mostly a statement about BF, not a cost of amortization. Budgets differ and are not
matched: ~51M simulations per BF case against ~256M for the one checkpoint covering
every case and family.

## Provenance warning

The DDM and CDM files here come from a run whose **sources no longer exist**:
`outputs/figures/c2st/` and `outputs/figures/model_class/cf_paired/` are absent,
there are zero `outputs/data/model_family/*_data.npz`, and 1 of 15 BayesFlow
checkpoints survives. **RDM was never run** — yet `amortization_gap.tex` quotes RDM
numbers, so that run happened and its artifacts were lost.

Re-deriving any of this is a full rebuild (~15 BayesFlow trainings):

```bash
bash run_experiments.sh exp1a
```

`MANIFEST.csv` therefore carries forward rows whose sources are gone. Re-running this
assembler never removes files it cannot regenerate.

## Excluded

`full` (dropped by `model_family_c2st`) and `mixed_predictor` (no BF baseline was
ever trained on it) have no BF arm to compare against, so they are absent by
construction rather than by omission. The 200-epoch baseline variant lands in
`e200/` via `experiments/finalize_experiment_1a_e200.py` — a separate budget, kept
in its own subtree and its own manifest so the two are never mixed.
