# amortization_gap_e200

BayesFlow per-family baselines retrained at **200 epochs x batch 256** (e200), under
the 2026-09-10 pipeline changes:

- `standardize_summaries` OFF -- inference_variables only, matching the predecessor
  project. The A/B measured standardizing the summary channel as a small regression
  (calibration error 0.166 vs 0.119 at 100 epochs).
- `--time_power_law_alpha 0.0` -- uniform flow-matching time. bayesflow 2.0.13 changed
  this default from 0.0 to 0.5, so a bare `FlowMatching()` no longer matches what the
  predecessor trained.
- `model_family_bf_validate` pins `ModelFamily(num_workers=1)`, so `seed_case` actually
  pins the validation set (its thread pool previously made draws non-reproducible).
- The global parameter `pi` is reported on its own axes: `bf_*_pi.pdf` / `bf_metrics_pi.csv`.
  The BF arm was the only one missing these; the `cf_*` arm already had them.

## Why this is not in `amortization_gap/`

`amortization_gap/` is a coherent snapshot: its `bf_*`, `cf_*` and C2ST files all
describe the 2000-epoch checkpoints. These figures come from different checkpoints and
cover only 2 of the 5 cases, so merging them would leave `orig_vs_bf.csv` describing a
comparison that no longer matches the figures beside it. **There is no C2ST arm here** --
that would need the paired CogFormer validate re-run against these npz files.

## Contents

`{ddm,cdm}/{intercept_only,regressed}/` with `bf_recovery`, `bf_coverage`, `bf_ecdf`,
`bf_metrics` (+ `_pi` companions) and `bf_evaluations.csv`.

## Headline

200 epochs beats 2000 on every case and metric, but remains far from the predecessor:

| run | calib err | contraction |
|---|---|---|
| ddm/intercept_only 200ep | 0.259 | 0.967 |
| ddm/intercept_only 2000ep | 0.422 | 0.996 |
| cdm/intercept_only 200ep | 0.214 | 0.973 |
| cdm/intercept_only 2000ep | 0.437 | 0.997 |
| ddm/regressed 200ep | 0.378 | 0.989 |
| cdm/regressed 200ep | 0.348 | 0.991 |
| predecessor (all four) | 0.011-0.027 | 0.848-0.922 |

Posteriors are centred but too narrow: empirical coverage of a nominal 95% interval is
0.61-0.65 on intercept_only and 0.32-0.38 on regressed. See MANIFEST.csv for the source
path and mtime of every file.

Snapshot git commit: `35c0b557e9d4919952f5d80712c4bfec5689f026`
