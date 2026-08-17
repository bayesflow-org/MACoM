# Paired LBA validation — 2026-08-16

Three-arm paired validation testing whether finetuning the model-class CogFormer on
LBA degrades recovery on the three families it was originally trained on.

Run as a chained driver, all arms `exit=0`. See `logs/mc_prod/paired_validate_driver.log`.

| Arm | Checkpoint evaluated | Families | Log |
|---|---|---|---|
| `zeroshot/` | original (pre-finetune) | LBA only | `logs/mc_prod/paired_zeroshot.log` |
| `finetuned/` | LBA-finetuned | LBA + DDM/CDM/RDM | `logs/mc_prod/paired_finetuned.log` |
| `original/` | original (pre-finetune) | DDM/CDM/RDM | `logs/mc_prod/paired_original.log` |

Wall clock: zeroshot 2026-08-15 16:18→18:39, finetuned 18:39→2026-08-16 03:52,
original 03:52→10:48 (all EDT).

## Checkpoint

`outputs/checkpoints/model_class/cf/lba_finetune/cogformer_lba_finetune_s2_A0.5_l8_h8_p256_s32_d128_b256_e2000_t100.pt`

Trained 2026-08-14 11:01 → 2026-08-15 08:21 EDT, 2000/2000 epochs, 21h 20m,
final loss 0.4524. Timing: `outputs/timing/lba_finetune.json`. Training log:
`logs/mc_prod/lba_finetune.log`.

The driver recorded `exit=1`, but that was cosmetic — training and the checkpoint
save both completed, then `trainer.finish()` raised a missing-`self` TypeError on
the final line. Since fixed in `experiments/model_class/model_class_cf_finetune_lba.py`.

## Result

Finetuning on LBA cut LBA error ~48% and cost <1% on the original three families —
no catastrophic forgetting.

LBA, mean NRMSE (lower is better):

| Scenario | zero-shot | finetuned | change |
|---|---|---|---|
| fixed | 0.4153 | 0.1376 | -0.2778 |
| fixed_regressed | 0.3076 | 0.0499 | -0.2577 |
| interaction | 0.1838 | 0.0989 | -0.0849 |
| intercept_only | 0.3424 | 0.1457 | -0.1967 |
| mixed_predictor | 0.2054 | 0.1601 | -0.0453 |
| regressed | 0.2617 | 0.0846 | -0.1771 |
| **overall** | **0.2386** | **0.1237** | **-48.2%** |

Original families, mean NRMSE:

| Family | original | finetuned | change |
|---|---|---|---|
| cdm | 0.0874 | 0.0877 | +0.3% |
| ddm | 0.0945 | 0.0954 | +1.0% |
| rdm | 0.1240 | 0.1246 | +0.5% |

Aggregated over per-parameter rows in each scenario's `*_metrics.csv`
(80 LBA rows per arm; 244 rows across the three original families).

## Layout

Each arm holds `<family>/<scenario>/` (zeroshot uses `<scenario>/` directly, since
it covers LBA only) with `*_recovery.pdf`, `*_coverage.pdf`, `*_ecdf.pdf`,
`*_metrics.{pdf,csv}`, their `*_global` variants, and 10 `*_posterior_N.pdf`.
Prediction `.npz` files are in each arm's `predictions/` and are gitignored.

## Provenance note

These three dirs were consolidated here on 2026-08-17 from:

- `outputs/figures/generalization/model_class_lba/cf/paired_zeroshot` → `zeroshot/`
- `outputs/figures/model_class/cf/lba_finetune/paired_ft` → `finetuned/`
- `outputs/figures/model_class/cf/paired_orig` → `original/`

Symlinks remain at all three old paths. The older LBA generalization scenarios that
shared the `generalization/model_class_lba/cf/` tree moved to
`outputs/figures/model_class/cf/lba_generalization/` (also symlinked back).

No driver script for this run was saved to the repo; the log above is the only record
of how it was invoked.
