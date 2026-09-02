# Experiment 3 — LBA zero-shot inference and finetuning

Frozen snapshot of the three-arm paired validation. Assembled
2026-09-02 18:58:27Z by
`experiments/finalize_experiment_3.py` — 560 files. Git commit `c9d6f9704b88c743d6b301c82b16f83bd7d28e97`.

`paired_validation_notes.md` is the original run's write-up and carries the headline
tables; `MANIFEST.csv` records per-file provenance.

## Arms

| folder | checkpoint evaluated | families |
|---|---|---|
| `zeroshot/` | original, pre-finetune | LBA only |
| `finetuned/` | LBA-finetuned | LBA + DDM/CDM/RDM |
| `original/` | original, pre-finetune | DDM/CDM/RDM |
| `c2st/` | both, compared | DDM/CDM/RDM, 5 scenarios |
| `predictions/` | — | posterior draws per arm |

Each case folder holds `recovery`, `coverage`, `ecdf`, `metrics` as PDF (plus
`metrics.csv`) and their `_pi` global-parameter companions.

## Headline

LBA mean NRMSE 0.2386 → 0.1237 (**-48.2%**); the three original families move
+0.3% / +1.0% / +0.5%.

**Read that second number alongside `c2st/forgetting_c2st_summary.csv`, not instead
of it.** Aggregate NRMSE barely moves, but the per-dataset C2ST columns in that file
report `perds_mean_excess` of 0.24-0.37 over the null with `perds_frac_excess_gt_05`
at 0.985-1.0 — on essentially every dataset the finetuned and original posteriors are
distinguishable. The matched-sample C2ST in the same file shows excess of only
0.009-0.021, which is the expected degeneracy: both models are calibrated, so pooled
draws approach the prior and the test goes to its null by construction. The two
statistics are not in conflict, but "no catastrophic forgetting" is a stronger claim
than aggregate NRMSE alone supports.

**The NRMSE metric is also not precise enough to carry that claim.**
`bf.diagnostics.metrics.root_mean_squared_error` is STOCHASTIC: three calls on
byte-identical arrays return

    [1.0041 0.2435 1.0260 0.1322 ...]
    [1.0088 0.2446 1.0172 0.1329 ...]
    [0.9887 0.2427 1.0222 0.1347 ...]

i.e. roughly 1-2% spread run to run. (`posterior_contraction` on the same arrays is
deterministic to 4 dp, so this is the RMSE estimator, not the data.) The +0.3% / +1.0%
/ +0.5% movements reported for CDM/DDM/RDM therefore sit INSIDE the metric's own
noise: they cannot distinguish "no forgetting" from "a small amount of forgetting" in
either direction. Any re-derivation of those numbers should average over repeated
calls, or use contraction, which is reproducible.

## Excluded

0 posterior pairplots (0 files, 10 per case in `original/`) were
deleted at source and are excluded here. This assembler uses an ALLOWLIST of plot
stems rather than copying directory contents, so a future run that forgets
`--skip_posteriors` cannot reintroduce them silently.

## Naming

Family/scenario/arm infixes (`_fm_`, `_ft_`, `_mid1_`) are stripped — the path already
carries them. `_global` is renamed `_pi`; that panel is the global-parameter view,
which today is exactly pi. The zeroshot arm's single extra variant level (`mid_rdm/`)
is flattened.
