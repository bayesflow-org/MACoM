# Experiment 3 — LBA zero-shot inference and finetuning

Frozen snapshot of the three-arm paired validation. Assembled
2026-09-15 05:54:53Z by
`experiments/finalize_experiment_3.py` — 593 files. Git commit `22fad3d5961a6a3bae464878935cc39b227ffa7c`.

`paired_validation_notes.md` is the original run's write-up and carries the headline
tables; `MANIFEST.csv` records per-file provenance.

## ppc/ — LBA on empirical data

Posterior predictive checks for the finetuned checkpoint applied to the empirical
lexical-decision dataset (17 subjects, 500 stored draws each). It sits in experiment 3
rather than experiment 2 because experiment 2 is the DDM/RDM/CDM real-data tree — the
three families the model class was trained on — while LBA is the held-out family this
experiment exists to study. Same checkpoint as `finetuned/`.

**Two variants, same posterior draws.** They differ only in what the forward
simulation holds fixed:

| folder | s_v | s_tau | what it is |
|---|---|---|---|
| `ppc/free/` | estimated | estimated | the model's own fit, simulated as-is |
| `ppc/conventional/` | 1.0 | 0.0 | the standard LBA identifiability constraint |

`posteriors.npz` is stored ONCE, at `ppc/`, not per variant: both PPCs read the same
draws, so duplicating it would imply two fits where there is one.

Which variant belongs in the paper is a modelling decision and has NOT been made here.
Both are kept in labelled subtrees precisely so that decision stays visible — a single
unlabelled `ppc/` would make the constraint unrecoverable from the tree.

Plot names follow experiment 2 exactly, so the same check is called the same thing in
both real-data trees: `rt_ppc_*` (RT distribution), `choice_ppc_*` (choice
proportions), `rt_qq_*`, `rt_quantiles_*`, `choice_quantiles_*`, suffixed by condition
(`condition`, `frequency`, `lexicality`).

## trajectory/ — finetuning path, NOT just its endpoints

Complete: all five checkpoints (0/500/1000/1500/2000).

The three arms above are ENDPOINTS: pre-finetune and post-finetune. `trajectory/` adds
the path between them, evaluating diagnostics at epochs 0/500/1000/1500/2000 of a
re-run finetune. It answers a question the endpoints assert but cannot show: whether
LBA's gain is gradual or front-loaded, and whether the three trained families drift at
any point rather than only at the end.

- `lba_finetuning_trajectory*.pdf` — NRMSE, posterior contraction and calibration
  error against epoch, one line per family. Note contraction runs the OPPOSITE way to
  the other two (higher is better), which is stated on each panel.
- `trajectory_summary.csv` — the plotted means, aggregated over parameters then over
  scenarios.
- `trajectory_per_scenario.csv` — one row per (epoch, family, scenario, metric). Kept
  because the family mean hides genuine disagreement between scenarios: LBA's
  per-scenario NRMSE spans 0.20-0.48 at epoch 0.

Epoch 0 is not "untrained": it is the pre-finetune checkpoint with LBA's embedding row
warm-started by COPYING RDM's, which is exactly the zero-shot condition. Every epoch was
scored on IDENTICAL datasets (`seed_case` with a fixed seed), verified by byte-comparing
`true_set` across epochs, so movement along a line is the checkpoint changing and not
the test set.

**Lineage warning.** The trajectory comes from a SECOND finetune run (2026-09-09,
seeded), while `finetuned/` above still comes from the 2026-08-14 run, which set no
seed. The two are therefore not the same model, and the trajectory's endpoint will not
exactly equal the `finetuned/` numbers. Reconciling them -- rebuilding the arms from the
seeded lineage -- is pending.

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

1 posterior pairplots (1 files, 10 per case in `original/`) were
deleted at source and are excluded here. This assembler uses an ALLOWLIST of plot
stems rather than copying directory contents, so a future run that forgets
`--skip_posteriors` cannot reintroduce them silently.

## Naming

Family/scenario/arm infixes (`_fm_`, `_ft_`, `_mid1_`) are stripped — the path already
carries them. `_global` is renamed `_pi`; that panel is the global-parameter view,
which today is exactly pi. The zeroshot arm's single extra variant level (`mid_rdm/`)
is flattened.
