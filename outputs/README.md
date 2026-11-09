# Final results

These are the retained October 26, 2026 study and its subsequent continuation
results. The former `outputs/10-26-2026/outputs/` and date container were flattened;
result arrays, checkpoints and figures were moved without recomputation.

- `finals/experiment-{1,1a,2,3,4}/`: assembled paper figures, tables, captions and source manifests.
- `figures/`: source figures, grouped by analysis and run tag.
- `data/`: generated benchmark datasets, predictions and encoder summaries.
- `checkpoints/`: trained CogFormer and BayesFlow models.
- `evaluations/`: evaluation records, including both 1,000-configuration sweeps.
- `timing/`, `logs/`, `wandb/`: timing and local run histories.
- `study.env`: retained design generator and structural settings.
- `run_manifest.json`, `source_snapshot/`, `checks/`: original provenance and source checks.
- `continuation/`: continuation provenance, source snapshots and sweep launchers.

The repository's experiment modules and `run_experiments.sh` default to this
folder. Empirical inputs are in the repository's separate `data/` directory.
See the repository README for replotting and reproduction commands. Regeneration
writes into each analysis's source directory; `bash run_experiments.sh finalize`
assembles those files into `finals/`. Each experiment's `appendix_captions` module
and `experiments.experiment_1a.gap_tables` refresh its text and tables.

Current metadata paths and internal result links were relocated with the files.
Frozen source snapshots and original logs retain their historical paths and source
hashes; they document how the results were produced. `launch_command.txt` is the
original launch record; use the current repository launcher to reproduce the full
pipeline. The ensemble continuation launchers read their relocated manifests.
