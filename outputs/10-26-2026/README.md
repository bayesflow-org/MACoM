# Experiment 1: Gaussian-softmax structure rerun

Launched on October 25, 2026 (America/New_York), using the requested `10-26-2026`
folder. The detached pipeline runs Experiment 1 only: shared benchmark generation,
training, per-case validation, random-design ensemble, benchmark ensemble,
design information gain, and final assembly. Experiment 1a and Experiments 2–4
are outside this run.

Training retains the previous full-run budget and architecture: 15,000 epochs ×
100 steps × batch 128 = 192 million datasets; DDM/RDM/CDM; 0–4 regressors;
2–4 categories; 200–1,000 observations. The design generator is
`gaussian_softmax`. The new settings are optional parameter-specific interactions
with a per-design Beta(2,2) inclusion probability, and encoder structure conditioning.
This run does not enable the diagnostic-only prior-coordinate, count or
known-zero-column masking variants.

Acceleration matches the previous Gaussian-softmax full training: GPU simulation,
six spawned simulation workers, BF16 autocast, fused AdamW, three decoder groups,
and separately compiled encoder/decoder blocks with
`max-autotune-no-cudagraphs`. The process uses GPU 0 and CPU affinity 0–23.
Single-GPU training preserves batch 128; the second GPU is available.

W&B: [calgww80](https://wandb.ai/jerrymhuang-rensselaer-polytechnic-institute/cogformer-model-class/runs/calgww80).
The study explicitly selects Jerry's original entity. No credentials were changed.

Files:

- `study.env`: the generator, structural settings and W&B destination.
- `experiment_1.log`: pipeline progress; `experiment_1.pid`: launcher PID.
- `logs/finals/20261025_214220/e1_train.log`: training progress and compiler messages.
- `outputs/`: checkpoints, data, figures and `finals/experiment-1/`.
- `wandb/`: local tracking files.
- `run_manifest.json`: source hashes, versions, budget, GPU settings and benchmark hashes.
- `source_snapshot/`: the actual source used by this run, including uncommitted work.
- `source_changes.patch`: tracked changes relative to the recorded Git commit.
- `checks/`: passing structural and orchestration checks; `dry_run.log`: planned commands.

The pipeline is running from the saved source, so later source edits in the working
tree do not change training or its queued evaluation. The 20 benchmark files were
generated once in this study and hashed; reuse these exact files for future paired
BayesFlow comparisons. Training-time validation retains the existing monitoring
protocol; the standalone scoring stages receive `gaussian_softmax` and the new
encoder flag. No calibration improvement is established until evaluation finishes.

To inspect progress:

```bash
tail -n 10 outputs/10-26-2026/experiment_1.log
tail -n 10 outputs/10-26-2026/logs/finals/20261025_214220/e1_train.log
```

If the process exits, first inspect its logs and ensure it is no longer running.
`launch_command.txt` records its exact launcher and environment settings. Running
the same launcher again resumes training from its last saved checkpoint and skips
completed stages. The first checkpoint is scheduled at epoch 1,000; subsequent
checkpoints are scheduled every 2,500 epochs. A checkpoint is complete only after
the pipeline writes its `.done` sentinel.
