# BayesFlow metrics replay of the 1,000-configuration evaluation

This evaluation calls the installed BayesFlow 2.0.14 public
`root_mean_squared_error`, `posterior_contraction`, and `calibration_error`
functions directly. NRMSE explicitly uses `normalize="range"`; all three use
median aggregation. Calibration uses 20 interval levels from 0.005 to 0.995.
No log-gamma calculation is performed.

The plan is copied exactly from `../ensemble_1000/run.json`: DDM 334, RDM 333,
CDM 333. The checkpoint, base seed, simulator backend, 200 datasets of 500 observations,
200 posterior draws, 50-step RK4, architecture, and chunk size match that run.
This evaluation uses fresh datasets: the original seed helper reset NumPy and
PyTorch but omitted Numba's separate RNG, which the Gaussian-softmax generator
uses for LKJ correlation matrices. Exact original datasets cannot be recovered
from those seeds. This run resets all three RNGs per configuration, using the
same derived seed for Numba as PyTorch, and records old and new input hashes.
Only datasets supporting a coefficient enter its metrics, and local coefficients
and global pi remain separate. Headline means give each configuration equal weight.

BayesFlow's installed RMSE takes the root mean squared error across posterior
draws per dataset, then the median across datasets. The previous local metric
implementation reduced across datasets first, then took the median across draws.
Consequently, the NRMSE results can change even with identical predictions.

The original run discarded its posterior draws. This replay therefore repeats
simulation and posterior inference. It stores fresh metrics in this directory;
the original `../ensemble_1000` results are preserved. The previous run's recorded
processing time was about 26 hours on two RTX 6000 Ada GPUs; actual runtime varies.

`progress.json` records live status, and `configs/NNNN.json` stores per-cell BF
metrics, support counts, metric version/commit, new and reference input hashes,
and whether inputs matched. This is a fresh-data evaluation of the same design
plan, not a paired metric-only comparison on identical old posterior draws.
Reports are written at 200, 400, 600, 800, and 1,000 completed configurations.
The final report will be `reports/metrics_1000.md`.

Launch or resume with the same frozen source and settings:

```bash
.venv/bin/python outputs/continuation/ensemble_metrics_1000_bayesflow_20261105/launch.py
```

The continuation directory contains the frozen source, command, environment,
source hashes, and launch record. The coordinator refuses to mix local-metric
results or different sampling settings into this BayesFlow run. The initial
strict-hash startup and RNG diagnostic evidence are archived under
`startup_hash_mismatch` in the continuation directory; no results were saved by
that attempt.
