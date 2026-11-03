# Experiment 1: 1,000 random design configurations

The original pretrained model-class checkpoint is evaluated on 1,000 random
configurations: DDM 334, RDM 333, CDM 333. Families are interleaved, giving
67 DDM, 67 RDM, and 66 CDM in the first 200. GPU workers use disjoint config
indices; neither begins config 201 until the report for the first 200 is saved.

Per configuration: 200 independently simulated datasets, 500 observations each,
200 posterior draws; 50-step RK4 with chunk size 5 (subject to the sampler's
existing token cap). Two regressors, K=2 categorical levels, optional interactions
with a per-design Beta(2,2) inclusion probability, free_prob=0.5, Gaussian-softmax
designs, and encoder structure conditioning match the existing ensemble protocol.
The random set for each family is conditioned on covering the estimable cells,
as in the existing ensemble. Configurations are sampled with replacement; they
are draws from the declared design distribution, not a uniform enumeration of
unique structural masks. CPU design generation uses one worker and stable per-config
NumPy/Torch seeds; forward simulation uses the GPU backend.

Only NRMSE, posterior contraction, and calibration error are computed, using the
existing metric functions: range-normalized NRMSE, median aggregation within each
coefficient cell, and 20 calibration interval levels from 0.005 to 0.995. There is
no log-gamma calculation. Local coefficients and global pi are reported separately.
Headline means average active cells within each config and give every config equal
weight. Per-parameter tables retain the number of contributing configs and SDs.

- `run.json`: checkpoint hash, exact settings, and all sampled structural designs.
- `configs/NNNN.json`: per-cell metrics, design, input hash, elapsed time, and GPU peaks.
- `progress.json`: live status and worker PIDs.
- `reports/metrics_0200.md`: first 200 configurations; also CSV, JSON, per-cell data,
  and per-parameter aggregation. Reports repeat at 400, 600, 800, and 1,000.
- `reports/metrics_1000.md`: final report.
- `coordinator.log`, `worker_0.log`, `worker_1.log`: live logs.

Posterior draws are discarded after each config's atomic result save. The run can
resume from completed config records with identical settings. Source snapshot,
launch record and resume entry point are in
`../../../continuation/ensemble_metrics_1000_20261029/`. Use its `launch.py` to
resume; the absolute source snapshot path is recorded in the launch manifest.

## Realized-support handling (November 2 recovery)

A fixed structural design can have an empty categorical interaction cell in some
datasets. Each coefficient is scored only on datasets whose realized mask marks it
active. Per-cell `n_datasets` records that support; fewer than two supported datasets
is insufficient for variance-based metrics, so such cells are omitted. Earlier
records without `n_datasets` had homogeneous masks and therefore use the full batch
size (200). The first 18 completed configs are preserved unchanged.
