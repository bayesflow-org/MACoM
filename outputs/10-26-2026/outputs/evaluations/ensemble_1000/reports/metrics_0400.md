# Ensemble metrics after 400 configurations

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses the configured normalization (range for this study). SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 134 | 0.07005 | 0.91654 | 0.01985 |
| DDM | pi | 134 | 0.09141 | 0.91224 | 0.02588 |
| RDM | local | 133 | 0.09467 | 0.85920 | 0.02024 |
| RDM | pi | 133 | 0.08876 | 0.91867 | 0.02502 |
| CDM | local | 133 | 0.07059 | 0.91533 | 0.02196 |
| CDM | pi | 133 | 0.08814 | 0.91730 | 0.02046 |
| ALL | local | 400 | 0.07842 | 0.89707 | 0.02068 |
| ALL | pi | 400 | 0.08944 | 0.91606 | 0.02379 |
