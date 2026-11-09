# Ensemble metrics after 1000 configurations

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses the configured normalization (range for this study). SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 334 | 0.07058 | 0.91700 | 0.02029 |
| DDM | pi | 334 | 0.09407 | 0.91103 | 0.02553 |
| RDM | local | 333 | 0.09429 | 0.86116 | 0.02070 |
| RDM | pi | 333 | 0.08900 | 0.92024 | 0.02645 |
| CDM | local | 333 | 0.06747 | 0.92283 | 0.02155 |
| CDM | pi | 333 | 0.08975 | 0.91593 | 0.02147 |
| ALL | local | 1000 | 0.07744 | 0.90035 | 0.02085 |
| ALL | pi | 1000 | 0.09094 | 0.91573 | 0.02449 |
