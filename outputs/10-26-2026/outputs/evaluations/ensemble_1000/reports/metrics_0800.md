# Ensemble metrics after 800 configurations

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses the configured normalization (range for this study). SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 267 | 0.07116 | 0.91450 | 0.02024 |
| DDM | pi | 267 | 0.09383 | 0.91052 | 0.02560 |
| RDM | local | 267 | 0.09446 | 0.86032 | 0.02052 |
| RDM | pi | 267 | 0.08903 | 0.91996 | 0.02608 |
| CDM | local | 266 | 0.06654 | 0.92537 | 0.02180 |
| CDM | pi | 266 | 0.08914 | 0.91654 | 0.02095 |
| ALL | local | 800 | 0.07740 | 0.90003 | 0.02085 |
| ALL | pi | 800 | 0.09067 | 0.91567 | 0.02422 |
