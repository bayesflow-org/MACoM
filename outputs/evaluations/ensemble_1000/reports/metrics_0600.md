# Ensemble metrics after 600 configurations

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses the configured normalization (range for this study). SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 200 | 0.07068 | 0.91524 | 0.02016 |
| DDM | pi | 200 | 0.09237 | 0.91194 | 0.02571 |
| RDM | local | 200 | 0.09450 | 0.86047 | 0.02048 |
| RDM | pi | 200 | 0.09016 | 0.91954 | 0.02626 |
| CDM | local | 200 | 0.06746 | 0.92314 | 0.02174 |
| CDM | pi | 200 | 0.08908 | 0.91688 | 0.02063 |
| ALL | local | 600 | 0.07755 | 0.89962 | 0.02080 |
| ALL | pi | 600 | 0.09054 | 0.91612 | 0.02420 |
