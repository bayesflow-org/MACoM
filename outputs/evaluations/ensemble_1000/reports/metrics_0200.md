# Ensemble metrics after 200 configurations

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses the configured normalization (range for this study). SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 67 | 0.06977 | 0.91630 | 0.01946 |
| DDM | pi | 67 | 0.09302 | 0.91016 | 0.02553 |
| RDM | local | 67 | 0.09661 | 0.85313 | 0.02123 |
| RDM | pi | 67 | 0.09047 | 0.91789 | 0.02513 |
| CDM | local | 66 | 0.07306 | 0.90945 | 0.02142 |
| CDM | pi | 66 | 0.08835 | 0.91516 | 0.02183 |
| ALL | local | 200 | 0.07985 | 0.89288 | 0.02070 |
| ALL | pi | 200 | 0.09063 | 0.91440 | 0.02417 |
