# Ensemble metrics after 600 configurations

Metrics computed by BayesFlow 2.0.14 using its public root_mean_squared_error, posterior_contraction, and calibration_error functions.

Freshly simulated datasets for the reference structural plan; original datasets and posterior draws were not retained. NumPy, PyTorch, and Numba are seeded per configuration.

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses range normalization. SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 200 | 0.04901 | 0.91523 | 0.02004 |
| DDM | pi | 200 | 0.06763 | 0.91241 | 0.02487 |
| RDM | local | 200 | 0.07159 | 0.86036 | 0.02056 |
| RDM | pi | 200 | 0.06590 | 0.91994 | 0.02665 |
| CDM | local | 200 | 0.04383 | 0.92307 | 0.02207 |
| CDM | pi | 200 | 0.06531 | 0.91669 | 0.02131 |
| ALL | local | 600 | 0.05481 | 0.89955 | 0.02089 |
| ALL | pi | 600 | 0.06628 | 0.91635 | 0.02428 |
