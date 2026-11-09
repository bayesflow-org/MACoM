# Ensemble metrics after 400 configurations

Metrics computed by BayesFlow 2.0.14 using its public root_mean_squared_error, posterior_contraction, and calibration_error functions.

Freshly simulated datasets for the reference structural plan; original datasets and posterior draws were not retained. NumPy, PyTorch, and Numba are seeded per configuration.

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses range normalization. SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 134 | 0.04850 | 0.91646 | 0.02002 |
| DDM | pi | 134 | 0.06676 | 0.91243 | 0.02470 |
| RDM | local | 133 | 0.07211 | 0.85907 | 0.02032 |
| RDM | pi | 133 | 0.06511 | 0.91887 | 0.02613 |
| CDM | local | 133 | 0.04620 | 0.91514 | 0.02217 |
| CDM | pi | 133 | 0.06453 | 0.91724 | 0.02091 |
| ALL | local | 400 | 0.05558 | 0.89694 | 0.02083 |
| ALL | pi | 400 | 0.06547 | 0.91617 | 0.02392 |
