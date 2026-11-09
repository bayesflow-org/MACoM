# Ensemble metrics after 800 configurations

Metrics computed by BayesFlow 2.0.14 using its public root_mean_squared_error, posterior_contraction, and calibration_error functions.

Freshly simulated datasets for the reference structural plan; original datasets and posterior draws were not retained. NumPy, PyTorch, and Numba are seeded per configuration.

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses range normalization. SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 267 | 0.04936 | 0.91438 | 0.02031 |
| DDM | pi | 267 | 0.06889 | 0.91090 | 0.02503 |
| RDM | local | 267 | 0.07163 | 0.86037 | 0.02053 |
| RDM | pi | 267 | 0.06505 | 0.92017 | 0.02622 |
| CDM | local | 266 | 0.04301 | 0.92531 | 0.02194 |
| CDM | pi | 266 | 0.06539 | 0.91631 | 0.02099 |
| ALL | local | 800 | 0.05468 | 0.89999 | 0.02093 |
| ALL | pi | 800 | 0.06644 | 0.91579 | 0.02408 |
