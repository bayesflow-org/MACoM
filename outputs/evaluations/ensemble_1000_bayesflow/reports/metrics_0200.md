# Ensemble metrics after 200 configurations

Metrics computed by BayesFlow 2.0.14 using its public root_mean_squared_error, posterior_contraction, and calibration_error functions.

Freshly simulated datasets for the reference structural plan; original datasets and posterior draws were not retained. NumPy, PyTorch, and Numba are seeded per configuration.

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses range normalization. SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 67 | 0.04824 | 0.91651 | 0.01970 |
| DDM | pi | 67 | 0.06813 | 0.91036 | 0.02438 |
| RDM | local | 67 | 0.07370 | 0.85316 | 0.02030 |
| RDM | pi | 67 | 0.06659 | 0.91780 | 0.02741 |
| CDM | local | 66 | 0.04807 | 0.90890 | 0.02249 |
| CDM | pi | 66 | 0.06538 | 0.91490 | 0.02228 |
| ALL | local | 200 | 0.05671 | 0.89278 | 0.02082 |
| ALL | pi | 200 | 0.06671 | 0.91435 | 0.02470 |
