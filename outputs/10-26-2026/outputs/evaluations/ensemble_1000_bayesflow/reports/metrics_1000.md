# Ensemble metrics after 1000 configurations

Metrics computed by BayesFlow 2.0.14 using its public root_mean_squared_error, posterior_contraction, and calibration_error functions.

Freshly simulated datasets for the reference structural plan; original datasets and posterior draws were not retained. NumPy, PyTorch, and Numba are seeded per configuration.

Means give equal weight to each configuration; within a configuration, active coefficient cells have equal weight. Local coefficients and global pi are reported separately. NRMSE uses range normalization. SD and SE in the CSV describe variation between config-level averages, not uncertainty within an individual dataset.

| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |
|---|---|---:|---:|---:|---:|
| DDM | local | 334 | 0.04887 | 0.91689 | 0.02030 |
| DDM | pi | 334 | 0.06896 | 0.91127 | 0.02462 |
| RDM | local | 333 | 0.07136 | 0.86120 | 0.02064 |
| RDM | pi | 333 | 0.06494 | 0.92037 | 0.02653 |
| CDM | local | 333 | 0.04381 | 0.92285 | 0.02184 |
| CDM | pi | 333 | 0.06572 | 0.91588 | 0.02155 |
| ALL | local | 1000 | 0.05467 | 0.90033 | 0.02093 |
| ALL | pi | 1000 | 0.06654 | 0.91584 | 0.02423 |
