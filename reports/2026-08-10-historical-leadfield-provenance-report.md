# Historical lead-field provenance

This is a truth-only provenance audit and is not an honest localization estimator.

| Dataset | Rows | Domain | Rebuilt lower | Median rebuilt−Anna | Historical dependency |
|---|---:|---|---:|---:|---|
| `meg_flow_dataset_10000_new_leadfield_error.npz` | 300 | `rebuilt_like` | 0.9967 | -0.051239 | 4.589 mm sent to Anna; oracle-contaminated localization result |
| `test_dataset_good_idxs_mesh.npz` | 240 | `rebuilt_like` | 1.0000 | -0.050701 | good_idxs three-model report (240 samples) and model ranking |
| `meg_flow_dataset_10000_new.npz` | 300 | `rebuilt_like` | 0.9967 | -0.045063 | training domain of the evaluated CFM model |

The training dataset does not persist G. Its classification is direct evidence about the generating sensor data, but the exact historical matrix remains unrecoverable as a file.
