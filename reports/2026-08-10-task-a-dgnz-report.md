# Truth-free sensor-residual posterior rescoring

- Scenario: `${MEG_CFM_WORKSPACE}/data_anna_shift/anna_dgnz.npz`
- Localization G: `${MEG_CFM_WORKSPACE}/data_anna_shift/anna_dg0.npz` element 0, verified constant
- Scenario `G_for_simulation` was not loaded or used by the estimator.
- Samples: 315; draws per input: 100
- `residual_selected`, `single_draw`, `posterior_mean_naive`, and `posterior_medoid` are truth-free.
- Oracle rows and draw-error/Spearman columns are diagnostic-only.

## Honest and diagnostic localization metrics

| Input | Estimator | Honest | MAE euc, mm | MAE geo, mm | Geo non-finite |
|---|---|---:|---:|---:|---:|
| original | `residual_selected` | yes | 2.5202 | 4.7397 | 14 |
| original | `single_draw` | yes | 5.4972 | 10.7230 | 25 |
| original | `posterior_mean_naive` | yes | 4.3996 | 8.1097 | 22 |
| original | `posterior_medoid` | yes | 4.5876 | 8.1329 | 22 |
| original | `oracle_best_euc` | no | 1.6320 | 2.8242 | 8 |
| original | `oracle_best_geo` | no | 1.8314 | 2.2784 | 1 |
| shifted | `residual_selected` | yes | 2.4826 | 4.0714 | 16 |
| shifted | `single_draw` | yes | 5.4968 | 10.5621 | 23 |
| shifted | `posterior_mean_naive` | yes | 4.3681 | 7.5903 | 19 |
| shifted | `posterior_medoid` | yes | 4.4907 | 8.1656 | 19 |
| shifted | `oracle_best_euc` | no | 1.6286 | 2.7959 | 5 |
| shifted | `oracle_best_geo` | no | 1.7684 | 2.1353 | 1 |

## Residual/error rank association

| Input | Error | Mean per-sample Spearman | Pooled raw | Pooled within-sample rank |
|---|---|---:|---:|---:|
| original | euclidean | 0.5049 | 0.4256 | 0.5048 |
| original | geodesic | 0.4180 | 0.2976 | 0.4213 |
| shifted | euclidean | 0.5231 | 0.4196 | 0.5228 |
| shifted | geodesic | 0.4516 | 0.3165 | 0.4539 |

## Interpretation boundary

Residual selection uses only Y, the persisted Anna localization G, mesh geometry, and spatial parameters. Truth is used only after selection for this audit. Absolute CFM MAE on Anna-domain data remains a cross-domain number and must not be compared directly with matched-G RAP MUSIC until the in-domain control is available.
