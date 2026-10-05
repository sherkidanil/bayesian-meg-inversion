# Truth-free sensor-residual posterior rescoring

- Scenario: `${MEG_CFM_WORKSPACE}/data_anna_shift/anna_dg0.npz`
- Localization G: `${MEG_CFM_WORKSPACE}/data_anna_shift/anna_dg0.npz` element 0, verified constant
- Scenario `G_for_simulation` was not loaded or used by the estimator.
- Samples: 315; draws per input: 100
- `residual_selected`, `single_draw`, `posterior_mean_naive`, and `posterior_medoid` are truth-free.
- Oracle rows and draw-error/Spearman columns are diagnostic-only.

## Honest and diagnostic localization metrics

| Input | Estimator | Honest | MAE euc, mm | MAE geo, mm | Geo non-finite |
|---|---|---:|---:|---:|---:|
| original | `residual_selected` | yes | 2.3362 | 4.5470 | 11 |
| original | `single_draw` | yes | 5.2732 | 10.2482 | 21 |
| original | `posterior_mean_naive` | yes | 4.0512 | 7.1668 | 20 |
| original | `posterior_medoid` | yes | 4.2689 | 7.7394 | 18 |
| original | `oracle_best_euc` | no | 1.4574 | 2.4589 | 6 |
| original | `oracle_best_geo` | no | 1.6529 | 1.9412 | 1 |
| shifted | `residual_selected` | yes | 2.0818 | 3.5129 | 13 |
| shifted | `single_draw` | yes | 5.2503 | 9.6569 | 15 |
| shifted | `posterior_mean_naive` | yes | 4.0005 | 6.9389 | 15 |
| shifted | `posterior_medoid` | yes | 4.1710 | 7.3003 | 14 |
| shifted | `oracle_best_euc` | no | 1.3889 | 2.4838 | 7 |
| shifted | `oracle_best_geo` | no | 1.5771 | 1.8671 | 1 |

## Residual/error rank association

| Input | Error | Mean per-sample Spearman | Pooled raw | Pooled within-sample rank |
|---|---|---:|---:|---:|
| original | euclidean | 0.5124 | 0.4287 | 0.5122 |
| original | geodesic | 0.4226 | 0.3096 | 0.4247 |
| shifted | euclidean | 0.5462 | 0.4367 | 0.5457 |
| shifted | geodesic | 0.4721 | 0.3302 | 0.4745 |

## Interpretation boundary

Residual selection uses only Y, the persisted Anna localization G, mesh geometry, and spatial parameters. Truth is used only after selection for this audit. Absolute CFM MAE on Anna-domain data remains a cross-domain number and must not be compared directly with matched-G RAP MUSIC until the in-domain control is available.
