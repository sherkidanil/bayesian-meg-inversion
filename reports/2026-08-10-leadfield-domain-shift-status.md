# Lead-field domain-shift status

## Established result

The rebuilt and Anna lead fields are structurally different in the projector geometry used by data-residual localization. Global scaling leaves a relative residual of 0.3555. For real Gaussian patches, the median principal angle is 7.84° and the P90 of the maximum angle per case is 60.44°.

The within-dataset truth-only calibration classifies fixed test as `rebuilt_like`: rebuilt `G` wins 298/300 fixed samples, while Anna `G` wins 313/315 `anna_dg0` samples. The training dataset did not persist `G`; the training-domain conclusion remains an indirect inference from the five-day proximity of the training and fixed artifacts.

## Interpretation rule

- Absolute CFM MAE on Anna-domain data is directly contaminated by lead-field mismatch. A comparison with RAP MUSIC using its matched `G` is not symmetric.
- Paired ΔX′ is less contaminated because both members of a pair share the same mismatch. It is not assumed invariant.
- Therefore the stability conclusion remains usable; the absolute-accuracy conclusion is suspended pending the in-domain control.

## Next computations

1. A matched 315-pair server-`G` dataset and paired domain-shift cost measurement.
2. Tail decomposition and reconciliation with Anna's exact `MAE Geo` definition.

## Task A result

The residual criterion positively ranks good posterior draws without access to truth. On `dG=0`, pooled within-sample rank Spearman is 0.5122 for Euclidean error and 0.4247 for geodesic error. On `dG!=0`, the corresponding values are 0.5048 and 0.4213.

For original inputs, residual selection changes honest mean MAE as follows:

| Scenario | Posterior mean euc | Residual-selected euc | Posterior mean geo | Residual-selected geo |
|---|---:|---:|---:|---:|
| `dG=0` | 4.0512 | **2.3362** | 7.1668 | **4.5470** |
| `dG!=0` | 4.3996 | **2.5202** | 8.1097 | **4.7397** |

The Euclidean oracle remains at 1.4574/1.6320 mm. Thus the posterior contains better solutions, and the sensor residual recovers much of that gap. This supports a decoder defect rather than retraining as the necessary explanation. These absolute MAE values are still Anna-domain measurements and do not remove the lead-field domain-shift caveat.

Detailed reports: [dG=0](2026-08-10-task-a-dg0-report.md) and [dG!=0](2026-08-10-task-a-dgnz-report.md).

## Post-Task-A paired result

Residual selection improves absolute localization but does not materially improve the primary Euclidean paired stability metric:

| Scenario | Previous posterior-mean ΔX′ | Residual-selected ΔX′ | Change | Residual-selected geodesic ΔX′ |
|---|---:|---:|---:|---:|
| `dG=0` | 6.205 | 6.186 | −0.019 | 12.638 (295/315 finite) |
| `dG!=0` | 6.366 | 6.443 | +0.077 | 12.966 (291/315 finite) |

Thus localization accuracy and paired stability are empirically distinct decoder properties. The fixed-shift neighbor-correction branch is inferred in exactly 46/315 retained samples (14.60%), with zero disagreement between the generator rule replay and changed centroids. It does not explain the full original/shifted difficulty asymmetry:

| Scenario | Subset | Original MAE geo | Shifted MAE geo | Shifted−original |
|---|---|---:|---:|---:|
| `dG=0` | no correction (269) | 4.648 | 3.372 | −1.276 |
| `dG=0` | correction (46) | 3.965 | 4.315 | +0.350 |
| `dG!=0` | no correction (269) | 4.915 | 3.913 | −1.001 |
| `dG!=0` | correction (46) | 3.719 | 4.989 | +1.270 |

The mean minimum pair spacing rises only by 0.668/0.681 mm after shifting. The correction subset reverses the MAE asymmetry instead of causing it, while the larger no-correction subset retains a strong shifted advantage. The proposed neighbor-correction mechanism is therefore rejected as the primary explanation; ΔX′ still mixes response to the shift with an unresolved directional/configuration difficulty effect.

Detailed paired reports: [dG=0](2026-08-10-post-task-a-paired-dg0-report.md) and [dG!=0](2026-08-10-post-task-a-paired-dgnz-report.md).

## Residual-selected fixed and Fresh IID result

The fixed test was rescored with the explicit rebuilt lead field and the existing `trial81_push_3771` checkpoint. At N=100, residual selection changes MAE_geo from 9.093 to **5.4065 mm** (MAE_euc 2.9302 mm; 16/300 non-finite geodesic rows).

The paired Fresh IID-100 × 3 reranking is no longer a tie:

| Rank | Model | MAE geo, mm | MAE euc, mm | Non-finite geo |
|---:|---|---:|---:|---:|
| 1 | `trial81_honest_4965` | **4.3057 ± 0.0799** | **2.5864 ± 0.1383** | 17 |
| 2 | `trial81_push_3771` | 4.7815 ± 0.2607 | 2.6974 ± 0.2325 | 14 |
| 3 | `trial81_original` | 5.2502 ± 0.3628 | 2.9388 ± 0.1101 | 13 |

For `honest_4965 − push_3771`, the paired geodesic differences on data seeds 202603260/1/2 are −0.2587, −0.4534, and −0.7153 mm: `honest_4965` wins 3/3, with mean advantage 0.4758 mm. It also beats `original` 3/3. This is sufficient to replace `push_3771` with `honest_4965` as the accuracy leader under the corrected truth-free decoder. Task A and the current residual-selected ΔX′ rows still refer to `push_3771`; they must not be silently relabeled as results of the new leader.

Detailed reports: [fixed test](2026-08-10-residual-selected-fixed-report.md) and [paired IID ranking](2026-08-10-residual-selected-iid-ranking-report.md).

## Historical provenance result

All audited historical datasets are `rebuilt_like`:

| Dataset | Rows | Rebuilt wins | Median residual rebuilt−Anna | Consequence |
|---|---:|---:|---:|---|
| `meg_flow_dataset_10000_new_leadfield_error.npz` | 300/10000 | 99.67% | −0.05124 | 4.589 mm was in-domain, but remains oracle-contaminated |
| `test_dataset_good_idxs_mesh.npz` | 240/240 | 100% | −0.05070 | March three-model report is not contaminated by Anna-domain shift |
| `meg_flow_dataset_10000_new.npz` | 300/10000 | 99.67% | −0.04506 | Direct sensor-data evidence for a rebuilt-like training domain; exact G file remains lost |

Detailed table: [historical lead-field provenance](2026-08-10-historical-leadfield-provenance-report.md).
