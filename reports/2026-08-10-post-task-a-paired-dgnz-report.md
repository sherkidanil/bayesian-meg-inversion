# Post-Task-A paired audit: dGnz

## Paired ΔX′ after residual selection

- Euclidean, per-sample mean: 6.4432 (median 6.1313; n=315, nonfinite=0) mm
- Geodesic, per-sample mean: 12.9662 (median 10.2738; n=291, nonfinite=24) mm
- Previous posterior-mean Euclidean ΔX′: 6.366 mm
- Predictions were aligned independently to the corresponding original and shifted truths; truth is used only to define the controlled paired stability metric.

## Shift-construction asymmetry

- Neighbor-correction branch inferred: 46/315 (14.60%).
- Actual shifted centroids differ from the simple nearest snap: 46/315.
- Original minimum pair spacing: 49.1436 (median 47.1880; n=315, nonfinite=0) mm
- Shifted minimum pair spacing: 49.8251 (median 46.4927; n=315, nonfinite=0) mm
- Shifted minus original minimum spacing: 0.6815 (median 0.3850; n=315, nonfinite=0) mm

## Residual-selected localization MAE

| subset | input | MAE euc, mm | MAE geo, mm |
|---|---|---:|---:|
| all | original | 2.5202 (median 1.9536; n=315, nonfinite=0) | 4.7397 (median 2.0971; n=301, nonfinite=14) |
| all | shifted | 2.4826 (median 2.1628; n=315, nonfinite=0) | 4.0714 (median 2.2025; n=299, nonfinite=16) |
| no_correction | original | 2.5394 (median 2.0652; n=269, nonfinite=0) | 4.9145 (median 2.1698; n=257, nonfinite=12) |
| no_correction | shifted | 2.4547 (median 2.1763; n=269, nonfinite=0) | 3.9131 (median 2.0806; n=255, nonfinite=14) |
| correction | original | 2.4083 (median 1.6733; n=46, nonfinite=0) | 3.7186 (median 1.6733; n=44, nonfinite=2) |
| correction | shifted | 2.6454 (median 2.1295; n=46, nonfinite=0) | 4.9886 (median 2.4792; n=44, nonfinite=2) |

The correction split diagnoses whether ΔX′ mixes sensitivity to the input shift with a systematic change in source separability. It does not alter the estimator.
