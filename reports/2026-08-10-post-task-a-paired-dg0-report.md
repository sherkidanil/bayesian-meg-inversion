# Post-Task-A paired audit: dG0

## Paired ΔX′ after residual selection

- Euclidean, per-sample mean: 6.1860 (median 5.9640; n=315, nonfinite=0) mm
- Geodesic, per-sample mean: 12.6381 (median 10.1145; n=295, nonfinite=20) mm
- Previous posterior-mean Euclidean ΔX′: 6.205 mm
- Predictions were aligned independently to the corresponding original and shifted truths; truth is used only to define the controlled paired stability metric.

## Shift-construction asymmetry

- Neighbor-correction branch inferred: 46/315 (14.60%).
- Actual shifted centroids differ from the simple nearest snap: 46/315.
- Original minimum pair spacing: 49.0359 (median 47.0262; n=315, nonfinite=0) mm
- Shifted minimum pair spacing: 49.7042 (median 46.4567; n=315, nonfinite=0) mm
- Shifted minus original minimum spacing: 0.6683 (median 0.3408; n=315, nonfinite=0) mm

## Residual-selected localization MAE

| subset | input | MAE euc, mm | MAE geo, mm |
|---|---|---:|---:|
| all | original | 2.3362 (median 1.7498; n=315, nonfinite=0) | 4.5470 (median 1.9297; n=304, nonfinite=11) |
| all | shifted | 2.0818 (median 1.6040; n=315, nonfinite=0) | 3.5129 (median 1.6325; n=302, nonfinite=13) |
| no_correction | original | 2.3788 (median 1.8397; n=269, nonfinite=0) | 4.6480 (median 1.9584; n=259, nonfinite=10) |
| no_correction | shifted | 2.0273 (median 1.5987; n=269, nonfinite=0) | 3.3724 (median 1.5563; n=257, nonfinite=12) |
| correction | original | 2.0871 (median 1.4878; n=46, nonfinite=0) | 3.9653 (median 1.6781; n=45, nonfinite=1) |
| correction | shifted | 2.4005 (median 2.0058; n=46, nonfinite=0) | 4.3149 (median 2.0325; n=45, nonfinite=1) |

The correction split diagnoses whether ΔX′ mixes sensitivity to the input shift with a systematic change in source separability. It does not alter the estimator.
