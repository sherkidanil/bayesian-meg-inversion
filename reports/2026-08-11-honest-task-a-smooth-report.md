# Honest checkpoint: Task A and paired ΔX′

Date: 2026-08-11

The canonical best-model alias was switched to `trial81_honest_4965` before
these runs. Both Anna scenarios used exactly 100 posterior draws per input,
seed `20260810`, and Anna's persisted `dG=0` lead field for localization and
residual scoring. The `dG≠0` simulation lead field was not exposed to the
estimator.

The smooth decoder was frozen on the independent matched-server dev split:
quadratic score, `tau=0.019385232728949937`, selection SHA-256
`24305dbb3ac57460c5ca7039cfcbb2d025fded461bf6f7b8b727a86a63ab4120`.
No temperature or score-form search was run on Anna's data.

## Localization accuracy

Geodesic means are finite-only; the final column gives excluded non-finite
sample metrics.

| scenario | estimator | input | MAE euc, mm | MAE geo, mm | geo non-finite |
|---|---|---|---:|---:|---:|
| `dG=0` | argmin | original | 2.5103 | 4.5321 | 14 |
| `dG=0` | argmin | shifted | 2.4827 | 4.0123 | 8 |
| `dG=0` | frozen smooth | original | 2.5275 | 4.5531 | 12 |
| `dG=0` | frozen smooth | shifted | 2.5347 | 4.1003 | 7 |
| `dG≠0` | argmin | original | 2.7673 | 5.0028 | 9 |
| `dG≠0` | argmin | shifted | 2.9189 | 4.9325 | 7 |
| `dG≠0` | frozen smooth | original | 2.7286 | 4.7237 | 11 |
| `dG≠0` | frozen smooth | shifted | 2.9490 | 4.8037 | 7 |

## Paired stability

Euclidean ΔX′ is the primary complete metric. Geodesic ΔX′ is finite-only and
has the listed number of non-finite sample means.

| scenario | estimator | ΔX′ euc, mm | ΔX′ geo, mm | geo non-finite |
|---|---|---:|---:|---:|
| `dG=0` | argmin | 6.4091 | 12.9757 | 23 |
| `dG=0` | frozen smooth | **6.3745** | 13.1053 | 21 |
| `dG≠0` | argmin | 6.6367 | 13.7723 | 16 |
| `dG≠0` | frozen smooth | **6.5330** | 13.6416 | 18 |

Relative to argmin, frozen smoothing changes primary ΔX′ by −0.0346 mm at
`dG=0` and −0.1037 mm at `dG≠0`. The direction agrees with the discontinuity
hypothesis, but the effect is small: most paired motion remains after the
winner-selection discontinuity is removed.

Relative to the previous `push_3771` argmin results, the honest checkpoint's
argmin ΔX′ changes from 6.186 to 6.409 mm at `dG=0` and from 6.443 to 6.637 mm
at `dG≠0`. The earlier ΔX′ values must not be relabelled as results of the new
best model.

## Price of dG inside Anna's paired protocol

For original inputs, `dG≠0 − dG=0` changes MAE_euc by +0.2570 mm (argmin) and
+0.2011 mm (smooth). For shifted inputs the changes are +0.4361 and +0.4143
mm. Primary ΔX′ changes by +0.2277 mm (argmin) and +0.1585 mm (smooth).

The residual/error association remains strongly positive: mean within-sample
Spearman for Euclidean draw error is 0.511/0.517 on original/shifted `dG=0`
and 0.485/0.520 on `dG≠0`. This independently preserves Task A's conclusion
that the data distinguish better and worse posterior draws.

## Interpretation

- The decoder defect is real: residual scoring roughly halves localization
  error relative to the old posterior-mean estimator, and smooth weighting
  preserves nearly all of argmin's accuracy.
- Argmin discontinuity contributes to ΔX′ but is not its dominant component;
  the earlier dev decomposition already showed about 5.75 mm of same-key
  conditional trajectory motion.
- Accuracy and stability remain asymmetric under lead-field mismatch. Absolute
  Anna-domain MAE is cross-domain; paired ΔX′ is much less contaminated because
  both members of each pair share the same mismatch.
- The new numbers remain close to Anna's reported 4.05/4.78 mm, but parity is
  not claimed until the definition and disconnected-component policy of her
  `MAE Geo` are aligned.

## Provenance

Both Task A summaries contain `n_samples=315`,
`repository_changed_during_run=false`, and
`model_key=trial81_honest_4965`.

- `dG=0` summary SHA-256: `54a0e0ba06a202dbfd23025ed081b9de7723ac46f06a2e467f89572d2d8f9df8`
- `dG=0` argmin predictions: `86d6fd3efe06ab8e0921ea307aa05213672ac2aa4ff066ba8879438bc6ff146c`
- `dG=0` smooth predictions: `418eaf0450278a76956a4c229c7ffd857c11633656878bf0bda7c774b6bcf0d0`
- `dG≠0` summary SHA-256: `0f06c31040b483871b95844966cdc5332057566d2a8999137e7817bd4c121012`
- `dG≠0` argmin predictions: `fac27d8f5170f83ddbc4e8514696f576fc609541bafa2f193bc7dc2af2877485`
- `dG≠0` smooth predictions: `f96895cec1eeb8e4a863e9634152e2993f89efdf70b604f38de09b5121c950af`

Full CSV and NPZ artifacts remain under
`/mnt/local/tester/meg/reports/honest_smooth_anna_{dg0,dgnz}_20260811/`.
