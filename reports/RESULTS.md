# Canonical results: source-model specification dominates method choice

Date: 2026-08-20

Source-model specification dominates method choice by an order of magnitude: changing
the classical dictionary from point topographies to matched Gaussian patches moves
geodesic MAE from 11.25 to 1.118 mm, whereas the CFM-versus-classical differences
are within 1.5 mm. With correct radii patch-RAP is best (1.125 versus 2.716 mm on
the strict misspecification mask); under 0.7× radii CFM is better (5.845 versus
7.135 mm), and under 1.3× radii patch-RAP is again better (3.018 versus 4.985 mm).
Although CFM degrades proportionally less at both perturbations, absolute
degradation reverses at 1.3× (+2.269 mm for CFM versus +1.893 mm for patch-RAP),
and equal weighting of nominal, 0.7×, and 1.3× regimes gives 3.759 mm for
patch-RAP versus 4.515 mm for CFM. Thus the learned prior is proportionally softer
under radius misspecification, but not enough to reverse the overall ranking:
source-model specification matters more than algorithm selection, and patch-RAP
remains the practical choice under this three-regime model.

## Final result

On the matched server-lead-field corpus, the temporal-attention CFM with the
truth-free residual-weighted cortical medoid, 400 posterior draws, and a
dev-selected mesh-refinement policy achieves geodesic MAE
`2.618 ± 0.288 mm` across three independently trained seeds. The pass counts
selected before held-out was opened are `0`, `0`, and `1`. This is more
accurate than the standard free-orientation, point-dipole MNE RAP MUSIC
reference (`3.836 mm`), but it is not the final equal-prior comparison.

All three seeds favour CFM. On the intersection of cases with finite geodesic
metrics for RAP MUSIC and every CFM seed (`n=290`), the seed-and-case mean
difference is `-1.202 mm`. A two-level bootstrap that resamples both training
seeds and cases gives a 95% interval of `[-1.905, -0.517] mm`. A paired
one-sided Wilcoxon test on the per-case CFM error averaged over seeds gives
`p=0.000385`.

When RAP is given the same Gaussian-patch source dictionary as the CFM
decoder, it reaches `1.118 mm` geodesic MAE on the common finite mask versus
`2.646 mm` for CFM. Patch-RAP minus CFM is `-1.527 mm`, with hierarchical 95%
interval `[-2.143,-0.901] mm` and one-sided Wilcoxon `p=1.85e-25`. The
pre-registered outcome is therefore **source shape explains the apparent CFM
accuracy advantage**. The network remains an amortized proposal mechanism and
may retain a speed trade-off, but it does not beat exhaustive patch-RAP on
localization accuracy in this benchmark. Delta X' is numerically lower than
standard point RAP but is not significant (`p=0.101`), so no formal Pareto
dominance claim remains.

## S=400 base decoder before mesh refinement

Each seed freezes its own quadratic residual temperature on dev before
held-out is opened. The draw bank is prefix-stable: the first 100 and 200 draws
are exact prefixes of the 400-draw bank.

| training seed | decoded Euc, mm | decoded Geo, mm | Delta X' Euc, mm | oracle Geo, mm | finite pairs on frozen mask | tau |
|---:|---:|---:|---:|---:|---:|---:|
| 303 | 1.607 | 2.883 | 5.601 | 1.290 | 294/300 | 0.009479 |
| 304 | 1.476 | 2.659 | 5.568 | 1.212 | 296/300 | 0.020223 |
| 305 | 1.418 | 2.530 | 5.457 | 1.146 | 296/300 | 0.009929 |
| mean ± SD | **1.500 ± 0.097** | **2.691 ± 0.179** | **5.542 ± 0.076** | **1.216 ± 0.072** | **295.3 ± 1.2** | — |

Per-seed paired comparisons to RAP MUSIC:

| seed | mean CFM−RAP, mm | bootstrap 95% CI, mm | one-sided Wilcoxon p | paired n |
|---:|---:|---:|---:|---:|
| 303 | -0.904 | [-1.671, -0.147] | 0.000915 | 294 |
| 304 | -1.179 | [-1.906, -0.510] | 0.000207 | 296 |
| 305 | -1.210 | [-1.904, -0.542] | 0.00000308 | 296 |

The common-mask seed-specific mean differences are `-0.963`, `-1.217`, and
`-1.214 mm`; hence the aggregate result is not created by one favourable seed.

## Prepublication tests and final estimator

The number of mesh passes was selected separately for each seed on dev by the
preregistered rule: minimize Delta X' subject to Euclidean MAE no greater than
`1.10 × MAE(0 passes)`, with ties resolved by lower MAE and then fewer passes.
Held-out remained unopened until the choice was persisted.

| training seed | selected passes | final Euc, mm | final Geo, mm | final Delta X' Euc, mm |
|---:|---:|---:|---:|---:|
| 303 | 0 | 1.607 | 2.883 | 5.601 |
| 304 | 0 | 1.476 | 2.659 | 5.568 |
| 305 | 1 | 1.205 | 2.312 | 5.461 |
| mean ± SD | — | **1.429 ± 0.205** | **2.618 ± 0.288** | **5.543 ± 0.073** |

On the locked `n=290` mask, final CFM and RAP MUSIC geodesic levels are
`2.649` and `3.851 mm`; the paired CFM-minus-RAP result is `-1.202 mm`,
hierarchical 95% interval `[-1.905,-0.517] mm`, one-sided Wilcoxon
`p=0.000385`. For Delta X', the corresponding levels are `5.577` and
`6.009 mm`; the mean difference is `-0.432 mm` and the hierarchical interval is
`[-0.814,-0.080] mm`, but Wilcoxon gives `p=0.101`. Therefore the numerical
stability improvement is reported descriptively, not as an established
advantage.

## Lead-field sensitivity of the zero-refinement S=400 decoder

The CFM side was recomputed with V1, the cortical medoid, `S=400`, zero mesh
refinement, and each seed's own dev-frozen temperature. RAP MUSIC inference was
not rerun: the persisted `dG=0` and `dG!=0` prediction artifacts are reused.
This section predates the later `0/0/1` pass selection and is not used as a
headline claim of robustness.
The historical four-sided convention was kept—both branches must be finite for
both methods in both perturbation states. Reproducing the old CFM configuration
recovers the published mask counts (`297` MAE pairs, `292` Delta X' pairs) and
RAP MUSIC cost (`+0.9473156 mm`) exactly.

The final comparison uses the stricter intersection across RAP MUSIC and all
three V1 seeds (`n=286` for MAE and `n=282` for Delta X'). Exact reuse of the
old rows would require imputing new cross-component failures, so no such
imputation is made.

| quantity | seed 303 | seed 304 | seed 305 | CFM mean ± SD | RAP MUSIC |
|---|---:|---:|---:|---:|---:|
| Geo MAE cost of `dG`, mm | 0.864 | 0.573 | 0.702 | **0.713 ± 0.146** | 0.930 |
| Delta X' Euc cost of `dG`, mm | 0.392 | 0.142 | 0.279 | **0.271 ± 0.125** | 0.260 |

For geodesic MAE, the hierarchical seed-and-case interval for CFM is
`[0.407, 1.031] mm`. The point sensitivity ratio is `0.930/0.713 = 1.30`, but
the paired RAP-minus-CFM difference is `+0.217 mm` with interval
`[-0.359, 0.804] mm`; lower CFM sensitivity is therefore suggested but not
statistically resolved. For Delta X', CFM's interval is `[0.071, 0.469] mm`
and the RAP-minus-CFM difference is `-0.012 mm`, interval
`[-0.331, 0.311] mm`. The methods are indistinguishable on this response.

## Architecture result

At 100 draws, temporal attention improves every diagnostic level relative to
the matched V0 control:

| architecture | seeds | single-draw Geo, mm | decoded Geo, mm | decoded Euc, mm | oracle Geo, mm | finite pairs |
|---|---:|---:|---:|---:|---:|---:|
| V0 baseline | 2 | 10.105 ± 0.155 | 4.963 ± 0.340 | 2.711 ± 0.062 | 2.453 ± 0.001 | 292.0 ± 2.8 |
| V1 temporal attention | 3 | **8.801 ± 0.287** | **3.616 ± 0.266** | **1.999 ± 0.127** | **1.805 ± 0.073** | **296.0 ± 1.0** |

For the common training seeds, the paired V1−V0 decoded-geodesic differences
are `-0.909 mm` (`p=0.00148`) for seed 303 and `-1.870 mm`
(`p=4.42e-12`) for seed 304. Temporal attention therefore improves the raw
distribution, practical decoding, oracle coverage, and cross-hemisphere tail;
the result is not only a decoder effect.

The previous `4.723` versus `4.187 mm` V0 discrepancy is not a seed estimate.
Its dominant term (`+0.748 mm`) comes from the changed training/evaluation
harness and checkpoint; mask and temperature terms are much smaller. See
`docs/v0-discrepancy-decomposition.md`.

## Decoder ablations

### Cortical aggregation

Replacing Euclidean posterior averaging by a residual-weighted geodesic medoid
improves seed-303 decoded geodesic MAE from `3.907` to `3.736 mm` and Euclidean
MAE from `2.144` to `2.083 mm`. A one-ring Fréchet expansion selects the same
held-out centroids, so it adds no benefit.

### Greedy mesh refinement

The earlier `3.295 mm` refinement result starts from the S=100 hard argmin and
cannot be compared directly with the three-seed S=400 medoid mean of
`2.691 mm`. A controlled, untuned diagnostic applies the same two graph passes
directly to each final S=400 medoid. It improves geodesic MAE for every seed:
`2.883→2.626`, `2.659→2.350`, and `2.530→2.309 mm`, for an exploratory mean of
`2.428 ± 0.173 mm`. Delta X' changes only slightly,
`5.542 ± 0.076→5.561 ± 0.084 mm`.

The criterion is nevertheless imperfect locally. Among branches that move,
the residual decreases in 100%, while true geodesic error increases in
`32.5%`, `25.0%`, and `29.6%` across the three seeds. Thus the data support
local residual/truth mismatch, not the stronger claim that the residual
minimum is globally shifted or that the oracle gap is unreachable by local
refinement. The confirmatory dev selection chooses `0/0/1` passes rather than
two passes for every seed. It replaces the headline by `2.618 ± 0.288 mm`; the
fixed two-pass `2.428 mm` result remains exploratory.

### Signal-subspace denoising

Truncating the sensor covariance does not improve draw ranking. Mean per-sample
Spearman correlation between residual and true geodesic draw error is `0.4593`
for the full signal, `0.4594` at rank 3, and `0.4592` at rank 6. This line is
closed as a negative result.

### Number of draws

Increasing the prefix-stable draw bank improves accuracy, stability, and the
oracle together:

| draws | decoded Euc, mm | decoded Geo, mm | Delta X' Euc, mm | oracle Geo, mm | paired CFM−RAP, mm |
|---:|---:|---:|---:|---:|---:|
| 100 | 2.028 | 3.774 | 5.833 | 1.810 | -0.053 |
| 200 | 1.868 | 3.412 | 5.752 | 1.530 | -0.426 |
| 400 | **1.607** | **2.883** | **5.601** | **1.290** | **-0.904** |

The 400-draw result is the first point whose paired interval lies entirely
below zero. More draws help because the posterior already covers excellent
solutions and the residual ranking is informative; the experiment does not
support the claim that the model must be retrained from scratch.

## Negative architecture results

Two independent compressions of the sensor signal are harmful:

- accumulated signal-subspace vertex features failed to reach comparable
  train loss across two encoders and three learning rates;
- covariance concatenation V5 worsened single-draw, decoded, and oracle
  geodesic MAE (`11.778`, `5.974`, and `2.824 mm`).

Together with the success of temporal attention, these results show that the
fixed temporal windows and frequency bands carry important source identity.
Physics-derived spatial summaries should not replace the raw temporal path in
this task.

## Point-source residual ablation

Replacing the Gaussian-patch basis by the point basis
`B=[g_c1,g_c2,g_c3]`, while retaining V1, the cortical medoid, `S=400`, and
dev-only temperature selection, degrades geodesic MAE to `6.459`, `5.749`, and
`6.842 mm` across seeds: `6.350 ± 0.555 mm`. Euclidean MAE is
`3.576 ± 0.274 mm` and Delta X' is `5.990 ± 0.154 mm`.

The ablation therefore remains decisive for the zero-refinement S=400 decoder:
the residual decoder is strongly sensitive to misspecifying extended Gaussian
sources as points. Standard MNE has no mirror switch for Gaussian patches, but
the independently validated scalar RAP implementation below now supplies that
equal-prior comparison.

## Raw-residual greedy scan: optimizer negative result

The earlier prepublication diagnostic precomputes three dictionaries
`G_patch = G @ Phi.T` of shape `3 × 102 × 4699` for the true radius multiset
`(10, 8, 10) mm`. Each greedy stage exhaustively scans every remaining
radius-slot/vertex pair using the same localization `G`, 2–25 Hz filtering,
Gaussian profile, and subspace residual as the CFM decoder. Each radius slot is
consumed once and a selected vertex cannot be repeated.

The raw-residual scan is substantially worse: finite-only geodesic MAE is
`7.880 mm`, with `84/700` non-finite branches and `39/290` failures on the
locked headline mask. On the common finite subset (`n=251`), CFM is
`2.693 mm`, the patch scan `7.762 mm`, and RAP MUSIC `3.689 mm`. The
seed-and-case CFM-minus-scan difference is `-5.068 mm`, interval
`[-6.318,-3.899] mm`, Wilcoxon `p=2.91e-14`. This measures the weakness of
greedy minimization of the raw residual. It is not a classical RAP MUSIC
baseline and cannot establish that network proposals add localization
information.

Precomputation takes `0.420 s`; the 700-branch scan takes `4.425 s` wall time
with eight CPU workers (`0.0485 s` accumulated per branch on average). The
complete run including filtering, metrics, and artifact writes takes `21.9 s`.

## RAP MUSIC with a matched Gaussian-patch dictionary

The equal-prior baseline implements scalar RAP directly. From the top-three
left singular vectors `E_s` of the filtered sensor data, each stage scores a
candidate topography `b` after projecting away previously selected
topographies:

`rho(b) = ||(P_perp E_s)^T(P_perp b)|| / ||P_perp b||`.

Unlike TRAP MUSIC, RAP retains all three signal-subspace vectors at every
stage. The implementation applies the same rank-99 MNE SSP projector to the
data and dictionaries. With the point dictionary and fixed orientations, its
ordered predictions match MNE RAP MUSIC on all `700/700` branches. This
validation also exposes a large orientation effect: on the shared finite mask
(`n=277`), fixed-minus-free geodesic MAE is `+7.416 mm`, 95% interval
`[6.479,8.372] mm`, `p=1.24e-37`.

Replacing only the point dictionary by the three precomputed Gaussian-patch
dictionaries gives finite-only geodesic MAE `1.346 mm` over all branches. On
the common patch/CFM mask (`n=288`), the levels are:

| method | geodesic MAE, mm |
|---|---:|
| Gaussian-patch RAP MUSIC | **1.118** |
| final refined CFM, seed average | 2.646 |
| standard free-orientation point RAP MUSIC | 3.701 |

Patch-RAP minus CFM is `-1.527 mm`; the three seed-specific differences are
`-1.767`, `-1.525`, and `-1.290 mm`. The hierarchical seed-and-case interval is
`[-2.143,-0.901] mm`, the direct paired bootstrap interval is
`[-2.043,-0.980] mm`, and one-sided Wilcoxon gives `p=1.85e-25`. Patch-RAP also
beats free point RAP by `-2.583 mm`, interval `[-2.987,-2.204] mm`, two-sided
`p=6.28e-34`.

Under the outcome rule frozen before inspecting patch-RAP, the result is
classified as `source_shape_explains_cfm_advantage` because patch-RAP is below
`2.6 mm`. Equalizing the source model reverses the accuracy comparison: the
classical exhaustive subspace scan is more accurate than the network pipeline.
The vectorized patch scan averages `0.0879 s` of CPU time per branch and
`11.36 s` wall time for 700 branches with eight workers, excluding the shared
`16.31 s` filtering step. Therefore the previously reported CFM speedup over
MNE must not be transferred to this custom equal-prior implementation without
a dedicated end-to-end timing protocol.

## Source-radius misspecification: proportional and absolute robustness

The final experiment perturbs the analytic radii by the same `±30%` in both
methods. Patch-RAP rebuilds its exhaustive topography dictionary at
`(7,5.6,7) mm` or `(13,10.4,13) mm`. CFM reuses the three trained V1 networks,
their prefix-stable `S=400` posterior banks, and each seed's nominal dev-frozen
quadratic-medoid temperature; only the radii inside the truth-free residual
decoder are changed. The networks are not retrained. This directly tests
whether the learned proposal distribution can compensate for an incorrect
analytic source model.

All levels below use one strict mask: the locked headline mask intersected with
finite geodesic pairs for nominal, `0.7×`, and `1.3×` patch-RAP and all three
CFM seeds (`n=274`). Relative degradation is computed from aggregate means on
this same mask.

| decoder radii | patch-RAP Geo, mm | patch degradation | CFM Geo, mm | CFM degradation | patch−CFM relative degradation |
|---|---:|---:|---:|---:|---:|
| nominal | 1.125 | — | 2.716 | — | — |
| `0.7×` | 7.135 | +6.010 mm; **+533.9%** | 5.845 | +3.129 mm; **+115.2%** | +418.7 pp |
| `1.3×` | 3.018 | +1.893 mm; **+168.1%** | 4.985 | +2.269 mm; **+83.6%** | +84.6 pp |

At `0.7×`, the hierarchical interval for the relative-degradation difference
is `[218.3,859.5]` percentage points and one-sided Wilcoxon gives
`p=2.32e-37`. At `1.3×`, the interval is `[10.2,241.8]` points and
`p=1.35e-6`. Every CFM seed shows the same qualitative result: its relative
degradation is `106.1–124.6%` at `0.7×` and `76.0–98.3%` at `1.3×`.
The outcome rule was frozen before these runs and requires both scales, a
strictly positive hierarchical interval, and `p<0.05`; it classifies the
result as `learned_prior_more_robust` on the preregistered relative scale.
The planned fallback variant is not warranted.

Non-finite branch counts out of 700 are `17/27/18` for patch-RAP at
nominal/`0.7×`/`1.3×`. For CFM they are `22/27/24`, `15/26/23`, and
`20/22/27` for seeds `303`, `304`, and `305`, respectively. The strict common
mask prevents any method-specific failure handling from entering the reported
accuracy contrast.

The absolute and relative measures reverse at `1.3×`: relative degradation
favours CFM (`83.6% < 168.1%`), but absolute degradation favours patch-RAP
(`1.893 < 2.269 mm`). This is caused by the different nominal denominators
(`1.125` versus `2.716 mm`). CFM is more accurate in only one of the three
regimes, `0.7×`, where both methods are already strongly degraded. Patch-RAP
is more accurate at nominal and `1.3×`.

With equal probability assigned to nominal, `0.7×`, and `1.3×`, expected
geodesic MAE is `(1.125+7.135+3.018)/3 = 3.759 mm` for patch-RAP and
`(2.716+5.845+4.985)/3 = 4.515 mm` for CFM. Thus the classical exhaustive
method remains the practical choice under this three-regime model. The learned
prior degrades proportionally more softly, but that does not reverse the
overall accuracy ranking.

The previous `6.350 ± 0.555 mm` CFM number is the point-source `R→0` ablation,
not the `±30%` result. Historical `5.122/4.722 mm` values used a single seed,
`S=100` hard argmin, Anna-G/Anna-domain data, and are not comparable to this
three-seed server-domain `S=400` experiment.

## Metric and reporting conventions

- `single_draw` measures the raw sampler.
- `decoded` is the truth-free practical estimator.
- `oracle` uses truth only to measure posterior coverage and is never used for
  selection.
- Delta X' is reported separately as the paired response to the imposed input
  shift.
- Geodesic failures on disconnected components are never silently converted
  to finite distances; finite branch and pair counts are always reported.
- Temperatures and decoder variants are selected on dev. Held-out is loaded
  only after selections are persisted.
- The final aggregate uses a common finite mask across all three CFM seeds and
  RAP MUSIC. It does not treat the same case under three seeds as three
  independent observations.

## Reproducibility

Primary artifact root:

`/mnt/storage/data/tester/meg_beat_rapmusic_20260819/v1_followup/`

Finalization artifact root:

`/mnt/storage/data/tester/meg_finalize_v1_20260819/`

Prepublication artifact root:

`/mnt/storage/data/tester/meg_prepublication_20260819/`

Matched patch-RAP artifact root:

`/mnt/storage/data/tester/meg_patch_rapmusic_20260819/`

Source-radius misspecification artifact root:

`/mnt/storage/data/tester/meg_source_misspec_20260820/`

Key artifacts:

- `seed_aggregate/summary.json`: V0/V1 seed audit at 100 draws;
- `surface_decoder_seed303/summary.json`: cortical aggregation;
- `greedy_decoder_seed303/summary.json`: mesh refinement;
- `rank_decoder_seed303/summary.json` and
  `rank_spearman_seed303/summary.json`: denoising diagnostics;
- `s_ablation_seed303/summary.json`: prefix draw-count ablation;
- `s_ablation_seed304/` and `s_ablation_seed305/`: confirmatory evaluations;
- `s400_seed_aggregate/summary.json`: final seed-aware comparison;
- `s400_seed_aggregate/per_seed_errors.npz`: per-case inputs to final
  statistics;
- `dg_aggregate/summary.json`: three-seed final `dG` comparison and
  hierarchical intervals;
- `point_ablation_seed{303,304,305}/`: final point-source ablation;
- `refinement_shift_seed{303,304,305}/`: controlled S=400 refinement
  diagnostics;
- `deltax_test/summary.json`: locked-mask Delta X' test;
- `gaussian_patch_baseline/`: three precomputed patch dictionaries, predictions,
  metrics, and timings;
- `refinement_seed{303,304,305}/`: dev-selected pass counts and held-out results;
- `aggregate/summary.json`: final common-mask comparisons.
- `final_run/summary.json`: validated point- and patch-RAP branch metrics;
- `mne_fixed_validation/summary.json`: independent MNE fixed-orientation run;
- `aggregate/summary.json` under the matched patch-RAP root: paired patch/CFM,
  point/MNE, and fixed/free-orientation tests;
- `aggregate/masks_and_errors.npz`: exact masks and per-case errors;
- `sensor_projector.npz`: persisted rank-99 MNE SSP projector.
- `run/patch_dictionary_scale_{070,130}.npz`: perturbed patch dictionaries;
- `run/patch_scale_{070,130}.npz` and `run/cfm_*_scale_{070,130}.npz`:
  per-case misspecification predictions and metrics;
- `aggregate/summary.json`: frozen outcome, hierarchical intervals, and tests;
- `aggregate/masks_and_errors.npz` under the misspecification root: strict
  common mask and exact per-case errors.

The final code is on branch `codex/source-misspec`. No additional computational
experiments are required; the next phase is manuscript consolidation.
