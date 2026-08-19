# Findings: what an evaluation audit changed

A record of four things this project got wrong, how each was detected, and what it cost. Written because the detection methods are reusable and the failure modes are not specific to MEG.

The frozen numerical source of truth is [`reports/RESULTS.md`](../reports/RESULTS.md).
This document explains the audit logic; it does not supersede the canonical
tables there.

---

## 1. The evaluation protocol was an oracle

**What happened.** The function that produced the project's headline numbers drew `n_post_samples` from the posterior and kept the one with the lowest error *against ground truth*:

```python
for _ in range(n_post_samples):
    x = sample_fn(y, key)
    m = compute_metrics_one(C_true=c_true, C_pred=project(x), ...)
    if m["mae_geo_mm"] < best_score:
        best_score, best_m = m["mae_geo_mm"], m
```

This is not a localiser. It is an upper bound on sample quality, computable only when the answer is already known.

**Where it spread.** Not only in final evaluation scripts, but in the hyperparameter sweep itself, whose result was written to `trial.set_user_attr("eval_mae_geo_mm", ...)` — the metric by which the best trial was crowned.

**How much it inflated.** The selected trial drew 5 oracle samples. Grouping completed trials by draw count shows the expected monotone advantage:

| Posterior draws | Trials | Best oracle MAE (mm) | Median (mm) |
|---:|---:|---:|---:|
| 1 | 4 | 13.70 | 66.07 |
| 3 | 37 | 5.12 | 8.83 |
| 5 | 61 | 4.60 | 7.86 |

More lottery tickets, better score. The 1-draw group is small and confounded by the sampler's allocation, so this is indicative rather than conclusive — but the mechanism is not in doubt.

**What survived.** The search itself optimised honest validation loss, not the oracle metric. The selected trial ranks 5th of 102 by validation loss, with Spearman correlation 0.543 between the honest and oracle rankings. The oracle inflated the winner's margin; it did not invent the winner.

**Lesson.** Any selection step that touches ground truth contaminates everything downstream of it. Grep for `argmin` over candidates whose score depends on the label.

---

## 2. The defect was in the decoder, not the weights

Removing the oracle made honest error jump to ~9 mm, which looked like a case for retraining. It was not.

**Diagnosis by elimination.** Five decoders were compared on identical posterior draws:

| Decoder | MAE geo (mm) | Mechanism tested |
|---|---:|---|
| Single sample | 11.74 | raw posterior quality |
| Naive mean | 9.09 | averaging |
| Permutation-aligned mean | 9.09 | slot swapping between draws |
| Medoid | 9.31 | consensus among draws |
| Residual-scored | 5.41 | agreement with the data |
| Oracle best-of-100 | 3.00 | what the posterior contains |

The permutation hypothesis — that output slots swap between draws, so averaging smears each estimate toward the centroid of all sources — was refuted directly: non-identity permutations occurred in 0.014% of draws, and the aligned mean is numerically identical to the naive one. Code review explains the small rate: each slot has a fixed time window and frequency band in every sample, so the sources are labelled by their temporal signatures. This does not guarantee permutation stability in a dataset that randomizes those signatures.

The decisive observation was that the posterior mean barely improves with more samples (9.47 at N=25, 9.09 at N=100), while oracle best-of-N reaches 3.00. An unbiased noisy estimator would concentrate as $1/\sqrt{N}$. This one does not, so its centre of mass is simply in the wrong place — yet the distribution covers the truth. That points at selection, not averaging.

**What worked.** Scoring candidates by unexplained sensor energy (see `docs/math.md` §5.5). Within-sample Spearman correlation between residual and true error is ≈ 0.51, which is enough for selection over 100 candidates to recover most of the oracle gap.

Matching the RAP MUSIC 2–25 Hz Butterworth preprocessing is subspace-safe:
$YF=B(AF)+\eta F$, so filtering time does not change $\operatorname{col}(B)$.
Refreezing the temperature after filtering improved the Anna-G working-point
accuracy from 2.273/4.383 to 2.067/3.655 mm Euc/Geo on `dG=0` original inputs.
Spearman itself increased by only 0.001–0.002, so the filter did not reveal a
previously hidden ranking signal; most of the working-point gain is the colder
dev-selected temperature.

**Lesson.** For a posterior sampler, the point estimator is a modelling decision of the same weight as the architecture. Report which one you used.

---

## 3. Two labs, two forward models

**Detection.** Comparing lead-field matrices elementwise is the wrong test: localisation depends on $G$ only through $\operatorname{col}(G\Phi(c)^{\top})$, which is invariant under $B \mapsto BD$ for invertible $D$. The right measure is principal angles between those subspaces, evaluated on real centroid triplets.

| Comparison | Value |
|---|---:|
| Raw relative Frobenius difference | 0.368 |
| After optimal global rescaling | 0.356 |
| Median principal angle | 7.84° |
| P90 of per-case maximum angle | 60.44° |
| Deliberate `dG` perturbation, same metric | 0.068 |

The unintentional mismatch is 5.4× larger than the perturbation whose effect the study was measuring. Its exact cause cannot be attributed because the historical generation environment and training lead field were not persisted; current rebuilds record MNE, nibabel, and sample-data versions.

**Attribution.** Which dataset belongs to which lead field is decided by the same residual, evaluated at *known* centroids — a diagnostic, not an estimator. Comparison is made within a dataset, since the absolute noise floor depends on the simulation's noise level:

| Dataset | Median residual, rebuilt $G$ | Median residual, collaborator $G$ | Verdict |
|---|---:|---:|---|
| Fixed test | 0.2256 | 0.2791 | rebuilt |
| Collaborator paired | 0.2772 | 0.2232 | collaborator |

Symmetric and unambiguous — the correct lead field wins in ≥ 99% of samples in both directions.

**Independent confirmation.** Two checkpoints separated by a clear margin on in-domain data (4.31 vs 4.78 mm across three paired seeds) become indistinguishable on cross-domain data (4.553 vs 4.547). The mismatch dominates the difference between models — evidence that does not overlap with the subspace-angle argument.

**Lesson.** When two groups each rebuild a physical operator from "the same" source data, verify it. Compare the quantity the method actually consumes, not the raw matrix.

---

## 4. The lead field was never saved

Training and test datasets stored sources, sensors, mesh and parameters — but not $G$. The historical matrix is therefore unrecoverable, and the conclusion that training used a rebuilt-like lead field rests on indirect evidence: the fixed test set classifies as rebuilt-like, and it was generated five days after the training set, presumably in one environment.

Sensor-level evidence supports it — the training set itself classifies as rebuilt-like in 99.67% of sampled rows — but the exact matrix is gone.

Every dataset generated since persists $G$ alongside the data.

**Lesson.** Persist the operator, not just its outputs. Regeneration is not reproduction.

---

## 5. Controlled retraining separated decoder error from domain error

Once the decoder was fixed, retraining was repeated as a controlled experiment:
the mesh, latent sources, simulator settings, seeds, hyperparameters, and split
policy were held fixed, and only the persisted lead field changed. No sweep was
run.

| Training domain | Server-G test, Geo mm | Anna-G test, Geo mm |
|---|---:|---:|
| Server-G model | **4.306 ± 0.080** | 4.850 ± 0.202 |
| Anna-G model | 4.946 ± 0.068 | **4.513 ± 0.327** |

The hard-decoder domain cost is +0.544 mm in one direction and +0.433 mm in the
other. With independently frozen smooth decoders, the full sensitivity range
is +0.404 to +1.160 mm. Retraining therefore removes part of a physical domain
mismatch; it does not retroactively prove that the original in-domain weights
were undertrained.

On all 315 collaboration pairs, the filtered Anna-G smooth decoder gives
$\Delta X'=5.958$ mm at `dG=0` and 6.153 mm at `dG≠0`. Excluding its own
cross-component failures, symmetrically with the RAP MUSIC code, gives 5.914
(`n=305`) and 6.149 mm (`n=301`). These remain numerically aligned with RAP
MUSIC 5.85 and 6.13 mm, but the subsets are method-specific. A common-mask
comparison is unavailable, so no superiority claim is made.

The earlier filtered geodesic `dG` costs, +0.286 mm for CFM and +0.73 mm for
RAP MUSIC, came from different corpora and cannot define a sensitivity ratio.
The final same-corpus result is given in §6.

The absolute accuracy comparison has four disclosed asymmetries: the residual
decoder knows the true Gaussian patch radii and profile while RAP MUSIC fits
point dipoles; CFM uses fixed cortical-normal orientation while RAP uses free
orientation; temporal source signatures are fixed across samples; and RAP
estimates noise from the known empty 1.7–1.8 s window. A ±30% radius ablation
degraded CFM hard-decoder geodesic MAE from 3.587 to 4.722–5.122 mm, confirming
that patch-shape knowledge is material.

**Lesson.** A controlled retraining result is a model-by-domain matrix, not one
improved number. The diagonal measures in-domain quality; the off-diagonal
measures mismatch in both directions.

---

## Method notes carried forward

- **Selection sets and reporting sets are disjoint.** The decoder temperature was frozen on an independent development split, with the selection rule fixed before the Pareto front was inspected, and its choice recorded by hash.
- **Truth-free estimators are tested for leakage.** Unit tests fail if ground truth reaches an estimator's API.
- **Paired comparisons use paired statistics.** An earlier apparent ranking change dissolved when per-seed differences were examined: the challenger won the aggregate by 0.100 mm while losing two of three seeds. A later reranking, decisive on all three seeds, was accepted.
- **Provenance travels with results.** Each run records input hashes, git HEAD, and whether the repository changed mid-run.
- **Stopping rules are part of rigor.** The accuracy--stability frontier is measured, five decoders are enough, three asymmetry hypotheses were rejected, and no further estimator, sweep, or post-hoc explanation is part of this closeout.

## 6. Temporal attention beats standard point RAP, not the equal-prior baseline

The matched RAP MUSIC baseline removes the forward-domain mismatch: its fixed
simulation matrix is byte-for-byte equal to the persisted server matrix, while
its free-orientation inverse operator comes from the same forward solution.
Replacing the old CFM configuration by temporal-attention V1, a cortical
medoid, 400 prefix-stable posterior draws, and dev-selected mesh refinement
produces a clear advantage over standard free-orientation point RAP.

Across three training seeds, CFM gives `1.429 ± 0.205 mm` Euclidean MAE,
`2.618 ± 0.288 mm` geodesic MAE, and `5.543 ± 0.073 mm` Delta X', versus RAP
MUSIC at `1.993`, `3.836`, and `5.982 mm`. On the common finite set (`n=290`),
the hierarchical seed-and-case interval for CFM−RAP geodesic error is
`[-1.905,-0.517] mm` and paired `p=0.000385`. All three seeds favour CFM.
For Delta X', however, Wilcoxon gives `p=0.101`; its numerical improvement is
not an established second axis of dominance.

This comparison is real but not source-model matched. Section 10 shows that it
reverses when RAP is given the same Gaussian-patch dictionary.

**Lesson.** A strong truth-free decoder can reveal a sampler whose posterior
already covers good solutions, and temporal attention improves single-draw,
decoded, oracle, and failure-tail metrics together. Neither fact identifies
the source of an advantage over a classical method until prior information is
matched.

## 7. Three negative or corrective results constrain the next step

1. **Signal-subspace vertex features did not converge.** Replacing the raw
   temporal path by accumulated subspace maps failed across two encoders and
   three learning rates. This is an optimization failure, not evidence against
   physics-informed features in general.
2. **Covariance concatenation V5 worsened every level.** Single-draw, decoded,
   and oracle Geo all degraded (`11.778`, `5.974`, `2.824 mm`). Together with
   the first result, this says compressed spatial summaries displace the fixed
   temporal signatures that distinguish sources in this dataset.
3. **Residual refinement is locally, not globally, misaligned.** The old
   `3.295` versus `2.691 mm` comparison mixed S=100 hard argmin with S=400
   medoid. Controlled two-pass refinement from the final medoid instead improves
   all three seeds to `2.428 ± 0.173 mm`, while Delta X' changes only
   `5.542→5.561 mm`. Yet in `25.0–32.5%` of moved finite branches the residual
   decreases while truth error rises. The criterion has local false directions,
   but the stronger claim of a globally shifted optimum is unsupported.

The fixed two-pass result remains exploratory. An independently frozen dev rule
selects `0/0/1` passes for seeds `303/304/305` and yields the final
`2.618 ± 0.288 mm` headline. The conservative rule rejects the larger
two-pass gain for two seeds because it worsens Delta X'.

## 8. Source-model ablation remains decisive in the S=400 base decoder

With V1, medoid, S=400, dev-only temperatures, and no optional refinement,
replacing Gaussian-patch residuals by point-source residuals degrades Geo to
`6.350 ± 0.555 mm` over three seeds, from `2.691 ± 0.179 mm`. The decoder is
therefore strongly
sensitive to misspecifying extended sources as points. MNE has no standard
mirror switch for Gaussian patches, but the asymmetry is removed by the
independently validated scalar RAP implementation in section 10.

## 9. The final dG estimate preserves direction but not significance

The historical mask and RAP cost reproduce exactly (`n=297/292`, `+0.9473156
mm`). Under the same four-sided rule and the stricter intersection of RAP and
all final CFM seeds, CFM's geodesic `dG` cost is `0.713 ± 0.146 mm` versus
RAP's `0.930 mm`, a point ratio of `1.30`. The hierarchical interval for the
RAP−CFM difference is `[-0.359,0.804] mm`, so the robustness difference is not
resolved. Delta X' costs are effectively equal: `0.271 ± 0.125` versus
`0.260 mm`.

The historical perturbation's global Frobenius change is about `6.6%`, so the
natural `36.8%` mismatch is roughly `5.4x` larger, not two orders of magnitude
larger.

## 10. Equalizing the Gaussian source class reverses the accuracy claim

The first attempt was a three-stage greedy minimization of the raw residual.
It gave `7.880 mm` finite-only Geo and many failures. That result diagnoses a
weak optimizer; it is not RAP MUSIC and says nothing decisive about the value
of network proposals.

The actual mirror baseline implements scalar RAP MUSIC. At each recursive
stage it maximizes projected signal-subspace correlation while retaining the
full rank-three signal subspace. Applying the same MNE SSP projector to data
and dictionary is essential. With a point, fixed-orientation dictionary, the
implementation matches MNE's ordered predictions on `700/700` branches. The
same validation measures the orientation effect: fixed-minus-free Geo is
`+7.416 mm` on `n=277`, interval `[6.479,8.372]`, `p=1.24e-37`.

Only then is the point dictionary replaced by three Gaussian-patch dictionaries
for `(10,8,10) mm`. On the common patch-RAP/CFM mask (`n=288`), patch-RAP is
`1.118 mm` Geo and final CFM is `2.646 mm`. Patch-RAP minus CFM is `-1.527 mm`,
hierarchical interval `[-2.143,-0.901]`, one-sided Wilcoxon `p=1.85e-25`; all
three training seeds lose separately. Patch-RAP also beats standard free-point
MNE by `-2.583 mm`, interval `[-2.987,-2.204]`.

**Lesson.** The pre-registered outcome is
`source_shape_explains_cfm_advantage`. Once source-shape knowledge is equalized,
the classical exhaustive subspace method is substantially more accurate than
CFM. The network's remaining case is computational amortization, not superior
localization accuracy on this benchmark.
