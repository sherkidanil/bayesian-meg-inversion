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

The filtered geodesic `dG` cost averages +0.286 mm across original and shifted
branches for the smooth decoder, versus the reported +0.73 mm for RAP MUSIC;
normalized sensitivity is about 2.55× lower, consistent with the previous
order-2.4 conclusion. This is an internal difference within each method and is
not confounded by the absolute comparison conditions below.

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
