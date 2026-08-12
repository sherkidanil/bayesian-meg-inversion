# Canonical results: conditional flow matching for MEG localization

Frozen: 2026-08-11. This is the single source of truth for project results.
The dated reports remain an audit trail and are superseded by this document.

## 1. Problem and notation

The simulator generates MEG sensor data

\[
Y = G\Phi(c)^\top A + \eta,
\]

where `G` is the lead field, `c` contains three cortical source centroids,
`Φ(c)` contains their Gaussian surface patches, and `A` contains unknown time
courses. A conditional flow-matching model draws samples from its learned
posterior over centroid coordinates. The central question is how to turn those
draws into one localization without using the true centroids.

The primary complete accuracy metric is Euclidean source-assignment MAE.
Geodesic MAE is also reported, but means are finite-only because the mesh has
disconnected hemispheres. The primary stability metric is paired Euclidean
`ΔX′`, the displacement between estimates from original and shifted sensor
inputs after each estimate is assigned to its own true source ordering. The
mean physical input displacement is `ΔX = 4.5663 mm` for `dG=0` and `4.5567
mm` for `dG≠0`.

Full definitions and derivations are in `docs/math.md` in the article
repository. Figure 1 summarizes the implemented inference path.

![Truth-free CFM and residual-decoder pipeline](../paper/imgs/fig1_pipeline.png)

## 2. The decoder, not the trained weights, was the main accuracy defect

Historical evaluation drew `N` posterior samples and kept the one closest to
ground truth. This `oracle_best_of_N` statistic is an upper bound on sample
quality, not a deployable localizer. Removing it initially made performance
look much worse, but five truth-free decoders showed that good solutions were
already present in the posterior.

For candidate centroids `c`, the residual decoder builds

\[
B(c)=G\Phi(c)^\top,
\qquad
r(c)=\frac{\lVert(I-P_B)Y\rVert_F}{\lVert Y\rVert_F}.
\]

It uses only `Y`, the persisted localization `G`, mesh geometry, and the
dataset-level spatial patch parameters. Ground truth is excluded from the
estimator API. On the final Anna-G model and the 315 `dG=0` original inputs:

| decoder | honest? | MAE euc, mm | MAE geo, mm | interpretation |
|---|:---:|---:|---:|---|
| single draw | yes | 5.1636 | 9.7359 | raw posterior sample |
| posterior mean | yes | 3.8222 | 7.5168 | averages all modes |
| posterior medoid | yes | 4.1285 | 8.2366 | densest draw mode |
| residual argmin | yes | **2.0307** | **3.6298** | best sensor-subspace fit |
| frozen smooth | yes | 2.2727 | 4.3826 | continuous Gibbs decoder |
| oracle best-of-100 | **no** | 1.3143 | 1.6849 | truth-selected upper bound only |

The residual is scientifically informative, not merely a post-hoc score. Its
mean within-sample Spearman association with true Euclidean draw error is
`0.518` on original and `0.539` on shifted `dG=0` inputs. The association stays
positive after `dG` perturbation (`0.507/0.525`). Thus the measured data can
rank better and worse posterior draws without access to truth.

![Decoder comparison](../paper/imgs/fig2_decoders.png)

![Residual versus true draw error](../paper/imgs/fig6_residual_vs_error.png)

The corrected paired IID-100×3 ranking selected `trial81_honest_4965`: its
geodesic MAE was `4.3057 ± 0.0799 mm`, versus `4.7815 ± 0.2607 mm` for
`push_3771` and `5.2502 ± 0.3628 mm` for the original checkpoint. It won all
three paired seeds. This is coherent with the training procedure: the model
selected honestly by validation became first once evaluation was also honest.

## 3. Forward-model domain shift and controlled retraining

The historical server and Anna lead fields are structurally different:

- raw relative Frobenius difference: `36.8%`;
- after optimal global scaling: `35.55%` (`α = 0.907`);
- median projector-aware principal angle on real centroid triplets: `7.84°`;
- 90th percentile of the maximum triplet angle: `60.44°`.

Truth-centroid residual calibration classified the fixed server test as
rebuilt-like in `298/300` cases and Anna's `dG=0` data as Anna-like in
`313/315`. The training dataset did not persist `G`; its rebuilt-like domain is
supported directly by its sensor data and artifact dates, but the exact
historical matrix is unrecoverable. Every new generated dataset now persists
its `G`.

A controlled retraining changed only the forward model. Mesh, latent source
locations, simulation settings, seeds, hyperparameters, and split policy were
held fixed; no sweep was run. The central paired matrix uses the truth-free
residual argmin with 100 draws:

| training model | server-G test, Geo mm | Anna-G test, Geo mm |
|---|---:|---:|
| server-G `honest_4965` | **4.3057 ± 0.0799** | 4.8497 ± 0.2022 |
| Anna-G `honest_4965_annaG` | 4.9463 ± 0.0681 | **4.5131 ± 0.3273** |

The paired geodesic mismatch cost is `+0.5441 mm` from server to Anna and
`+0.4332 mm` in the reverse direction. Under frozen smoothing the corresponding
range is `+0.4037` to `+1.1603 mm`; this wider, directional result is a real
decoder×domain interaction, not solely the use of different temperatures.
The closeout domain-cost interval is therefore **+0.404 to +1.160 mm Geo**
across decoder and direction (Euclidean: `+0.022` to `+0.550 mm`).

![Model by data-domain matrix](../paper/imgs/fig4_model_domain_matrix.png)

The experiment closes the retraining question: retraining is useful for
matching a changed physical operator, but the earlier error was not evidence
that the original weights had failed in their own domain.

## 4. Accuracy and stability next to RAP MUSIC

The final Anna-G model uses a quadratic Gibbs decoder whose temperature was
frozen on an independent 350-pair development split by the preregistered rule
“minimize `ΔX′` subject to `MAE_euc ≤ 1.10 × MAE_euc(argmin)`.” The selected
temperature was `τ = 0.0383041`; no Anna collaboration pair was used for this
choice.

| method / scenario | samples | accuracy | paired `ΔX′` euc, mm | comparability in this table |
|---|---:|---|---:|---|
| CFM Anna-G, `dG=0` | 315 pairs | 2.2727 Euc / 4.3826 Geo | **5.8471** | complete Euc; Geo finite-only |
| RAP MUSIC, reported `dG=0` | 315 pairs | **4.05 “MAE Geo”** | 5.85 | same collaboration simulations; graph-based Geo |
| CFM Anna-G, `dG≠0` | 315 pairs | 2.3959 Euc / 4.3991 Geo | **5.9380** | localization uses unperturbed persisted Anna G |
| RAP MUSIC, reported `dG≠0` | 315 pairs | **4.78 “MAE Geo”** | 6.13 | same collaboration simulations; graph-based Geo |

Anna confirmed that `4.05/4.78` and `5.85/6.13` were computed on the same
collaboration simulations and that the geodesic accuracy uses the real mesh
graph. The results are therefore numerically comparable at the dataset and
metric-family level. The stability values are essentially aligned; the
accuracy values are also of the same order. Exact disconnected-component and
source-matching conventions were not collected and are retained as an accepted
implementation-level limitation rather than an open experimental task. We do
not claim statistical superiority without per-sample RAP MUSIC outputs.

## 5. Accuracy–stability frontier

The development curve spans the measured limits `τ→0` (residual argmin) and
`τ→∞` (posterior mean). Its non-dominated points are:

| decoder point | `τ` | MAE euc, mm | `ΔX′` euc, mm |
|---|---:|---:|---:|
| quadratic `2^-4` | 0.0191521 | **2.0409** | 5.5626 |
| frozen working point, `2^-3` | 0.0383041 | 2.2322 | 5.5624 |
| quadratic `2^-2` | 0.0766083 | 2.5828 | **5.4103** |
| argmin limit | 0 | 2.0674 | 5.7304 |
| posterior-mean limit | ∞ | 3.7490 | 5.6459 |

![Accuracy–stability frontier](../paper/imgs/fig3_accuracy_stability_frontier.png)

The reported RAP point `(4.05, 5.85)` uses geodesic accuracy, whereas the CFM
frontier uses Euclidean accuracy. It is therefore shown for numerical context,
not assigned a formal Pareto status. No new estimator is needed: the achievable
CFM set is measured.

## 6. Stability under `dG` and Monte Carlo sampling

For the final Anna-G model, the smooth paired output distributions are:

| scenario | mean `ΔX`, mm | mean `ΔX′`, mm | median | p90 | maximum |
|---|---:|---:|---:|---:|---:|
| `dG=0` | 4.5663 | **5.8471** | 5.7616 | 9.0169 | 12.8436 |
| `dG≠0` | 4.5567 | **5.9380** | 5.8039 | 9.2150 | 15.2087 |

![Paired output-displacement distributions](../paper/imgs/fig5_deltax_distributions.png)

Across decoder and input branch, `dG` changes geodesic localization MAE by
`+0.017` to `+0.433 mm`, reasonably summarized as order `+0.3 mm`. Its paired
Euclidean `ΔX′` cost is `+0.070` to `+0.091 mm`, order `+0.09 mm`. These
intervals are smaller than the natural server-G/Anna-G mismatch cost.

The Monte Carlo control is a discrete-tail phenomenon: median jitter is zero
at 100, 300, and 1000 draws, while occasional estimates jump by millimetres.
On the first 30 exact pairs, the nonzero-jump fraction fell from `43.3%` to
`16.7%` on `dG=0` and from `33.3%` to `20.0%` on `dG≠0` as draws increased
from 100 to 1000. At 100 draws, restricting to zero-jitter samples reduced
`ΔX′` from `6.70` to `6.43 mm` (`dG=0`) and from `6.64` to `5.83 mm`
(`dG≠0`). Finite sampling inflates a tail, but it does not explain away the
response to the input shift. This `n=30` subset is diagnostic and does not
replace the `n=315` primary estimates.

## 7. Limitations, closed questions, and open collaboration checks

### Closed without further work

- **Oracle contamination:** twelve independent truth-selected evaluator
  locations are catalogued. Key historical values are retained and explicitly
  renamed `oracle_best_of_N_*`; same-corpus honest replacements are used where
  available.
- **Permutation swapping between draws:** non-identity rate `0.014%`; rejected
  as the reason posterior averaging failed.
- **`min_perm_loss`:** no invalid minimum arithmetic. The implementation chooses
  one permutation after batch averaging, so it is batch-global rather than a
  per-sample set loss. No repair or retraining is part of closeout.
- **Original/shifted difficulty asymmetry:** source spacing, lead-field norm and
  normal-angle observability, and the neighbor-correction branch were tested
  and rejected as primary explanations. The fixed head-coordinate shift
  remains a documented protocol limitation that can affect both methods. No
  fourth hypothesis is pursued.
- **Retraining, new decoders, and sweeps:** stopped. The controlled retraining
  and measured frontier answer the questions they were meant to answer.

### Accepted comparison boundary

Anna confirmed that both methods' reported values use the same collaboration
simulations and that RAP MUSIC's geodesic accuracy is evaluated on the real
mesh graph. Exact disconnected-component and source-matching implementation
details were not collected. They remain a documented limitation; no further
external clarification or computation is required for closeout.

## Provenance index

- Phase-1 closeout: `reports/closeout/phase1-summary.md`
- Oracle, `dG`, and `min_perm_loss` audit: `reports/closeout/phase2-debt-audit.md`
- Disk review, no deletions: `reports/closeout/disk-cleanup-candidates.md`
- Controlled Anna-G experiment: `reports/2026-08-11-anna-g-retrain-scientific-report.md`
- Figure generator: `closeout_figures.py`
- Figure manifest with SHA-256 hashes: `paper/imgs/manifest.json`

The public-facing manuscript should cite this file, not extract headline
numbers directly from the dated audit reports.
