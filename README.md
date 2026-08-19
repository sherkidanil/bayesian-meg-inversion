# Bayesian MEG Inversion with Conditional Flow Matching

Amortised posterior inference for the MEG inverse problem: a conditional flow-matching model learns to sample source configurations $\pi(\theta \mid Y)$ from sensor data, and a physics-based decoder turns those samples into a single localisation.

Companion code to ongoing work at Skoltech / HSE / AIRI, following [Bayesian Inverse Problems Meet Flow Matching](https://arxiv.org/abs/2503.01375) (arXiv:2503.01375).

> **Status: source-matched validation complete.** The frozen numerical source of
> truth is [`reports/RESULTS.md`](reports/RESULTS.md). The 2–25 Hz recomputation
> and final comparison boundaries are documented in
> [`reports/METHOD.md`](reports/METHOD.md).

---

## The problem

Given magnetometer recordings $Y \in \mathbb{R}^{102 \times 1800}$, recover the locations of $K = 3$ extended cortical sources. The forward model is

$$Y = G\,\Phi(c)^{\top} A + \eta$$

where $G$ is the lead field, $\Phi(c)$ are Gaussian patches around centroid vertices $c$, and $A$ are unknown time courses. The inverse problem is ill-posed: 102 measurements against 4699 candidate vertices.

Full derivations are in [`docs/math.md`](docs/math.md).

## The approach

**Sampling.** A conditional flow-matching model is trained on simulations to sample from the posterior. Inference integrates an ODE from noise, so each random seed yields a different plausible source configuration.

**Decoding.** A distribution of 400 samples must collapse to one answer. The
decoder scores each candidate $c$ by how much sensor energy its patches fail to
explain:

$$r(c) = \frac{\|(I - P_B)\,Y\|_F}{\|Y\|_F}, \qquad B = G\,\Phi(c)^{\top}, \qquad P_B = B(B^{\top}B)^{-1}B^{\top}$$

Since $Y = BA$ for unknown $A$, every time sample must lie in the $K$-dimensional column space of $B$. Minimising over all time courses is analytic, so $A$ is never computed. This is subspace fitting — the same geometry MUSIC exploits — evaluated on network proposals instead of a grid scan. It uses only $Y$, $G$ and the patch model; no ground truth.

The final residual path uses the same zero-phase fourth-order 2–25 Hz
Butterworth filter as RAP MUSIC. If the temporal filter is $F$, then
$Y_f=YF=B(AF)+\eta F$, so filtering leaves $\operatorname{col}(B)$ unchanged.
The CFM sampler still receives unfiltered sensor data.

Quadratic Gibbs weights, $w_s \propto \exp(-r_s^2 / 2\tau^2)$, are converted
to a cortical geodesic medoid for every source slot. This keeps the estimate on
the cortical surface and avoids averaging between opposite banks of a sulcus.
Each training seed freezes its own $\tau$ and optional mesh-refinement pass
count on dev before held-out is opened.

## Results

On the matched server-lead-field benchmark, temporal-attention CFM with the
cortical medoid, `S=400`, and dev-selected mesh refinement is more accurate
than standard free-orientation point-dipole MNE RAP MUSIC:

| Method | MAE euc | MAE geo | $\Delta X'$ euc |
|---|---:|---:|---:|
| RAP MUSIC | 1.993 | 3.836 | 5.982 |
| **CFM, 3 seeds** | **1.429 ± 0.205** | **2.618 ± 0.288** | 5.543 ± 0.073 |

All three independently trained CFM seeds favour CFM in geodesic accuracy. On
the common finite mask (`n=290`), the seed-and-case mean geodesic difference
CFM−RAP is `-1.202 mm`; a hierarchical bootstrap gives
`[-1.905,-0.517] mm`, and the one-sided paired Wilcoxon test gives
`p=0.000385`. Delta X' is numerically lower by `0.432 mm`, but its one-sided
Wilcoxon result is not significant (`p=0.101`). This comparison is not matched
for source-shape information.

A source-matched classical baseline changes the conclusion. We implemented
scalar RAP MUSIC over an explicit candidate dictionary and first validated it
against MNE with point topographies and fixed orientation: ordered predictions
match on all `700/700` branches after applying the same rank-99 SSP projector.
Replacing only the point dictionary by the simulated Gaussian-patch
dictionaries gives `1.118 mm` geodesic MAE, versus `2.646 mm` for CFM on the
common finite mask (`n=288`). Patch-RAP minus CFM is `-1.527 mm`, with a
hierarchical 95% interval `[-2.143,-0.901] mm` and one-sided Wilcoxon
`p=1.85e-25`. All three CFM seeds lose separately. The pre-registered outcome
is therefore that source-shape knowledge explains and reverses the apparent
CFM accuracy advantage.

The earlier raw-residual greedy patch scan (`7.880 mm`) remains useful only as
a negative optimizer result. It is not RAP MUSIC and does not establish that
network proposals add localization information.

The zero-refinement `S=400` same-corpus lead-field perturbation result is more
cautious. On the
four-sided intersection of RAP MUSIC and all three CFM seeds, CFM's Geo cost is
`0.713 ± 0.146 mm` versus RAP's `0.930 mm`; the point ratio is `1.30`, but the
interval for RAP−CFM, `[-0.359,0.804] mm`, crosses zero. Delta-X costs are
effectively equal (`0.271 ± 0.125` versus `0.260 mm`).

Orientation is material: on the common finite mask, fixed-orientation point RAP
is `+7.416 mm` worse than free-orientation MNE. Replacing the base CFM residual
basis by point sources degrades Geo to `6.350 ± 0.555 mm`; the validated
Gaussian-patch RAP run is the symmetric source-family control.

## What this repository also documents

An audit that changed the project's conclusions, written up in [`docs/findings.md`](docs/findings.md):

- **The historical evaluation protocol was an oracle.** It drew several posterior samples and kept the one closest to ground truth. Every headline number produced before this audit is an upper bound, not a localiser score.
- **The defect was in the decoder, not the weights.** The trained model was fine; the way answers were extracted from it was not.
- **Two labs were unknowingly using different forward models.** Rebuilding the lead field from the same MNE source data produced a matrix differing by 36.8% in relative norm — 5.4× larger than the perturbation whose effect was being measured. Detected via principal angles between the subspaces $\operatorname{col}(G\Phi(c)^{\top})$ that localisation actually depends on.
- **Training datasets did not persist `G`.** The exact historical lead field is unrecoverable. Every generated dataset now stores it.

## Layout

```
src/meg_cfm/       reusable modules: metrics, simulator, decoders, provenance
experiments/       reproducible scripts, one per audit or measurement
tests/             unit tests, including leakage guards on truth-free estimators
docs/              mathematical appendix and findings write-up
reports/           canonical RESULTS.md plus dated provenance logs
paper/             manuscript source (Typst, draft)
```

## Install

```bash
pip install -e .
pytest
```

Requires Python 3.11+, JAX with CUDA for training, MNE-Python for lead-field construction.

## Reproducing

Large artefacts — datasets, checkpoints, Optuna databases — are deliberately not tracked. Scripts take explicit paths:

```bash
python experiments/estimator_audit.py --model_dir <ckpt> --dataset_npz <data> --out_dir reports/audit
python experiments/leadfield_provenance.py --datasets <a.npz> <b.npz> --out_dir reports/provenance
python experiments/shift_stability_paired.py --dataset_npz <paired.npz> --estimator smooth --out_dir reports/shift
```

Every run writes `summary.json` with input SHA-256 hashes, git HEAD, and a flag recording whether the repository changed mid-run.

## Comparison boundary

Code review closed the graph, source-matching, dataset, and metric-definition
questions, and final claims use explicit common finite masks. Standard MNE RAP
and CFM remain a full-pipeline comparison, while patch-RAP equalizes the
simulated source family and reverses the accuracy result. Both equal-prior
methods receive simulation-known radii. Results are from one synthetic subject
with fixed temporal signatures and therefore are not universal claims about
algorithm classes.

## Authors

Daniil Sherki (Skoltech, Sber) · Anna Kubyak (HSE) · Alexei Ossadtchi (HSE, AIRI) · Ivan Oseledets (Skoltech, AIRI) · Ekaterina Muravleva (Skoltech, Sber)

The paired-shift simulation protocol was designed by Anna Kubyak; her simulator and RAP MUSIC baseline are not included here.

## License

MIT — see [LICENSE](LICENSE).
