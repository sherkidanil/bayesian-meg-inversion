# Bayesian MEG Inversion with Conditional Flow Matching

Amortised posterior inference for the MEG inverse problem: a conditional flow-matching model learns to sample source configurations $\pi(\theta \mid Y)$ from sensor data, and a physics-based decoder turns those samples into a single localisation.

Companion code to ongoing work at Skoltech / HSE / AIRI, following [Bayesian Inverse Problems Meet Flow Matching](https://arxiv.org/abs/2503.01375) (arXiv:2503.01375).

> **Status: research in progress.** Numbers below are current best estimates under an explicitly documented evaluation protocol. Several comparisons are still blocked on cross-lab metric alignment — see [Open questions](#open-questions).

---

## The problem

Given magnetometer recordings $Y \in \mathbb{R}^{102 \times 1800}$, recover the locations of $K = 3$ extended cortical sources. The forward model is

$$Y = G\,\Phi(c)^{\top} A + \eta$$

where $G$ is the lead field, $\Phi(c)$ are Gaussian patches around centroid vertices $c$, and $A$ are unknown time courses. The inverse problem is ill-posed: 102 measurements against 4699 candidate vertices.

Full derivations are in [`docs/math.md`](docs/math.md).

## The approach

**Sampling.** A conditional flow-matching model is trained on simulations to sample from the posterior. Inference integrates an ODE from noise, so each random seed yields a different plausible source configuration.

**Decoding.** The interesting part. A distribution of 100 samples must collapse to one answer, and the choice of decoder dominates accuracy — more than the trained weights do. The decoder scores each candidate $c$ by how much sensor energy its patches fail to explain:

$$r(c) = \frac{\|(I - P_B)\,Y\|_F}{\|Y\|_F}, \qquad B = G\,\Phi(c)^{\top}, \qquad P_B = B(B^{\top}B)^{-1}B^{\top}$$

Since $Y = BA$ for unknown $A$, every time sample must lie in the $K$-dimensional column space of $B$. Minimising over all time courses is analytic, so $A$ is never computed. This is subspace fitting — the same geometry MUSIC exploits — evaluated on network proposals instead of a grid scan. It uses only $Y$, $G$ and the patch model; no ground truth.

Samples are then combined by Gibbs weighting, $w_s \propto \exp(-r_s^2 / 2\tau^2)$, which keeps argmin's accuracy while staying continuous in $Y$.

## Results

Evaluated on 315 paired simulations (original and shifted source configurations sharing identical time courses and noise).

**Decoder comparison**, fixed test set, geodesic MAE:

| Decoder | MAE (mm) | Uses ground truth |
|---|---:|:-:|
| Single posterior sample | 11.74 | no |
| Naive posterior mean | 9.09 | no |
| Permutation-aligned mean | 9.09 | no |
| Posterior medoid | 9.31 | no |
| **Residual-scored** | **5.41** | **no** |
| Oracle best-of-100 | 3.00 | yes — upper bound only |

Residual scoring roughly halves the error of the naive estimator without retraining anything.

**Localisation accuracy and shift stability**, at input shift $\Delta X \approx 4.57$ mm:

| Scenario | MAE euc | MAE geo | $\Delta X'$ euc | $\Delta X'/\Delta X$ |
|---|---:|---:|---:|---:|
| Matched lead field | 2.53 | 4.55 | 6.37 | 1.40 |
| Perturbed lead field | 2.73 | 4.72 | 6.53 | 1.43 |

A lead-field perturbation of the magnitude studied here (6.8% relative Frobenius norm) costs +0.20 mm accuracy and +0.16 mm stability.

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
reports/           dated result reports with provenance hashes
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

## Open questions

1. Cross-lab alignment of the geodesic metric — specifically the handling of infinite distances between disconnected hemispheres, which drives the gap between 2.53 mm Euclidean and 4.55 mm geodesic.
2. A single shared lead-field file, so the two methods are compared under one forward model.
3. Separating the decoder's own Monte Carlo jitter from genuine shift sensitivity in $\Delta X'$.
4. Shifted configurations localise systematically better than original ones across every decoder tested. Source separation, sensor visibility, and the minimum-distance correction branch have each been ruled out.

## Authors

Daniil Sherki (Skoltech, Sber) · Anna Kubyak (HSE) · Alexei Ossadtchi (HSE, AIRI) · Ivan Oseledets (Skoltech, AIRI) · Ekaterina Muravleva (Skoltech, Sber)

The paired-shift simulation protocol was designed by Anna Kubyak; her simulator and RAP MUSIC baseline are not included here.

## License

MIT — see [LICENSE](LICENSE).
