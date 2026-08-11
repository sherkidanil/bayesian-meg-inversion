# Bayesian MEG Inversion with Conditional Flow Matching

Amortised posterior inference for the MEG inverse problem: a conditional flow-matching model learns to sample source configurations $\pi(\theta \mid Y)$ from sensor data, and a physics-based decoder turns those samples into a single localisation.

Companion code to ongoing work at Skoltech / HSE / AIRI, following [Bayesian Inverse Problems Meet Flow Matching](https://arxiv.org/abs/2503.01375) (arXiv:2503.01375).

> **Status: scientific closeout complete.** The frozen numerical source of
> truth is [`reports/RESULTS.md`](reports/RESULTS.md). Comparison with RAP MUSIC
> remains conditional on three cross-lab definition checks listed below.

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

Evaluated on 315 paired simulations with the final Anna-lead-field model.

**Decoder comparison**, `dG=0` original inputs:

| Decoder | MAE euc | MAE geo | Uses ground truth |
|---|---:|---:|:-:|
| Single posterior sample | 5.164 | 9.736 | no |
| Naive posterior mean | 3.822 | 7.517 | no |
| Posterior medoid | 4.129 | 8.237 | no |
| **Residual argmin** | **2.031** | **3.630** | **no** |
| Frozen smooth | 2.273 | 4.383 | no |
| Oracle best-of-100 | 1.314 | 1.685 | yes — upper bound only |

Residual scoring roughly halves the error of the naive estimator without
access to truth. The mean within-sample Spearman association between residual
and true draw error is 0.518.

**Localisation accuracy and shift stability**, at input shift $\Delta X \approx 4.57$ mm:

| Scenario | MAE euc | MAE geo | $\Delta X'$ euc |
|---|---:|---:|---:|
| Anna-G model, `dG=0` | 2.273 | 4.383 | **5.847** |
| Anna-G model, `dG≠0` | 2.396 | 4.399 | **5.938** |

Across decoder and input branch, `dG` costs +0.017 to +0.433 mm in geodesic
accuracy and +0.070 to +0.091 mm in paired stability. The full
accuracy–stability frontier and model-by-domain matrix are in the canonical
results report.

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

## Open questions

Only cross-lab definitions remain open:

1. Did the reported RAP MUSIC accuracy values 4.05 and 4.78 mm use these same
   315 pairs or an earlier 300-sample corpus?
2. What graph and disconnected-component policy defines RAP `MAE Geo`?
3. How were the three sources matched for RAP $\Delta X'$?

The project does not open new estimators, retraining runs, sweeps, or post-hoc
asymmetry hypotheses during closeout.

## Authors

Daniil Sherki (Skoltech, Sber) · Anna Kubyak (HSE) · Alexei Ossadtchi (HSE, AIRI) · Ivan Oseledets (Skoltech, AIRI) · Ekaterina Muravleva (Skoltech, Sber)

The paired-shift simulation protocol was designed by Anna Kubyak; her simulator and RAP MUSIC baseline are not included here.

## License

MIT — see [LICENSE](LICENSE).
