# Bayesian MEG Inversion with Conditional Flow Matching

Amortised posterior inference for the MEG inverse problem: a conditional flow-matching model learns to sample source configurations $\pi(\theta \mid Y)$ from sensor data, and a physics-based decoder turns those samples into a single localisation.

Companion code to ongoing work at Skoltech / HSE / AIRI, following [Bayesian Inverse Problems Meet Flow Matching](https://arxiv.org/abs/2503.01375) (arXiv:2503.01375).

> **Status: scientific closeout complete.** The frozen numerical source of
> truth is [`reports/RESULTS.md`](reports/RESULTS.md). The 2–25 Hz recomputation
> and comparison boundaries are documented in
> [`reports/2026-08-15-filtered-residual-recompute-report.md`](reports/2026-08-15-filtered-residual-recompute-report.md).

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

The final residual path uses the same zero-phase fourth-order 2–25 Hz
Butterworth filter as RAP MUSIC. If the temporal filter is $F$, then
$Y_f=YF=B(AF)+\eta F$, so filtering leaves $\operatorname{col}(B)$ unchanged.
The CFM sampler still receives unfiltered sensor data.

Samples are then combined by Gibbs weighting, $w_s \propto \exp(-r_s^2 / 2\tau^2)$, which keeps argmin's accuracy while staying continuous in $Y$.

## Results

Evaluated on 315 paired simulations with the final Anna-lead-field model.

**Decoder comparison**, `dG=0` original inputs:

| Decoder | MAE euc | MAE geo | Uses ground truth |
|---|---:|---:|:-:|
| Single posterior sample | 5.164 | 9.736 | no |
| Naive posterior mean | 3.822 | 7.517 | no |
| Posterior medoid | 4.129 | 8.237 | no |
| **Residual argmin** | **2.003** | **3.587** | **no** |
| Frozen smooth | 2.067 | 3.655 | no |
| Oracle best-of-100 | 1.314 | 1.685 | yes — upper bound only |

Residual scoring roughly halves the error of the naive estimator without
access to truth. The mean within-sample Spearman association between residual
and true draw error is 0.519.

**Localisation accuracy and shift stability**, at input shift $\Delta X \approx 4.57$ mm:

| Scenario | MAE euc | MAE geo | $\Delta X'$ euc |
|---|---:|---:|---:|
| Anna-G model, `dG=0` | 2.067 | 3.655 | **5.958** |
| Anna-G model, `dG≠0` | 2.199 | 4.022 | **6.153** |

After excluding CFM's own cross-component failures, symmetrically with the RAP
MUSIC implementation, stability is 5.914 mm (`n=305`) and 6.149 mm (`n=301`).
The method-specific RAP subsets give 5.85 and 6.13 mm. Across input branches,
the filtered smooth decoder's geodesic `dG` cost is +0.286 mm versus the
reported RAP change +0.73 mm, approximately 2.55-times lower normalized
sensitivity. The full accuracy–stability frontier and model-by-domain matrix
are in the canonical results report.

Absolute accuracy is not an algorithm-only comparison: the CFM decoder knows
the true Gaussian patch family and radii, while RAP MUSIC fits point dipoles;
fixed/free orientation and noise-window information also differ. A ±30% radius
ablation degrades CFM hard-decoder geodesic MAE from 3.587 to 4.722–5.122 mm.

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
questions. Each method nevertheless excludes its own cross-component failures,
and no common-intersection mask is available. The pipelines also differ in
source-shape prior, orientation model, fixed temporal signatures, and RAP's
empty-window noise estimate. Accordingly, the repository claims lower
within-method sensitivity to `dG`, not absolute statistical superiority over
RAP MUSIC.

## Authors

Daniil Sherki (Skoltech, Sber) · Anna Kubyak (HSE) · Alexei Ossadtchi (HSE, AIRI) · Ivan Oseledets (Skoltech, AIRI) · Ekaterina Muravleva (Skoltech, Sber)

The paired-shift simulation protocol was designed by Anna Kubyak; her simulator and RAP MUSIC baseline are not included here.

## License

MIT — see [LICENSE](LICENSE).
