# MEG Inversion with Conditional Flow Matching

Amortised candidate generation for the MEG inverse problem: a conditional
flow-matching model generates source configurations from sensor data, and a
physics-based decoder turns the candidate pool into a single localisation. The
generated pool is not assumed to be a calibrated posterior distribution.

Companion code to ongoing work at Skoltech / HSE / AIRI, following [Bayesian Inverse Problems Meet Flow Matching](https://arxiv.org/abs/2503.01375) (arXiv:2503.01375).

> **Status: prepublication validation complete.** The frozen numerical source of
> truth is [`reports/RESULTS.md`](reports/RESULTS.md). The 2–25 Hz recomputation
> and final comparison boundaries are documented in
> [`reports/METHOD.md`](reports/METHOD.md).

---

## Start here

- [`reports/EXECUTIVE_SUMMARY.md`](reports/EXECUTIVE_SUMMARY.md) gives the short scientific conclusion.
- [`reports/METHOD.md`](reports/METHOD.md) defines the forward model, decoder, metrics, and comparison boundary.
- [`reports/RESULTS.md`](reports/RESULTS.md) is the frozen numerical source of truth.
- [`docs/math.md`](docs/math.md) contains the mathematical derivations.
- [`docs/testing.md`](docs/testing.md) documents the test environment and GPU precision policy.

The repository contains source code, tests, manuscript sources, and compact
result summaries. Large datasets, checkpoints, and generated run directories
are excluded intentionally.

## The problem

Given magnetometer recordings $Y \in \mathbb{R}^{102 \times 1800}$, recover the locations of $K = 3$ extended cortical sources. The forward model is

$$Y = G\,\Phi(c)^{\top} A + \eta$$

where $G$ is the lead field, $\Phi(c)$ are Gaussian patches around centroid vertices $c$, and $A$ are unknown time courses. The inverse problem is ill-posed: 102 measurements against 4699 candidate vertices.

Full derivations are in [`docs/math.md`](docs/math.md).

## The approach

**Sampling.** A conditional flow-matching model is trained on simulations to
generate conditional source configurations. Inference integrates an ODE from
noise, so each random seed yields a different candidate configuration.

**Decoding.** A distribution of 400 samples must collapse to one answer. The
decoder scores each candidate $c$ by how much sensor energy its patches fail to
explain:

$$r(c) = \frac{\|(I - P_B)\,Y\|_F}{\|Y\|_F}, \qquad B = G\,\Phi(c)^{\top}, \qquad P_B = B(B^{\top}B)^{-1}B^{\top}$$

Since $Y = BA$ for unknown $A$, every time sample must lie in the $K$-dimensional column space of $B$. Minimising over all time courses is analytic, so $A$ is never computed. This is subspace fitting — the same geometry MUSIC exploits — evaluated on network proposals instead of a grid scan. It uses only $Y$, $G$ and the patch model; no ground truth.

The final residual path uses the same zero-phase fourth-order 2–25 Hz
Butterworth filter as RAP MUSIC. If the temporal filter is $F$, then
$Y_f=YF=B(AF)+\eta F$, so filtering leaves $\mathrm{col}(B)$ unchanged.
The CFM sampler still receives unfiltered sensor data.

Quadratic Gibbs weights, $w_s \propto \exp(-r_s^2 / 2\tau^2)$, are converted
to a cortical geodesic medoid for every source slot. This keeps the estimate on
the cortical surface and avoids averaging between opposite banks of a sulcus.
Each training seed freezes its own $\tau$ and optional mesh-refinement pass
count on dev before held-out is opened.

## Results

The main practical result is about model specification, not a universal winner:
changing the classical dictionary from point topographies to matched Gaussian
patches moves geodesic MAE from about `11.25` to `1.118 mm`. That order-of-magnitude
change is much larger than the remaining CFM-versus-classical gap.

On the matched server-lead-field benchmark, temporal-attention CFM with the
cortical medoid, `S=400`, and dev-selected mesh refinement is more accurate
than the matched RAP MUSIC pipeline:

| Method | MAE euc | MAE geo | $\Delta X'$ euc |
|---|---:|---:|---:|
| RAP MUSIC | 1.993 | 3.836 | 5.982 |
| **CFM, 3 seeds** | **1.429 ± 0.205** | **2.618 ± 0.288** | 5.543 ± 0.073 |

All three independently trained CFM seeds favour CFM in geodesic accuracy. On
the common finite mask (`n=290`), the seed-and-case mean geodesic difference
CFM−RAP is `-1.202 mm`; a hierarchical bootstrap gives
`[-1.905,-0.517] mm`, and the one-sided paired Wilcoxon test gives
`p=0.000385`. Delta X' is numerically lower by `0.432 mm`, but its one-sided
Wilcoxon result is not significant (`p=0.101`), so the final claim is accuracy
and speed rather than three-axis Pareto dominance.

A validated scalar RAP implementation was first checked against MNE on all
`700/700` point-source branches, then rerun with matched Gaussian-patch
dictionaries. On the locked common finite mask, matched patch-RAP gives
`1.118 mm` geodesic MAE versus `2.646 mm` for CFM. Thus the earlier CFM advantage
over standard point/free-orientation RAP MUSIC does not survive source-model
matching: specification of the source family dominates algorithm selection.

Under a frozen ±30% radius-misspecification test, CFM degrades less in percentage
terms at both perturbations, but absolute accuracy is mixed. At `0.7×`, CFM is
better (`5.845` versus `7.135 mm`); at nominal radii and `1.3×`, patch-RAP is
better (`1.125` versus `2.716 mm`, and `3.018` versus `4.985 mm`). Equal weighting
of the three regimes gives `3.759 mm` for patch-RAP and `4.515 mm` for CFM.

The zero-refinement `S=400` same-corpus lead-field perturbation result is more
cautious. On the
four-sided intersection of RAP MUSIC and all three CFM seeds, CFM's Geo cost is
`0.713 ± 0.146 mm` versus RAP's `0.930 mm`; the point ratio is `1.30`, but the
interval for RAP−CFM, `[-0.359,0.804] mm`, crosses zero. Delta-X costs are
effectively equal (`0.271 ± 0.125` versus `0.260 mm`).

The standard MNE comparison remains a full-pipeline comparison: CFM uses fixed
cortical-normal extended sources, while MNE RAP MUSIC fits point dipoles with
free orientation. The matched scalar patch-RAP control isolates the source
dictionary on the shared fixed-orientation forward model.

### Fixed-K Transformer extension

A separate `K=3`, `S=800` experiment compares MLP and Transformer estimators on
an identical set of records without disconnected, non-finite, or capped pairs.
The metric is the ordinary uncapped geodesic MAE; coverage is reported
separately.

| Method | Common-valid MAE geo | Individual coverage |
|---|---:|---:|
| MLP, Gibbs medoid | 1.164 mm | 80.57% |
| Transformer, Gibbs medoid | **0.812 mm** | 79.71% |
| Transformer, max-ρ | 0.661 mm | 92.57% |
| Fine-tuned Transformer, max-ρ | **0.620 mm** | **93.71%** |

The architecture-only comparison uses 244 common records; the four-pipeline
comparison uses 242. This extension has its own frozen inputs, manifests,
recalculation script, and interpretation in
[`reports/transformer-common-valid/`](reports/transformer-common-valid/README.md).
It is a separate protocol from the CFM-versus-RAP comparison above.

### Variable-cardinality extension

The [`megflow/`](megflow/README.md) package contains the current PyTorch
implementation for an unknown number of cortical sources. It supports
`K=1,...,10`, predicts coordinates and radii, and includes two ways to infer
cardinality: a count head and null slots. The true `K` is not supplied at
inference. The package contains dataset generation, training, candidate-bank
and single-draw evaluation, geodesic metrics, configurations, and tests.

Fixed-`K=3` MLP and Transformer comparisons use a separate protocol, described
in [`reports/transformer-common-valid/`](reports/transformer-common-valid/README.md).
Datasets and model checkpoints are intentionally excluded from Git; see the
package README for commands and required inputs.

## What this repository also documents

An audit that changed the project's conclusions, written up in [`docs/findings.md`](docs/findings.md):

- **The historical evaluation protocol was an oracle.** It drew several posterior samples and kept the one closest to ground truth. Every headline number produced before this audit is an upper bound, not a localiser score.
- **The defect was in the decoder, not the weights.** The trained model was fine; the way answers were extracted from it was not.
- **Two labs were unknowingly using different forward models.** Rebuilding the lead field from the same MNE source data produced a matrix differing by 36.8% in relative norm — 5.4× larger than the perturbation whose effect was being measured. Detected via principal angles between the subspaces $\mathrm{col}(G\Phi(c)^{\top})$ that localisation actually depends on.
- **Training datasets did not persist `G`.** The exact historical lead field is unrecoverable. Every generated dataset now stores it.

## Layout

```
src/meg_cfm/       reusable modules: metrics, simulator, decoders, provenance
experiments/       reproducible scripts, one per audit or measurement
tests/             unit tests, including leakage guards on truth-free estimators
docs/              mathematical appendix and findings write-up
reports/           executive summary, method, results, and provenance logs
paper/             manuscript source (Typst, draft)
megflow/           variable-cardinality PyTorch implementation and tests
```

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

The code requires Python 3.11 or newer. Training uses JAX and benefits from a
CUDA-capable GPU; the unit test suite runs on CPU without paper artifacts. See
[`docs/testing.md`](docs/testing.md) for numerical precision details.

## Reproducing

Large artefacts — datasets, checkpoints, Optuna databases — are deliberately not tracked. Scripts take explicit paths:

```bash
python experiments/estimator_audit.py \
  --model_dir <checkpoint-directory> \
  --dataset_npz <dataset.npz> \
  --out_dir <output-directory>

python experiments/historical_leadfield_provenance.py \
  --anna_dg0_npz <paired-dataset.npz> \
  --calibration_summary <calibration-summary.json> \
  --datasets <dataset-a.npz> <dataset-b.npz> \
  --out_dir <output-directory>

python experiments/shift_stability_paired.py \
  --dataset_npz <paired-dataset.npz> \
  --out_dir <output-directory>
```

Historical report paths use `${MEG_CFM_WORKSPACE}` for the original experiment
checkout and `${MEG_CFM_ARTIFACT_ROOT}` for generated server artifacts. Every
new run should record input hashes, the Git revision, and whether the checkout
changed during execution.

## Comparison boundary

Code review closed the graph, source-matching, dataset, and metric-definition
questions, and final claims use an explicit common finite mask. The measured
dominance applies to two complete pipelines on one synthetic subject. The
pipelines still differ in source-shape prior, orientation model, fixed temporal
signatures, and RAP's empty-window noise estimate, so it is not a universal
comparison of algorithm classes.

## Authors

Daniil Sherki (Skoltech, Sber) · Anna Kubyak (HSE) · Alexei Ossadtchi (HSE, AIRI) · Ivan Oseledets (Skoltech, AIRI) · Ekaterina Muravleva (Skoltech, Sber)

The paired-shift simulation protocol was designed by Anna Kubyak; her simulator and RAP MUSIC baseline are not included here.

## License

MIT — see [LICENSE](LICENSE).
