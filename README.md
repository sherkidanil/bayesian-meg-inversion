# MEG inversion with conditional flow matching

Code for generating candidate cortical source configurations from MEG recordings
and selecting a final estimate using the MEG forward model. The repository
contains two related implementations:

- `src/meg_cfm/` and `experiments/`: fixed-`K=3` conditional flow matching,
  physics-based decoding, and comparison scripts.
- [`megflow/`](megflow/README.md): PyTorch models for an unknown number of
  sources (`K=1,...,10`), including count-head and null-slots Transformers,
  fixed-`K=3` MLP and Transformer baselines, dataset generation, training,
  candidate-bank decoding, and evaluation.

Each source is represented by a cortical location and radius. In the
variable-cardinality models, the true number of sources is not provided at
inference. Model outputs are candidate configurations; a residual-based
decoder can rank or combine candidates using the observed recording and lead
field.

## Installation

Python 3.11 or newer is recommended. Install the fixed-`K` package from the
repository root:

```bash
python -m pip install -e .
```

For the variable-cardinality package and its training dependencies:

```bash
python -m pip install -e './megflow[v15]'
```

## Data and evaluation

The code expects an anatomy bundle and generated recordings supplied by the
user. Datasets, lead fields, model checkpoints, and run outputs are not
included in this repository. The [`megflow` guide](megflow/README.md) shows
the dataset, training, and evaluation commands and required inputs.

The evaluation code reports source-count accuracy, localisation and radius
errors, and disconnected geodesic pairs. Comparisons should use the same
records, candidate-bank size, decoder, and validity rule.

## Reference

This work builds on [Bayesian Inverse Problems Meet Flow Matching](https://arxiv.org/abs/2503.01375).

## License

MIT — see [LICENSE](LICENSE).
