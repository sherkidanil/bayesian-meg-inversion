# Variable-cardinality MEG inversion

This package implements conditional flow matching for cortical source sets with
an unknown number of sources (`K=1,...,10`). Each source token contains a
location and a log radius. The recording encoder supplies the condition; the
true `K` is not passed to the model at inference. Two cardinality mechanisms
are implemented: a categorical count head and active/null source slots.

The package also retains the fixed-`K=3` MLP and Transformer baselines used
for controlled architecture comparisons. Candidate-bank decoding, geodesic
evaluation, and the frozen source simulator are included alongside the models.

## Contents

- `megflow/`: simulator, geometry, dataset handling, models, training, decoding,
  and evaluation metrics.
- `scripts/`: command-line entry points for geometry export, dataset extension,
  fixed-`K` comparisons, null-slot training, and evaluation.
- `config/`: experiment configurations, including the `K_max=10` variants.
- `references/`: the supplied 15 September dataset generator and its provenance.
- `tests/`: CPU unit tests for simulation, models, training, and evaluation.

## Install and test

Python 3.10 or newer is required. From the repository root:

```sh
python -m pip install -e './megflow[v15,test]'
python -m pytest megflow/tests -q
```

Training normally uses a CUDA GPU; the test suite runs on CPU. The package
does not include anatomical bundles, generated recordings, or checkpoints.
Paths to those inputs are provided explicitly to the commands below.

## Data, training, and evaluation

The V15 generator uses an existing anatomy bundle and separate train and
evaluation groups. For example:

```sh
python -m megflow.v15_dataset \
  --config megflow/config/v15_k10_4s.yaml \
  --anatomy /path/to/anatomy.npz \
  --output /path/to/dataset \
  --group train

python -m megflow.v15_dataset \
  --config megflow/config/v15_k10_4s.yaml \
  --anatomy /path/to/anatomy.npz \
  --output /path/to/dataset \
  --group eval

python -m megflow.v15_dataset \
  --config megflow/config/v15_k10_4s.yaml \
  --anatomy /path/to/anatomy.npz \
  --output /path/to/dataset \
  --group audit

python -m megflow.train \
  --config megflow/config/v15_null_slots.yaml \
  --dataset /path/to/dataset \
  --output /path/to/runs \
  --architecture transformer

python -m megflow.v15_evaluate \
  --config megflow/config/v15_null_slots.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --dataset-root /path/to/dataset \
  --anatomy /path/to/anatomy.npz \
  --output /path/to/evaluation \
  --mode null_slots --split heldout
```

Use `--mode count_head` with the corresponding checkpoint and configuration
for the count-head model. `python -m megflow.v15_candidate_bank --help` lists
the additional inputs for candidate-bank evaluation. The evaluation code
reports exact-cardinality accuracy, OSPA, radius error, and geodesic errors;
record validity and disconnected pairs are retained for separate analysis.

The experimental protocols use fixed seeds and disjoint train, validation,
development, and held-out splits. Results from different candidate-bank sizes
or decoding rules should be compared only on the same records and under the
same validity definition.
