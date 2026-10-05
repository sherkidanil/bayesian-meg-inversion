# Testing and numerical precision

The repository test suite contains unit tests for metrics, source matching,
truth-free decoders, provenance checks, paired stability analysis, and the
fixed-orientation RAP MUSIC implementation.

## Recommended environment

Use Python 3.11 or newer and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

The public baseline is 113 passing tests on Python 3.12. Tests do not require
the large datasets or checkpoints used for the paper.

## CPU and GPU precision

Some projector and backprojection checks compare JAX matrix products with
NumPy references at tight tolerances. GPU backends may use TensorFloat-32 for
matrix multiplication unless precision is configured explicitly. Run these
mathematical unit tests on CPU or set:

```bash
JAX_DEFAULT_MATMUL_PRECISION=highest pytest -q
```

Production runs may use their normal precision policy; record that policy in
the run metadata.

## External artifacts

Datasets, checkpoints, and generated result bundles are intentionally excluded
from Git. Scripts accept their locations through command-line arguments. Tests
that require an external artifact should skip cleanly when it is unavailable;
unit tests must not assume a particular server checkout or directory layout.
