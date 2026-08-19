# Diagnostics for the four pre-existing full-suite failures

Date: 2026-08-19

The prepublication branch reproduces four failures from the untouched
`codex/finalize-v1` baseline. The new scoped suite is green; these failures are
not caused by the patch scanner, statistics, or refinement code.

## 1–3. Backprojection precision tests

Failing tests:

- `test_accumulated_projection_features_match_projector_reference`
- `test_accumulated_features_are_sign_invariant_and_stable_at_block_boundaries`
- `test_rms_backprojection_matches_explicit_reference`

All three compare JAX `float32` matrix products against NumPy references with
`2e-6` tolerances. On the server GPU, JAX's default matmul precision permits
reduced-precision TensorFloat-32 products. The observed discrepancies are
`4e-4` to `7e-4`, much larger than the test tolerance but consistent with this
backend policy.

The root cause is confirmed by two one-variable reproductions:

```text
GPU, default precision                         -> 3 failures
GPU, JAX_DEFAULT_MATMUL_PRECISION=highest     -> 3 passed
CPU, JAX_PLATFORMS=cpu                        -> 3 passed
```

The implementation is therefore not violating sign/projector invariance; the
test silently assumes a precision policy that it never sets. These tests were
missed because the repository has no full-suite CI matrix and historical work
ran scoped tests under inconsistent CPU/GPU environments.

Recommended repair after submission: run mathematical unit tests on CPU or
wrap their operations in `jax.default_matmul_precision("highest")`; keep a
separate GPU tolerance/smoke test. Do not change production precision and then
reinterpret the already completed negative architecture experiments.

## 4. Checkpoint-path test

Failing test:

- `test_checkpoint_specs_define_expected_three_models`

`checkpoint_specs()` builds paths relative to the current Git worktree. The
required 4.5 MB checkpoint and companion artifacts exist under the primary
server checkout, but `cnn_posterior_outputs/` is ignored by Git and therefore
is absent from isolated worktrees:

```text
/mnt/local/tester/meg/new_version_mesh_experiments/cnn_posterior_outputs/...  exists
/mnt/local/tester/meg/.worktrees/prepublication/...                           missing
```

The test is an environment-dependent integration check presented as a unit
test. It was missed for the same reason: scoped tests in the primary checkout
saw local ignored artifacts, while no clean-clone/worktree CI existed.

Recommended repair after submission: make the artifact root an explicit CLI or
environment input, skip the integration test when that root is unavailable,
and add a pure unit test that validates model keys and relative filenames
without asserting local files exist.

## Current decision

No production change is made in this prepublication branch for these four
historical tests. The branch records the causes and reproducible passing
conditions. Automatic merge to `main` remains disabled until the user reviews
the diff and chooses whether to repair the test harness or merge with these
known environmental failures documented.
