"""Terminal-state contracts for the bounded null-slot runner."""

import json

import pytest


def test_complete_and_valid_budget_stop_are_successful(tmp_path):
    from megflow.dataset import sha256
    from scripts.run_v15_null_slots import accepted_terminal_result

    run = tmp_path / "run"
    run.mkdir()
    (run / "last.pt").write_bytes(b"checkpoint")
    (run / "validated-ema.pt").write_bytes(b"validated")
    complete = dict(status="complete", step=20000, validation_step=20000)
    bounded = dict(
        status="budget_exhausted",
        step=18000,
        validation_step=17500,
        validated_checkpoint_step=17500,
        validated_checkpoint_sha256=sha256(run / "validated-ema.pt"),
    )

    assert accepted_terminal_result(complete, run)
    assert accepted_terminal_result(bounded, run)


@pytest.mark.parametrize(
    "result",
    [
        dict(status="failed", step=10, validation_step=0),
        dict(status="interrupted", step=10, validation_step=0),
        dict(status="budget_exhausted", step=100, validation_step=0),
    ],
)
def test_failed_or_unvalidated_terminal_states_are_rejected(tmp_path, result):
    from scripts.run_v15_null_slots import accepted_terminal_result

    run = tmp_path / "run"
    run.mkdir()
    (run / "last.pt").write_bytes(b"checkpoint")
    assert not accepted_terminal_result(result, run)


def test_missing_checkpoint_is_rejected(tmp_path):
    from scripts.run_v15_null_slots import accepted_terminal_result

    run = tmp_path / "run"
    run.mkdir()
    assert not accepted_terminal_result(
        dict(status="complete", step=20000, validation_step=20000), run
    )


def test_find_new_run_requires_exactly_one_new_directory(tmp_path):
    from scripts.run_v15_null_slots import find_new_run

    runs = tmp_path / "runs"
    runs.mkdir()
    old = runs / "transformer-old"
    old.mkdir()
    before = {old.resolve()}
    new = runs / "transformer-new"
    new.mkdir()
    assert find_new_run(runs, before) == new

    another = runs / "transformer-another"
    another.mkdir()
    with pytest.raises(RuntimeError, match="exactly one"):
        find_new_run(runs, before)


def test_queue_records_actual_budget_terminal_state(tmp_path):
    from scripts.run_v15_null_slots import terminal_queue_payload

    result = dict(status="budget_exhausted", step=18000, validation_step=17500)
    payload = terminal_queue_payload(result, "run-id")
    assert payload["status"] == "budget_exhausted"
    assert payload["run_id"] == "run-id"
    assert payload["step"] == 18000
    assert json.dumps(payload)


def test_budget_stop_requires_the_exact_validated_checkpoint(tmp_path):
    from megflow.dataset import sha256
    from scripts.run_v15_null_slots import accepted_terminal_result

    run = tmp_path / "run"
    run.mkdir()
    (run / "last.pt").write_bytes(b"unvalidated-last-step")
    result = dict(
        status="budget_exhausted",
        step=18000,
        validation_step=17500,
        validated_checkpoint_step=17500,
        validated_checkpoint_sha256="missing",
    )
    assert not accepted_terminal_result(result, run)

    validated = run / "validated-ema.pt"
    validated.write_bytes(b"validated")
    result["validated_checkpoint_sha256"] = sha256(validated)
    assert accepted_terminal_result(result, run)
