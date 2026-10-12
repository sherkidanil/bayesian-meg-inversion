#!/usr/bin/env python3
"""Run one bounded null-slot Transformer and preserve valid budget stops."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from megflow.dataset import sha256, write_json


def accepted_terminal_result(result, run_directory):
    run_directory = Path(run_directory)
    status = result.get("status")
    if status == "complete":
        return (
            (run_directory / "last.pt").is_file()
            and result.get("step", 0) > 0
            and result.get("validation_step") == result.get("step")
        )
    if status == "budget_exhausted":
        checkpoint = run_directory / "validated-ema.pt"
        return (
            checkpoint.is_file()
            and result.get("step", 0) > 0
            and result.get("validation_step", 0) > 0
            and result.get("validated_checkpoint_step") == result["validation_step"]
            and result.get("validated_checkpoint_sha256") == sha256(checkpoint)
        )
    return False


def find_new_run(runs_directory, before):
    runs_directory = Path(runs_directory)
    after = {path.resolve() for path in runs_directory.glob("transformer-*") if path.is_dir()}
    created = sorted(after - set(before))
    if len(created) != 1:
        raise RuntimeError("training must create exactly one new run directory")
    return Path(created[0])


def terminal_queue_payload(result, run_id):
    return dict(
        status=result["status"],
        run_id=run_id,
        step=result["step"],
        validation_step=result["validation_step"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hard-seconds", type=int, default=7000)
    args = parser.parse_args(argv)
    if args.hard_seconds < 1:
        raise ValueError("hard runtime limit must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    runs = args.root / "runs"
    runs.mkdir(exist_ok=True)
    queue = args.root / "queue.json"
    if queue.exists():
        raise FileExistsError(queue)
    before = {path.resolve() for path in runs.glob("transformer-*") if path.is_dir()}
    state = dict(
        status="running",
        code_commit=args.code_commit,
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(queue, state)
    command = [
        sys.executable,
        "-m",
        "megflow.train",
        "--config",
        str(args.config),
        "--dataset",
        str(args.dataset),
        "--output",
        str(runs),
        "--architecture",
        "transformer",
        "--device",
        args.device,
        "--code-commit",
        args.code_commit,
    ]
    process = None
    stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
        if process is not None and process.poll() is None:
            process.terminate()

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        with (args.root / "train-transformer.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + args.hard_seconds
            while process.poll() is None:
                if stop_requested:
                    process.terminate()
                if time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=100)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError("null-slot training exceeded its hard runtime budget")
                time.sleep(1)
            returncode = process.returncode
        if stop_requested:
            raise RuntimeError("null-slot runner was interrupted")
        run = find_new_run(runs, before)
        result_path = run / "result.json"
        if not result_path.is_file():
            raise RuntimeError("training did not write result.json")
        result = json.loads(result_path.read_text())
        expected_code = 0 if result.get("status") == "complete" else 3
        if returncode != expected_code or not accepted_terminal_result(result, run):
            raise RuntimeError(
                f"training terminal state is not acceptable: code={returncode}, result={result}"
            )
        state.update(terminal_queue_payload(result, run.name))
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(queue, state)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
