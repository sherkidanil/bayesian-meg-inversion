#!/usr/bin/env python3
"""Run two fixed-config M1 arms sequentially, with a hard deadline for each."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import signal
import time

from megflow.dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    queue = args.root / 'queue.json'
    if queue.exists():
        raise FileExistsError(queue)
    state = dict(status='running', order=['transformer', 'mlp'], finished=[],
                 started_utc=datetime.now(timezone.utc).isoformat(), code_commit=args.code_commit)
    active, stop_requested_at = None, None

    def stop_queue(*_):
        nonlocal stop_requested_at
        if stop_requested_at is None:
            stop_requested_at = time.monotonic()
        if active is not None and active.poll() is None:
            active.terminate()

    previous = {sig: signal.signal(sig, stop_queue) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for architecture in state['order']:
            if stop_requested_at is not None:
                break
            state['current'] = architecture
            write_json(queue, state)
            command = [sys.executable, '-m', 'megflow.train', '--config', str(args.config),
                       '--dataset', str(args.root / 'dataset'), '--output', str(args.root / 'runs'),
                       '--architecture', architecture, '--code-commit', args.code_commit]
            with (args.root / f'train-{architecture}.log').open('x') as log:
                with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
                    active = process
                    deadline = time.monotonic() + 7000
                    termination_started = None
                    while True:
                        now = time.monotonic()
                        if stop_requested_at is not None:
                            termination_started = stop_requested_at
                            process.terminate()
                        elif now >= deadline and termination_started is None:
                            termination_started = now
                            process.terminate()
                        if termination_started is not None and now - termination_started >= 100:
                            process.kill()
                        try:
                            returncode = process.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    active = None
                    if termination_started is not None and stop_requested_at is None:
                        raise RuntimeError(f'{architecture} exceeded its hard runtime budget')
            if stop_requested_at is not None:
                break
            if returncode:
                raise RuntimeError(f'{architecture} exited with status {returncode}')
            state['finished'].append(architecture)
        state.update(status='interrupted' if stop_requested_at is not None else 'complete', current=None)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(queue, state)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
