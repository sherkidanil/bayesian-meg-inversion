#!/usr/bin/env python3
"""Run M0 from a checkout without installing the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from megflow.milestone import main

if __name__ == '__main__':
    raise SystemExit(main())
