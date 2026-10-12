#!/usr/bin/env python3
"""Export checked MNE sample anatomy in the portable megflow bundle format."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from megflow.anatomy import main

if __name__ == '__main__':
    main()
