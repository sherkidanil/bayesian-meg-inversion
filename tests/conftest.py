"""Make the curated repository layout importable during source-tree tests."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for module_root in (ROOT / "experiments", ROOT / "src" / "meg_cfm", ROOT / "src"):
    module_path = str(module_root)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
