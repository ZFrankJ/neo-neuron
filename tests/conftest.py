"""Pytest configuration for Neo tests."""

import sys
from pathlib import Path

NEO_ROOT = Path(__file__).resolve().parents[1]
if str(NEO_ROOT) not in sys.path:
    sys.path.insert(0, str(NEO_ROOT))
