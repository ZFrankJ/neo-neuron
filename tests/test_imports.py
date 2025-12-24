import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_core_imports():
    for module in [
        "src",
        "src.models",
        "src.neurons",
        "src.train",
        "src.utils",
    ]:
        importlib.import_module(module)
