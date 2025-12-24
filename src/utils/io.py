"""I/O helpers."""

from pathlib import Path
from typing import Union
import os
import tempfile

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_bytes(path: PathLike, data: bytes) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    return path


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))
