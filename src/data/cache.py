"""Cache path utilities for datasets and tokenized corpora."""

from pathlib import Path
import os
from typing import Optional, Union

PathLike = Union[str, Path]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_root(root: Optional[PathLike] = None) -> Path:
    if root is not None:
        return Path(root)
    env_root = os.getenv("NEO_DATA_DIR")
    if env_root:
        return Path(env_root)
    return get_repo_root() / "data"


def dataset_dir(dataset_config: str, root: Optional[PathLike] = None) -> Path:
    return get_data_root(root) / dataset_config


def token_cache_path(dataset_config: str, split: str, root: Optional[PathLike] = None) -> Path:
    return dataset_dir(dataset_config, root) / f"{split}_gpt2_ids.pt"
