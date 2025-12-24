"""Unified WikiText loading with token caching."""

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .cache import dataset_dir, get_data_root, token_cache_path

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - configuration guard
    raise RuntimeError(
        "This module requires 'datasets'.\n"
        "Install via: pip install datasets"
    ) from exc

WT103_BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
WT103_REMOTE_FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def load_autoregressive_corpus(
    dataset_name: str,
    dataset_config: str,
    train_split: str,
    val_split: str,
    test_split: str,
    tokenizer,
    cache_root: Optional[Path] = None,
    chunk_size: int = 1_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset_name != "wikitext":
        raise ValueError(f"Unsupported dataset_name '{dataset_name}'.")

    try:
        datasets = {
            "train": load_dataset(dataset_name, dataset_config, split=train_split),
            "validation": load_dataset(dataset_name, dataset_config, split=val_split),
            "test": load_dataset(dataset_name, dataset_config, split=test_split),
        }
    except Exception as err:
        datasets = _load_dataset_from_local_files(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            cache_root=cache_root,
            load_error=err,
        )

    sep_tokens = tokenizer("\n\n", add_special_tokens=False, return_attention_mask=False)["input_ids"]

    train_ids = _load_or_tokenize(
        datasets["train"], "train", dataset_config, tokenizer, sep_tokens, cache_root, chunk_size
    )
    val_ids = _load_or_tokenize(
        datasets["validation"], "validation", dataset_config, tokenizer, sep_tokens, cache_root, chunk_size
    )
    test_ids = _load_or_tokenize(
        datasets["test"], "test", dataset_config, tokenizer, sep_tokens, cache_root, chunk_size
    )
    return train_ids, val_ids, test_ids


def _load_or_tokenize(
    dataset,
    split: str,
    dataset_config: str,
    tokenizer,
    sep_tokens,
    cache_root: Optional[Path],
    chunk_size: int,
) -> torch.Tensor:
    cache_path = token_cache_path(dataset_config, split, root=cache_root)
    if cache_path.exists():
        try:
            return torch.load(cache_path)
        except Exception:
            pass

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    token_ids = _tokenize_dataset(dataset, tokenizer, sep_tokens, chunk_size)
    torch.save(token_ids, cache_path)
    return token_ids


def _tokenize_dataset(dataset, tokenizer, sep_tokens, chunk_size: int) -> torch.Tensor:
    parts = []
    buffer = []
    first = True

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return
        parts.append(torch.tensor(buffer, dtype=torch.long))
        buffer = []

    for row in dataset:
        if not first:
            buffer.extend(sep_tokens)
        ids = tokenizer(row["text"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        buffer.extend(ids)
        first = False
        if len(buffer) >= chunk_size:
            flush_buffer()
    flush_buffer()

    if not parts:
        return torch.tensor([], dtype=torch.long)
    return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]


def _load_dataset_from_local_files(
    dataset_name: str,
    dataset_config: str,
    train_split: str,
    val_split: str,
    test_split: str,
    cache_root: Optional[Path],
    load_error: Exception,
):
    if dataset_name != "wikitext":
        raise load_error
    if dataset_config == "wikitext-103-raw-v1":
        return _prepare_wikitext103_local(
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            cache_root=cache_root,
        )
    raise RuntimeError(
        f"No offline loader available for {dataset_name}/{dataset_config}. "
        "Please download it manually or adjust the dataset configuration."
    ) from load_error


def _download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _prepare_wikitext103_local(
    train_split: str,
    val_split: str,
    test_split: str,
    cache_root: Optional[Path],
) -> Dict[str, object]:
    if not (train_split == "train" and val_split == "validation" and test_split == "test"):
        raise RuntimeError(
            "Automatic WikiText-103 download only supports the default train/validation/test splits."
        )

    cache_dir = dataset_dir("wikitext-103-raw-v1", root=cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_files: Dict[str, list] = {}
    for split, filenames in WT103_REMOTE_FILES.items():
        local_paths = []
        for fname in filenames:
            target = cache_dir / fname
            if not target.exists():
                url = f"{WT103_BASE_URL}/{fname}?download=true"
                try:
                    _download_file(url, target)
                except urllib.error.URLError as err:
                    raise RuntimeError(f"Failed to download {fname} from {url}: {err}") from err
            local_paths.append(str(target))
        data_files[split] = local_paths

    dataset_dict = load_dataset("parquet", data_files=data_files)
    return {
        "train": dataset_dict["train"],
        "validation": dataset_dict["validation"],
        "test": dataset_dict["test"],
    }
