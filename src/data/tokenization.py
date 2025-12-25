"""Tokenizer helpers."""

from typing import Optional

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - configuration guard
    raise RuntimeError(
        "This module requires 'transformers'.\n"
        "Install via: pip install transformers"
    ) from exc


def build_gpt2_tokenizer(name: str = "gpt2", use_fast: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
    tokenizer.pad_token = tokenizer.eos_token
    # Avoid max length warnings when tokenizing long corpora.
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def build_tokenizer(name: str = "gpt2", use_fast: bool = True):
    return build_gpt2_tokenizer(name=name, use_fast=use_fast)
