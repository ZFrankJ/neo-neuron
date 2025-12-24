"""Model registry helpers."""

from typing import Dict

from .lstm_lm import LSTMLM
from .neo_lm import NeoLM
from .transformer_lm import TransformerLM

MODEL_REGISTRY: Dict[str, object] = {
    "lstm": LSTMLM,
    "neo": NeoLM,
    "transformer": TransformerLM,
}


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
