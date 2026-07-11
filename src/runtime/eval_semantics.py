"""Shared evaluation-regime contract."""

from typing import Any


EVAL_REGIMES = ("block_reset", "streaming")


def resolve_eval_regime(cfg: Any) -> str:
    if isinstance(cfg, dict):
        value = cfg.get("eval_regime", "block_reset")
    else:
        value = getattr(cfg, "eval_regime", "block_reset")
    regime = str(value).strip().lower()
    if regime not in EVAL_REGIMES:
        choices = ", ".join(EVAL_REGIMES)
        raise ValueError(f"eval_regime must be one of: {choices}; got {value!r}")
    return regime
