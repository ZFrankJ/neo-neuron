import pickle
from pathlib import Path

import pytest
import torch

from src.runtime.backends import torch_backend
from src.runtime.checkpoint_compat import REQUIRED_ALIGNED_NEO_CFG_KEYS


def _neo_cfg(**overrides):
    cfg = {
        "model_name": "neo",
        "vocab_size": 32,
        "d_model": 16,
        "d_embed": 8,
        "n_layers": 2,
        "dropout": 0.0,
        "tie_embeddings": True,
        "cell_type": "cortical",
        "activation_id": "id5",
        "recurrent_norm": "rmsnorm",
        "recurrent_norm_place": "all",
        "rmsnorm_eps": 1e-5,
        "use_checkpoint": False,
        "weight_decay_policy": "table",
        "reference_backend": "mlx",
    }
    cfg.update(overrides)
    return cfg


def _save_checkpoint(path: Path, cfg: dict) -> None:
    model = torch_backend.build_model(cfg, "neo")
    torch_backend.save_checkpoint_entry(path, model, None, None, epoch=1, global_step=2, cfg=cfg)


def test_neo_checkpoint_preserves_alignment_metadata(tmp_path):
    cfg = _neo_cfg()
    path = tmp_path / "neo.pt"

    _save_checkpoint(path, cfg)

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    for key in REQUIRED_ALIGNED_NEO_CFG_KEYS:
        assert payload["cfg"][key] == cfg[key]


def test_neo_checkpoint_load_warns_for_legacy_missing_metadata(tmp_path):
    cfg = _neo_cfg()
    path = tmp_path / "legacy.pt"
    _save_checkpoint(path, cfg)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["cfg"].pop("reference_backend")
    payload["cfg"].pop("weight_decay_policy")
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    model = torch_backend.build_model(cfg, "neo")
    with pytest.warns(UserWarning, match="legacy/provisional"):
        torch_backend.load_checkpoint_entry(path, model, device=torch.device("cpu"), cfg=cfg)


def test_neo_checkpoint_load_fails_for_incompatible_metadata(tmp_path):
    cfg = _neo_cfg()
    path = tmp_path / "conflict.pt"
    _save_checkpoint(path, cfg)

    model = torch_backend.build_model(cfg, "neo")
    incompatible_cfg = dict(cfg, activation_id="id4")
    with pytest.raises(ValueError, match="activation_id"):
        torch_backend.load_checkpoint_entry(path, model, device=torch.device("cpu"), cfg=incompatible_cfg)
