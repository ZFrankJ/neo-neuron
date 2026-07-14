from pathlib import Path

import pytest
import yaml

from src.runtime.backends import torch_backend


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DOC_PATH = REPO_ROOT / "docs/training.md"
EXPECTED_TRAINABLE_PARAMS = 60_024_343
PROFILE_PATHS = {
    "matched": REPO_ROOT
    / "configs/wt103/lstm_60m_matched_no_layer_dropout.yaml",
    "standard_init": REPO_ROOT
    / "configs/wt103/lstm_60m_standard_init_no_layer_dropout.yaml",
}
EXPECTED_SHARED_PROFILE = {
    "backend": "mlx",
    "reference_backend": "mlx",
    "model_name": "lstm",
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "train_split": "train",
    "val_split": "validation",
    "test_split": "test",
    "d_model": 790,
    "d_embed": 192,
    "n_layers": 10,
    "dropout": 0.1,
    "lstm_layer_dropout": 0.0,
    "tie_embeddings": True,
    "recurrent_norm": "rmsnorm",
    "recurrent_norm_place": "all",
    "rmsnorm_eps": 1e-5,
    "lstm_bias_mode": "single",
    "vocab_size": 50527,
    "block_size": 256,
    "batch_size": 20,
    "epochs": 12,
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "weight_decay_policy": "table",
    "transformer_weight_decay": 1e-2,
    "proj_weight_decay": 1e-3,
    "recurrent_weight_decay": 0.0,
    "embed_weight_decay": 0.0,
    "grad_clip": 1.0,
    "tbptt_len": 256,
    "train_regime": "streaming",
    "stream_state": True,
    "eval_regime": "streaming",
    "cosine": True,
    "warmup_epochs": 2,
    "min_lr": 3e-5,
    "seed": 42,
    "use_checkpoint": False,
    "save_dir": "checkpoints",
    "mem_report_interval": 10000,
    "mem_clear_interval": 200,
    "save_each_epoch": True,
    "resume_path": "",
}
EXPECTED_PROFILE_FIELDS = {
    "matched": {
        "profile_label": "matched no-layer-dropout RMSNorm-LSTM",
        "run_tag": "wt103_lstm_60m_matched_no_layer_dropout",
    },
    "standard_init": {
        "profile_label": "standard-init no-layer-dropout RMSNorm-LSTM",
        "forget_bias_init": 1.0,
        "recurrent_init": "orthogonal",
        "run_tag": "wt103_lstm_60m_standard_init_no_layer_dropout",
    },
}
EXPECTED_HISTORICAL_BASE = {
    "backend": "mlx",
    "model_name": "lstm",
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "train_split": "train",
    "val_split": "validation",
    "test_split": "test",
    "d_model": 790,
    "d_embed": 192,
    "dropout": 0.1,
    "tie_embeddings": True,
    "recurrent_norm": "rmsnorm",
    "recurrent_norm_place": "all",
    "vocab_size": 50527,
    "block_size": 256,
    "batch_size": 20,
    "epochs": 12,
    "lr": "3e-4",
    "weight_decay": "1e-2",
    "weight_decay_policy": "table",
    "transformer_weight_decay": "1e-2",
    "proj_weight_decay": "1e-3",
    "recurrent_weight_decay": 0.0,
    "embed_weight_decay": 0.0,
    "grad_clip": 1.0,
    "tbptt_len": 256,
    "train_regime": "streaming",
    "stream_state": True,
    "cosine": True,
    "warmup_epochs": 2,
    "min_lr": "3e-5",
    "seed": 42,
    "save_dir": "checkpoints",
    "mem_report_interval": 10000,
    "mem_clear_interval": 200,
    "save_each_epoch": True,
    "resume_path": "",
}
HISTORICAL_PROFILES = {
    "20m": {"n_layers": 2, "run_tag": "wt103_lstm_20m"},
    "30m": {"n_layers": 4, "run_tag": "wt103_lstm_30m"},
    "50m": {"n_layers": 8, "run_tag": "wt103_lstm_50m"},
}


def _load_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.mark.parametrize("profile_name", PROFILE_PATHS)
def test_wt103_lstm_diagnostic_profile_freezes_policy(profile_name):
    cfg = _load_yaml(PROFILE_PATHS[profile_name])
    expected = EXPECTED_SHARED_PROFILE | EXPECTED_PROFILE_FIELDS[profile_name]

    assert cfg == expected
    assert cfg["tbptt_len"] >= cfg["block_size"]

    if profile_name == "matched":
        assert "forget_bias_init" not in cfg
        assert "recurrent_init" not in cfg


@pytest.mark.parametrize("profile_name", PROFILE_PATHS)
def test_wt103_lstm_diagnostic_profile_has_equal_backend_counts(profile_name):
    cfg = _load_yaml(PROFILE_PATHS[profile_name])
    torch_model = torch_backend.build_model(cfg, "lstm")
    assert torch_backend.count_params(torch_model) == EXPECTED_TRAINABLE_PARAMS

    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    from src.runtime.backends import mlx_backend

    mlx_model = mlx_backend.build_model(cfg, "lstm")
    assert mlx_backend.count_params(mlx_model) == EXPECTED_TRAINABLE_PARAMS


def test_historical_wt103_lstm_profiles_remain_exact():
    for size, profile_fields in HISTORICAL_PROFILES.items():
        cfg = _load_yaml(REPO_ROOT / f"configs/wt103/lstm_{size}.yaml")
        assert cfg == EXPECTED_HISTORICAL_BASE | profile_fields


def test_diagnostic_run_tags_do_not_collide_with_historical_artifacts():
    historical_tags = {
        profile_fields["run_tag"] for profile_fields in HISTORICAL_PROFILES.values()
    }
    diagnostic_tags = {
        _load_yaml(profile_path)["run_tag"] for profile_path in PROFILE_PATHS.values()
    }

    assert len(diagnostic_tags) == len(PROFILE_PATHS)
    assert diagnostic_tags.isdisjoint(historical_tags)


def test_training_docs_freeze_epoch_four_validation_gate():
    training_docs = TRAINING_DOC_PATH.read_text(encoding="utf-8")

    assert "same-geometry historical epoch-4 streaming validation PPL: `84.54`" in training_docs
    assert "preceding eight-layer epoch-4 streaming validation PPL: `82.57`" in training_docs
    assert "continue the matched profile when PPL is at most `83.54`" in training_docs
    assert "do not inspect test PPL" in training_docs
    assert "run only the standard-init fallback" in training_docs
