from pathlib import Path

import pytest
import yaml

from src.runtime.backends import torch_backend


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs/alignment/lstm_standard_init_trial.yaml"
EXPECTED_TRAINABLE_PARAMS = 3_546_833


def _load_profile():
    with PROFILE_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_alignment_trial_profile_freezes_scientific_contract():
    cfg = _load_profile()

    assert cfg["profile_label"] == "standard-init no-layer-dropout RMSNorm-LSTM"
    assert cfg["backend"] == "mlx"
    assert cfg["reference_backend"] == "mlx"
    assert cfg["lstm_bias_mode"] == "single"
    assert cfg["recurrent_norm"] == "rmsnorm"
    assert cfg["rmsnorm_eps"] == pytest.approx(1e-5)
    assert cfg["lstm_layer_dropout"] == pytest.approx(0.0)
    assert cfg["forget_bias_init"] == pytest.approx(1.0)
    assert cfg["recurrent_init"] == "orthogonal"
    assert cfg["eval_regime"] == "streaming"
    assert cfg["use_checkpoint"] is False
    assert cfg["dataset_config"] == "wikitext-2-raw-v1"
    assert cfg["run_tag"] == "wt2_lstm_aligned_standard_init_trial"


def test_alignment_trial_profile_has_equal_backend_parameter_counts():
    cfg = _load_profile()
    torch_model = torch_backend.build_model(cfg, "lstm")

    assert torch_backend.count_params(torch_model) == EXPECTED_TRAINABLE_PARAMS

    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    from src.runtime.backends import mlx_backend

    mlx_model = mlx_backend.build_model(cfg, "lstm")
    assert mlx_backend.count_params(mlx_model) == EXPECTED_TRAINABLE_PARAMS
