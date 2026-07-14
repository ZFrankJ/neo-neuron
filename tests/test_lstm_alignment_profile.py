from pathlib import Path

import pytest
import torch
import yaml

from src.runtime.backends import torch_backend
from src.train.schedulers import build_scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs/alignment/lstm_standard_init_trial.yaml"
EXPECTED_TRAINABLE_PARAMS = 3_546_833
EXPECTED_PROFILE = {
    "profile_label": "standard-init no-layer-dropout RMSNorm-LSTM",
    "backend": "mlx",
    "reference_backend": "mlx",
    "model_name": "lstm",
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "train_split": "train",
    "val_split": "validation",
    "test_split": "test",
    "d_model": 128,
    "d_embed": 64,
    "n_layers": 2,
    "dropout": 0.1,
    "lstm_layer_dropout": 0.0,
    "tie_embeddings": True,
    "recurrent_norm": "rmsnorm",
    "recurrent_norm_place": "all",
    "rmsnorm_eps": 1e-5,
    "lstm_bias_mode": "single",
    "forget_bias_init": 1.0,
    "recurrent_init": "orthogonal",
    "vocab_size": 50257,
    "block_size": 128,
    "batch_size": 16,
    "epochs": 1,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "weight_decay_policy": "table",
    "transformer_weight_decay": 1e-2,
    "proj_weight_decay": 1e-3,
    "recurrent_weight_decay": 0.0,
    "embed_weight_decay": 0.0,
    "grad_clip": 1.0,
    "tbptt_len": 128,
    "train_regime": "streaming",
    "stream_state": True,
    "eval_regime": "streaming",
    "cosine": True,
    "warmup_epochs": 0.1,
    "min_lr": 2e-5,
    "seed": 20260714,
    "use_checkpoint": False,
    "save_dir": "checkpoints",
    "run_tag": "wt2_lstm_aligned_standard_init_trial",
    "save_each_epoch": True,
    "resume_path": "",
}


def _load_profile():
    with PROFILE_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_alignment_trial_profile_freezes_scientific_contract():
    cfg = _load_profile()

    assert cfg == EXPECTED_PROFILE
    assert cfg["tbptt_len"] >= cfg["block_size"]


def test_alignment_trial_profile_reaches_cosine_decay_and_min_lr():
    cfg = _load_profile()
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=cfg["lr"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=100)
    lr_lambda = scheduler.lr_lambdas[0]

    assert lr_lambda(0) == pytest.approx(0.1)
    assert lr_lambda(9) == pytest.approx(1.0)
    assert lr_lambda(54) == pytest.approx(0.55)
    assert lr_lambda(99) == pytest.approx(cfg["min_lr"] / cfg["lr"])


def test_alignment_trial_profile_has_equal_backend_parameter_counts():
    cfg = _load_profile()
    torch_model = torch_backend.build_model(cfg, "lstm")

    assert torch_backend.count_params(torch_model) == EXPECTED_TRAINABLE_PARAMS

    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    from src.runtime.backends import mlx_backend

    mlx_model = mlx_backend.build_model(cfg, "lstm")
    assert mlx_backend.count_params(mlx_model) == EXPECTED_TRAINABLE_PARAMS
