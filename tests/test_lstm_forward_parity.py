from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.runtime.backends import torch_backend
from src.runtime.checkpoint_compat import (
    REQUIRED_ALIGNED_LSTM_CFG_KEYS,
    map_model_state,
    validate_checkpoint_metadata,
)


def _cfg(**overrides):
    cfg = {
        "model_name": "lstm",
        "vocab_size": 19,
        "d_model": 4,
        "d_embed": 4,
        "n_layers": 1,
        "dropout": 0.0,
        "tie_embeddings": True,
        "recurrent_norm": "rmsnorm",
        "recurrent_norm_place": "all",
        "rmsnorm_eps": 1e-5,
        "lstm_bias_mode": "single",
        "reference_backend": "mlx",
        "use_checkpoint": False,
    }
    cfg.update(overrides)
    return cfg


def _batch():
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.int64)
    return x, y


def _torch_forward(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, state = model(torch.from_numpy(x).long(), None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            torch.from_numpy(y).reshape(-1),
        )
    return (
        logits.detach().numpy(),
        (state[0].detach().numpy(), state[1].detach().numpy()),
        float(loss),
    )


def _mlx_forward(model, x, y):
    mx = pytest.importorskip("mlx.core")
    mxnn = pytest.importorskip("mlx.nn")
    model.eval()
    logits, state = model(mx.array(x), None)
    loss = mx.mean(
        mxnn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            mx.array(y.astype(np.int32)).reshape(-1),
        )
    )
    mx.eval(logits, state, loss)
    return (
        np.asarray(logits),
        (np.asarray(state[0]), np.asarray(state[1])),
        float(loss.item()),
    )


def _assert_forward_close(torch_result, mlx_result):
    torch_logits, torch_state, torch_loss = torch_result
    mlx_logits, mlx_state, mlx_loss = mlx_result
    np.testing.assert_allclose(torch_logits, mlx_logits, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(torch_state[0], mlx_state[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(torch_state[1], mlx_state[1], rtol=1e-5, atol=1e-6)
    assert torch_loss == pytest.approx(mlx_loss, rel=1e-6, abs=1e-6)


def test_torch_lstm_explicit_rmsnorm_epsilon_preserves_legacy_omission():
    explicit = torch_backend.build_model(_cfg(), "lstm")
    legacy_cfg = _cfg()
    legacy_cfg.pop("rmsnorm_eps")
    legacy = torch_backend.build_model(legacy_cfg, "lstm")

    assert explicit.lstm.pre_norms[0].eps == pytest.approx(1e-5)
    assert explicit.lstm.stack_norm.eps == pytest.approx(1e-5)
    assert legacy.lstm.pre_norms[0].eps is None
    assert legacy.lstm.stack_norm.eps is None


def test_mlx_lstm_rejects_unsupported_explicit_rmsnorm_epsilon():
    pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    mlx_backend.build_model(_cfg(rmsnorm_eps=1e-5), "lstm")
    with pytest.raises(ValueError, match="rmsnorm_eps.*1e-5"):
        mlx_backend.build_model(_cfg(rmsnorm_eps=1e-6), "lstm")


@pytest.mark.parametrize("recurrent_norm", ["none", "layernorm", "rmsnorm"])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_mapped_weight_lstm_forward_state_and_loss_parity(recurrent_norm, n_layers):
    pytest.importorskip("mlx.core")
    mlx_utils = pytest.importorskip("mlx.utils")
    from src.runtime.backends import mlx_backend

    cfg = _cfg(recurrent_norm=recurrent_norm, n_layers=n_layers)
    mlx_model = mlx_backend.build_model(cfg, "lstm")
    torch_model = torch_backend.build_model(cfg, "lstm")
    mlx_state = {
        name: np.asarray(value)
        for name, value in mlx_utils.tree_flatten(mlx_model.parameters())
    }
    mapped, warnings = map_model_state(
        model_name="lstm",
        src_backend="mlx",
        dst_backend="torch",
        src_state_np=mlx_state,
        dst_template=torch_model.state_dict(),
        cfg=cfg,
    )
    assert warnings == []
    torch_model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in mapped.items()}
    )

    x, y = _batch()
    _assert_forward_close(
        _torch_forward(torch_model, x, y),
        _mlx_forward(mlx_model, x, y),
    )


def test_lstm_checkpoint_conversion_preserves_loss_in_both_directions(tmp_path: Path):
    pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    cfg = _cfg(n_layers=2)
    x, y = _batch()

    mlx_source = mlx_backend.build_model(cfg, "lstm")
    mlx_checkpoint = tmp_path / "mlx-lstm.pkl"
    mlx_backend.save_checkpoint_entry(
        mlx_checkpoint,
        mlx_source,
        None,
        None,
        epoch=1,
        global_step=2,
        cfg=cfg,
    )
    torch_from_mlx = torch_backend.build_model(cfg, "lstm")
    torch_backend.load_checkpoint_entry(
        mlx_checkpoint,
        torch_from_mlx,
        device=torch.device("cpu"),
        cfg=cfg,
    )
    _assert_forward_close(
        _torch_forward(torch_from_mlx, x, y),
        _mlx_forward(mlx_source, x, y),
    )

    torch_checkpoint = tmp_path / "torch-lstm.pkl"
    torch_backend.save_checkpoint_entry(
        torch_checkpoint,
        torch_from_mlx,
        None,
        None,
        epoch=2,
        global_step=3,
        cfg=cfg,
    )
    mlx_from_torch = mlx_backend.build_model(cfg, "lstm")
    with pytest.warns(UserWarning, match="optimizer resume is not equivalent"):
        mlx_backend.load_checkpoint_entry(torch_checkpoint, mlx_from_torch, cfg=cfg)
    _assert_forward_close(
        _torch_forward(torch_from_mlx, x, y),
        _mlx_forward(mlx_from_torch, x, y),
    )


def test_lstm_checkpoint_metadata_warns_when_legacy_fields_are_missing():
    payload = {"cfg": {"model_name": "lstm"}}
    expected = _cfg()

    with pytest.warns(UserWarning, match="legacy/provisional"):
        missing = validate_checkpoint_metadata(
            payload,
            expected_cfg=expected,
            model_name="lstm",
        )

    assert missing == list(REQUIRED_ALIGNED_LSTM_CFG_KEYS)


@pytest.mark.parametrize(
    ("key", "checkpoint_value"),
    [
        ("lstm_bias_mode", "split"),
        ("recurrent_norm", "layernorm"),
        ("recurrent_norm_place", "pre"),
        ("rmsnorm_eps", 1e-6),
    ],
)
def test_lstm_checkpoint_metadata_rejects_aligned_contract_conflicts(
    key, checkpoint_value
):
    checkpoint_cfg = _cfg()
    checkpoint_cfg[key] = checkpoint_value

    with pytest.raises(ValueError, match=key):
        validate_checkpoint_metadata(
            {"cfg": checkpoint_cfg},
            expected_cfg=_cfg(),
            model_name="lstm",
        )


def test_lstm_checkpoint_persists_aligned_metadata(tmp_path: Path):
    cfg = _cfg()
    path = tmp_path / "aligned-lstm.pkl"
    model = torch_backend.build_model(cfg, "lstm")
    torch_backend.save_checkpoint_entry(
        path,
        model,
        None,
        None,
        epoch=1,
        global_step=2,
        cfg=cfg,
    )

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    for key in REQUIRED_ALIGNED_LSTM_CFG_KEYS:
        if key == "rmsnorm_eps":
            assert math.isclose(payload["cfg"][key], cfg[key])
        else:
            assert payload["cfg"][key] == cfg[key]
