from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.runtime.backends import torch_backend
from src.runtime.checkpoint_compat import map_model_state
from src.train.optim import build_optimizer


def _cfg(**overrides):
    cfg = {
        "model_name": "lstm",
        "vocab_size": 16,
        "d_model": 4,
        "d_embed": 4,
        "n_layers": 1,
        "dropout": 0.0,
        "tie_embeddings": True,
        "recurrent_norm": "none",
        "recurrent_norm_place": "all",
        "reference_backend": "mlx",
        "weight_decay_policy": "table",
        "lr": 1e-3,
    }
    cfg.update(overrides)
    return cfg


def test_single_bias_mode_matches_mapped_mlx_effective_bias_gradient():
    mx = pytest.importorskip("mlx.core")
    mxnn = pytest.importorskip("mlx.nn")
    mlx_utils = pytest.importorskip("mlx.utils")
    from src.runtime.backends import mlx_backend

    cfg = _cfg(lstm_bias_mode="single")
    torch_model = torch_backend.build_model(cfg, "lstm")
    mlx_model = mlx_backend.build_model(cfg, "lstm")
    mlx_state = {
        name: np.asarray(value)
        for name, value in mlx_utils.tree_flatten(mlx_model.parameters())
    }
    mapped, _ = map_model_state(
        model_name="lstm",
        src_backend="mlx",
        dst_backend="torch",
        src_state_np=mlx_state,
        dst_template=torch_model.state_dict(),
        cfg=cfg,
    )
    torch_model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in mapped.items()}
    )

    x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    y_np = np.array([[2, 3], [4, 5], [6, 7]], dtype=np.int64)
    torch_logits, _ = torch_model(torch.from_numpy(x_np).long(), None)
    torch_loss = F.cross_entropy(
        torch_logits.reshape(-1, cfg["vocab_size"]),
        torch.from_numpy(y_np).reshape(-1),
    )
    torch_loss.backward()

    def mlx_loss_fn(batch_x, batch_y):
        logits, _ = mlx_model(batch_x, None)
        return mx.mean(
            mxnn.losses.cross_entropy(
                logits.reshape(-1, cfg["vocab_size"]),
                batch_y.reshape(-1),
            )
        )

    mlx_loss, mlx_grads = mxnn.value_and_grad(mlx_model, mlx_loss_fn)(
        mx.array(x_np), mx.array(y_np.astype(np.int32))
    )
    mx.eval(mlx_loss, mlx_grads)
    mlx_bias_grad = dict(mlx_utils.tree_flatten(mlx_grads))["lstm_layers.0.bias"]

    trainable_bias_grads = [
        parameter.grad
        for name, parameter in torch_model.named_parameters()
        if name in {"lstm.bias_ih_l0", "lstm.bias_hh_l0"} and parameter.requires_grad
    ]
    effective_torch_grad = sum(trainable_bias_grads)

    assert float(torch_loss.detach()) == pytest.approx(float(mlx_loss.item()), abs=1e-6)
    np.testing.assert_allclose(
        effective_torch_grad.detach().numpy(),
        np.asarray(mlx_bias_grad),
        rtol=1e-5,
        atol=1e-6,
    )
    assert len(trainable_bias_grads) == 1
    assert torch_model.lstm.bias_hh_l0.grad is None


def test_single_bias_mode_matches_mlx_trainable_parameter_count():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    from src.runtime.backends import mlx_backend

    cfg = _cfg(lstm_bias_mode="single", n_layers=2)
    torch_model = torch_backend.build_model(cfg, "lstm")
    mlx_model = mlx_backend.build_model(cfg, "lstm")

    assert torch_backend.count_params(torch_model) == mlx_backend.count_params(mlx_model)
    assert mlx_model.lstm_bias_mode == "single"

    with pytest.raises(ValueError, match="supports only lstm_bias_mode='single'"):
        mlx_backend.build_model(_cfg(lstm_bias_mode="split"), "lstm")


def test_bias_mode_defaults_to_legacy_split_and_rejects_unknown_values():
    legacy = torch_backend.build_model(_cfg(reference_backend=None), "lstm")
    single = torch_backend.build_model(_cfg(lstm_bias_mode="single"), "lstm")

    assert legacy.lstm.lstm_bias_mode == "split"
    assert legacy.lstm.bias_ih_l0.requires_grad
    assert legacy.lstm.bias_hh_l0.requires_grad
    assert single.lstm.lstm_bias_mode == "single"
    assert single.lstm.bias_ih_l0.requires_grad
    assert not single.lstm.bias_hh_l0.requires_grad

    optimizer = build_optimizer(single, _cfg(lstm_bias_mode="single"))
    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(single.lstm.bias_ih_l0) in optimized_parameters
    assert id(single.lstm.bias_hh_l0) not in optimized_parameters

    with pytest.raises(ValueError, match="lstm_bias_mode"):
        torch_backend.build_model(_cfg(lstm_bias_mode="combined"), "lstm")


def test_single_mode_loads_legacy_split_bias_checkpoint_for_evaluation(tmp_path: Path):
    legacy_cfg = _cfg(reference_backend=None)
    legacy = torch_backend.build_model(legacy_cfg, "lstm")
    with torch.no_grad():
        legacy.lstm.bias_ih_l0.copy_(torch.linspace(-0.4, 0.3, 16))
        legacy.lstm.bias_hh_l0.copy_(torch.linspace(0.2, -0.1, 16))
    checkpoint = tmp_path / "legacy-split.pkl"
    torch_backend.save_checkpoint_entry(
        checkpoint,
        legacy,
        None,
        None,
        epoch=1,
        global_step=2,
        cfg=legacy_cfg,
    )

    single_cfg = _cfg(lstm_bias_mode="single")
    single = torch_backend.build_model(single_cfg, "lstm")
    torch_backend.load_checkpoint_entry(
        checkpoint,
        single,
        device=torch.device("cpu"),
        cfg=single_cfg,
    )

    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long)
    legacy.eval()
    single.eval()
    with torch.no_grad():
        legacy_logits, legacy_state = legacy(x, None)
        single_logits, single_state = single(x, None)

    torch.testing.assert_close(single_logits, legacy_logits)
    torch.testing.assert_close(single_state[0], legacy_state[0])
    torch.testing.assert_close(single_state[1], legacy_state[1])
    torch.testing.assert_close(single.lstm.bias_ih_l0, legacy.lstm.bias_ih_l0)
    torch.testing.assert_close(single.lstm.bias_hh_l0, legacy.lstm.bias_hh_l0)
    assert not single.lstm.bias_hh_l0.requires_grad


def test_legacy_split_bias_conversion_warns_about_optimizer_resume():
    d_model = 2
    gate_dim = 4 * d_model
    src = {
        "lstm.weight_ih_l0": np.zeros((gate_dim, d_model), dtype=np.float32),
        "lstm.weight_hh_l0": np.zeros((gate_dim, d_model), dtype=np.float32),
        "lstm.bias_ih_l0": np.full((gate_dim,), 0.25, dtype=np.float32),
        "lstm.bias_hh_l0": np.full((gate_dim,), 0.5, dtype=np.float32),
    }
    dst = {
        "lstm_layers.0.Wx": np.zeros((gate_dim, d_model), dtype=np.float32),
        "lstm_layers.0.Wh": np.zeros((gate_dim, d_model), dtype=np.float32),
        "lstm_layers.0.bias": np.zeros((gate_dim,), dtype=np.float32),
    }

    with pytest.warns(UserWarning, match="optimizer resume is not equivalent"):
        mapped, warnings = map_model_state(
            model_name="lstm",
            src_backend="torch",
            dst_backend="mlx",
            src_state_np=src,
            dst_template=dst,
            cfg={"n_layers": 1},
        )

    np.testing.assert_array_equal(
        mapped["lstm_layers.0.bias"],
        src["lstm.bias_ih_l0"] + src["lstm.bias_hh_l0"],
    )
    assert any("optimizer resume is not equivalent" in warning for warning in warnings)
