import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.backends.torch_backend import build_model


def _config(**overrides):
    cfg = {
        "vocab_size": 32,
        "d_model": 8,
        "d_embed": 4,
        "n_layers": 2,
        "dropout": 0.2,
        "tie_embeddings": True,
        "recurrent_norm": "rmsnorm",
    }
    cfg.update(overrides)
    return cfg


def test_standard_init_sets_xavier_input_forget_bias_and_orthogonal_recurrence(
    monkeypatch,
):
    xavier_sentinel = 0.125

    def fake_xavier_uniform_(value):
        with torch.no_grad():
            return value.fill_(xavier_sentinel)

    monkeypatch.setattr(torch.nn.init, "xavier_uniform_", fake_xavier_uniform_)
    model = build_model(
        _config(forget_bias_init=1.0, recurrent_init="orthogonal"),
        "lstm",
    )

    for layer in range(model.n_layers):
        bias_ih = getattr(model.lstm, f"bias_ih_l{layer}")
        bias_hh = getattr(model.lstm, f"bias_hh_l{layer}")
        input_gate, forget_gate, cell_gate, output_gate = bias_ih.chunk(4)

        torch.testing.assert_close(input_gate, torch.zeros_like(input_gate))
        torch.testing.assert_close(forget_gate, torch.ones_like(forget_gate))
        torch.testing.assert_close(cell_gate, torch.zeros_like(cell_gate))
        torch.testing.assert_close(output_gate, torch.zeros_like(output_gate))
        torch.testing.assert_close(bias_hh, torch.zeros_like(bias_hh))

        input_weights = getattr(model.lstm, f"weight_ih_l{layer}")
        torch.testing.assert_close(
            input_weights,
            torch.full_like(input_weights, xavier_sentinel),
        )
        recurrent = getattr(model.lstm, f"weight_hh_l{layer}")
        identity = torch.eye(model.d_model, dtype=recurrent.dtype)
        for gate_matrix in recurrent.chunk(4, dim=0):
            torch.testing.assert_close(
                gate_matrix.T @ gate_matrix,
                identity,
                rtol=1e-5,
                atol=1e-5,
            )


def test_layer_dropout_is_explicit_and_legacy_default_is_preserved():
    legacy_model = build_model(_config(dropout=0.3), "lstm")
    aligned_model = build_model(_config(dropout=0.3, lstm_layer_dropout=0.0), "lstm")

    assert legacy_model.lstm.layer_dropout == pytest.approx(0.3)
    assert legacy_model.drop.p == pytest.approx(0.3)
    assert aligned_model.lstm.layer_dropout == pytest.approx(0.0)
    assert aligned_model.drop.p == pytest.approx(0.3)


@pytest.mark.parametrize("recurrent_init", ["xavier", "unsupported"])
def test_recurrent_init_rejects_unknown_values(recurrent_init):
    with pytest.raises(ValueError, match="recurrent_init"):
        build_model(_config(recurrent_init=recurrent_init), "lstm")


def test_mlx_standard_init_sets_xavier_input_zero_bias_and_orthogonal_recurrence(
    monkeypatch,
):
    mx = pytest.importorskip("mlx.core")
    mxnn = pytest.importorskip("mlx.nn")
    from src.runtime.backends import mlx_backend

    xavier_sentinel = 0.125

    def fake_glorot_uniform():
        return lambda value: mx.full(value.shape, xavier_sentinel, dtype=value.dtype)

    monkeypatch.setattr(mlx_backend.mxnn.init, "glorot_uniform", fake_glorot_uniform)
    model = mlx_backend.build_model(
        _config(forget_bias_init=1.0, recurrent_init="orthogonal"),
        "lstm",
    )

    for layer in model.lstm_layers:
        np.testing.assert_array_equal(
            np.asarray(layer.Wx),
            np.full(layer.Wx.shape, xavier_sentinel, dtype=np.float32),
        )
        bias = np.asarray(layer.bias)
        input_gate, forget_gate, cell_gate, output_gate = np.split(bias, 4)
        np.testing.assert_array_equal(input_gate, np.zeros_like(input_gate))
        np.testing.assert_array_equal(forget_gate, np.ones_like(forget_gate))
        np.testing.assert_array_equal(cell_gate, np.zeros_like(cell_gate))
        np.testing.assert_array_equal(output_gate, np.zeros_like(output_gate))

        identity = np.eye(model.d_model, dtype=np.float32)
        for gate_matrix in np.split(np.asarray(layer.Wh), 4, axis=0):
            np.testing.assert_allclose(
                gate_matrix.T @ gate_matrix,
                identity,
                rtol=1e-5,
                atol=1e-5,
            )


def test_mlx_standard_init_supports_xavier_recurrent_matrices(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    from src.runtime.backends import mlx_backend

    xavier_sentinel = 0.125

    def fake_glorot_uniform():
        return lambda value: mx.full(value.shape, xavier_sentinel, dtype=value.dtype)

    monkeypatch.setattr(mlx_backend.mxnn.init, "glorot_uniform", fake_glorot_uniform)
    model = mlx_backend.build_model(
        _config(forget_bias_init=0.0, recurrent_init="xavier_uniform"),
        "lstm",
    )

    for layer in model.lstm_layers:
        for value in (layer.Wx, layer.Wh):
            np.testing.assert_array_equal(
                np.asarray(value),
                np.full(value.shape, xavier_sentinel, dtype=np.float32),
            )
        np.testing.assert_array_equal(
            np.asarray(layer.bias),
            np.zeros(layer.bias.shape, dtype=np.float32),
        )


def test_mlx_missing_init_controls_preserve_native_lstm_parameters(monkeypatch):
    pytest.importorskip("mlx.core")
    mxnn = pytest.importorskip("mlx.nn")
    from src.runtime.backends import mlx_backend

    native_lstm = mxnn.LSTM
    native_parameters = []

    def tracked_native_lstm(*args, **kwargs):
        layer = native_lstm(*args, **kwargs)
        native_parameters.append(
            tuple(np.asarray(value).copy() for value in (layer.Wx, layer.Wh, layer.bias))
        )
        return layer

    monkeypatch.setattr(mlx_backend.mxnn, "LSTM", tracked_native_lstm)
    model = mlx_backend.build_model(_config(), "lstm")

    assert len(native_parameters) == model.n_layers
    for layer, expected in zip(model.lstm_layers, native_parameters):
        for actual, native in zip((layer.Wx, layer.Wh, layer.bias), expected):
            np.testing.assert_array_equal(np.asarray(actual), native)


def test_mlx_layer_dropout_is_explicit_and_output_dropout_is_preserved():
    pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    legacy_model = mlx_backend.build_model(_config(dropout=0.3), "lstm")
    aligned_model = mlx_backend.build_model(
        _config(dropout=0.3, lstm_layer_dropout=0.0),
        "lstm",
    )

    assert legacy_model.layer_drop == pytest.approx(0.3)
    assert legacy_model.drop._p_1 == pytest.approx(0.7)
    assert aligned_model.layer_drop == pytest.approx(0.0)
    assert aligned_model.drop._p_1 == pytest.approx(0.7)


@pytest.mark.parametrize("recurrent_init", ["xavier", "unsupported"])
def test_mlx_recurrent_init_rejects_unknown_values(recurrent_init):
    pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    with pytest.raises(ValueError, match="recurrent_init"):
        mlx_backend.build_model(_config(recurrent_init=recurrent_init), "lstm")
