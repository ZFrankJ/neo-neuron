import sys
from pathlib import Path

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


def test_standard_init_sets_forget_bias_and_gatewise_orthogonal_recurrence():
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


def test_mlx_backend_rejects_pytorch_only_lstm_controls():
    pytest.importorskip("mlx")
    from src.runtime.backends.mlx_backend import build_model as build_mlx_model

    with pytest.raises(ValueError, match="PyTorch-only"):
        build_mlx_model(_config(forget_bias_init=1.0), "lstm")
