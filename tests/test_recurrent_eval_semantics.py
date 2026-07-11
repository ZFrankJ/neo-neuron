import math

import pytest
import torch
from torch import nn

from src.train.eval import eval_perplexity


class _StateSentinel(nn.Module):
    """Record incoming recurrent state while emitting valid binary logits."""

    def __init__(self):
        super().__init__()
        self.seen_states = []

    def init_state(self, batch_size: int, device: torch.device):
        return torch.zeros(1, batch_size, 1, device=device)

    def forward(self, idx: torch.Tensor, state: torch.Tensor):
        self.seen_states.append(state.detach().cpu().clone())
        logits = torch.zeros(*idx.shape, 2, device=idx.device)
        return logits, state + 1


@pytest.mark.parametrize(
    ("regime", "expected_states"),
    [
        ("block_reset", [0.0, 0.0, 0.0]),
        ("streaming", [0.0, 1.0, 2.0]),
    ],
)
def test_recurrent_eval_regime_controls_state_lifetime(regime, expected_states):
    model = _StateSentinel()
    ids = torch.tensor([0, 1] * 7, dtype=torch.long)
    cfg = {"block_size": 2, "batch_size": 2, "vocab_size": 2, "eval_regime": regime}

    ppl = eval_perplexity(model, ids, cfg, torch.device("cpu"))

    assert ppl == pytest.approx(math.exp(math.log(2.0)))
    assert [float(state[0, 0, 0]) for state in model.seen_states] == expected_states


def test_recurrent_eval_regime_defaults_to_block_reset():
    model = _StateSentinel()
    cfg = {"block_size": 2, "batch_size": 2, "vocab_size": 2}

    eval_perplexity(model, torch.tensor([0, 1] * 7), cfg, torch.device("cpu"))

    assert all(torch.count_nonzero(state) == 0 for state in model.seen_states)


def test_recurrent_eval_regime_rejects_unknown_value():
    with pytest.raises(ValueError, match="eval_regime must be one of"):
        eval_perplexity(
            _StateSentinel(),
            torch.tensor([0, 1] * 7),
            {"block_size": 2, "batch_size": 2, "vocab_size": 2, "eval_regime": "continuous"},
            torch.device("cpu"),
        )


def test_streaming_eval_reduces_batch_size_for_short_split():
    model = _StateSentinel()
    cfg = {"block_size": 128, "batch_size": 16, "vocab_size": 2, "eval_regime": "streaming"}

    ppl = eval_perplexity(model, torch.tensor([0, 1] * 150), cfg, torch.device("cpu"))

    assert ppl == pytest.approx(2.0)
    assert len(model.seen_states) == 1


@pytest.mark.parametrize(
    ("regime", "expected_states"),
    [
        ("block_reset", [0.0, 0.0, 0.0]),
        ("streaming", [0.0, 1.0, 2.0]),
    ],
)
def test_mlx_recurrent_eval_regime_controls_state_lifetime(regime, expected_states):
    mx = pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    class MlxStateSentinel:
        def __init__(self):
            self.seen_states = []

        def eval(self):
            return self

        def init_state(self, batch_size):
            return mx.zeros((1, batch_size, 1))

        def __call__(self, idx, state):
            self.seen_states.append(float(state[0, 0, 0].item()))
            return mx.zeros((*idx.shape, 2)), state + 1

    model = MlxStateSentinel()
    metrics = mlx_backend.eval_metrics_entry(
        model,
        torch.tensor([0, 1] * 7),
        {"block_size": 2, "batch_size": 2, "vocab_size": 2, "eval_regime": regime},
    )

    assert metrics["eval_regime"] == regime
    assert metrics["ppl"] == pytest.approx(2.0)
    assert model.seen_states == expected_states


def test_mlx_streaming_eval_reduces_batch_size_for_short_split():
    mx = pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    class MlxStateSentinel:
        def __init__(self):
            self.call_count = 0

        def eval(self):
            return self

        def init_state(self, batch_size):
            return mx.zeros((1, batch_size, 1))

        def __call__(self, idx, state):
            self.call_count += 1
            return mx.zeros((*idx.shape, 2)), state + 1

    model = MlxStateSentinel()
    metrics = mlx_backend.eval_metrics_entry(
        model,
        torch.tensor([0, 1] * 150),
        {"block_size": 128, "batch_size": 16, "vocab_size": 2, "eval_regime": "streaming"},
    )

    assert metrics["ppl"] == pytest.approx(2.0)
    assert model.call_count == 1
