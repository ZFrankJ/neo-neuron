import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neurons.activations import cortical_activation, fused_cortical_step
from src.neurons.neo_cell import CorticalNeuron, _parse_activation_id


def test_parse_gain_tanh_aliases():
    assert _parse_activation_id("gain_tanh") == 103
    assert _parse_activation_id("one_plus_tanh") == 103
    assert _parse_activation_id("1+tanh") == 103


def test_gain_tanh_forward_shapes():
    cell = CorticalNeuron(input_dim=4, output_dim=6, activation_id="gain_tanh")
    x = torch.randn(3, 4)

    output, state, aux = cell(x)

    assert output.shape == (3, 6)
    assert state.shape == (3, 6)
    assert aux is None


def test_gain_tanh_fused_step_uses_output_gain():
    f_x = torch.tensor([[0.25, -0.5, 1.0]])
    s_prev = torch.tensor([[0.75, 0.25, -1.5]])
    g_x = torch.tensor([[2.0, -3.0, 0.5]])

    output, state = fused_cortical_step(f_x, s_prev, g_x, 103)

    hidden = f_x + s_prev
    expected_state = torch.tanh(hidden)
    expected_output = (1.0 + expected_state) * g_x
    assert torch.allclose(state, expected_state)
    assert torch.allclose(output, expected_output)


def test_gain_tanh_standalone_activation_returns_state():
    x = torch.tensor([[-1.0, 0.0, 1.0]])

    assert torch.allclose(cortical_activation(x, 103), torch.tanh(x))
