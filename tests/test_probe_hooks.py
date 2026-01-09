import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import LSTMLM, NeoLM
from src.probe import capture_traces, select_random_neurons


def test_probe_neo_traces():
    vocab = 40
    model = NeoLM(
        vocab_size=vocab,
        d_model=12,
        d_embed=6,
        n_layers=2,
        dropout=0.0,
        tie_embeddings=True,
        cell_type="mode_c",
        cell_kwargs={
        },
        use_checkpoint=False,
    )
    idx = torch.randint(0, vocab, (5, 2))
    layer_indices, selected = select_random_neurons(model, n_layers_to_watch=1, neurons_per_layer=2, seed=0)
    records = capture_traces(model, idx, layer_indices, selected)

    layer = layer_indices[0]
    neuron = records[layer]["neuron_indices"][0]
    assert len(records[layer]["state"][neuron]) == idx.size(0)
    assert len(records[layer]["output"][neuron]) == idx.size(0)


def test_probe_lstm_gates():
    vocab = 50
    model = LSTMLM(
        vocab_size=vocab,
        d_model=10,
        d_embed=6,
        n_layers=2,
        dropout=0.0,
        tie_embeddings=True,
    )
    idx = torch.randint(0, vocab, (6, 2))
    layer_indices, selected = select_random_neurons(model, n_layers_to_watch=1, neurons_per_layer=2, seed=1)
    records = capture_traces(model, idx, layer_indices, selected)

    layer = layer_indices[0]
    neuron = records[layer]["neuron_indices"][0]
    assert len(records[layer]["hidden_state"][neuron]) == idx.size(0)
    assert len(records[layer]["cell_state"][neuron]) == idx.size(0)
    for gate in ("i", "f", "g", "o"):
        assert len(records[layer]["gates"][gate][neuron]) == idx.size(0)
