import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import LSTMLM, NeoLM, TransformerLM


def test_lstm_forward_shapes():
    vocab = 50
    model = LSTMLM(
        vocab_size=vocab,
        d_model=16,
        d_embed=8,
        n_layers=2,
        dropout=0.0,
        tie_embeddings=True,
    )
    idx = torch.randint(0, vocab, (6, 3))
    state = model.init_state(idx.size(1), torch.device("cpu"))
    logits, new_state = model(idx, state)
    assert logits.shape == (6, 3, vocab)
    assert isinstance(new_state, tuple)
    assert new_state[0].shape == (2, 3, 16)
    assert new_state[1].shape == (2, 3, 16)


def test_neo_forward_shapes():
    vocab = 60
    model = NeoLM(
        vocab_size=vocab,
        d_model=16,
        d_embed=8,
        n_layers=2,
        dropout=0.0,
        tie_embeddings=True,
        cell_type="mode_c",
        cell_kwargs={
            "exp_factor": 1.0,
            "neg_quad": 1.0,
            "exp_clip": 6.0,
            "eps": 1e-4,
        },
        use_checkpoint=False,
    )
    idx = torch.randint(0, vocab, (5, 2))
    state = model.init_state(idx.size(1), torch.device("cpu"))
    logits, new_state = model(idx, state)
    assert logits.shape == (5, 2, vocab)
    assert new_state.shape == (2, 2, 16)


def test_transformer_forward_shapes():
    vocab = 64
    model = TransformerLM(
        vocab_size=vocab,
        d_model=16,
        n_layers=2,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        tie_embeddings=True,
        max_seq_len=32,
    )
    idx = torch.randint(0, vocab, (7, 4))
    logits, state = model(idx, None)
    assert logits.shape == (7, 4, vocab)
    assert state is None
