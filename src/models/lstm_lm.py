"""LSTM language model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LSTMConfig:
    vocab_size: int
    d_model: int
    d_embed: int
    n_layers: int
    dropout: float
    tie_embeddings: bool = True
    output_norm: str = "layernorm"


def _build_output_norm(norm_type: str, d_model: int) -> nn.Module:
    norm = str(norm_type).strip().lower()
    if norm in ("none", "off", "identity"):
        return nn.Identity()
    if norm in ("layernorm", "layer_norm", "ln"):
        return nn.LayerNorm(d_model)
    if norm in ("rmsnorm", "rms_norm", "rms"):
        rms_norm = getattr(nn, "RMSNorm", None)
        if rms_norm is None:
            raise ValueError("RMSNorm is not available in this torch version.")
        return rms_norm(d_model)
    raise ValueError(f"Unsupported output_norm '{norm_type}'.")


class LSTMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_embed: int,
        n_layers: int,
        dropout: float,
        tie_embeddings: bool = True,
        output_norm: str = "layernorm",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.tie_embeddings = tie_embeddings
        self.output_norm_type = str(output_norm)

        self.emb = nn.Embedding(vocab_size, d_embed)
        self.in_proj = nn.Linear(d_embed, d_model) if d_embed != d_model else nn.Identity()
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, dropout=dropout, batch_first=False)
        self.out_norm = _build_output_norm(self.output_norm_type, d_model)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_embed) if d_embed != d_model else nn.Identity()

        if tie_embeddings:
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
            self.head = None
        else:
            self.head = nn.Linear(d_model, vocab_size)
            self.output_bias = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_state(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.n_layers, batch_size, self.d_model, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.d_model, device=device)
        return (h0, c0)

    def forward(self, idx: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # idx: [T, B]
        x = self.emb(idx)
        if not isinstance(self.in_proj, nn.Identity):
            x = self.in_proj(x)
        y, new_state = self.lstm(x, state)
        y = self.out_norm(y)
        y = self.drop(y)

        if self.tie_embeddings:
            if not isinstance(self.out_proj, nn.Identity):
                y = self.out_proj(y)
            logits = F.linear(y, self.emb.weight, self.output_bias)
        else:
            logits = self.head(y)
        return logits, new_state
