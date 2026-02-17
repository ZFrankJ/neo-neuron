"""Neo recurrent language model."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..neurons.neo_stack import CorticalRecurrentStack


@dataclass
class NeoConfig:
    vocab_size: int
    d_model: int
    d_embed: int
    n_layers: int
    dropout: float
    tie_embeddings: bool
    cell_type: str
    cell_kwargs: Dict[str, float]
    output_norm: str = "layernorm"
    norm_place: str = "all"
    use_checkpoint: bool = False


class NeoLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_embed: int,
        n_layers: int,
        dropout: float,
        tie_embeddings: bool,
        cell_type: str,
        cell_kwargs: Dict[str, float],
        output_norm: str = "layernorm",
        norm_place: str = "all",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.tie_embeddings = tie_embeddings

        self.emb = nn.Embedding(vocab_size, d_embed)
        self.in_proj = nn.Linear(d_embed, d_model) if d_embed != d_model else nn.Identity()
        self.recurrent = CorticalRecurrentStack(
            d_model=d_model,
            n_layers=n_layers,
            cell_type=cell_type,
            output_norm=output_norm,
            norm_place=norm_place,
            cell_kwargs=cell_kwargs,
            use_checkpoint=use_checkpoint,
        )
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
        return self.recurrent.init_state(batch_size, device=device, dtype=self.emb.weight.dtype)

    def forward(self, idx: torch.Tensor, state: Optional[torch.Tensor]):
        x = self.emb(idx)
        if not isinstance(self.in_proj, nn.Identity):
            x = self.in_proj(x)
        y, new_state = self.recurrent(x, state)
        y = self.drop(y)

        if self.tie_embeddings:
            if not isinstance(self.out_proj, nn.Identity):
                y = self.out_proj(y)
            logits = F.linear(y, self.emb.weight, self.output_bias)
        else:
            logits = self.head(y)
        return logits, new_state
