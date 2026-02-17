"""LSTM language model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    norm_place: str = "all"


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


def _parse_norm_place(norm_place: str) -> str:
    place = str(norm_place).strip().lower()
    if place in ("all", "pre", "stack"):
        return place
    raise ValueError(f"Unsupported norm placement '{norm_place}'. Use one of: all, pre, stack.")


class LSTMStack(nn.Module):
    """Stacked LSTM core with internal per-layer pre-norm and final stack norm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        output_norm: str,
        layer_dropout: float,
        norm_place: str = "all",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_layers = int(n_layers)
        self.layer_dropout = float(layer_dropout)
        place = _parse_norm_place(norm_place)
        self.pre_norms = nn.ModuleList(
            [
                _build_output_norm(output_norm, d_model) if place in ("all", "pre") else nn.Identity()
                for _ in range(n_layers)
            ]
        )
        self.stack_norm = _build_output_norm(output_norm, d_model) if place in ("all", "stack") else nn.Identity()

        gate_dim = 4 * d_model
        for li in range(n_layers):
            setattr(self, f"weight_ih_l{li}", nn.Parameter(torch.empty(gate_dim, d_model)))
            setattr(self, f"weight_hh_l{li}", nn.Parameter(torch.empty(gate_dim, d_model)))
            setattr(self, f"bias_ih_l{li}", nn.Parameter(torch.empty(gate_dim)))
            setattr(self, f"bias_hh_l{li}", nn.Parameter(torch.empty(gate_dim)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for li in range(self.num_layers):
            w_ih = getattr(self, f"weight_ih_l{li}")
            w_hh = getattr(self, f"weight_hh_l{li}")
            b_ih = getattr(self, f"bias_ih_l{li}")
            b_hh = getattr(self, f"bias_hh_l{li}")
            nn.init.xavier_uniform_(w_ih)
            nn.init.xavier_uniform_(w_hh)
            nn.init.zeros_(b_ih)
            nn.init.zeros_(b_hh)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [T, B, D]
        T, B, _ = x.shape
        if state is None:
            h_prev = [x.new_zeros(B, self.d_model) for _ in range(self.num_layers)]
            c_prev = [x.new_zeros(B, self.d_model) for _ in range(self.num_layers)]
        else:
            h_prev = list(state[0].unbind(0))
            c_prev = list(state[1].unbind(0))

        layer_input = x
        next_h: List[torch.Tensor] = []
        next_c: List[torch.Tensor] = []

        for li in range(self.num_layers):
            x_norm = self.pre_norms[li](layer_input)
            w_ih = getattr(self, f"weight_ih_l{li}")
            w_hh = getattr(self, f"weight_hh_l{li}")
            b_ih = getattr(self, f"bias_ih_l{li}")
            b_hh = getattr(self, f"bias_hh_l{li}")

            # Fuse the input affine across all timesteps: one large matmul per layer.
            x_gates = F.linear(x_norm.reshape(T * B, self.d_model), w_ih, b_ih).reshape(
                T, B, 4 * self.d_model
            )
            h_t = h_prev[li]
            c_t = c_prev[li]
            layer_output = x_norm.new_empty((T, B, self.d_model))
            for t in range(T):
                gates = x_gates[t] + F.linear(h_t, w_hh, b_hh)
                i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                g_gate = torch.tanh(g_gate)
                o_gate = torch.sigmoid(o_gate)
                c_t = f_gate * c_t + i_gate * g_gate
                h_t = o_gate * torch.tanh(c_t)
                layer_output[t] = h_t

            if self.layer_dropout > 0.0 and self.training and li < (self.num_layers - 1):
                layer_input = F.dropout(layer_output, p=self.layer_dropout, training=True)
            else:
                layer_input = layer_output
            next_h.append(h_t)
            next_c.append(c_t)

        y = self.stack_norm(layer_input)
        new_state = (torch.stack(next_h, dim=0), torch.stack(next_c, dim=0))
        return y, new_state


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
        norm_place: str = "all",
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
        self.lstm = LSTMStack(
            d_model=d_model,
            n_layers=n_layers,
            output_norm=self.output_norm_type,
            layer_dropout=dropout,
            norm_place=norm_place,
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
        h0 = torch.zeros(self.n_layers, batch_size, self.d_model, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.d_model, device=device)
        return (h0, c0)

    def forward(self, idx: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # idx: [T, B]
        x = self.emb(idx)
        if not isinstance(self.in_proj, nn.Identity):
            x = self.in_proj(x)
        y, new_state = self.lstm(x, state)
        y = self.drop(y)

        if self.tie_embeddings:
            if not isinstance(self.out_proj, nn.Identity):
                y = self.out_proj(y)
            logits = F.linear(y, self.emb.weight, self.output_bias)
        else:
            logits = self.head(y)
        return logits, new_state
