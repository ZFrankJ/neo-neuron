"""Causal Transformer language model."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ff_mult: int
    dropout: float
    tie_embeddings: bool = True


def _make_causal_mask(T: int, device: torch.device):
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, optimized_causal: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = float(dropout)
        self.optimized_causal = bool(optimized_causal)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=not optimized_causal)
        self.out_proj = nn.Linear(d_model, d_model, bias=not optimized_causal)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # x: [T, B, D]
        T, B, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(T, B, self.n_heads, self.head_dim).transpose(1, 2)  # [T, H, B, Hd]
        k = k.view(T, B, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(T, B, self.n_heads, self.head_dim).transpose(1, 2)

        q = q.transpose(0, 2)  # [B, H, T, Hd]
        k = k.transpose(0, 2)
        v = v.transpose(0, 2)

        if self.optimized_causal:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            return self.dropout(self.out_proj(out.transpose(0, 1)))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, T, Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = out.transpose(0, 1)  # [T, B, D]
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ff_mult: int, dropout: float, optimized_causal: bool = False
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout, optimized_causal)
        self.ln2 = nn.LayerNorm(d_model)
        mlp = [
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        ]
        if not optimized_causal:
            mlp.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        tie_embeddings: bool = True,
        max_seq_len: int = 2048,
        transformer_variant: str = "legacy",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_mult = ff_mult
        self.tie_embeddings = tie_embeddings
        self.transformer_variant = str(transformer_variant).strip().lower()
        if self.transformer_variant not in {"legacy", "gpt2"}:
            raise ValueError("transformer_variant must be 'legacy' or 'gpt2'")

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    ff_mult,
                    dropout,
                    optimized_causal=self.transformer_variant == "gpt2",
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

        if tie_embeddings:
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
            self.head = None
        else:
            self.head = nn.Linear(d_model, vocab_size)
            self.output_bias = None

        self.apply(self._init_weights)
        if self.transformer_variant == "gpt2":
            residual_std = 0.02 / (2 * n_layers) ** 0.5
            for block in self.blocks:
                nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.mlp[3].weight, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.transformer_variant == "gpt2":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, state=None):
        # idx: [T, B]
        T, B = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.emb(idx) + self.pos_emb(pos)[:, None, :]
        x = self.drop(x)
        mask = _make_causal_mask(T, device=idx.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.ln_f(x)

        if self.tie_embeddings:
            logits = F.linear(x, self.emb.weight, self.output_bias)
        else:
            logits = self.head(x)
        return logits, state
