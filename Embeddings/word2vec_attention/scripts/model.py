"""CBOW + self-attention word embedding model (see ../plan.md).

Standard CBOW replaces the naive "average the context word embeddings"
step with a single Transformer-style block (self-attention -> residual +
LayerNorm -> feedforward -> residual + LayerNorm) before pooling and
scoring against the vocabulary via negative sampling -- the CBOW "hidden
layer" (the input embedding table) is kept exactly as in classic word2vec,
just processed further before prediction instead of being averaged
directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_up_to_multiple(n: int, multiple: int = 64) -> int:
    """Pads the embedding table to a tiling-friendly row count (same
    convention as LM_DiD/scripts/model.py and CBOW/SkipGram's model.py)."""
    return ((n + multiple - 1) // multiple) * multiple


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """x: (batch, seq, dim), attention_mask: (batch, seq) bool, True = real
    token. Averages only over real (non-padded) positions -- a plain mean
    would be diluted toward zero by padding for short-context rows, which
    are the common case here (short comments), not the exception.
    """
    mask = attention_mask.unsqueeze(-1).to(x.dtype)  # (batch, seq, 1)
    summed = (x * mask).sum(dim=1)  # (batch, dim)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (batch, 1) -- avoid div by zero
    return summed / counts


class CBOWAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int = 20_000,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padded_vocab_size = round_up_to_multiple(vocab_size)
        # The CBOW "hidden layer" -- context-word input embeddings.
        self.input_embeddings = nn.Embedding(self.padded_vocab_size, embed_dim, padding_idx=pad_id)
        # Separate output/negative-sampling embedding matrix -- standard
        # word2vec convention (two embedding tables, only the input one is
        # kept as "the word vectors" after training).
        self.output_embeddings = nn.Embedding(self.padded_vocab_size, embed_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.uniform_(self.input_embeddings.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.output_embeddings.weight)

    def encode_context(self, context_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """context_ids: (batch, window) token ids, padded with pad_id.
        attention_mask: (batch, window) bool, True = real (non-padded) token.
        Returns: (batch, embed_dim) -- one pooled vector per row.
        """
        x = self.input_embeddings(context_ids)  # (batch, window, embed_dim)

        # nn.MultiheadAttention's key_padding_mask: True = ignore that position.
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=~attention_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))  # residual + LayerNorm (attention block)

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))  # residual + LayerNorm (FFN block)

        return masked_mean_pool(x, attention_mask)

    def forward(
        self,
        context_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        center_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """center_ids: (batch,). negative_ids: (batch, num_negative).
        Returns scalar negative-sampling loss (SGNS objective):
        -log sigmoid(ctx . pos) - sum_k log sigmoid(-ctx . neg_k)
        """
        ctx = self.encode_context(context_ids, attention_mask)  # (batch, embed_dim)

        pos_emb = self.output_embeddings(center_ids)  # (batch, embed_dim)
        pos_score = (ctx * pos_emb).sum(dim=-1)  # (batch,)

        neg_emb = self.output_embeddings(negative_ids)  # (batch, num_negative, embed_dim)
        neg_score = torch.bmm(neg_emb, ctx.unsqueeze(-1)).squeeze(-1)  # (batch, num_negative)

        pos_loss = -F.logsigmoid(pos_score)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=-1)
        return (pos_loss + neg_loss).mean()
