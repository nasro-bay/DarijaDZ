"""Classic CBOW word embedding model (see ../../plan.md) -- literally
word2vec_attention's CBOWAttention with the self-attention+FFN block
removed: context-word embeddings are mean-pooled directly, the way
original word2vec CBOW does it, instead of being refined by a
Transformer-style block first.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_up_to_multiple(n: int, multiple: int = 64) -> int:
    """Pads the embedding table to a tiling-friendly row count (same
    convention as LM_DiD/scripts/model.py) -- the extra rows never
    receive gradient since token ids never reach them, this only helps
    GPU kernel tiling."""
    return ((n + multiple - 1) // multiple) * multiple


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """x: (batch, window, dim), attention_mask: (batch, window) bool, True
    = real token. Averages only over real (non-padded) positions -- a
    plain mean would be diluted toward zero by padding for short-context
    rows, the common case here (short comments). Same helper as
    word2vec_attention/scripts/model.py's version.
    """
    mask = attention_mask.unsqueeze(-1).to(x.dtype)  # (batch, window, 1)
    summed = (x * mask).sum(dim=1)  # (batch, dim)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (batch, 1) -- avoid div by zero
    return summed / counts


class CBOW(nn.Module):
    def __init__(self, vocab_size: int = 20_000, embed_dim: int = 128, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padded_vocab_size = round_up_to_multiple(vocab_size)
        # The CBOW "hidden layer" -- context-word input embeddings. Kept as
        # "the word vectors" after training, same convention as word2vec_attention.
        self.input_embeddings = nn.Embedding(self.padded_vocab_size, embed_dim, padding_idx=pad_id)
        # Separate output/negative-sampling embedding matrix -- standard
        # word2vec convention (two embedding tables, only the input one is
        # used downstream).
        self.output_embeddings = nn.Embedding(self.padded_vocab_size, embed_dim)

        nn.init.uniform_(self.input_embeddings.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.output_embeddings.weight)

    def encode_context(self, context_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """context_ids: (batch, window) token ids, padded with pad_id.
        attention_mask: (batch, window) bool, True = real (non-padded) token.
        Returns: (batch, embed_dim) -- one pooled vector per row, the plain
        average of the context words' input embeddings (no attention/FFN
        refinement, unlike CBOWAttention).
        """
        x = self.input_embeddings(context_ids)  # (batch, window, embed_dim)
        return masked_mean_pool(x, attention_mask)

    def forward(
        self,
        context_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        center_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """center_ids: (batch,). negative_ids: (batch, num_negative).
        Returns scalar negative-sampling loss (SGNS objective), identical
        shape to word2vec_attention's:
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
