"""Classic Skip-gram word embedding model (see ../../plan.md). No window
aggregation/pooling at all, unlike CBOW: a center word's own input
embedding is scored directly against each individual context word's
output embedding (and negatives) -- the reverse prediction direction
from CBOW (center predicts context, not context predicts center).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_up_to_multiple(n: int, multiple: int = 64) -> int:
    """Pads the embedding table to a tiling-friendly row count (same
    convention as LM_DiD/scripts/model.py and CBOW's model.py)."""
    return ((n + multiple - 1) // multiple) * multiple


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int = 20_000, embed_dim: int = 128, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padded_vocab_size = round_up_to_multiple(vocab_size)
        # Center-word input embeddings -- kept as "the word vectors" after
        # training, same convention as CBOW/CBOWAttention.
        self.input_embeddings = nn.Embedding(self.padded_vocab_size, embed_dim, padding_idx=pad_id)
        # Separate output/negative-sampling embedding matrix, scores
        # context words (and negatives) against a center word's vector.
        self.output_embeddings = nn.Embedding(self.padded_vocab_size, embed_dim)

        nn.init.uniform_(self.input_embeddings.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """center_ids: (batch,). context_ids: (batch,) -- ONE context word
        per pair (unlike CBOW's whole window per pair). negative_ids:
        (batch, num_negative). Returns scalar negative-sampling loss
        (SGNS objective), same shape as CBOW's:
        -log sigmoid(center . pos) - sum_k log sigmoid(-center . neg_k)
        """
        center = self.input_embeddings(center_ids)  # (batch, embed_dim)

        pos_emb = self.output_embeddings(context_ids)  # (batch, embed_dim)
        pos_score = (center * pos_emb).sum(dim=-1)  # (batch,)

        neg_emb = self.output_embeddings(negative_ids)  # (batch, num_negative, embed_dim)
        neg_score = torch.bmm(neg_emb, center.unsqueeze(-1)).squeeze(-1)  # (batch, num_negative)

        pos_loss = -F.logsigmoid(pos_score)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=-1)
        return (pos_loss + neg_loss).mean()
