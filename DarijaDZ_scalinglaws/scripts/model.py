"""Decoder-only Transformer LM for the DarijaDZ scaling-laws lab (Part 1:
data scaling). Adapted directly from `LM_DiD/scripts/model.py` -- same
architectural tweaks (RoPE, pre-norm, RMSNorm, no biases, SwiGLU, fast
SDPA attention, tiling-friendly shapes, z-loss) -- with two deliberate
differences for this experiment:

1. **`d_model=128`, not 256** -- the scaling-laws lab wants the smallest
   model that stays safely in the data-limited regime up to a 10M-token
   ceiling, not LM_DiD's own (larger) production config. `num_heads=2`
   keeps `head_dim=64` (the flash-attention tiling convention LM_DiD's
   docstring explains), rather than the `num_heads=4` that `d_model=128`
   would otherwise default to (which gives `head_dim=32`, off the
   tile-friendly size).
2. **Embeddings are NOT tied** (Press & Wolf 2017's trick, which LM_DiD
   uses) -- kept as two separate weight matrices (`tok_embedding` and
   `lm_head`) specifically so each one can be initialized independently
   from `Embeddings/word2vec/skip-gram`'s two matrices (its
   `input_embeddings`/`output_embeddings` split maps naturally onto this
   model's input-embedding/LM-head split -- see `load_word2vec_init()`).

See `plan.md` in this directory for the full rationale, including why
`unigram_20000` (not `bpe_10000`, the lab doc's original suggestion) is
this experiment's tokenizer -- it's the exact tokenizer the reused
word2vec checkpoint was trained against, which is what makes vocabulary
ids line up between the two models without any remapping.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_up_to_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10_000.0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x1 * sin + x2 * cos
    out = torch.stack([rotated_1, rotated_2], dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv_proj(x).view(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(attn_out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, multiple_of: int = 64):
        super().__init__()
        hidden = round_up_to_multiple(int(2 / 3 * 4 * d_model), multiple_of)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 2,
        num_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
        z_loss_weight: float = 1e-4,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.z_loss_weight = z_loss_weight
        self.head_dim = d_model // num_heads

        self.padded_vocab_size = round_up_to_multiple(vocab_size, 64)
        self.real_vocab_size = vocab_size

        self.tok_embedding = nn.Embedding(self.padded_vocab_size, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)

        # NOT tied (unlike LM_DiD) -- a separate LM head so it can be
        # independently initialized from word2vec's output_embeddings.
        self.lm_head = nn.Linear(d_model, self.padded_vocab_size, bias=False)

        cos, sin = build_rope_cache(max_seq_len, self.head_dim, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def load_word2vec_init(self, skipgram_checkpoint_path: str | Path) -> dict:
        """Initializes `tok_embedding` from word2vec skip-gram's
        `input_embeddings` and `lm_head` from its `output_embeddings` --
        both (vocab=20000, dim=128) matrices, which is why this model uses
        `d_model=128` and the `unigram_20000` tokenizer specifically (see
        module docstring / plan.md). Only rows for the real (non-padded)
        vocab are copied; the padded rows keep their random init since
        they're never real tokens. Returns a small report dict for logging
        rather than asserting silently.
        """
        ckpt = torch.load(skipgram_checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        in_emb = sd["input_embeddings.weight"]
        out_emb = sd["output_embeddings.weight"]
        assert in_emb.shape[1] == self.tok_embedding.weight.shape[1], (
            f"embed_dim mismatch: word2vec={in_emb.shape[1]} vs model d_model={self.tok_embedding.weight.shape[1]}"
        )
        n = min(in_emb.shape[0], self.real_vocab_size)
        self.tok_embedding.weight[:n].copy_(in_emb[:n])
        self.lm_head.weight[:n].copy_(out_emb[:n])
        return {
            "word2vec_vocab_rows": int(in_emb.shape[0]),
            "rows_copied": int(n),
            "model_real_vocab_size": self.real_vocab_size,
            "model_padded_vocab_size": self.padded_vocab_size,
        }

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = input_ids.shape
        assert t <= self.max_seq_len, f"sequence length {t} exceeds max_seq_len={self.max_seq_len}"

        x = self.tok_embedding(input_ids)
        cos = self.rope_cos[:t].to(x.device)
        sin = self.rope_sin[:t].to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None, None

        flat_logits = logits.view(-1, self.padded_vocab_size)
        flat_targets = targets.view(-1)
        ce_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.pad_id)

        logsumexp = torch.logsumexp(flat_logits, dim=-1)
        pad_mask = flat_targets != self.pad_id
        z_loss = logsumexp[pad_mask].pow(2).mean()

        # Return the pure cross-entropy loss separately from the
        # z-loss-augmented training objective: z-loss is a training
        # stabilizer (keeps logit magnitudes bounded), not part of the
        # actual predictive loss a scaling law should be fit against --
        # conflating them would bias every fitted exponent by whatever
        # z_loss_weight happens to be set to.
        return logits, ce_loss, ce_loss + self.z_loss_weight * z_loss
