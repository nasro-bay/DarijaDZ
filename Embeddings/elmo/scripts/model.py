"""ELMo-style bidirectional language model (Peters et al. 2018), plan.md section 3.

Structure (per the paper, scaled to this machine):
- token representation shared by both directions: character embeddings -> conv filters of widths
  1..7 -> max-pool over character positions -> 2 highway layers -> linear to `proj`;
- a **forward** stack and a **backward** stack, deliberately separate modules (a single bidirectional
  `nn.LSTM` would let layer 2 of the forward LM see future words). Each stack = 2 x projected LSTM
  (`hidden` cells, `proj`-dim output) with a residual connection from layer 1 to layer 2;
- one softmax layer shared by both directions;
- loss = forward CE (state at k predicts token k+1) + backward CE (state at k predicts token k-1).

Implementation choices that come from this project's optimization work:
- **Exact-length batches** (see dataset.py): every sequence in a batch has the same length, so there
  is no padding and no masking; the backward stack is just the forward computation on the
  time-flipped input.
- **Time-major (T, B, .) layout** end to end (cuDNN LSTM's native layout -- no transposes).
- **Char-CNN once per unique word type in the batch** (`torch.unique` + gather): word frequencies are
  Zipfian, so a batch of ~8K positions contains far fewer distinct types; gradients flow through the
  gather.
- Output targets are derived on the device from input ids via `type_to_out`, so batches carry only
  ids.
- Dims are multiples of 64 (hidden, proj, filter counts, vocab padded to 20,032).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_CHARS = 20
PAD_OUT, UNK_OUT = 0, 1


@dataclass
class BiLMConfig:
    hidden: int = 2048
    proj: int = 256
    filters: tuple = (16, 16, 32, 64, 128, 256, 512)   # widths 1..7 -> 1,024 filters
    char_dim: int = 16
    n_chars: int = 256
    vocab_out: int = 20_032                             # 20,000 real + padding to a multiple of 64
    dropout: float = 0.1
    highway_layers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


CONFIGS = {
    "S": dict(hidden=1024, proj=256, filters=(8, 8, 16, 32, 64, 128, 256)),
    "M": dict(hidden=2048, proj=256, filters=(16, 16, 32, 64, 128, 256, 512)),
    "M+": dict(hidden=2048, proj=512, filters=(16, 16, 32, 64, 128, 256, 512)),
}


class Highway(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 2 * d)
        with torch.no_grad():
            self.lin.bias[d:].fill_(-2.0)   # gate starts mostly "carry" (Srivastava et al. 2015)

    def forward(self, x):
        h, g = self.lin(x).chunk(2, -1)
        g = torch.sigmoid(g)
        return g * F.relu(h) + (1 - g) * x


class CharCNN(nn.Module):
    def __init__(self, cfg: BiLMConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.n_chars, cfg.char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(cfg.char_dim, f, w) for w, f in enumerate(cfg.filters, 1)])
        d = sum(cfg.filters)
        self.highway = nn.Sequential(*[Highway(d) for _ in range(cfg.highway_layers)])
        self.proj = nn.Linear(d, cfg.proj)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:      # (U, MAX_CHARS) long -> (U, proj)
        x = self.emb(chars).transpose(1, 2)
        pooled = [F.relu(c(x)).max(-1).values for c in self.convs]
        return self.proj(self.highway(torch.cat(pooled, -1)))


class Stack(nn.Module):
    """One direction: two projected LSTMs, residual connection layer 1 -> layer 2."""

    def __init__(self, cfg: BiLMConfig):
        super().__init__()
        self.l1 = nn.LSTM(cfg.proj, cfg.hidden, proj_size=cfg.proj)     # time-major (T, B, .)
        self.l2 = nn.LSTM(cfg.proj, cfg.hidden, proj_size=cfg.proj)
        self.drop = nn.Dropout(cfg.dropout)
        for lstm in (self.l1, self.l2):
            with torch.no_grad():
                lstm.bias_ih_l0[cfg.hidden:2 * cfg.hidden].fill_(1.0)   # forget-gate bias 1

    def forward(self, x: torch.Tensor):
        h1 = self.l1(x)[0]
        h2 = self.l2(self.drop(h1))[0] + h1
        return h1, h2


class BiLM(nn.Module):
    def __init__(self, cfg: BiLMConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn = CharCNN(cfg)
        self.fwd = Stack(cfg)
        self.bwd = Stack(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.softmax = nn.Linear(cfg.proj, cfg.vocab_out)               # shared by both directions
        # Lookup tables live on the device but are not part of the checkpoint (they come from data/).
        self.register_buffer("char_table", torch.zeros(1, MAX_CHARS, dtype=torch.uint8), persistent=False)
        self.register_buffer("type_to_out", torch.zeros(1, dtype=torch.long), persistent=False)

    def set_tables(self, char_table: np.ndarray, type_to_out: np.ndarray) -> None:
        dev = self.softmax.weight.device
        self.char_table = torch.from_numpy(np.ascontiguousarray(char_table)).to(dev)
        self.type_to_out = torch.from_numpy(type_to_out.astype(np.int64)).to(dev)

    # ---- pieces ---------------------------------------------------------------------------------
    def token_repr(self, ids_tb: torch.Tensor) -> torch.Tensor:
        """(T, B) input type ids -> (T, B, proj). Char-CNN runs once per unique type in the batch."""
        flat = ids_tb.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rep = self.cnn(self.char_table[uniq].long())
        return rep[inv].view(*ids_tb.shape, -1)

    def layers(self, in_ids: torch.Tensor):
        """in_ids (B, T) -> token layer x (T,B,proj), forward (h1,h2), backward (h1,h2), all in the
        ORIGINAL time order (the backward stack runs on the flipped sequence and is flipped back)."""
        ids = in_ids.t().contiguous()
        x = self.token_repr(ids)
        f1, f2 = self.fwd(x)
        b1, b2 = self.bwd(x.flip(0))
        return x, (f1, f2), (b1.flip(0), b2.flip(0))

    def forward(self, in_ids: torch.Tensor, detailed: bool = False):
        """in_ids (B, T) long. Returns (loss_forward, loss_backward) means; with `detailed=True`,
        per-position losses and target ids (used for held-out perplexity with/without <unk>)."""
        out_ids = self.type_to_out[in_ids].t()                            # (T, B) targets
        _, (_, f2), (_, b2) = self.layers(in_ids)
        V = self.cfg.vocab_out
        logits_f = self.softmax(self.drop(f2[:-1])).float().reshape(-1, V)   # state k -> token k+1
        logits_b = self.softmax(self.drop(b2[1:])).float().reshape(-1, V)    # state k -> token k-1
        tgt_f, tgt_b = out_ids[1:].reshape(-1), out_ids[:-1].reshape(-1)
        if detailed:
            return (F.cross_entropy(logits_f, tgt_f, reduction="none"), tgt_f,
                    F.cross_entropy(logits_b, tgt_b, reduction="none"), tgt_b)
        return F.cross_entropy(logits_f, tgt_f), F.cross_entropy(logits_b, tgt_b)

    def count_params(self) -> dict:
        n = lambda m: sum(p.numel() for p in m.parameters())
        total = n(self)
        return {"total": total, "softmax": n(self.softmax), "lstm": n(self.fwd) + n(self.bwd),
                "char_cnn": n(self.cnn), "non_softmax": total - n(self.softmax)}


def build(config: str | dict = "M", **overrides) -> BiLM:
    kw = dict(CONFIGS[config]) if isinstance(config, str) else dict(config)
    kw.update(overrides)
    kw["filters"] = tuple(kw["filters"])
    return BiLM(BiLMConfig(**kw))
