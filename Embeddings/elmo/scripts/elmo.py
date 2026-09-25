"""ELMo representations from a trained biLM (Peters et al. 2018, eq. 1):

    ELMo_k = gamma * sum_j softmax(s)_j * h_{k,j},   j = 0 (token layer), 1, 2

`h_{k,j}` for j >= 1 is the concatenation of the forward and backward states at word k, so each layer is
2*proj dims wide; the token layer (j = 0) is duplicated to the same width. `ScalarMix` holds the task-specific
s_j and gamma (gamma is mandatory per the paper); the biLM itself stays frozen.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import model as M

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
N_LAYERS = 3
BOS_CHAR, EOS_CHAR = 252, 253


class ScalarMix(nn.Module):
    def __init__(self, n_layers: int = N_LAYERS, layer_norm: bool = False):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(()))
        self.layer_norm = layer_norm

    def forward(self, layers: torch.Tensor) -> torch.Tensor:     # (n_layers, B, T, D) -> (B, T, D)
        if self.layer_norm:
            layers = torch.nn.functional.layer_norm(layers, layers.shape[-1:])
        w = torch.softmax(self.w, 0)
        return self.gamma * (w.view(-1, 1, 1, 1) * layers).sum(0)

    def l2_penalty(self) -> torch.Tensor:                         # lambda * ||w||^2 (paper, section 3.2)
        return (self.w ** 2).sum()


class ElmoEncoder:
    """Loads a checkpoint and the vocabulary tables and turns raw text into per-layer word vectors.
    `checkpoint=None` + `cfg=...` builds a randomly initialised biLM (control baseline for probes)."""

    def __init__(self, checkpoint: Path | None, data_dir: Path, device: str = "cuda", cfg: dict | None = None, seed: int = 0):
        if checkpoint is not None:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model = M.BiLM(M.BiLMConfig(**{**ck["cfg"], "filters": tuple(ck["cfg"]["filters"])}))
            self.model.load_state_dict(ck["model"])
            self.info = {k: ck.get(k) for k in ("step", "epoch", "history")}
        else:
            torch.manual_seed(seed)
            self.model = M.build(cfg)
            self.info = {}
        self.device = torch.device(device)
        self.model.to(self.device).eval().requires_grad_(False)
        self.data_dir = Path(data_dir)
        self.model.set_tables(np.load(self.data_dir / "char_table.npy"), np.load(self.data_dir / "type_to_out.npy"))
        self.dim = 2 * self.model.cfg.proj

    def word_chars(self, word: str) -> np.ndarray:
        """Char ids exactly as prepare_data.py builds them: BOW=2, <=18 codepoints (4.. by frequency,
        1 = unknown), EOW=3, zero padded."""
        if not hasattr(self, "_cmap"):
            keep = json.loads((self.data_dir / "char_vocab.json").read_text(encoding="utf-8"))
            self._cmap = {c: i + 4 for i, c in enumerate(keep)}
        ids = [2] + [self._cmap.get(c, 1) for c in word[:M.MAX_CHARS - 2]] + [3]
        return np.array(ids + [0] * (M.MAX_CHARS - len(ids)), dtype=np.uint8)

    @staticmethod
    def _boundary(kind: int) -> np.ndarray:
        row = np.zeros(M.MAX_CHARS, dtype=np.uint8)
        row[0], row[1], row[2] = 2, BOS_CHAR if kind == 0 else EOS_CHAR, 3
        return row

    @torch.no_grad()
    def _run(self, sentences: list[list[str]], max_len: int = 64, chunk: int = 256):
        """Yields (indices, layers) per same-length group of at most `chunk` sentences; `layers` is a GPU
        tensor (3, B, L, D): layer 0 = token repr (fwd/bwd copy), 1 and 2 = concatenated fwd|bwd states.
        Boundary markers are stripped. Sentences are truncated to `max_len`; empty ones are skipped."""
        by_len: dict[int, list[int]] = {}
        for i, s in enumerate(sentences):
            if s:
                by_len.setdefault(min(len(s), max_len), []).append(i)
        for L, all_idxs in by_len.items():
            for c in range(0, len(all_idxs), chunk):
                idxs = all_idxs[c:c + chunk]
                chars = np.stack([np.stack([self._boundary(0)] + [self.word_chars(w) for w in sentences[i][:L]] + [self._boundary(1)])
                                  for i in idxs])                                # (B, L+2, MAX_CHARS)
                B = len(idxs)
                flat = torch.from_numpy(chars.reshape(-1, M.MAX_CHARS)).to(self.device).long()
                x = self.model.cnn(flat).view(B, L + 2, -1).transpose(0, 1).contiguous()    # (T, B, proj)
                f1, f2 = self.model.fwd(x)
                b1, b2 = self.model.bwd(x.flip(0))
                b1, b2 = b1.flip(0), b2.flip(0)
                layers = torch.stack([torch.cat([x, x], -1), torch.cat([f1, b1], -1), torch.cat([f2, b2], -1)])
                yield idxs, layers[:, 1:-1].transpose(1, 2)                      # (3, B, L, D)

    def encode(self, sentences: list[list[str]], max_len: int = 64) -> list[torch.Tensor | None]:
        """Per sentence a CPU tensor (3, n_words, D) -- memory-heavy; prefer the pooled/positional variants."""
        out: list[torch.Tensor | None] = [None] * len(sentences)
        for idxs, layers in self._run(sentences, max_len):
            for j, i in enumerate(idxs):
                out[i] = layers[:, j].cpu()
        return out

    def encode_positions(self, sentences: list[list[str]], positions: list[int]) -> torch.Tensor:
        """Vectors of one chosen word per sentence -> CPU tensor (N, 3, D) (zeros for skipped sentences)."""
        out = torch.zeros(len(sentences), 3, self.dim)
        for idxs, layers in self._run(sentences):
            pos = torch.tensor([positions[i] for i in idxs], device=self.device)
            picked = layers[:, torch.arange(len(idxs), device=self.device), pos]     # (3, B, D)
            out[torch.tensor(idxs)] = picked.transpose(0, 1).cpu()
        return out

    def encode_mean(self, sentences: list[list[str]], max_len: int = 64) -> torch.Tensor:
        """Mean over words per layer -> CPU tensor (N, 3, D): the sentence features for classification."""
        out = torch.zeros(len(sentences), 3, self.dim)
        for idxs, layers in self._run(sentences, max_len):
            out[torch.tensor(idxs)] = layers.mean(2).transpose(0, 1).cpu()
        return out


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)
