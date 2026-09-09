"""Random fixed-length-slice sampling over the packed token stream
(build_training_data.py's tokens.bin) -- nanoGPT-style: the file is
memory-mapped (never fully loaded into RAM), and each `__getitem__` just
slices out `seq_len + 1` contiguous uint16s at a random offset. Input =
slice[:-1], target = slice[1:] (next-token prediction at every position).

Sampling a random offset rather than iterating rows means a training
example can span a `<bos>...<eos>` boundary (i.e. include the tail of one
comment and the head of the next) -- accepted, standard for this packing
scheme (same as nanoGPT/GPT-2 training data prep): the model still sees
`<eos>`/`<bos>` as real tokens marking the boundary, and it's a small
price for never wasting batch capacity on padding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_meta() -> dict:
    return json.loads((DATA_DIR / "tokens_meta.json").read_text(encoding="utf-8"))


class PackedTokenDataset(Dataset):
    def __init__(self, seq_len: int, epoch_size: int):
        """seq_len: length of each training sequence (excluding the extra
        target-shift token). epoch_size: how many random slices count as
        "one epoch" -- arbitrary for this sampling scheme (there's no
        natural notion of a pass over a randomly-sliced stream), just
        controls how often train.py's per-epoch logging/checkpointing
        fires.
        """
        self.meta = load_meta()
        self.seq_len = seq_len
        self.epoch_size = epoch_size
        self.tokens = np.memmap(DATA_DIR / "tokens.bin", dtype=np.uint16, mode="r")
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(f"token stream ({len(self.tokens):,}) shorter than seq_len+1 ({seq_len + 1})")

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.seq_len - 1
        start = np.random.randint(0, max_start)
        window = self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        input_ids = torch.from_numpy(window[:-1].copy())
        targets = torch.from_numpy(window[1:].copy())
        return input_ids, targets
