"""Near-duplicate detection over character shingles via MinHash/LSH.

Exact-hash dedup is intentionally not used — near-dup (MinHash/LSH) alone
is applied, since it also catches exact duplicates as a special case. The
LSH index is persisted so dedup is cumulative across pipeline runs, not
just within a single batch.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from ._atomic import replace_with_retry

NUM_PERM = 128
SHINGLE_SIZE = 4
DEFAULT_THRESHOLD = 0.8

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_shingles(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    normalized = normalize_for_shingles(text)
    if not normalized:
        return set()
    if len(normalized) <= k:
        return {normalized}
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def text_to_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def minhash_digest(mh: MinHash) -> str:
    return "".join(f"{v:08x}" for v in mh.hashvalues[:8])


def load_lsh(path: Path, threshold: float = DEFAULT_THRESHOLD, num_perm: int = NUM_PERM) -> MinHashLSH:
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return MinHashLSH(threshold=threshold, num_perm=num_perm)


def save_lsh(lsh: MinHashLSH, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(lsh, f)
    replace_with_retry(tmp_path, path)


def is_near_duplicate(lsh: MinHashLSH, key: str, minhash: MinHash) -> bool:
    """Checks `minhash` against the index; if not a near-dup, inserts it under `key`."""
    if lsh.query(minhash):
        return True
    lsh.insert(key, minhash)
    return False
