"""Builds the corpus JSON schema for a single TikTok comment/reply.

Uses the nested `source_metadata` schema -- the current project-wide
convention (see Mountada_djelfa_scrap's schema.py), not Youtube_scrap's
older flat schema (written before that convention existed).

`script` and `darija_confidence` are left as `null` placeholders -- the
language/dialect filter is deferred to a later phase, same as the other
two sources. `dedup_hash` is also `null` by default -- dedup.py is a
standalone opt-in pass here too, same reasoning as Youtube_scrap's
pipeline.py (sequential LSH insert/query was the dominant cost at scale
there; kept disabled by default here from the start rather than
discovering the same thing the hard way).
"""
from __future__ import annotations

from typing import Optional


def build_document(
    *,
    doc_id: str,
    text: str,
    video_id: str,
    channel: str,
    is_reply: bool,
    parent_comment_id: Optional[str],
    scrape_date: str,
    dedup_hash: Optional[str],
) -> dict:
    return {
        "id": doc_id,
        "text": text,
        "source": "tiktok",
        "source_type": "reply" if is_reply else "comment",
        "source_metadata": {
            "video_id": video_id,
            "channel": channel,
            "parent_comment_id": parent_comment_id,
        },
        "scrape_date": scrape_date,
        "script": None,
        "darija_confidence": None,
        "char_count": len(text),
        "token_count": len(text.split()),
        "dedup_hash": dedup_hash,
    }
