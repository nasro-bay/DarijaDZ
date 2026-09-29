"""Text cleaning rules for YouTube comments, derived empirically from a
1,000-comment representative sample — see
Notebooks/02_cleaning_rules_youtube.ipynb for the prevalence data and
rationale behind each rule (including amendments found by validating this
implementation against that sample, in the notebook's Section 10), plus
Notebooks/03_build_vocabulary.ipynb for the Unicode-normalization rule
(found via the vocabulary's "other" script bucket). Wired into pipeline.py.

Rules preserve expressive language (elongated letters, emoji) rather than
stripping it outright, per the project's "preserve natural variation"
policy — only excess repetition is collapsed, and near-empty results are
signaled for the caller to drop, not silently emptied.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@[\w.]+")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

# Zero-width/invisible/bidi-control characters: ZWSP, ZWNJ, LRM, RLM, the
# directional-isolate controls (LRI/RLI/FSI/PDI — confirmed present in real
# data, e.g. wrapping flag emoji), and BOM. ZWJ (U+200D) is handled
# separately since it's legitimate *inside* compound emoji sequences
# (e.g. "🤷‍♂️") and shouldn't be stripped there.
_ZERO_WIDTH_NO_ZWJ_RE = re.compile(
    "[​‌‎‏⁦⁧⁨⁩﻿]"
)
_ZWJ = "‍"
_VS16 = "️"  # emoji variation selector

# Arabic diacritics (tachkil/tashkeel: fatha, damma, kasra, sukun, shadda,
# tanwin, etc.) plus the superscript alef (dagger alef). Not a Darija
# feature — casual Darija is written essentially undiacritized; the rare
# diacritized fragments found in real samples were religious/Quranic
# quotes or formal text bleeding in, not deliberate dialectal expression.
# Stripped outright (not collapsed) since they carry no dialectal signal
# and just fragment tokenization (e.g. "السلام" vs "السَّلَام" as distinct
# tokens for the same word).
TACHKIL_RE = re.compile("[ً-ٰٟ]")

EMOJI_CLASS = (
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U0001F1E6-\U0001F1FF"
    "☀-➿"
)
EMOJI_CHAR_RE = re.compile(f"[{EMOJI_CLASS}]")
# A full emoji "grapheme" — one emoji, optionally ZWJ-joined to more emoji
# (e.g. "🤷‍♀️" = person + ZWJ + female-sign + variation-selector) — treated
# as a single atomic unit so run-collapsing can't truncate a compound emoji
# mid-sequence and leave a dangling ZWJ (confirmed bug: an earlier version
# that treated "emoji + optional single modifier" as the unit did exactly
# that on real data — "😂🤷‍♀️" collapsed to "😂🤷‍", an orphaned ZWJ).
_EMOJI_GRAPHEME_RE = re.compile(f"[{EMOJI_CLASS}]{_VS16}?(?:{_ZWJ}[{EMOJI_CLASS}]{_VS16}?)*")
EMOJI_RUN_RE = re.compile(f"(?:{_EMOJI_GRAPHEME_RE.pattern}){{3,}}")

# Letters only (excludes digits/underscore) — a repeated letter is
# expressive stretching ("hhhhh", "ااااا"); a repeated digit is just a
# number (e.g. "1000" in a price) and shouldn't be touched.
ELONGATION_RE = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)

# A single word repeated 5+ times in a row, separated only by whitespace —
# keyboard/copy-paste spam ("تم تم تم تم ... تم" x40), distinct from
# ELONGATION_RE (a repeated *character* inside one word, e.g. "هههه").
# Threshold ({4,} = 5+ total occurrences) is set well above the 2-3x
# repeats used for real emphasis in Darija ("لا لا لا") per the project's
# "preserve natural variation" rule — confirmed empirically on real TikTok
# data (found via a 1,000-doc sample review), a real but rare (~0.2%)
# pattern.
WORD_REPEAT_RE = re.compile(r"(?P<word>\S+)(?:\s+(?P=word)\b){4,}")

# A multi-character phrase/sentence pasted back-to-back 3+ times (promo-spam
# replies, e.g. a whole sentence glued to itself ~30x with no separator).
# Minimum unit length of 12 keeps this from ever matching short expressive
# repeats (well above anything WORD_REPEAT_RE or ELONGATION_RE would catch).
# Non-greedy so it finds the smallest repeating unit rather than swallowing
# the whole match as one "unit".
PHRASE_REPEAT_RE = re.compile(r"(.{12,}?)\1{2,}", re.DOTALL)

PERIOD_RUN_RE = re.compile(r"\.{2,}")
OTHER_PUNCT_RUN_RE = re.compile(r"([!?؟,،])\1{2,}")  # "،" = Arabic comma, distinct from "," (found on real data)

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_NON_WORD_RE = re.compile(r"[^\w؀-ۿ]", re.UNICODE)

MIN_RESIDUAL_LETTERS = 2  # below this (after stripping emoji/punctuation), drop the doc


def normalize_unicode_forms(text: str) -> str:
    """NFKC-normalizes compatibility characters to their canonical form --
    catches Arabic Presentation Forms ligatures (found via the vocabulary
    notebook's "other" script bucket) that the plain [؀-ۿ] Arabic
    range doesn't recognize as Arabic at all, e.g. "ﷺ" (a single
    religious-phrase ligature) -> "صلى الله عليه وسلم", "ﷲ" -> "الله",
    "ﻻ" -> "لا". Also normalizes presentation-form letter variants
    (e.g. "ﻣﻦ" -> "من") and full-width digits/Latin to their
    plain equivalents. Verified empirically to leave Arabic-Indic digits,
    ASCII digits, tatweel, ZWJ-joined compound emoji, tachkil, and French
    accented letters unchanged -- only touches compatibility-equivalent
    characters, not the real content those other rules depend on.
    """
    return unicodedata.normalize("NFKC", text)


def strip_invisible_chars(text: str) -> str:
    text = _ZERO_WIDTH_NO_ZWJ_RE.sub("", text)
    if _ZWJ not in text:
        return text
    out = []
    for i, ch in enumerate(text):
        if ch == _ZWJ:
            prev_emoji = i > 0 and bool(EMOJI_CHAR_RE.match(text[i - 1]))
            next_emoji = i + 1 < len(text) and bool(EMOJI_CHAR_RE.match(text[i + 1]))
            if prev_emoji and next_emoji:
                out.append(ch)  # legitimate compound emoji (e.g. "🤷‍♂️")
            continue
        out.append(ch)
    return "".join(out)


def strip_tachkil(text: str) -> str:
    return TACHKIL_RE.sub("", text)


def replace_mentions(text: str) -> str:
    return MENTION_RE.sub(" [MENTION] ", text)


def replace_urls(text: str) -> str:
    return URL_RE.sub(" [URL] ", text)


def strip_timestamps(text: str) -> str:
    return TIMESTAMP_RE.sub(" ", text)


def collapse_repeated_phrases(text: str) -> str:
    return PHRASE_REPEAT_RE.sub(lambda m: m.group(1) * 2, text)


def collapse_repeated_words(text: str) -> str:
    return WORD_REPEAT_RE.sub(lambda m: f"{m.group('word')} {m.group('word')}", text)


def collapse_elongation(text: str) -> str:
    return ELONGATION_RE.sub(r"\1\1", text)


def normalize_punctuation(text: str) -> str:
    text = PERIOD_RUN_RE.sub("...", text)
    text = OTHER_PUNCT_RUN_RE.sub(r"\1\1", text)
    return text


def collapse_emoji_runs(text: str) -> str:
    def _collapse(match: re.Match) -> str:
        graphemes = _EMOJI_GRAPHEME_RE.findall(match.group(0))
        return "".join(graphemes[:2])

    return EMOJI_RUN_RE.sub(_collapse, text)


def normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


_PLACEHOLDER_RE = re.compile(r"\[MENTION\]|\[URL\]")


def residual_letter_count(text: str) -> int:
    """Length of `text` after stripping placeholder tokens, emoji, and all
    non-word characters -- used to decide whether a comment is near-empty
    (emoji-only, punctuation-only, or *just* a bare [MENTION]/[URL] tag with
    no actual reply text). Placeholder tokens must be stripped before this
    count: their own literal letters ("MENTION"/"URL") would otherwise read
    as real content and prevent an otherwise-empty reply from being dropped
    -- confirmed on real TikTok data, ~1.2% of processed docs were nothing
    but a bare "[MENTION]" (plus emoji/punctuation)."""
    no_placeholders = _PLACEHOLDER_RE.sub("", text)
    no_emoji = EMOJI_CHAR_RE.sub("", no_placeholders)
    no_punct = _NON_WORD_RE.sub("", no_emoji)
    return len(no_punct.strip())


def clean(text: str) -> Optional[str]:
    """Applies all rules in order and returns the cleaned text, or `None`
    if the result should be dropped (near-empty after cleaning).

    Unicode normalization runs first so every other rule (regex-based, all
    matching specific codepoint ranges) sees canonical text rather than
    presentation-form ligatures that wouldn't match them.
    """
    text = normalize_unicode_forms(text)
    text = strip_invisible_chars(text)
    text = strip_tachkil(text)
    text = replace_mentions(text)
    text = replace_urls(text)
    text = strip_timestamps(text)
    text = collapse_repeated_phrases(text)
    text = collapse_repeated_words(text)
    text = collapse_elongation(text)
    text = normalize_punctuation(text)
    text = collapse_emoji_runs(text)
    text = normalize_whitespace(text)

    if residual_letter_count(text) < MIN_RESIDUAL_LETTERS:
        return None
    return text
