---
title: Algerian Darija Stochastic Arabizi Transliterator
emoji: 🔀
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---
# Algerian Darija Stochastic Arabizi Transliterator

A web application and API that transliterates Algerian Darija text from Arabic script to Arabizi (Latin script with numerals representing specific Arabic sounds), e.g., `راني عارف` &rarr; `rani 3aref`.

## Core Design and Stochastic Spelling

Real Arabizi does not have a standard orthography. The same word is regularly written in multiple ways by different speakers, or even by the same speaker within the same conversation. To reflect this natural variation and avoid generating artificially uniform, "too clean" text:
- The transliterator is **non-deterministic** by design.
- Each run samples spelling variants using empirical probability distributions derived from real social media comment corpora (YouTube).
- Repeated submissions of the same input text will output different valid Arabizi orthographies.

## Architecture

The transliteration pipeline follows a 3-step priority logic:
1. **Lexicon Lookup**: High-frequency words are matched against an empirically calibrated lexicon seeded with top unigrams.
2. **Context-Sensitive Madd Rules**: Long vowels (`ا`, `ي`, `و`) are mapped using surrounding character bigram contexts to distinguish between vowel lengthening and consonantal roles.
3. **Character-Level Fallback**: Any remaining characters are mapped using character-level weighted probability tables.

## Limitations
- **Dialectal Variation**: Currently optimized for Algerian Darija text distributions observed in general YouTube comments. Local regional dialects (e.g., Oran vs. Algiers) may have spelling nuances not fully captured by general weights.
- **Lexicon Coverage**: Out-of-vocabulary words rely entirely on the character-level fallback, which may occasionally produce spelling combinations less common in native writing.
