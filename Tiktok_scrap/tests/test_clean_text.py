"""Unit tests for clean_text.py's tachkil (Arabic diacritics) stripping.

Added as a permanent regression test after a real bug was found while
writing TACHKIL_RE: typing the raw diacritic-range Unicode characters
directly into the regex got silently scrambled (RTL-aware text handling)
into a wider, wrong range that would have deleted Arabic-Indic digits
(٠-٩) along with the diacritics. Caught by direct codepoint verification
before it ever reached real data — this test guards against it recurring.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from darija_tiktok import clean_text  # noqa: E402


class TachkilTests(unittest.TestCase):
    def test_diacritics_stripped_base_letters_kept(self):
        self.assertEqual(clean_text.strip_tachkil("السَّلَام"), "السلام")
        self.assertEqual(clean_text.strip_tachkil("شُكْرًا"), "شكرا")

    def test_ascii_digits_survive(self):
        self.assertEqual(clean_text.strip_tachkil("عندي 1500 دج"), "عندي 1500 دج")

    def test_arabic_indic_digits_survive(self):
        # Regression: an earlier (buggy) version of TACHKIL_RE's range
        # accidentally covered U+0660-U+0669 and would have deleted these.
        self.assertEqual(clean_text.strip_tachkil("عندي ١٥٠٠ دج"), "عندي ١٥٠٠ دج")

    def test_tachkil_range_excludes_arabic_indic_digits(self):
        for cp in range(0x0660, 0x066A):
            self.assertFalse(
                clean_text.TACHKIL_RE.match(chr(cp)),
                f"TACHKIL_RE must not match Arabic-Indic digit U+{cp:04X}",
            )

    def test_clean_strips_tachkil_and_preserves_content(self):
        result = clean_text.clean("السَّلَام عَلَيْكُم وَرَحْمَة الله")
        self.assertEqual(result, "السلام عليكم ورحمة الله")


class UnicodeNormalizationTests(unittest.TestCase):
    """NFKC normalization, added after Notebooks/03_build_vocabulary.ipynb's
    "other" script bucket turned up Arabic Presentation Forms ligatures
    (a different Unicode block than the plain Arabic range every other
    rule checks against, so they were invisible to those rules)."""

    def test_religious_phrase_ligature_expands(self):
        self.assertEqual(clean_text.normalize_unicode_forms("ﷺ"), "صلى الله عليه وسلم")

    def test_allah_ligature_expands(self):
        self.assertEqual(clean_text.normalize_unicode_forms("ﷲ"), "الله")

    def test_lam_alef_ligature_expands(self):
        self.assertEqual(clean_text.normalize_unicode_forms("ﻻ"), "لا")

    def test_positional_letter_forms_normalize(self):
        self.assertEqual(clean_text.normalize_unicode_forms("ﻣﻦ"), "من")
        self.assertEqual(clean_text.normalize_unicode_forms("ﻓﻲ"), "في")

    def test_ascii_and_arabic_indic_digits_untouched(self):
        self.assertEqual(clean_text.normalize_unicode_forms("1500 و ١٥٠٠"), "1500 و ١٥٠٠")

    def test_tatweel_emoji_tachkil_untouched(self):
        text = "مرحبـــا 😂🤷‍♀️ شُكْرًا"
        self.assertEqual(clean_text.normalize_unicode_forms(text), text)

    def test_clean_expands_ligature_and_keeps_rest_of_pipeline_working(self):
        result = clean_text.clean("ربي يرحمو ﷺ وربي يبارك ﷲ فيك")
        self.assertEqual(result, "ربي يرحمو صلى الله عليه وسلم وربي يبارك الله فيك")


class RepeatSpamTests(unittest.TestCase):
    """Added after a 1,000-doc TikTok sample review turned up copy-paste
    spam (a word or phrase pasted dozens of times in one comment) slipping
    through -- MinHash/LSH dedup only catches near-duplicates *across*
    documents, not repetition within one. Thresholds are set well above
    the 2-3x repeats used for real Darija emphasis, per "preserve natural
    variation"."""

    def test_word_repeated_many_times_collapses_to_two(self):
        self.assertEqual(clean_text.collapse_repeated_words("تم " * 40), "تم تم ")

    def test_word_repeated_three_times_untouched(self):
        text = "لا لا لا نروح"
        self.assertEqual(clean_text.collapse_repeated_words(text), text)

    def test_word_repeated_two_times_untouched(self):
        text = "يا حبيبي حبيبي"
        self.assertEqual(clean_text.collapse_repeated_words(text), text)

    def test_phrase_glued_many_times_collapses_to_two(self):
        phrase = "خالد العليان مرحبا بك في الجزائر أفضل صانع محتوى"
        spam = phrase * 30
        self.assertEqual(clean_text.collapse_repeated_phrases(spam), phrase * 2)

    def test_short_phrase_repeated_twice_untouched(self):
        # below the 12-char minimum unit length -- must not touch it
        text = "هه هه"
        self.assertEqual(clean_text.collapse_repeated_phrases(text), text)

    def test_normal_sentence_untouched(self):
        text = "صحابي عايشين في إمارات رجعوا للجزائر"
        self.assertEqual(clean_text.clean(text), text)

    def test_clean_end_to_end_collapses_word_spam(self):
        result = clean_text.clean("تم " * 40)
        self.assertEqual(result, "تم تم")


if __name__ == "__main__":
    unittest.main()
