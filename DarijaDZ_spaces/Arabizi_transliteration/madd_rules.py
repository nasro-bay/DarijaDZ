# madd_rules.py
# Context-sensitive rules for Madd letters (ا, ي, و) to decide vowel vs consonant representations.

def apply_madd_rules(word: str, index: int) -> list:
    """
    Given a word and the index of the character (which is one of 'ا', 'ي', 'و'),
    returns a list of (rendering, weight) pairs based on the surrounding context.
    If no specific context matches, returns None to fallback to standard character mapping.
    """
    char = word[index]
    prev_char = word[index - 1] if index > 0 else None
    next_char = word[index + 1] if index < len(word) - 1 else None

    vowels = {'ا', 'ي', 'و', 'أ', 'إ', 'آ', 'ى', 'ئ', 'ؤ', 'ء'}
    def is_consonant(c):
        if not c:
            return False
        return c not in vowels

    if char == 'ي':
        # Case 1: Word-initial -> consonantal 'y' (e.g. يبارك -> ybarek)
        if index == 0:
            return [("y", 1.00)]
        # Case 2: Word-final -> vowel 'i' or 'y' (e.g. ربي, راني -> rabi, rani)
        if index == len(word) - 1:
            return [("i", 0.90), ("y", 0.10)]
        # Case 3: Medial between consonants -> long vowel 'i' or 'e' (e.g. دير -> dir)
        if is_consonant(prev_char) and is_consonant(next_char):
            return [("i", 1.00)]
        # Case 4: Adjacent to another vowel -> consonantal 'y'
        return [("y", 1.00)]

    elif char == 'و':
        # Case 1: Word-initial -> consonantal 'w' (e.g. واحد, وين -> wahed, win)
        if index == 0:
            return [("w", 1.00)]
        # Case 2: Word-final -> 'ou' or 'o' or 'u'
        if index == len(word) - 1:
            return [("ou", 0.75), ("o", 0.25)]
        # Case 3: Medial between consonants (e.g. شكون -> chkoun) -> 'ou' or 'o'
        if is_consonant(prev_char) and is_consonant(next_char):
            return [("ou", 0.75), ("o", 0.25)]
        return [("w", 1.00)]

    elif char == 'ا':
        # Case 1: Word-initial -> 'a' or 'e'
        if index == 0:
            return [("a", 0.85), ("e", 0.15)]
        # Case 2: Word-final -> 'a'
        if index == len(word) - 1:
            return [("a", 1)]
        # Case 3: Medial -> long vowel 'a' or 'aa'
        return [("a", 0.90), ("aa", 0.10)]

    return None
