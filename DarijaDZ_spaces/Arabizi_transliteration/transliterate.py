# transliterate.py
import random
import re

try:
    from .lexicon import LEXICON
    from .madd_rules import apply_madd_rules
    from .mapping import MAPPING
except ImportError:
    from lexicon import LEXICON
    from madd_rules import apply_madd_rules
    from mapping import MAPPING

def sample_rendering(choices_list):
    if not choices_list:
        return ""
    renderings, weights = zip(*choices_list)
    return random.choices(renderings, weights=weights)[0]

def transliterate_word(word: str) -> str:
    # 1. Lexicon lookup first
    if word in LEXICON:
        return sample_rendering(LEXICON[word])
    
    # 2. Transliterate letter by letter
    result = []
    
    # Handle "ال" prefix
    start_idx = 0
    if word.startswith("ال") and len(word) > 2:
        # Option to render as "el", "l", or "al"
        el_choice = random.choices(["el", "l", "al"], weights=[0.60, 0.30, 0.10])[0]
        result.append(el_choice)
        start_idx = 2
        
    i = start_idx
    while i < len(word):
        char = word[i]
        
        # Check special case of "لا" ligature
        if char == 'ل' and i < len(word) - 1 and word[i+1] == 'ا':
            result.append(sample_rendering(MAPPING.get("لا", [("la", 1.0)])))
            i += 2
            continue
            
        # Check Madd context rules
        if char in {'ا', 'ي', 'و'}:
            madd_choices = apply_madd_rules(word, i)
            if madd_choices:
                result.append(sample_rendering(madd_choices))
                i += 1
                continue
                
        # Character fallback
        char_choices = MAPPING.get(char, [(char, 1.0)])
        result.append(sample_rendering(char_choices))
        i += 1
        
    return "".join(result)

def transliterate(text: str) -> str:
    """
    Takes Arabic-script text and returns a stochastically-generated Arabizi rendering.
    """
    # Split text while preserving punctuation, emojis, and whitespace
    tokens = re.split(r'(\s+|[^\w\u0600-\u06FF]+)', text)
    result = []
    for token in tokens:
        if re.match(r'^[\u0600-\u06FF]+$', token):
            result.append(transliterate_word(token))
        else:
            result.append(token)
    return "".join(result)
