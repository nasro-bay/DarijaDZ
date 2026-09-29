import json
import re
from collections import Counter

arabic_char_re = re.compile(r'[\u0600-\u06FF]')
latin_char_re = re.compile(r'[a-zA-Z]')

input_file = r"c:\Users\ASUS\Desktop\summer 2026\Darija\Data\sample_youtube.jsonl"
output_file = r"c:\Users\ASUS\Desktop\summer 2026\Darija\Arabizi_transliteration\arabic_comments.jsonl"

arabic_comments = []
all_words = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        text = data.get('text', '')
        # Filter to comments that contain Arabic and do not contain Latin letters (to get pure Arabic script comments)
        if arabic_char_re.search(text) and not latin_char_re.search(text):
            arabic_comments.append(data)
            # Tokenize words (simple whitespace split, removing punctuation)
            words = re.findall(r'[\u0600-\u06FF]+', text)
            all_words.extend(words)

# Write filtered comments
with open(output_file, 'w', encoding='utf-8') as f:
    for comment in arabic_comments:
        f.write(json.dumps(comment, ensure_ascii=False) + '\n')

print(f"Total Arabic script comments: {len(arabic_comments)}")
print(f"Total words: {len(all_words)}")

# Count unigrams
word_counts = Counter(all_words)
with open("unigrams.txt", "w", encoding="utf-8") as f_uni:
    for word, count in word_counts.most_common(100):
        f_uni.write(f"{word}: {count}\n")

