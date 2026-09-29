import json
import re
from collections import Counter

latin_char_re = re.compile(r'[a-zA-Z]')
arabic_char_re = re.compile(r'[\u0600-\u06FF]')

input_file = r"c:\Users\ASUS\Desktop\summer 2026\Darija\Data\sample_youtube.jsonl"

latin_comments = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        text = data.get('text', '').strip()
        # Find comments that contain Latin characters and contain very few/no Arabic characters
        if latin_char_re.search(text) and len(arabic_char_re.findall(text)) < 3:
            latin_comments.append(text.lower())

# Join all comments to tokenize
all_text = " ".join(latin_comments)
words = re.findall(r'[a-z0-9]+', all_text)

word_counts = Counter(words)

# Let's check frequencies for options of interest
target_patterns = {
    "في (fi)": ["fi"],
    "على (ela/3la/ala)": ["ela", "3la", "ala"],
    "ربي (rabbi/rbby/rabi)": ["rabbi", "rbby", "rabi"],
    "لي (li/ly)": ["li", "ly", "lli"],
    "ما (ma)": ["ma"],
    "يا (ya)": ["ya"],
    "لا (la)": ["la"],
    "كي (ki)": ["ki"],
    "تاع (ta3/ta3e/te3)": ["ta3", "ta3e", "te3"],
    "والله (wallah)": ["wallah"],
    "هذا (hada/hada)": ["hada"],
    "غير (ghir/5ir/ghire)": ["ghir", "5ir", "ghire"],
    "ولا (wla/oula/wlla)": ["wla", "oula", "wlla"],
    "راني (rani)": ["rani"],
    "بصح (besah/bsh/bessah/bezah)": ["besah", "bsh", "bessah"],
    "بزاف (bezzaf/bzaf)": ["bezzaf", "bzaf"],
    "واحد (wahed/wahad/wa7ed)": ["wahed", "wahad", "wa7ed"],
    "وين (win)": ["win"],
    "كان (kan)": ["kan"],
    "راك (rak)": ["rak"],
    "كاين (kayen/kain)": ["kayen", "kain"],
    "خويا (khouya/5oya)": ["khouya", "5oya"],
    "مع (مع/m3a/ma3a)": ["m3a", "ma3a"],
    "واش (wach/wesh)": ["wach", "wesh"],
    "عندي (3andi/endi)": ["3andi", "endi"],
    "علاه (3lah/elah)": ["3lah", "elah"],
    "انا (ana)": ["ana"],
    "حاجة (haja/7aja)": ["haja", "7aja"],
    "دير (dir)": ["dir"],
    "صح (sah/sa7)": ["sah", "sa7"],
    "ماشي (machi/machy)": ["machi", "machy"],
    "كيما (kima)": ["kima"],
    "شكون (chkoun)": ["chkoun"],
    "راهي (rahi)": ["rahi"],
    "حنا (hna)": ["hna"],
    "راه (rah)": ["rah"],
    "رانا (rana)": ["rana"],
    "يكون (ykoun)": ["ykoun"],
    "عندك (3andek/endek)": ["3andek", "endek"]
}

output_lines = []
for label, options in target_patterns.items():
    res = []
    for opt in options:
        res.append(f"{opt}: {word_counts[opt]}")
    output_lines.append(f"{label} -> {', '.join(res)}")

with open("arabizi_frequencies.txt", "w", encoding="utf-8") as f_out:
    f_out.write("\n".join(output_lines))

print("Done counting.")
