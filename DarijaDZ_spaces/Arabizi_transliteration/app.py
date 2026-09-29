import base64
from pathlib import Path

import gradio as gr
import spaces
from transliterate import transliterate

LOGO_B64 = base64.b64encode((Path(__file__).parent / "logo.png").read_bytes()).decode()

description = (
    "This tool transliterates Algerian Darija text from Arabic script to Arabizi (Latin script). "
    "Because Arabizi has no standard orthography and speakers write the same words in multiple ways, "
    "the output is deliberately non-deterministic. Submit the input multiple times or try different "
    "runs to explore different valid spelling variations for the same text."
)

examples = [
    ["راني عارف"],
    ["عليهم"],
    ["نصرو شاك داير فيها"],
    ["ربي يحفظك خويا"],
    ["الدار في وهران غير واه وشتاكاين"],
]

# Same Algeria-flag palette + Cairo font + topbar/flag-strip identity as the
# rest of the DarijaDZ ecosystem (Embeddings/intrinsic_eval,
# DarijaDZ_DialectID_Classifier, LM_DiD/webapp).
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&display=swap');

:root {
  --dz-green: #04663a;
  --dz-red: #c8102e;
  --dz-white: #ffffff;
  --dz-gold: #b8922f;
  --dz-cream: #f7f4ee;
  --ink: #1e2723;
  --muted: #74827b;
  --border: #e2e0d8;
}

.gradio-container {
  font-family: "Cairo", "Segoe UI", Arial, sans-serif !important;
  background: var(--dz-cream) !important;
  max-width: 760px !important;
}

#dz-topbar {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 4px 6px;
}
#dz-topbar img {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: 1px solid var(--border);
  object-fit: cover;
  flex-shrink: 0;
}
#dz-topbar h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--ink);
}
#dz-topbar p {
  margin: 2px 0 0;
  font-size: 0.8rem;
  color: var(--muted);
}
#dz-topbar .flag-mark {
  margin-left: auto;
  color: var(--dz-green);
}

#dz-flag-strip {
  height: 4px;
  display: flex;
  margin: 0 0 18px;
  border-radius: 2px;
  overflow: hidden;
}
#dz-flag-strip span { flex: 1; display: block; }
#dz-flag-strip span:nth-child(1) { background: var(--dz-green); }
#dz-flag-strip span:nth-child(2) { background: var(--dz-white); }
#dz-flag-strip span:nth-child(3) { background: var(--dz-red); }

.dz-desc {
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 6px;
}

button.primary {
  background: var(--dz-green) !important;
  border: none !important;
}
button.primary:hover { background: #054d2c !important; }

.gr-box, textarea, .block {
  border-radius: 6px !important;
}
"""

FLAG_MARK_SVG = """<span class="flag-mark">
<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
  <path d="M14.5 3.5a8.5 8.5 0 1 0 0 17 8.7 8.7 0 0 1 0-17z"/>
  <path d="m17.6 8.2.9 2.7h2.9l-2.3 1.7.9 2.7-2.4-1.7-2.3 1.7.9-2.7-2.4-1.7h2.9z"/>
</svg>
</span>"""

TOPBAR_HTML = f"""
<div id="dz-topbar">
  <img src="data:image/png;base64,{LOGO_B64}" alt="DarijaDZ">
  <div>
    <h1>Darija &rarr; Arabizi Transliterator</h1>
    <p>Part of DarijaDZ &mdash; an NLP ecosystem for Algerian Darija</p>
  </div>
  {FLAG_MARK_SVG}
</div>
<div id="dz-flag-strip"><span></span><span></span><span></span></div>
"""


@spaces.GPU
def run(arabic_text):
    return transliterate(arabic_text)


with gr.Blocks(css=CUSTOM_CSS, title="Darija Arabizi Transliterator") as demo:
    gr.HTML(TOPBAR_HTML)
    gr.Markdown(description, elem_classes="dz-desc")

    input_box = gr.Textbox(
        label="Input Arabic Script (Algerian Darija)",
        placeholder="Type here (e.g., راني عارف)...",
        lines=4,
        rtl=True,
    )
    submit_btn = gr.Button("Transliterate", variant="primary")
    output_box = gr.Textbox(label="Transliterated Arabizi", lines=4)

    submit_btn.click(fn=run, inputs=input_box, outputs=output_box)
    input_box.submit(fn=run, inputs=input_box, outputs=output_box)

    gr.Examples(examples=examples, inputs=input_box)

if __name__ == "__main__":
    demo.launch()
