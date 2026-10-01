---
language:
  - ar

license: cc-by-4.0

tags:
  - algerian-darija
  - sentiment-analysis
  - dialect

task_categories:
  - text-classification

pretty_name: DarijaDZ_SA

size_categories:
  - 1K<n<10K
---

# DarijaDZ_SA

**DarijaDZ_SA** is a sentiment-labeled subset of the [DarijaDZ](https://huggingface.co/datasets/nasrellahkharroubi/DarijaDz) corpus of Algerian Darija text. Each document is labeled with one of four sentiment classes: **POS** (positive), **NEG** (negative), **NEU** (neutral), or **MIX** (mixed/both positive and negative).

---

## Dataset Description

### Motivation

Sentiment analysis for Algerian Darija is held back by the lack of labeled data. DarijaDZ_SA provides a labeled sample drawn from DarijaDZ to support:

* sentiment classification model training and evaluation
* fine-tuning word/sentence embeddings for sentiment
* Darija NLP research
* studying sentiment expression in informal, code-switched, and multi-script text

---

## Dataset Statistics

| Metric                     |          Value |
| -------------------------- | -------------: |
| **Documents**              |           9,579 |

### Label distribution

| Label                   | Count  | Share |
| ------------------------ | -----: | ----: |
<!-- DISTRIBUTION_TABLE_START -->
| POS                     |  4,484 |  46.8% |
| NEU                     |  2,437 |  25.4% |
| NEG                     |  2,473 |  25.8% |
| MIX                     |    185 |   1.9% |
<!-- DISTRIBUTION_TABLE_END -->

---

## Labeling Methodology

Documents were sampled from DarijaDZ with a stratified design to ensure adequate representation across script variants (Arabic script, Latin/Arabizi, and mixed/code-switched).

Labels were produced by **5 independent annotators**, each labeling a unique portion of the sample, with a shared overlap subset labeled by all 5 for inter-annotator agreement checks. Overlap items were resolved by majority vote (at least 3 of 5 annotators agreeing); items without a majority were resolved through manual review.

### Quality checks

* **Inter-annotator agreement** (Fleiss' kappa, computed on the shared overlap subset): **0.684**
* **Human evaluation**: a stratified sample of agreed-upon labels was independently rated by a human reviewer, blind to the original label. Agreement with the dataset's labels: **Cohen's kappa = 0.702** ("substantial agreement").

### Known limitations

* The **MIX** class is comparatively small and underrepresented relative to POS/NEG/NEU.
* Labels reflect sentence/comment-level sentiment and may not capture sarcasm, irony, or sentiment embedded in longer discourse context.
* A small number of originally ambiguous items were manually resolved; the dataset does not flag which items these were.

---

## Dataset Format

Distributed as a single JSONL file with the following schema:

| Field   | Type   | Description                              |
| ------- | ------ | ----------------------------------------- |
| `id`    | string | Unique document identifier                |
| `text`  | string | The document text                         |
| `label` | string | One of `POS`, `NEG`, `NEU`, `MIX`          |

```python
from datasets import load_dataset

dataset = load_dataset("nasrellahkharroubi/DarijaDZ_SA")

print(dataset)
```

---

## License

This dataset is released under the **CC BY 4.0** license.

---

## Citation

If you use DarijaDZ_SA in your research, please cite:

```text
Kharroubi Nasrellah.
DarijaDZ_SA: Sentiment-Labeled Algerian Darija Dataset.
2026.
```
