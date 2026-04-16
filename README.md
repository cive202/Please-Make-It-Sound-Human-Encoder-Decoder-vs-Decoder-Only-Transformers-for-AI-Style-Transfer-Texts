# Please Make It Sound Human: Encoder-Decoder vs. Decoder-Only Transformers for AI-to-Human Text Style Transfer

<p align="center">
  <a href="https://arxiv.org/abs/2604.11687"><img src="https://img.shields.io/badge/arXiv-2604.11687-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/cive202/humanize-ai-text-bart-large"><img src="https://img.shields.io/badge/🤗-BART Large-yellow" alt="BART Large"></a>
  <a href="https://huggingface.co/cive202/humanize-ai-text-bart-base"><img src="https://img.shields.io/badge/🤗-BART Base-yellow" alt="BART Base"></a>
  <a href="https://huggingface.co/cive202/humanize-ai-text-mistral-7b-lora"><img src="https://img.shields.io/badge/🤗-Mistral 7B LoRA-yellow" alt="Mistral LoRA"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
</p>

> Can AI-generated text be systematically rewritten to sound genuinely human? This repository benchmarks BART (encoder-decoder, full fine-tune) against Mistral 7B (decoder-only, QLoRA) on that exact question — with a 25,140-pair dataset and an evaluation framework spanning semantic, lexical, fluency, and 11 linguistic marker dimensions.

---

## Overview

| Model | Architecture | Training Strategy |
|---|---|---|
| `bart-base` | Encoder-decoder | Full fine-tune |
| `bart-large` | Encoder-decoder | Full fine-tune |
| `mistral-7b-instruct` | Decoder-only | QLoRA (4-bit / 8-bit) |

**Headline finding:** BART-large outperforms Mistral 7B despite being 17× smaller. Pretraining objective matters more than parameter scale for constrained style transfer.

---

## Resources

| Resource | Link |
|---|---|
| Paper | [arxiv.org/abs/2604.11687](https://arxiv.org/abs/2604.11687) |
| BART Large checkpoint | [cive202/humanize-ai-text-bart-large](https://huggingface.co/cive202/humanize-ai-text-bart-large) |
| BART Base checkpoint | [cive202/humanize-ai-text-bart-base](https://huggingface.co/cive202/humanize-ai-text-bart-base) |
| Mistral 7B LoRA adapter | [cive202/humanize-ai-text-mistral-7b-lora](https://huggingface.co/cive202/humanize-ai-text-mistral-7b-lora) |

---

## Key Contributions

- **25,140-pair dataset** of AI-generated and human-authored text across academic, technical, and creative domains
- **11 linguistic markers** operationalising what "human-like" writing actually means
- Introduction of **marker shift accuracy vs. magnitude** as a more faithful evaluation axis
- Evidence that **encoder-decoder architectures outperform decoder-only** for controlled rewriting tasks

---

## Results

### Reference similarity — best model: BART-large

| Metric | Score |
|---|---|
| BERTScore F1 | 0.924 |
| ROUGE-L | 0.566 |
| chrF++ | 55.92 |

### Fluency vs. authenticity

| Model | GPT-2 Perplexity |
|---|---|
| Human reference | 23.69 |
| BART-large | ~27 |
| Mistral 7B | 9.03 |

Mistral's near-perfect perplexity is a red flag, not a success signal. Unnaturally low perplexity indicates over-smoothed, predictable text — the opposite of authentic human writing.

### The overshoot problem

Mistral 7B consistently over-corrects: too many contractions, inflated word and sentence counts, wrong punctuation trends, and excessive simplification. High marker *shift* does not mean human-like; *accuracy* of shift direction is what matters.

---

## Evaluation Framework

Five complementary dimensions:

| Dimension | Metric(s) | What it captures |
|---|---|---|
| Semantic preservation | BERTScore F1 | Meaning retained after rewrite |
| Lexical similarity | ROUGE-L, chrF++ | Surface-level overlap with human reference |
| Fluency | GPT-2 perplexity | Naturalness of generated text |
| Vocabulary overlap | Jaccard similarity | Lexical diversity alignment |
| Stylistic realism | 11 linguistic marker shifts | Genuine human-writing behaviour |

---

## Quickstart

**Requirements:** Python 3.10+, PyTorch with CUDA, Hugging Face account with access to `mistralai/Mistral-7B-Instruct-v0.2`. A fresh virtual environment is recommended.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place `train.jsonl`, `val.jsonl`, and `test.jsonl` into `data/processed/`.

### 3. Train

```bash
# BART (base or large — set model name in configs/default.yaml)
python train_bart.py

# Mistral 7B QLoRA
python train_mistral_qlora.py
```

### 4. Evaluate

```bash
python evaluate.py --model-type bart --checkpoint outputs/bart-large-humanize
python evaluate.py --model-type mistral --checkpoint outputs/mistral-7b-lora-humanize
```

### 5. Inspect linguistic markers

```bash
python linguistic_markers.py --predictions outputs/bart-large-humanize/predictions.jsonl
```

### 6. Export qualitative examples

```bash
python qualitative_examples.py --checkpoint outputs/bart-large-humanize
```

---

## Repository Layout

```
BARTvsMistral/
├── configs/
│   └── default.yaml               # Paths and hyperparameters
├── data/
│   └── processed/                 # train/val/test.jsonl (not committed)
├── results/                       # Evaluation outputs
├── train_bart.py
├── train_mistral_qlora.py
├── evaluate.py
├── linguistic_markers.py
├── qualitative_examples.py
└── requirements.txt
```

---

## Citation

```bibtex
@article{paneru2026humanize,
  title   = {Please Make It Sound like Human: Encoder-Decoder vs Decoder-Only Transformers for AI-to-Human Text Style Transfer},
  author  = {Paneru, Utsav},
  journal = {arXiv preprint arXiv:2604.11687},
  year    = {2026}
}
```

---

<p align="center">
  <a href="https://arxiv.org/abs/2604.11687">Paper</a> ·
  <a href="https://huggingface.co/cive202/humanize-ai-text-bart-large">BART Large</a> ·
  <a href="https://huggingface.co/cive202/humanize-ai-text-bart-base">BART Base</a> ·
  <a href="https://huggingface.co/cive202/humanize-ai-text-mistral-7b-lora">Mistral 7B LoRA</a>
</p>
