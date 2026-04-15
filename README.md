# ✨ Please Make It Sound Human  
### *Encoder–Decoder vs. Decoder-Only Transformers for AI → Human Text Style Transfer*

<p align="center">

![Paper](https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge)  
![Model](https://img.shields.io/badge/Model-BART%20%7C%20Mistral-blueviolet?style=for-the-badge)  
![Task](https://img.shields.io/badge/Task-Text%20Style%20Transfer-success?style=for-the-badge)  
![Metrics](https://img.shields.io/badge/Metrics-BERTScore%20%7C%20ROUGE--L%20%7C%20chrF++-orange?style=for-the-badge)  

</p>

---

## 🧠 Overview

This project investigates a fundamental but underexplored question:

> **Can AI-generated text be systematically rewritten to sound genuinely human?**

We replicate and extend the findings of *From Machine to Human*, comparing:

- **BART (Encoder–Decoder)** — reconstruction-based generation  
- **Mistral 7B (Decoder-Only, QLoRA)** — large-scale autoregressive model  

---

## 🚀 Key Contributions

- 📊 **25,140 paired dataset** of AI-generated and human-authored text  
- 🔍 **11 linguistic markers** defining “human-like” writing  
- ⚖️ Introduction of **marker shift accuracy vs. magnitude**  
- 🏆 Demonstration that **BART-large outperforms Mistral-7B (17× smaller)**  
- 🧩 Insight: **pretraining objective > parameter scale** for style transfer  

---

## 🏗️ Architecture Comparison

| Model Type | Example | Strength |
|-----------|--------|----------|
| Encoder–Decoder | BART | Precise reconstruction & controlled rewriting |
| Decoder-Only | Mistral 7B | Strong generation, but prone to stylistic overshoot |

---

## 📊 Core Findings

### 🥇 Reference Similarity (Best Model: BART-large)

| Metric | Score |
|------|------|
| **BERTScore F1** | **0.924** |
| **ROUGE-L** | **0.566** |
| **chrF++** | **55.92** |

---

### ⚠️ The Overshoot Problem

Mistral-7B shows:

- ❌ Excessive contractions  
- ❌ Too many words & sentences  
- ❌ Wrong punctuation trends  
- ❌ Over-simplified, overly predictable text  

> High “shift” ≠ human-like  
> **Accuracy matters more than magnitude**

---

### 📉 Fluency vs Authenticity

| Model | GPT-2 Perplexity |
|------|----------------|
| Human | 23.69 |
| BART-large | ~27 |
| Mistral-7B | **9.03** |

---

## 🧪 Evaluation Framework

We evaluate across **5 dimensions**:

- 🧠 **BERTScore** → semantic preservation  
- 🔤 **ROUGE-L / chrF++** → lexical similarity  
- 📉 **Perplexity** → fluency vs naturalness  
- 📚 **Vocabulary Jaccard** → lexical overlap  
- 🧬 **Linguistic Marker Shift (11 features)** → stylistic realism  

---

## 🧩 Methodology Highlights

### Data
- Multi-domain corpus (academic, technical, creative)
- AI text generated using LLaMA-family models
- Sentence-aware chunking (~200 tokens)

### Models
- **BART-base / BART-large** → full fine-tuning  
- **Mistral 7B** → QLoRA (4-bit/8-bit)  

---

## 📁 Repository Structure

```
BARTvsMistral/
├── train_bart.py
├── train_mistral_qlora.py
├── evaluate.py
├── linguistic_markers.py
├── qualitative_examples.py
├── configs/
├── data/processed/
└── results/
```

---

## 🤖 Model Release

👉 https://huggingface.co/cive202/humanize-ai-text-mistral-7b-lora

---

## 📌 Research Takeaways

- 🔹 Bigger models ≠ better style transfer  
- 🔹 Encoder–decoder > decoder-only for constrained rewriting  
- 🔹 Evaluation must consider **accuracy, not just change magnitude**  
- 🔹 Human writing = **controlled variability, not smooth perfection**  

---

## 📖 Citation

```bibtex
@article{paneru2026humanize,
  title={Please Make It Sound like Human: Encoder-Decoder vs Decoder-Only Transformers for AI-to-Human Text Style Transfer},
  author={Paneru, Utsav},
  year={2026},
  journal={arXiv}
}
```

---

## 🌟 Final Note

> ✨ *Human-like writing is not about sounding perfect — it’s about sounding real.*
