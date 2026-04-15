---
language:
- en
license: mit
tags:
- text2text-generation
- style-transfer
- rewriting
- humanization
- seq2seq
- t5
- flan-t5
- evaluation
library_name: transformers
base_model: google/flan-t5-base
pipeline_tag: text2text-generation
---

# cive202/flan-t5-base-ai-text-humanization-seq2seq

Fine-tuned **FLAN-T5-base** (`google/flan-t5-base`) for **AI → Human “humanization”** rewriting.

Use this repo if you want a separately named T5 checkpoint focused on “humanization” (e.g., different instruction template, filtering, or training schedule) from:
- `cive202/flan-t5-base-ai-text-seq2seq`

## Quickstart

```bash
pip install -U "transformers>=4.40.0" torch sentencepiece
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "cive202/flan-t5-base-ai-text-humanization-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

ai_text = "Large language models often produce fluent, structured prose with recognizable regularities..."

# Use the exact instruction/prefix you trained with:
prefix = "humanize: "  # TODO: replace if your T5 template differs
inputs = tokenizer(prefix + ai_text, return_tensors="pt", truncation=True)

out = model.generate(
    **inputs,
    max_new_tokens=256,
    num_beams=4,
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training & configuration (REQUIRED FILL)

Your `BARTvsMistral/summary.json` does not include FLAN-T5 metrics/configs, so fill these fields from your actual run:

- **Instruction/prefix/template**: `[REQUIRED_FILL]`
- **Max lengths**: `[REQUIRED_FILL]`
- **Epochs or max_steps**: `[REQUIRED_FILL]`
- **LR / scheduler / warmup**: `[REQUIRED_FILL]`
- **Effective batch size**: `[REQUIRED_FILL]`
- **Precision**: `[REQUIRED_FILL]`

Differences vs `cive202/flan-t5-base-ai-text-seq2seq`:
- `[REQUIRED_FILL: e.g., different instruction template or longer training]`

## Dataset

If trained on the same parallel corpus as the project’s BART/Mistral runs:

- **Train**: 25,140 pairs
- **Validation**: 1,390 examples
- **Test**: 1,390 examples

Otherwise:
- **Train/Val/Test**: `[REQUIRED_FILL]`

## Evaluation (REQUIRED FILL)

Recommended evaluation suite for AI→Human rewriting:

- **Semantic similarity**: BERTScore (P/R/F1)
- **Overlap**: ROUGE-L, chrF++
- **Fluency proxy**: GPT-2 perplexity (output / AI input / human reference)
- **Lexical overlap**: Vocabulary Jaccard
- **Style movement**: 11 linguistic markers + directional marker shift, clipped to \([-1, 2]\)

Fill with your results:

| Metric | Value |
|---|---|
| BERTScore F1 | `[REQUIRED_FILL]` |
| ROUGE-L | `[REQUIRED_FILL]` |
| chrF++ | `[REQUIRED_FILL]` |
| GPT-2 PPL (output) | `[REQUIRED_FILL]` |
| Mean marker shift | `[REQUIRED_FILL]` |

## Research paper (unpublished)

This checkpoint is part of an unpublished manuscript (2026):

**“Rewriting the Machine: Encoder-Decoder vs. Decoder-Only Transformers for AI-to-Human Text Style Transfer”**

- Status: **not published yet**
- Link: *(add your PDF/arXiv link when available)*

## License

MIT is a placeholder—set the license to what you intend, consistent with `google/flan-t5-base` terms.

