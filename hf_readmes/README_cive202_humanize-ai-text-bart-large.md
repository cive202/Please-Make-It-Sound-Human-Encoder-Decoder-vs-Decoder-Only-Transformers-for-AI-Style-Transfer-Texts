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
- bart
- evaluation
- bertscore
- rouge
- chrf
library_name: transformers
base_model: facebook/bart-large
pipeline_tag: text2text-generation
---

# cive202/humanize-ai-text-bart-large

Fine-tuned **BART-large** (`facebook/bart-large`) for **AI → Human rewriting** (“humanization”). This model is designed for **constrained rewriting**: preserve meaning while shifting style toward human-authored text.

- **Architecture**: encoder–decoder (seq2seq)
- **Parameters**: ~406M
- **Task format**: `humanize: {ai_text}` → `{human_text}`

## Quickstart

```bash
pip install -U "transformers>=4.40.0" torch sentencepiece
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "cive202/humanize-ai-text-bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

ai_text = "Large language models often produce fluent, structured prose with recognizable regularities..."
inputs = tokenizer("humanize: " + ai_text, return_tensors="pt", truncation=True)

out = model.generate(
    **inputs,
    max_new_tokens=256,
    num_beams=4,
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training summary (from project config)

Full fine-tuning (no adapters) with a standard seq2seq cross-entropy objective:

- **LR / schedule**: `5e-5`, cosine scheduler
- **Warmup ratio**: `0.1`
- **Precision**: bf16
- **Effective batch size**: 16 (`per_device_train_batch_size = 2`, `gradient_accumulation_steps = 8`)
- **Epochs**: 5
- **Checkpoint selection**: best checkpoint by validation loss

## Dataset

Parallel chunk pairs created via sentence-aware chunking:

- **Train**: 25,140 pairs
- **Validation**: 1,390 examples
- **Test (evaluation subset)**: 1,390 examples

Preprocessing details (high-level):
- sentence tokenization (NLTK)
- greedy packing to a token budget (≤200 tokens measured with BART-base tokenizer)
- drop pairs with fewer than 10 words on either side
- document-disjoint splits (no `doc_id` overlap between splits)

## Evaluation (test n = 1,390)

All metrics computed on the same 1,390-example test subset.

### Reference similarity (higher is better)

- **BERTScore F1**: **0.9240**
- **ROUGE-L**: **0.5657**
- **chrF++**: **55.9219**

### Fluency proxy

- **GPT-2 perplexity (output)**: **27.1481**
- **GPT-2 perplexity (human reference)**: **23.6912**

### Linguistic marker shift (style movement)

- **Mean directional marker shift**: **0.8289**

Qualitative note:
- This run is characterized by comparatively **precise targeting** of human marker means on several features (e.g., average word length and lexical diversity were extremely close to human reference means in the project’s analysis).

## Limitations

- This model optimizes reference similarity and controlled rewriting; it may not “push style” as aggressively as decoder-only models that can overshoot.
- No guarantee of bypassing AI detectors.
- Generalization depends on domains/styles present in training data.

## Research paper (unpublished)

Part of an unpublished manuscript (2026):

**“Rewriting the Machine: Encoder-Decoder vs. Decoder-Only Transformers for AI-to-Human Text Style Transfer”**

- Status: **not published yet**
- Link: *(add your PDF/arXiv link when available)*

## License

MIT is a placeholder here—set this repo’s license to what you intend to distribute under, consistent with the base model’s terms.

