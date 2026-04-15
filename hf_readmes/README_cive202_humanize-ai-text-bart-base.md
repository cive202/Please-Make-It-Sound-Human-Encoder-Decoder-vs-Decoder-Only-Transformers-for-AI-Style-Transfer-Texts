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
base_model: facebook/bart-base
pipeline_tag: text2text-generation
---

# cive202/humanize-ai-text-bart-base

Fine-tuned **BART-base** (`facebook/bart-base`) for **AI → Human rewriting** (“humanization”) via prefix-based conditional generation.

- **Architecture**: encoder–decoder (seq2seq)
- **Parameters**: ~139M
- **Task format**: `humanize: {ai_text}` → `{human_text}`

## Quickstart

```bash
pip install -U "transformers>=4.40.0" torch sentencepiece
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "cive202/humanize-ai-text-bart-base"
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

## Training note (important)

This checkpoint corresponds to a **smoke-test / pipeline validation run**, not a full training run.

Saved config characteristics (from the project’s training config):
- `max_steps = 10`
- `max_train_samples = 128`
- `num_train_epochs = 1`

Interpret results below as a **constrained lower-bound baseline**, not as a fully optimized BART-base model.

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

### Reference similarity

- **BERTScore F1**: **0.9088**
- **ROUGE-L**: **0.4448**
- **chrF++**: **46.4131**

### Fluency proxy

- **GPT-2 perplexity (output)**: **26.6919**
- **GPT-2 perplexity (human reference)**: **23.6912**

### Linguistic marker shift

- **Mean directional marker shift**: **0.6513**

This baseline shifts markers partially toward the human distribution but is limited by the smoke-test training regime.

## Limitations

- Not a fully trained BART-base run (smoke-test config).
- Style shift may be modest compared to larger/fuller runs.
- No guarantee of bypassing AI detectors.

## Research paper (unpublished)

Part of an unpublished manuscript (2026):

**“Rewriting the Machine: Encoder-Decoder vs. Decoder-Only Transformers for AI-to-Human Text Style Transfer”**

- Status: **not published yet**
- Link: *(add your PDF/arXiv link when available)*

## License

MIT is a placeholder here—set this repo’s license to what you intend to distribute under, consistent with the base model’s terms.

