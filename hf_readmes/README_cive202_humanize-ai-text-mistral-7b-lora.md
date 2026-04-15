---
language:
- en
license: apache-2.0
tags:
- text-generation
- text2text-generation
- style-transfer
- rewriting
- humanization
- llm
- mistral
- mistral-7b
- instruct
- peft
- lora
- qlora
- bitsandbytes
- evaluation
- bertscore
- rouge
- chrf
library_name: transformers
base_model: mistralai/Mistral-7B-Instruct-v0.2
pipeline_tag: text-generation
---

# cive202/humanize-ai-text-mistral-7b-lora

**LoRA/QLoRA adapter** for **AI → Human text rewriting** (“humanization”) on top of **`mistralai/Mistral-7B-Instruct-v0.2`**.

- **What this repo contains**: *adapter weights only* (PEFT/LoRA), **not** the full base model
- **What it does**: rewrites AI-styled passages into more human-like writing while preserving meaning
- **How to use**: load the base model + attach this adapter via `peft`

## What “AI → Human humanization” means here

Given an AI-generated passage \(x\), the model generates a rewrite \(\hat{y}\) that aims to satisfy:

- **Semantic preservation** (keep meaning close to the human reference)
- **Stylistic transformation** (shift measurable markers toward the human distribution)

This repo is part of a larger comparison across architectures:
- encoder–decoder seq2seq (BART)
- decoder-only instruction-tuned LLM + adapters (Mistral)

## Quickstart (Transformers + PEFT)

> Mistral is a gated model. You may need to accept terms and run `huggingface-cli login`.

```bash
pip install -U "transformers>=4.40.0" "peft>=0.10.0" accelerate bitsandbytes torch
```

### Load base + adapter (4-bit inference recommended)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_id = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_id = "cive202/humanize-ai-text-mistral-7b-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)

base = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,  # QLoRA-style inference on consumer GPUs
)

model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

ai_text = "Large language models often produce fluent, structured prose with recognizable regularities..."

prompt = f\"\"\"### Instruction:
Rewrite the input so it sounds human-written while preserving meaning.

### Input:
{ai_text}

### Response:
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training summary (adapter config)

This adapter was trained with **QLoRA-style fine-tuning**:

- **Base model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Quantization**: 4-bit NF4, **double quantization**, **float16 compute dtype**
- **LoRA**: rank `r = 16`, scaling `α = 32`, dropout `0.05`
- **Target modules**: `q_proj, k_proj, v_proj, o_proj`
- **Max sequence length**: 512
- **Optimizer**: Paged AdamW 32-bit
- **LR / schedule**: `2e-4`, cosine scheduler, warmup ratio `0.05`
- **Training length**: max_steps `500` (checkpoints every 100 steps)
- **Effective batch size**: 8 (`per_device_train_batch_size = 2`, `gradient_accumulation_steps = 4`)
- **Loss masking**: completion-only loss on the `### Response:` span

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

Metadata fields include: `doc_id`, `chunk_idx`, `ai`, `human`, plus `style`, `model`, `prompt_id`.

## Evaluation (test n = 1,390)

All metrics were computed on the **same 1,390-example test subset**.

### Reference similarity (higher is better)

- **BERTScore F1**: **0.8980**
- **ROUGE-L**: **0.4642**
- **chrF++**: **55.6770**

### Fluency proxy (interpret cautiously)

- **GPT-2 perplexity (output)**: **9.0325**
- **GPT-2 perplexity (AI input)**: 37.8485
- **GPT-2 perplexity (human reference)**: **23.6912**

Very low perplexity can indicate text that is *highly predictable*, which is not always the same as “human-like.”

### Linguistic marker shift (style movement)

- **Mean directional marker shift**: **1.2788**

Per-marker shift highlights (directional shift, clipped to \([-1, 2]\)):
- multiple **overshoots** (capped at 2.0 on 5 markers)
- **Commas shift = −1.0** (**wrong direction**)

## Known failure modes / limitations

- **Overshoot**: can move *past* human means (becoming differently non-human).
- **Wrong-direction movement**: some markers can shift away from the human target.
- **Detector claims**: this is *not* guaranteed to fool AI detectors.
- **Domain dependence**: quality depends on domains/styles present in the parallel data.

## Reproducibility

This adapter’s training + evaluation pipeline is implemented in the companion codebase `BARTvsMistral` (unpublished research project).

If you have `summary.json` from evaluation, you can regenerate paper-ready figures via a script like `generate_figures.py` (not included in this model repo).

## Research paper (unpublished)

This model is part of an unpublished manuscript (2026):

**“Rewriting the Machine: Encoder-Decoder vs. Decoder-Only Transformers for AI-to-Human Text Style Transfer”**

- Status: **not published yet**
- Link: *(add your PDF/arXiv link when available)*

## License

Adapter weights: Apache-2.0 (as tagged above).  
Base model usage: follow `mistralai/Mistral-7B-Instruct-v0.2` terms.

