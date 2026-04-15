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

    # cive202/flan-t5-base-ai-text-seq2seq

    Fine-tuned **FLAN-T5-base** (`google/flan-t5-base`) for **AI → Human rewriting** (style transfer / “humanization”).

    This README is written to be *Hugging Face–ready*, but the **exact training/eval numbers for this specific T5 run must be filled from your own outputs**. (Your `summary.json` in `BARTvsMistral` contains metrics for BART-base, BART-large, and Mistral-7B; no T5 results are present there.)

    ## What this model does

    Given an AI-generated chunk/passage, the model generates a rewrite intended to:
    - preserve semantic content
    - reduce “LLM-polished” patterns
    - move measurable linguistic markers toward the human reference distribution

    ## Quickstart

    ```bash
    pip install -U "transformers>=4.40.0" torch sentencepiece
    ```

    ```python
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_id = "cive202/flan-t5-base-ai-text-seq2seq"
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

    ## Training (REQUIRED FILL)

    Fill this section with the config used for your T5 run:

    - **Objective**: supervised seq2seq rewriting (conditional generation)
    - **Instruction/prefix/template**: `[REQUIRED_FILL]`
    - **Max input length / max output length**: `[REQUIRED_FILL]`
    - **Epochs or max_steps**: `[REQUIRED_FILL]`
    - **Learning rate / scheduler / warmup**: `[REQUIRED_FILL]`
    - **Effective batch size**: `[REQUIRED_FILL]`
    - **Precision**: `[REQUIRED_FILL: fp16/bf16/fp32]`
    - **Hardware**: `[REQUIRED_FILL]`

    ## Dataset

    If you trained on the same parallel corpus as the BART/Mistral experiments:

    - **Train**: 25,140 pairs
    - **Validation**: 1,390 examples
    - **Test**: 1,390 examples

    Otherwise replace with your actual sizes:
    - **Train/Val/Test**: `[REQUIRED_FILL]`

    Preprocessing (typical in this project family):
    - sentence-aware chunking
    - token budget packing
    - filtering short pairs (<10 words)
    - document-disjoint splitting to avoid leakage

    ## Evaluation (REQUIRED FILL)

    Report the same evaluation suite used across the project:

    - **BERTScore** (P/R/F1)
    - **ROUGE-L**
    - **chrF++**
    - **GPT-2 perplexity** (output / AI input / human reference)
    - **Vocabulary Jaccard**
    - **11 linguistic markers** + **directional marker shift** (clipped to \([-1, 2]\))

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

