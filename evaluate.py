"""
evaluate.py
-----------
Full evaluation suite for BART-base, BART-large, and Mistral 7B.

Metrics computed:
  - BERTScore (P / R / F1)            reference-based semantic similarity
  - ROUGE-L                           n-gram overlap
  - chrF++                            character n-gram F-score (sacrebleu)
  - Perplexity (GPT-2)               reference-free fluency proxy
  - Linguistic marker shift (11)     how far output moved toward human distribution
  - Flesch Reading Ease              readability (via textstat)
  - Flesch-Kincaid Grade Level       readability (via textstat)
  - Vocabulary Jaccard               type-level overlap with human reference

Usage:
    python evaluate.py --test_data data/processed/test.jsonl --bart_base_ckpt results/bart_base/checkpoint-best --output_dir results/
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str, max_rows: int = None) -> List[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def vocab_jaccard(text_a: str, text_b: str) -> float:
    """Type-level Jaccard similarity between two texts."""
    import re
    def types(t):
        return set(re.findall(r"\b[a-zA-Z]+\b", t.lower()))
    a, b = types(text_a), types(text_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity_gpt2(texts: List[str], batch_size: int = 8,
                             device: str = "cpu") -> float:
    """Compute average per-token perplexity using a frozen GPT-2 model.

    Lower = more fluent (model assigns higher probability to the text).
    Uses 'gpt2' (124M params) as a lightweight, widely-available proxy.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading frozen GPT-2 for perplexity ...")
    ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    ppl_model.eval()
    ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

    total_log_prob = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = ppl_tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(device)
            labels = enc["input_ids"].clone()
            labels[labels == ppl_tokenizer.pad_token_id] = -100
            out = ppl_model(**enc, labels=labels)
            # out.loss is mean cross-entropy over non-masked tokens
            n_tokens = (labels != -100).sum().item()
            total_log_prob += out.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_ce = total_log_prob / max(total_tokens, 1)
    ppl = math.exp(avg_ce)
    return round(ppl, 4)


# ---------------------------------------------------------------------------
# BART inference
# ---------------------------------------------------------------------------

def generate_bart(checkpoint: str, texts: List[str], prefix: str = "humanize: ",
                  num_beams: int = 4, max_new_tokens: int = 128,
                  batch_size: int = 8) -> List[str]:
    import torch
    from transformers import AutoTokenizer, BartForConditionalGeneration

    # Handle LoRA checkpoints
    try:
        from peft import PeftModel, PeftConfig
        peft_cfg_path = os.path.join(checkpoint, "adapter_config.json")
        if os.path.exists(peft_cfg_path):
            peft_config = PeftConfig.from_pretrained(checkpoint)
            base_model = BartForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
            model = PeftModel.from_pretrained(base_model, checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            print(f"  Loaded BART + LoRA from {checkpoint}")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ImportError):
        model = BartForConditionalGeneration.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"  Loaded BART (full) from {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i: i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        outputs.extend(decoded)
        if (i // batch_size) % 5 == 0:
            print(f"    Generated {min(i+batch_size, len(texts))}/{len(texts)} ...")
    return outputs


# ---------------------------------------------------------------------------
# Mistral inference
# ---------------------------------------------------------------------------

def build_mistral_prompt(ai_text: str, instruction_template: str) -> str:
    base = instruction_template.split("{human_text}")[0]
    return base.format(ai_text=ai_text)


def generate_mistral(checkpoint: str, texts: List[str],
                     instruction_template: str,
                     num_beams: int = 4, max_new_tokens: int = 128,
                     batch_size: int = 4,
                     load_in_8bit: bool = True) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    print(f"  Loading Mistral from {checkpoint}")
    peft_config = PeftConfig.from_pretrained(checkpoint)
    base_name = peft_config.base_model_name_or_path
    use_8bit = load_in_8bit and torch.cuda.is_available()
    if use_8bit:
        print("  Base model: 8-bit bitsandbytes (fits 12GB VRAM)")
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        print("  Base model: fp16 (needs enough VRAM for 7B)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        prompts = [build_mistral_prompt(t, instruction_template) for t in batch_texts]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=400).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, seq in enumerate(out):
            prompt_len = enc["input_ids"].shape[1]
            new_tokens = seq[prompt_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(decoded.strip())
        if (i // batch_size) % 5 == 0:
            print(f"    Generated {min(i+batch_size, len(texts))}/{len(texts)} ...")
    return outputs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_bertscore(predictions: List[str], references: List[str]) -> Dict:
    from bert_score import score as bs_score
    print("  Computing BERTScore ...")
    P, R, F1 = bs_score(predictions, references, lang="en", rescale_with_baseline=False,
                         verbose=False)
    return {
        "bertscore_precision": round(float(P.mean()), 4),
        "bertscore_recall": round(float(R.mean()), 4),
        "bertscore_f1": round(float(F1.mean()), 4),
    }


def compute_rouge(predictions: List[str], references: List[str]) -> Dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    rl_f = np.mean([s["rougeL"].fmeasure for s in scores])
    return {"rouge_l": round(float(rl_f), 4)}


def compute_chrf(predictions: List[str], references: List[str]) -> Dict:
    import sacrebleu
    # chrF++ (word_order=2)
    result = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    return {"chrf_pp": round(result.score, 4)}


def compute_all_metrics(
    predictions: List[str],
    ai_inputs: List[str],
    human_refs: List[str],
    model_label: str,
    device: str = "cpu",
) -> Dict:
    """Compute the full metric suite for one model's predictions."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from linguistic_markers import (
        compute_markers, compute_marker_shift, average_markers, print_marker_table
    )

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_label}")
    print(f"{'='*60}")

    results = {"model": model_label, "n_samples": len(predictions)}

    # BERTScore
    results.update(compute_bertscore(predictions, human_refs))

    # ROUGE-L
    print("  Computing ROUGE-L ...")
    results.update(compute_rouge(predictions, human_refs))

    # chrF++
    print("  Computing chrF++ ...")
    results.update(compute_chrf(predictions, human_refs))

    # Perplexity (GPT-2)
    print("  Computing perplexity (GPT-2) ...")
    results["perplexity_gpt2"] = compute_perplexity_gpt2(predictions, device=device)

    # Perplexity of AI inputs and human refs for comparison
    print("  Computing perplexity of AI inputs ...")
    results["perplexity_ai_input"] = compute_perplexity_gpt2(ai_inputs, device=device)
    print("  Computing perplexity of human references ...")
    results["perplexity_human_ref"] = compute_perplexity_gpt2(human_refs, device=device)

    # Vocabulary Jaccard
    print("  Computing vocabulary Jaccard ...")
    jaccards = [vocab_jaccard(pred, ref) for pred, ref in zip(predictions, human_refs)]
    results["vocab_jaccard"] = round(float(np.mean(jaccards)), 4)

    # Linguistic markers
    print("  Computing linguistic markers ...")
    ai_markers_list = [compute_markers(t) for t in ai_inputs]
    out_markers_list = [compute_markers(t) for t in predictions]
    hum_markers_list = [compute_markers(t) for t in human_refs]

    ai_avg = average_markers(ai_markers_list)
    out_avg = average_markers(out_markers_list)
    hum_avg = average_markers(hum_markers_list)

    shifts = compute_marker_shift(ai_avg, out_avg, hum_avg)
    results["marker_shift_mean"] = round(float(np.mean(list(shifts.values()))), 4)
    results["marker_shifts"] = {k: round(v, 4) for k, v in shifts.items()}
    results["marker_averages_ai"] = ai_avg
    results["marker_averages_output"] = out_avg
    results["marker_averages_human"] = hum_avg

    print_marker_table(ai_avg, out_avg, hum_avg, model_label)

    # Print summary
    print(f"\n  Summary:")
    print(f"    BERTScore F1:      {results['bertscore_f1']:.4f}")
    print(f"    ROUGE-L:           {results['rouge_l']:.4f}")
    print(f"    chrF++:            {results['chrf_pp']:.4f}")
    print(f"    Perplexity (pred): {results['perplexity_gpt2']:.2f}")
    print(f"    Perplexity (AI):   {results['perplexity_ai_input']:.2f}  (baseline)")
    print(f"    Perplexity (hum):  {results['perplexity_human_ref']:.2f}  (target)")
    print(f"    Vocab Jaccard:     {results['vocab_jaccard']:.4f}")
    print(f"    Marker shift mean: {results['marker_shift_mean']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bart_base_ckpt", default=None)
    parser.add_argument("--bart_large_ckpt", default=None)
    parser.add_argument("--mistral_ckpt", default=None)
    parser.add_argument("--mistral_config", default="configs/mistral_qlora.yaml",
                        help="YAML with instruction_template for Mistral inference")
    parser.add_argument(
        "--mistral_fp16_base",
        action="store_true",
        help="Load Mistral base in fp16 (24GB+ VRAM). Default: 8-bit base on CUDA.",
    )
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Evaluate on first N test examples (None = all)")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test data
    print(f"Loading test data: {args.test_data}")
    test_rows = read_jsonl(args.test_data, args.n_samples)
    print(f"  {len(test_rows):,} examples")

    ai_inputs = [r["ai"] for r in test_rows]
    human_refs = [r["human"] for r in test_rows]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    # ------------------------------------------------------------------ #
    # BART-base
    # ------------------------------------------------------------------ #
    if args.bart_base_ckpt:
        print(f"\nGenerating - BART-base ...")
        preds = generate_bart(args.bart_base_ckpt, ai_inputs,
                              num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                              batch_size=args.batch_size)
        metrics = compute_all_metrics(preds, ai_inputs, human_refs, "BART-base", device)
        metrics["predictions"] = preds
        all_results.append(metrics)

        pred_path = os.path.join(args.output_dir, "bart_base", "predictions.jsonl")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, "w") as f:
            for ai, hum, pred in zip(ai_inputs, human_refs, preds):
                f.write(json.dumps({"ai": ai, "human": hum, "prediction": pred}) + "\n")

    # ------------------------------------------------------------------ #
    # BART-large
    # ------------------------------------------------------------------ #
    if args.bart_large_ckpt:
        print(f"\nGenerating - BART-large ...")
        preds = generate_bart(args.bart_large_ckpt, ai_inputs,
                              num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                              batch_size=max(1, args.batch_size // 2))
        metrics = compute_all_metrics(preds, ai_inputs, human_refs, "BART-large", device)
        metrics["predictions"] = preds
        all_results.append(metrics)

        pred_path = os.path.join(args.output_dir, "bart_large", "predictions.jsonl")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, "w") as f:
            for ai, hum, pred in zip(ai_inputs, human_refs, preds):
                f.write(json.dumps({"ai": ai, "human": hum, "prediction": pred}) + "\n")

    # ------------------------------------------------------------------ #
    # Mistral 7B
    # ------------------------------------------------------------------ #
    if args.mistral_ckpt:
        import yaml
        with open(args.mistral_config) as f:
            mistral_cfg = yaml.safe_load(f)
        instruction_template = mistral_cfg.get("instruction_template", (
            "### Instruction:\n"
            "Rewrite the following AI-generated text to sound natural and human-written.\n"
            "Keep the same meaning. Use a conversational tone. Vary sentence length naturally.\n\n"
            "### Input:\n{ai_text}\n\n"
            "### Response:\n{human_text}"
        ))

        print(f"\nGenerating - Mistral 7B ...")
        preds = generate_mistral(
            args.mistral_ckpt,
            ai_inputs,
            instruction_template,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=max(1, args.batch_size // 4),
            load_in_8bit=not args.mistral_fp16_base,
        )
        metrics = compute_all_metrics(preds, ai_inputs, human_refs, "Mistral-7B", device)
        metrics["predictions"] = preds
        all_results.append(metrics)

        pred_path = os.path.join(args.output_dir, "mistral_7b", "predictions.jsonl")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, "w") as f:
            for ai, hum, pred in zip(ai_inputs, human_refs, preds):
                f.write(json.dumps({"ai": ai, "human": hum, "prediction": pred}) + "\n")

    # ------------------------------------------------------------------ #
    # Save summary
    # ------------------------------------------------------------------ #
    if not all_results:
        print("No model checkpoints provided. Pass at least one of: "
              "--bart_base_ckpt, --bart_large_ckpt, --mistral_ckpt")
        sys.exit(1)

    # Strip predictions from summary (already saved per-model)
    summary = []
    for r in all_results:
        r_clean = {k: v for k, v in r.items() if k != "predictions"}
        summary.append(r_clean)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    cols = ["model", "bertscore_f1", "rouge_l", "chrf_pp",
            "perplexity_gpt2", "vocab_jaccard", "marker_shift_mean"]
    print(f"{'Model':<20} {'BS-F1':>8} {'ROUGE-L':>8} {'chrF++':>8} "
          f"{'PPL':>8} {'VocJac':>8} {'MkShift':>9}")
    print("-" * 80)
    for r in summary:
        print(f"  {r['model']:<18} {r.get('bertscore_f1',0):>8.4f} "
              f"{r.get('rouge_l',0):>8.4f} {r.get('chrf_pp',0):>8.2f} "
              f"{r.get('perplexity_gpt2',0):>8.2f} {r.get('vocab_jaccard',0):>8.4f} "
              f"{r.get('marker_shift_mean',0):>9.4f}")


if __name__ == "__main__":
    main()
