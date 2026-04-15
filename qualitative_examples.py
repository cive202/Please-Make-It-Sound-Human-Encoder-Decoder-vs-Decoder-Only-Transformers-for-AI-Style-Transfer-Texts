"""
qualitative_examples.py
-----------------------
Generates the qualitative comparison table for the paper (mirrors Table in Section 6.4).
Samples N examples from the test set, runs each model, prints and saves a Markdown table.

Usage:
    python qualitative_examples.py --test_data data/processed/test.jsonl --mistral_ckpt results/mistral_7b/checkpoint-final --n_examples 5 --output paper/qualitative_table.md
"""

import argparse
import json
import os
import sys

import yaml


def read_jsonl(path, max_rows=None):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line.strip()))
    return rows


def truncate(text, max_chars=300):
    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


def escape_pipe(text):
    return text.replace("|", "\\|").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--bart_base_ckpt", default=None)
    parser.add_argument("--bart_large_ckpt", default=None)
    parser.add_argument("--mistral_ckpt", default=None)
    parser.add_argument("--mistral_config", default="configs/mistral_qlora.yaml")
    parser.add_argument(
        "--mistral_fp16_base",
        action="store_true",
        help="Load Mistral base in fp16 (24GB+ VRAM). Default: 8-bit on CUDA.",
    )
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--output", default="paper/qualitative_table.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    rows = read_jsonl(args.test_data)
    # Sample diverse examples: pick evenly spaced for reproducibility
    step = max(1, len(rows) // args.n_examples)
    samples = rows[::step][: args.n_examples]

    ai_texts = [r["ai"] for r in samples]
    human_texts = [r["human"] for r in samples]

    model_preds = {}

    # BART-base
    if args.bart_base_ckpt:
        import torch
        from transformers import AutoTokenizer, BartForConditionalGeneration
        print("Loading BART-base ...")
        tokenizer = AutoTokenizer.from_pretrained(args.bart_base_ckpt)
        model = BartForConditionalGeneration.from_pretrained(args.bart_base_ckpt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        preds = []
        for ai in ai_texts:
            inp = tokenizer("humanize: " + ai, return_tensors="pt",
                            max_length=256, truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inp, num_beams=4, max_new_tokens=128)
            preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        model_preds["BART-base"] = preds
        del model

    # BART-large
    if args.bart_large_ckpt:
        import torch
        from transformers import AutoTokenizer, BartForConditionalGeneration
        print("Loading BART-large ...")
        tokenizer = AutoTokenizer.from_pretrained(args.bart_large_ckpt)
        model = BartForConditionalGeneration.from_pretrained(args.bart_large_ckpt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        preds = []
        for ai in ai_texts:
            inp = tokenizer("humanize: " + ai, return_tensors="pt",
                            max_length=256, truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inp, num_beams=4, max_new_tokens=128)
            preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        model_preds["BART-large"] = preds
        del model

    # Mistral 7B
    if args.mistral_ckpt:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, PeftConfig
        with open(args.mistral_config) as f:
            mistral_cfg = yaml.safe_load(f)
        instruction_template = mistral_cfg.get("instruction_template", (
            "### Instruction:\nRewrite the following AI-generated text to sound natural "
            "and human-written.\nKeep the same meaning.\n\n### Input:\n{ai_text}\n\n### Response:\n{human_text}"
        ))
        print("Loading Mistral 7B ...")
        peft_config = PeftConfig.from_pretrained(args.mistral_ckpt)
        base_name = peft_config.base_model_name_or_path
        from transformers import BitsAndBytesConfig
        if not args.mistral_fp16_base and torch.cuda.is_available():
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                quantization_config=bnb,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype=torch.float16, device_map="auto"
            )
        model = PeftModel.from_pretrained(base_model, args.mistral_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(args.mistral_ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        preds = []
        for ai in ai_texts:
            prompt = instruction_template.split("{human_text}")[0].format(ai_text=ai)
            inp = tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=128, num_beams=4,
                                     pad_token_id=tokenizer.eos_token_id)
            new = out[0][inp["input_ids"].shape[1]:]
            preds.append(tokenizer.decode(new, skip_special_tokens=True).strip())
        model_preds["Mistral-7B"] = preds
        del model

    # ------------------------------------------------------------------ #
    # Build markdown table
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model_cols = list(model_preds.keys())
    header_cols = ["AI Input"] + model_cols + ["Human Reference"]
    header = "| " + " | ".join(header_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(header_cols)) + " |"

    lines = [
        "# Qualitative Examples",
        "",
        "Each row shows one test passage. Columns show the AI input, each model's output, "
        "and the original human reference.",
        "",
        header,
        separator,
    ]

    for i in range(len(samples)):
        cells = [escape_pipe(truncate(ai_texts[i]))]
        for col in model_cols:
            pred = model_preds[col][i] if i < len(model_preds[col]) else "(not generated)"
            cells.append(escape_pipe(truncate(pred)))
        cells.append(escape_pipe(truncate(human_texts[i])))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("*Outputs truncated to 300 characters for readability.*")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nQualitative table saved → {args.output}")

    # Also print a plain-text version
    print("\n--- QUALITATIVE EXAMPLES ---")
    for i in range(len(samples)):
        print(f"\nExample {i+1}:")
        print(f"  AI:      {truncate(ai_texts[i], 150)}")
        for col in model_cols:
            pred = model_preds[col][i] if i < len(model_preds[col]) else "(n/a)"
            print(f"  {col:<14}: {truncate(pred, 150)}")
        print(f"  Human:   {truncate(human_texts[i], 150)}")


if __name__ == "__main__":
    main()
