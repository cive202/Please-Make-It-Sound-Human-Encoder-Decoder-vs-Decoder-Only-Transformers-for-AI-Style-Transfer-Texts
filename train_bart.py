"""
train_bart.py
-------------
Fine-tunes facebook/bart-base or facebook/bart-large for AI-to-Human style transfer.
Mirrors the T5 approach from the paper (Section 4.3–4.4) but uses BART seq2seq.

Usage:
    python train_bart.py --config bart_base.yaml
    python train_bart.py --config bart_base.yaml --smoke
    python train_bart.py --config bart_large.yaml --output_dir results/bart_large_run2
"""

import argparse
import json
import os
import sys

import yaml


def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 128 samples, 10 steps only")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, {"output_dir": args.output_dir})

    if args.smoke:
        cfg["max_train_samples"] = 128
        cfg["max_eval_samples"] = 32
        cfg["max_steps"] = 10
        cfg["num_train_epochs"] = 1
        cfg["logging_steps"] = 2
        cfg["save_strategy"] = "no"
        cfg["evaluation_strategy"] = "no"
        print("=== SMOKE TEST MODE ===")

    if args.max_train_samples:
        cfg["max_train_samples"] = args.max_train_samples

    # Lazy imports so --smoke on CPU doesn't pull in CUDA unnecessarily
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        BartForConditionalGeneration,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    # ------------------------------------------------------------------ #
    # Optional LoRA for BART
    # ------------------------------------------------------------------ #
    use_lora = cfg.get("use_lora", False)
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            print("ERROR: peft not installed. Run: pip install peft")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    def read_jsonl(path, max_rows=None):
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    print(f"Loading training data from {cfg['train_file']} ...")
    train_rows = read_jsonl(cfg["train_file"], cfg.get("max_train_samples"))
    val_rows = read_jsonl(cfg["val_file"], cfg.get("max_eval_samples"))
    print(f"  Train: {len(train_rows):,}  Val: {len(val_rows):,}")

    prefix = cfg.get("prefix", "humanize: ")

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    print(f"Loading tokenizer: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    max_in = cfg.get("max_input_length", 256)
    max_tgt = cfg.get("max_target_length", 256)

    def tokenize(batch):
        inputs = [prefix + t for t in batch["ai"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_in,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["human"],
                max_length=max_tgt,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_list(train_rows).map(
        tokenize, batched=True, remove_columns=Dataset.from_list(train_rows).column_names
    )
    val_ds = Dataset.from_list(val_rows).map(
        tokenize, batched=True, remove_columns=Dataset.from_list(val_rows).column_names
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print(f"Loading model: {cfg['model_name']}")
    model = BartForConditionalGeneration.from_pretrained(cfg["model_name"])

    if use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=0.05,
            target_modules=cfg.get("lora_target_modules",
                                   ["q_proj", "v_proj", "encoder_attn.q_proj", "encoder_attn.v_proj"]),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        print("LoRA enabled for BART.")

    # ------------------------------------------------------------------ #
    # Training arguments
    # ------------------------------------------------------------------ #
    output_dir = cfg.get("output_dir", "results/bart")
    os.makedirs(output_dir, exist_ok=True)

    # Build kwargs dynamically so max_steps overrides epochs when set in smoke
    training_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 5),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        seed=cfg.get("seed", 42),
        evaluation_strategy=cfg.get("evaluation_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        predict_with_generate=cfg.get("predict_with_generate", True),
        generation_num_beams=cfg.get("num_beams", 4),
        generation_max_new_tokens=cfg.get("max_new_tokens", 128),
        logging_steps=cfg.get("logging_steps", 50),
        report_to=cfg.get("report_to", "none"),
    )

    if "max_steps" in cfg and cfg["max_steps"] > 0:
        training_kwargs["max_steps"] = cfg["max_steps"]

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(f"\nStarting training - model: {cfg['model_name']}  output: {output_dir}")
    trainer.train()

    print("\nSaving final model ...")
    trainer.save_model(os.path.join(output_dir, "checkpoint-best"))
    tokenizer.save_pretrained(os.path.join(output_dir, "checkpoint-best"))

    # Save config alongside checkpoint for reproducibility
    with open(os.path.join(output_dir, "train_config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    print(f"\nDone.  Checkpoint saved to {output_dir}/checkpoint-best")

    # Quick smoke generation
    if args.smoke and val_rows:
        print("\n--- Smoke generation sample ---")
        model.eval()
        sample = val_rows[0]
        inputs = tokenizer(prefix + sample["ai"], return_tensors="pt",
                           max_length=max_in, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=4, max_new_tokens=64)
        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  AI input:    {sample['ai'][:120]}...")
        print(f"  Generated:   {generated[:120]}...")
        print(f"  Human ref:   {sample['human'][:120]}...")


if __name__ == "__main__":
    main()
