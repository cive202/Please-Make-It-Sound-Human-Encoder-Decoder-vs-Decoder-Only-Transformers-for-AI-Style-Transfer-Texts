"""
train_mistral_qlora.py
----------------------
Fine-tunes Mistral 7B for AI-to-Human style transfer with LoRA.
Default config uses 8-bit bitsandbytes (fits RTX 3060 12GB; higher fidelity than 4-bit NF4).
Optional: set load_in_4bit: true and load_in_8bit: false for 4-bit QLoRA or Unsloth.

CRITICAL — label masking:
  Loss is computed ONLY on the human_text continuation (### Response: section).
  Prompt positions are masked with labels = -100.
  This is validated in tests/test_label_mask.py.

Usage:
    python train_mistral_qlora.py --config mistral_qlora.yaml
    python train_mistral_qlora.py --config mistral_qlora.yaml --smoke

Setup:
  - huggingface-cli login (Mistral is gated)
  - pip install -r requirements-torch-gpu.txt && pip install -r requirements.txt
  - Optional Unsloth (4-bit only): pip install -r requirements-optional.txt
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


def build_prompt(ai_text: str, human_text: str, instruction_template: str) -> str:
    """Fill the instruction template for training (prompt + completion)."""
    return instruction_template.format(ai_text=ai_text, human_text=human_text)


def build_prompt_inference(ai_text: str, instruction_template: str) -> str:
    """Fill prompt for inference — no human_text, ends after '### Response:\n'."""
    # Strip everything from {human_text} onward, keep the response prefix
    base = instruction_template.split("{human_text}")[0]
    return base.format(ai_text=ai_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 128 samples, 10 steps")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, {"output_dir": args.output_dir})

    if args.smoke:
        cfg["max_train_samples"] = 128
        cfg["max_steps"] = 10
        cfg["logging_steps"] = 2
        cfg["save_strategy"] = "no"
        print("=== SMOKE TEST MODE ===")

    # ------------------------------------------------------------------ #
    # Imports (lazy to avoid CUDA errors in non-GPU environments)
    # ------------------------------------------------------------------ #
    import torch
    from transformers import AutoTokenizer, TrainingArguments

    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    load_in_8bit = cfg.get("load_in_8bit")
    if load_in_8bit is None:
        load_in_8bit = not load_in_4bit
    else:
        load_in_8bit = bool(load_in_8bit)
    if load_in_4bit and load_in_8bit:
        print("ERROR: set only one of load_in_4bit or load_in_8bit in the YAML.")
        sys.exit(1)

    _unsloth_imported = False
    try:
        from unsloth import FastLanguageModel
        _unsloth_imported = True
    except ImportError:
        FastLanguageModel = None  # type: ignore

    # Unsloth only supports the 4-bit path; 8-bit uses HuggingFace + bitsandbytes
    _USE_UNSLOTH = bool(
        _unsloth_imported and load_in_4bit and not load_in_8bit
    )
    if _unsloth_imported and load_in_8bit:
        print("Using HuggingFace + 8-bit bitsandbytes (Unsloth skipped for 8-bit training).")
    elif _USE_UNSLOTH:
        print("Unsloth detected - using FastLanguageModel (4-bit).")
    else:
        print("Using HuggingFace Transformers + bitsandbytes.")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model

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

    print(f"Loading data ...")
    train_rows = read_jsonl(cfg["train_file"], cfg.get("max_train_samples"))
    print(f"  Train: {len(train_rows):,}")

    instruction_template = cfg.get("instruction_template", (
        "### Instruction:\n"
        "Rewrite the following AI-generated text to sound natural and human-written.\n"
        "Keep the same meaning. Use a conversational tone. Vary sentence length naturally.\n\n"
        "### Input:\n{ai_text}\n\n"
        "### Response:\n{human_text}"
    ))
    response_template = cfg.get("response_template", "### Response:\n")

    max_seq_length = cfg.get("max_seq_length", 512)

    # ------------------------------------------------------------------ #
    # Model + tokenizer
    # ------------------------------------------------------------------ #
    model_name = cfg["model_name"]
    print(f"Loading model: {model_name}")

    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules",
                                   ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
            use_gradient_checkpointing=True,
            random_state=cfg.get("seed", 42),
        )
    else:
        from peft import prepare_model_for_kbit_training

        compute_dtype = torch.float16
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            bnb_config = None

        if bnb_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=compute_dtype,
            )
            model.config.use_cache = False
            model.enable_input_require_grads()
            model = prepare_model_for_kbit_training(model)
        else:
            _bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available() and _bf16
                else torch.float16,
            )
            model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        lora_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules",
                                   ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # Dataset tokenization with -100 label masking on prompt
    # ------------------------------------------------------------------ #
    from datasets import Dataset as HFDataset

    # Build text column for SFT-style training
    formatted_texts = []
    for row in train_rows:
        text = build_prompt(row["ai"], row["human"], instruction_template)
        formatted_texts.append({"text": text})

    train_dataset = HFDataset.from_list(formatted_texts)

    # ------------------------------------------------------------------ #
    # Training arguments
    # ------------------------------------------------------------------ #
    output_dir = cfg.get("output_dir", "results/mistral_7b")
    os.makedirs(output_dir, exist_ok=True)

    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        optim=cfg.get("optim", "paged_adamw_32bit"),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        seed=cfg.get("seed", 42),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        evaluation_strategy="no",
        logging_steps=cfg.get("logging_steps", 25),
        report_to=cfg.get("report_to", "none"),
    )

    if "max_steps" in cfg and cfg["max_steps"] > 0:
        training_kwargs["max_steps"] = cfg["max_steps"]
    else:
        training_kwargs["num_train_epochs"] = cfg.get("num_train_epochs", 3)

    training_args = TrainingArguments(**training_kwargs)

    # ------------------------------------------------------------------ #
    # Trainer (manual masking only; no TRL dependency)
    # ------------------------------------------------------------------ #
    from transformers import Trainer

    resp_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def tokenize_with_mask(examples):
        # Non-batched map: examples["text"] is a single string
        full_text = examples["text"]
        tokenized = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", None)

        labels = input_ids.copy()

        # Only search for the response template inside the non-padded region
        if attention_mask is not None:
            actual_len = int(sum(attention_mask))
        else:
            actual_len = len(input_ids)

        prompt_end = actual_len
        for i in range(max(0, actual_len - len(resp_ids) + 1)):
            if input_ids[i : i + len(resp_ids)] == resp_ids:
                prompt_end = i + len(resp_ids)
                break

        # Mask prompt (including response template header)
        labels[:prompt_end] = [-100] * prompt_end

        # Also mask padding tokens in labels
        if attention_mask is not None:
            labels = [-100 if am == 0 else lab for lab, am in zip(labels, attention_mask)]

        tokenized["labels"] = labels
        return tokenized

    train_dataset_tok = train_dataset.map(tokenize_with_mask, remove_columns=["text"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tok,
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print(f"\nStarting training - model: {model_name}  output: {output_dir}")
    print(f"  Effective batch size: "
          f"{cfg.get('per_device_train_batch_size',2) * cfg.get('gradient_accumulation_steps',4)}")
    trainer.train()

    print("\nSaving model ...")
    final_ckpt = os.path.join(output_dir, "checkpoint-final")
    trainer.save_model(final_ckpt)
    tokenizer.save_pretrained(final_ckpt)

    # Save config for reproducibility
    with open(os.path.join(output_dir, "train_config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    print(f"\nDone.  Checkpoint: {final_ckpt}")

    # ------------------------------------------------------------------ #
    # Smoke generation
    # ------------------------------------------------------------------ #
    if args.smoke:
        print("\n--- Smoke generation sample ---")
        sample_ai = "The implementation of autonomous vehicle technology presents significant challenges "
        "regarding infrastructure integration and regulatory compliance frameworks."
        prompt = build_prompt_inference(sample_ai, instruction_template)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"  AI:        {sample_ai[:100]}...")
        print(f"  Generated: {generated[:100]}...")


if __name__ == "__main__":
    main()
