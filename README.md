# Please Make It Sound Human: Encoder-Decoder vs Decoder-Only Transformers for AI Style Transfer Texts

Replication and extension of the paper *From Machine to Human* — **BART** (encoder–decoder) and **Mistral 7B** (decoder-only, LoRA on quantized weights), with BERTScore, linguistic markers, and readability metrics.

**Run all commands from the repository root** (where `prepare_data.py` and the YAML configs live).

---

## Project layout

```
BARTvsMistral/
├── combined_success_all_models.jsonl   # optional: your raw pairs (or put under data/raw/)
├── data/processed/                     # train.jsonl, val.jsonl, test.jsonl from prepare_data.py
├── prepare_data.py
├── train_bart.py
├── train_mistral_qlora.py
├── evaluate.py
├── qualitative_examples.py
├── linguistic_markers.py
├── bart_base.yaml
├── bart_large.yaml
├── mistral_qlora.yaml
├── test_label_mask.py
├── requirements.txt
├── requirements-torch-gpu.txt          # CUDA PyTorch (install first on GPU machines)
├── requirements-optional.txt           # optional Unsloth (4-bit only)
└── results/
```

---

## Arch Linux + RTX 3060 (12GB) setup

1. **Python 3.10+** and a virtual environment:

   ```bash
   cd /path/to/BARTvsMistral
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **CUDA PyTorch first** (matches RTX 30xx; adjust `cu124` if your driver needs `cu121` — see [pytorch.org](https://pytorch.org/get-started/locally/)):

   ```bash
   pip install -r requirements-torch-gpu.txt
   ```

3. **Rest of dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Hugging Face** (Mistral weights are gated):

   ```bash
   huggingface-cli login
   ```

5. **GPU check**:

   ```bash
   nvidia-smi
   python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
   ```

6. **NLTK** (first run of `prepare_data.py` can auto-download `punkt` / `punkt_tab`).

Optional: `export HF_HOME=/path/to/large/disk` for model cache.

**CPU-only machines:** install CPU PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) instead of `requirements-torch-gpu.txt`, then `pip install -r requirements.txt`. Training 7B without GPU is not practical.

---

## Precision and VRAM (defaults)

| Component | Default | Notes |
|-----------|---------|--------|
| **BART** | **bfloat16** (`fp16: false`, `bf16: true` in YAML) | Ampere/RTX 30xx supports bf16. |
| **Mistral 7B** | **8-bit weights + LoRA** | Full fp16 7B weights need ~14GB+ and typically **OOM on 12GB**. |
| **Mistral trainer** | `fp16: true` in YAML | Activations/optimizer; base stays 8-bit via bitsandbytes. |

To use **4-bit NF4 + Unsloth** (faster, lower VRAM): in `mistral_qlora.yaml` set `load_in_4bit: true`, `load_in_8bit: false`, then `pip install -r requirements-optional.txt`.

For **evaluation / qualitative** on a **24GB+** GPU, you can load the Mistral base in fp16: pass `--mistral_fp16_base`.

---

## Quick start

```bash
# 1) Data (paths relative to repo root)
python prepare_data.py --input combined_success_all_models.jsonl \
  --output_dir data/processed --chunk_size 200 --seed 42

# 2) BART-base (smoke)
python train_bart.py --config bart_base.yaml --smoke

# 3) BART-base (full)
python train_bart.py --config bart_base.yaml

# 4) BART-large (use LoRA on 12GB if OOM: use_lora: true in bart_large.yaml)
python train_bart.py --config bart_large.yaml

# 5) Mistral 7B (smoke)
python train_mistral_qlora.py --config mistral_qlora.yaml --smoke

# 6) Mistral 7B (full; saves checkpoint-final/)
python train_mistral_qlora.py --config mistral_qlora.yaml

# 7) Evaluate (checkpoint names match train_bart output: checkpoint-best)
python evaluate.py \
  --test_data data/processed/test.jsonl \
  --bart_base_ckpt results/bart_base/checkpoint-best \
  --bart_large_ckpt results/bart_large/checkpoint-best \
  --mistral_ckpt results/mistral_7b/checkpoint-final \
  --output_dir results/

# 8) Qualitative table
python qualitative_examples.py \
  --test_data data/processed/test.jsonl \
  --bart_base_ckpt results/bart_base/checkpoint-best \
  --mistral_ckpt results/mistral_7b/checkpoint-final \
  --n_examples 5 --output paper/qualitative_table.md
```

**Mistral masking test** (run before long training):

```bash
python test_label_mask.py
```

---

## Optional: Unsloth

Only used when `mistral_qlora.yaml` has **4-bit** enabled and **8-bit** disabled.

```bash
pip install -r requirements-optional.txt
```

---

## Hardware notes

| Setting | Notes |
|---------|--------|
| BART-base + bf16 | Comfortable on RTX 3060 12GB. |
| BART-large + bf16 | May need smaller batch or `use_lora: true`. |
| Mistral 7B + 8-bit LoRA | Default; fits 12GB with batch settings in YAML. |
| Smoke tests | `--smoke` uses few steps/samples. |

---

## Evaluation metrics

BERTScore (P/R/F1), ROUGE-L, chrF++, GPT-2 perplexity, 11 linguistic marker shifts, Flesch / Flesch–Kincaid (textstat), vocabulary Jaccard. See `evaluate.py` and `linguistic_markers.py`.

---

## Paper alignment

- Data chunking / splits: `prepare_data.py`
- BART seq2seq: `train_bart.py`
- Decoder-only LoRA: `train_mistral_qlora.py`
- Metrics: `evaluate.py` + `linguistic_markers.py`
- Qualitative table: `qualitative_examples.py`
