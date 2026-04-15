"""
prepare_data.py
---------------
Loads combined_success_all_models.jsonl (or any JSONL with 'ai' and 'human' fields),
chunks long passages (mirrors paper Section 4.2):

  - NLTK sentence segmentation
  - Subword token counts via a Hugging Face tokenizer (default: facebook/bart-base),
    not word-count heuristics, so ~400+ word passages split into multiple <=200-token chunks
  - If a single sentence exceeds the budget (common for run-on AI text), split at word
    boundaries into <=200-token pieces (paper avoids mid-sentence cuts when possible;
    this is the practical fallback)
  - When AI and human yield the same number of prepared segments, greedy joint packing:
    chunk i aligns chunk i (same sentence groups on both sides)
  - Otherwise independent greedy packing per side, then zip to min(lengths) (position alignment
    for the shared prefix of chunks)
  - Drop chunk pairs where either side has fewer than min_words words

splits by document id (so all styles of the same doc stay in one split),
deduplicates, and writes train/val/test JSONL files.

Usage:
    python prepare_data.py --input combined_success_all_models.jsonl --output_dir data/processed --chunk_size 200 --seed 42
    python prepare_data.py ... --max_rows 500   # smoke subset
"""

import argparse
import hashlib
import json
import os
import random
from typing import List, Tuple

import nltk

# Download punkt if not already present
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------------------------
# Chunking (Section 4.2 — tokenizer-based, sentence-aware)
# ---------------------------------------------------------------------------

def token_len(tokenizer, text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(tokenizer.encode(text.strip(), add_special_tokens=False))


def split_text_max_tokens(tokenizer, text: str, max_tokens: int) -> List[str]:
    """Split *text* into segments each with <= max_tokens subword tokens (word-boundary greedy)."""
    text = text.strip()
    if not text:
        return []
    if token_len(tokenizer, text) <= max_tokens:
        return [text]

    words = text.split()
    out: List[str] = []
    cur: List[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if token_len(tokenizer, trial) <= max_tokens:
            cur.append(w)
        else:
            if cur:
                out.append(" ".join(cur))
                cur = [w]
            else:
                # Single token/word still too long — keep as unavoidable segment
                out.append(w)
                cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def sentences_prepared(text: str, tokenizer, max_tokens: int) -> List[str]:
    """NLTK sentences, then split any oversize sentence into <=max_tokens pieces."""
    segs: List[str] = []
    for sent in sent_tokenize(text.strip()):
        if not sent.strip():
            continue
        if token_len(tokenizer, sent) <= max_tokens:
            segs.append(sent.strip())
        else:
            segs.extend(split_text_max_tokens(tokenizer, sent, max_tokens))
    return segs


def chunk_one_side_greedy(
    segments: List[str], tokenizer, max_tokens: int, min_words: int
) -> List[str]:
    """Greedily pack prepared segments into chunks, each side <= max_tokens."""
    if not segments:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    for seg in segments:
        trial = " ".join(cur + [seg]).strip()
        if cur and token_len(tokenizer, trial) > max_tokens:
            chunks.append(" ".join(cur).strip())
            cur = [seg]
        else:
            cur.append(seg)
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if len(c.split()) >= min_words]


def chunk_joint_from_segments(
    ai_segs: List[str],
    hu_segs: List[str],
    tokenizer,
    max_tokens: int,
    min_words: int,
) -> List[Tuple[str, str]]:
    """Greedy joint chunks: same grouping index on AI and human; each side <= max_tokens."""
    out: List[Tuple[str, str]] = []
    cur_a: List[str] = []
    cur_h: List[str] = []
    for a, h in zip(ai_segs, hu_segs):
        trial_a = " ".join(cur_a + [a]).strip()
        trial_h = " ".join(cur_h + [h]).strip()
        over = (
            cur_a
            and (
                token_len(tokenizer, trial_a) > max_tokens
                or token_len(tokenizer, trial_h) > max_tokens
            )
        )
        if over:
            sa = " ".join(cur_a).strip()
            sh = " ".join(cur_h).strip()
            if len(sa.split()) >= min_words and len(sh.split()) >= min_words:
                out.append((sa, sh))
            cur_a, cur_h = [a], [h]
        else:
            cur_a.append(a)
            cur_h.append(h)
    if cur_a:
        sa = " ".join(cur_a).strip()
        sh = " ".join(cur_h).strip()
        if len(sa.split()) >= min_words and len(sh.split()) >= min_words:
            out.append((sa, sh))
    return out


def chunk_pair(
    ai_text: str,
    human_text: str,
    tokenizer,
    max_tokens: int = 200,
    min_words: int = 10,
) -> List[Tuple[str, str]]:
    """Chunk AI/human passage pair; align by joint grouping when segment counts match."""
    ai_segs = sentences_prepared(ai_text, tokenizer, max_tokens)
    hu_segs = sentences_prepared(human_text, tokenizer, max_tokens)

    if len(ai_segs) == len(hu_segs) and ai_segs:
        pairs = chunk_joint_from_segments(ai_segs, hu_segs, tokenizer, max_tokens, min_words)
        return pairs, "joint"

    ac = chunk_one_side_greedy(ai_segs, tokenizer, max_tokens, min_words)
    hc = chunk_one_side_greedy(hu_segs, tokenizer, max_tokens, min_words)
    pairs = []
    for ai_c, h_c in zip(ac, hc):
        if len(ai_c.split()) >= min_words and len(h_c.split()) >= min_words:
            pairs.append((ai_c, h_c))
    return pairs, "zip_min"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_rows: int = None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_id(row: dict, i: int) -> str:
    """Return a stable document id, falling back to a hash of the human text."""
    if "id" in row and row["id"]:
        return str(row["id"])
    # Fall back to first-64-chars hash of human text for grouping
    key = (row.get("human") or row.get("human_text") or str(i))[:128]
    return hashlib.md5(key.encode()).hexdigest()[:16]


def get_text_fields(row: dict) -> Tuple[str, str]:
    """Resolve column names flexibly."""
    ai = row.get("ai") or row.get("ai_text") or row.get("ai_generated") or ""
    human = row.get("human") or row.get("human_text") or row.get("human_written") or ""
    return ai.strip(), human.strip()


def deduplicate_on_ai(pairs: List[dict]) -> List[dict]:
    """Remove rows with duplicate ai text (keeps first occurrence)."""
    seen = set()
    out = []
    for p in pairs:
        key = p["ai"][:200]  # hash prefix is cheap and sufficient
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def split_by_id(doc_ids: List[str], seed: int, train_r: float, val_r: float):
    """Assign unique doc ids to train / val / test splits."""
    unique_ids = list(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train: n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def write_jsonl(records: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw JSONL file")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--chunk_size", type=int, default=200,
                        help="Max subword tokens per chunk (via tokenizer; paper uses 200)")
    parser.add_argument(
        "--tokenizer_name",
        default="facebook/bart-base",
        help="HF tokenizer for token counting when chunking (align with BART training)",
    )
    parser.add_argument("--min_words", type=int, default=10,
                        help="Drop chunk pairs shorter than this many words")
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Load only first N rows (smoke / debug)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer for chunking: {args.tokenizer_name} ...")
    chunk_tok = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Loading {args.input} ...")
    rows = load_jsonl(args.input, args.max_rows)
    print(f"  Loaded {len(rows):,} rows")

    # Build (doc_id, ai, human) triples
    triples = []
    skipped = 0
    for i, row in enumerate(rows):
        ai, human = get_text_fields(row)
        if not ai or not human:
            skipped += 1
            continue
        doc_id = make_id(row, i)
        triples.append({"id": doc_id, "ai": ai, "human": human,
                        # preserve metadata for reproducibility
                        "style": row.get("style", ""),
                        "model": row.get("model", ""),
                        "prompt_id": row.get("prompt_id", "")})

    print(f"  {len(triples):,} valid pairs ({skipped} skipped - missing ai or human field)")

    # Deduplicate on ai text
    before_dedup = len(triples)
    triples = deduplicate_on_ai(triples)
    print(f"  Deduplicated: {before_dedup - len(triples)} duplicates removed -> {len(triples):,}")

    # Length check vs paper Section 4.2 (~418 words -> ~2.47 chunks/passage when using 200-token chunks)
    mean_ai_words = sum(len(t["ai"].split()) for t in triples) / max(len(triples), 1)
    mean_h_words = sum(len(t["human"].split()) for t in triples) / max(len(triples), 1)
    print(
        f"  Mean words/passage: AI {mean_ai_words:.1f}, human {mean_h_words:.1f} "
        f"(paper ~418 words; short passages stay single-chunk so expansion factor stays near 1.0)."
    )

    # Split by document id
    doc_ids = [t["id"] for t in triples]
    train_ids, val_ids, test_ids = split_by_id(doc_ids, args.seed, args.train_ratio, args.val_ratio)

    train_raw, val_raw, test_raw = [], [], []
    for t in triples:
        if t["id"] in train_ids:
            train_raw.append(t)
        elif t["id"] in val_ids:
            val_raw.append(t)
        else:
            test_raw.append(t)

    print(f"\nPassage-level split -> train {len(train_raw):,}  val {len(val_raw):,}  test {len(test_raw):,}")

    # Chunk each split
    joint_ct = 0
    zip_min_ct = 0

    def chunk_split(split: List[dict], label: str) -> List[dict]:
        nonlocal joint_ct, zip_min_ct
        out = []
        for row in split:
            pairs, mode = chunk_pair(
                row["ai"],
                row["human"],
                chunk_tok,
                args.chunk_size,
                args.min_words,
            )
            if mode == "joint":
                joint_ct += 1
            else:
                zip_min_ct += 1
            for j, (ai_c, h_c) in enumerate(pairs):
                out.append({
                    "doc_id": row["id"],
                    "chunk_idx": j,
                    "ai": ai_c,
                    "human": h_c,
                    "style": row.get("style", ""),
                    "model": row.get("model", ""),
                    "prompt_id": row.get("prompt_id", ""),
                })
        print(f"  {label}: {len(split):,} passages -> {len(out):,} chunks "
              f"(x{len(out)/max(len(split),1):.2f})")
        return out

    print("\nChunking (Section 4.2: NLTK + tokenizer, <=200 tokens per side) ...")
    train_chunks = chunk_split(train_raw, "train")
    val_chunks = chunk_split(val_raw, "val")
    test_chunks = chunk_split(test_raw, "test")
    total_p = joint_ct + zip_min_ct
    print(f"  Alignment: joint (equal NLTK segment counts) {joint_ct:,} passages "
          f"({100*joint_ct/max(total_p,1):.1f}%), "
          f"zip_min (unequal segment counts) {zip_min_ct:,} passages")

    # Write
    print("\nWriting ...")
    write_jsonl(train_chunks, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(val_chunks,   os.path.join(args.output_dir, "val.jsonl"))
    write_jsonl(test_chunks,  os.path.join(args.output_dir, "test.jsonl"))

    # Sanity stats
    total = len(train_chunks) + len(val_chunks) + len(test_chunks)
    print(f"\nDone.  Total chunks: {total:,}")
    print(f"  Train {len(train_chunks):,} ({100*len(train_chunks)/total:.1f}%)")
    print(f"  Val   {len(val_chunks):,} ({100*len(val_chunks)/total:.1f}%)")
    print(f"  Test  {len(test_chunks):,} ({100*len(test_chunks)/total:.1f}%)")

    # Verify no doc leaks
    train_docs = {c["doc_id"] for c in train_chunks}
    val_docs = {c["doc_id"] for c in val_chunks}
    test_docs = {c["doc_id"] for c in test_chunks}
    assert not train_docs & val_docs, "LEAK: same doc in train and val!"
    assert not train_docs & test_docs, "LEAK: same doc in train and test!"
    assert not val_docs & test_docs, "LEAK: same doc in val and test!"
    print("  Split integrity check PASSED - no document appears in multiple splits.")


if __name__ == "__main__":
    main()
