"""
merge_datasets.py
------------------
Merge multiple JSONL datasets containing at least:
  - `ai` (string)
  - `human` (string)

Optionally `id` (used for document-level split integrity).

This script prefixes each dataset's `id` with a user-provided tag to avoid id collisions
across datasets (important because `prepare_data.py` splits by `id`).

Example:
  python merge_datasets.py ^
    --inputs combined_success_all_models.jsonl ai_data_15k_converted.jsonl ^
    --prefixes base ai_data ^
    --output combined_success_all_models.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Iterator, List


def md5_short(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            # Some JSONL files may include a UTF-8 BOM or other invisible prefix.
            line = line.lstrip("\ufeff")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # Skip malformed lines rather than hard-failing the whole merge.
                print(
                    f"SKIP bad JSON at {path}:{idx}: {e}. "
                    # Keep the console output ASCII-only (Windows terminals often use cp1252).
                    f"line_start={ascii(line[:80])}"
                )
                continue


def process_row(row: Dict, prefix: str) -> Dict:
    # Ensure we have a doc-level id for prepare_data's split-by-id.
    raw_id = row.get("id", None)
    if raw_id is None or str(raw_id).strip() == "":
        human = row.get("human") or row.get("human_text") or ""
        row["id"] = f"{prefix}_{md5_short(human)}"
    else:
        row["id"] = f"{prefix}_{raw_id}"

    # Standardize source_dataset for later analysis (optional).
    if not row.get("source_dataset"):
        row["source_dataset"] = row.get("source") or prefix

    return row


def write_jsonl(records: List[Dict], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL files to merge")
    parser.add_argument("--prefixes", nargs="+", required=True, help="Prefix tags (same length as inputs)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    if len(args.inputs) != len(args.prefixes):
        raise SystemExit("--inputs and --prefixes must have the same number of elements")

    inputs: List[str] = args.inputs
    prefixes: List[str] = args.prefixes

    out_records: List[Dict] = []
    total_in = 0
    for path, prefix in zip(inputs, prefixes):
        for row in iter_jsonl(path):
            total_in += 1
            out_records.append(process_row(row, prefix))

    print(f"Merging {len(inputs)} files: {total_in} rows -> {len(out_records)} rows (ids prefixed).")
    write_jsonl(out_records, args.output)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

