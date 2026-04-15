#!/usr/bin/env python3
"""
Generate publication figures from evaluate.py output (summary.json).

Outputs (default: current directory):
  fig1_metrics.pdf — BERTScore F1, ROUGE-L, chrF++ (Section 6.1)
  fig2_shifts.pdf  — per-marker directional shift (Section 6.2)
  fig3_ppl.pdf     — GPT-2 perplexity (Section 6.3)

Figure numbering matches first-mention order in the paper.

Usage:
  python generate_figures.py
  python generate_figures.py --summary path/to/summary.json --outdir figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Order and labels aligned with Table 5 / linguistic_markers.py keys
MARKER_ORDER: list[tuple[str, str]] = [
    ("word_count", "Word count"),
    ("sentence_count", "Sentence count"),
    ("avg_word_length_chars", "Avg word length"),
    ("lexical_diversity", "Lexical diversity"),
    ("contractions_per_passage", "Contractions"),
    ("question_marks_per_passage", "Question marks"),
    ("exclamations_per_passage", "Exclamations"),
    ("commas_per_passage", "Commas"),
    ("sentence_length_variance", "Sent. length var."),
    ("flesch_reading_ease", "Flesch Reading Ease"),
    ("flesch_kincaid_grade_level", "F-K Grade Level"),
]

MODEL_ORDER = ["BART-base", "BART-large", "Mistral-7B"]
COLORS = ["#888888", "#222222", "#BBBBBB"]
HATCHES = ["", "", "//"]


def load_summary(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        sys.exit("summary.json must be a JSON array of model result objects.")
    by_name = {row["model"]: row for row in data}
    missing = [m for m in MODEL_ORDER if m not in by_name]
    if missing:
        sys.exit(f"Missing models in summary: {missing}. Found: {list(by_name.keys())}")
    return [by_name[m] for m in MODEL_ORDER]


def figure_1_metrics(rows: list[dict], out_path: Path, n_samples: int) -> None:
    bertscore_f1 = [rows[i]["bertscore_f1"] for i in range(3)]
    rouge_l = [rows[i]["rouge_l"] for i in range(3)]
    chrf_pp = [rows[i]["chrf_pp"] for i in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    metric_data = [bertscore_f1, rouge_l, chrf_pp]
    ylimits = [(0.85, 0.95), (0.35, 0.65), (40, 62)]
    ylabels = ["BERTScore F1", "ROUGE-L", "chrF++"]
    subtitles = ["(a) BERTScore F1", "(b) ROUGE-L", "(c) chrF++"]

    for ax, vals, ylim, ylabel, subtitle in zip(axes, metric_data, ylimits, ylabels, subtitles):
        bars = ax.bar(
            MODEL_ORDER,
            vals,
            color=COLORS,
            hatch=HATCHES,
            edgecolor="black",
            linewidth=0.8,
            width=0.55,
        )
        ax.set_ylim(ylim)
        ax.set_title(subtitle, pad=6)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        fmt = ".4f" if ylabel != "chrF++" else ".2f"
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.012,
                format(v, fmt),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        rf"Figure 1. Reference-based similarity on the test set ($n = {n_samples:,}$). "
        r"BART-large achieves the highest score on all three metrics "
        r"despite having 17$\times$ fewer parameters than Mistral-7B. "
        r"See Table 3 for exact values.",
        fontsize=10,
        y=0.01,
        va="bottom",
    )
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def figure_2_shifts(rows: list[dict], out_path: Path, n_samples: int) -> None:
    markers = [label for _, label in MARKER_ORDER]
    keys = [k for k, _ in MARKER_ORDER]
    shifts: dict[str, list[float]] = {}
    for row in rows:
        name = row["model"]
        m = row["marker_shifts"]
        shifts[name] = [float(m[k]) for k in keys]

    n_markers = len(markers)
    y = np.arange(n_markers)
    bar_h = 0.25

    fig, ax = plt.subplots(figsize=(10, 8))
    for model, color, hatch, offset in zip(MODEL_ORDER, COLORS, HATCHES, [-bar_h, 0, bar_h]):
        ax.barh(
            y + offset,
            shifts[model],
            height=bar_h,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.7,
            label=model,
        )

    ax.axvline(1.0, color="black", linewidth=1.5, linestyle="-", label="Perfect shift (= 1.0)")
    ax.axvspan(1.0, 2.05, alpha=0.06, color="gray", label="Overshoot zone (> 1)")
    ax.axvspan(-1.05, 0.0, alpha=0.06, color="red", label="Wrong direction (< 0)")

    commas_idx = markers.index("Commas")
    ax.annotate(
        r"Mistral-7B: $-1.0$" + "\n(wrong direction)",
        xy=(-1.0, commas_idx + bar_h),
        xytext=(0.35, commas_idx + bar_h + 0.65),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=9,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(markers)
    ax.set_xlabel(r"Directional marker shift (1.0 = exact human mean)")
    ax.set_xlim(-1.20, 2.20)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(
        rf"Figure 2. Per-marker directional shift for all 11 linguistic markers ($n = {n_samples:,}$)",
        pad=10,
    )

    caption = (
        r"Shift = (output $-$ AI) / (human $-$ AI), clipped to $[-1, 2]$. "
        "The vertical line at 1.0 marks exact match with the human distribution mean. "
        "Values $> 1$ = overshoot; values $< 0$ = wrong direction. "
        r"Mistral-7B is capped at 2.0 on five markers and reaches $-1.0$ on Commas "
        "(annotated). See Table 5."
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_3_ppl(rows: list[dict], out_path: Path, n_samples: int) -> None:
    ppl_vals = [rows[i]["perplexity_gpt2"] for i in range(3)]
    human_ref_ppl = float(rows[0]["perplexity_human_ref"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        MODEL_ORDER,
        ppl_vals,
        color=COLORS,
        hatch=HATCHES,
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    ax.axhline(
        human_ref_ppl,
        color="black",
        linewidth=1.4,
        linestyle="--",
        label=f"Human reference PPL = {human_ref_ppl:.2f}",
    )
    ax.set_ylabel("GPT-2 Perplexity (lower = more predictable)")
    ax.set_ylim(0, 34)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    for bar, v in zip(bars, ppl_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title(rf"Figure 3. GPT-2 perplexity of model outputs ($n = {n_samples:,}$)", pad=8)

    caption = (
        f"The dashed line shows human reference perplexity ({human_ref_ppl:.2f}). "
        f"BART-base ({ppl_vals[0]:.2f}) and BART-large ({ppl_vals[1]:.2f}) "
        "are close to this baseline; "
        f"Mistral-7B ({ppl_vals[2]:.2f}) is far below it, indicating unusually predictable "
        "outputs. Lower perplexity does not imply greater human-likeness; see Section 6.3."
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper figures from summary.json")
    p.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "summary.json",
        help="Path to summary.json from evaluate.py",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory for output PDFs",
    )
    args = p.parse_args()

    if not args.summary.is_file():
        sys.exit(f"Not found: {args.summary}")

    rows = load_summary(args.summary)
    n_samples = int(rows[0]["n_samples"])
    args.outdir.mkdir(parents=True, exist_ok=True)

    out1 = args.outdir / "fig1_metrics.pdf"
    out2 = args.outdir / "fig2_shifts.pdf"
    out3 = args.outdir / "fig3_ppl.pdf"

    figure_1_metrics(rows, out1, n_samples)
    print(f"Saved {out1}")
    figure_2_shifts(rows, out2, n_samples)
    print(f"Saved {out2}")
    figure_3_ppl(rows, out3, n_samples)
    print(f"Saved {out3}")
    print("Done.")


if __name__ == "__main__":
    main()
