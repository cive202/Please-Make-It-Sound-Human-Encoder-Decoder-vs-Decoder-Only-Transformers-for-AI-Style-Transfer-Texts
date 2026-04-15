"""
linguistic_markers.py
---------------------
Computes the 11 linguistic markers from the paper (Table 1) for a given text.
Used by both evaluate.py and prepare_data.py / qualitative analysis.

Markers:
  1.  word_count
  2.  sentence_count
  3.  avg_word_length_chars
  4.  lexical_diversity          (type-token ratio)
  5.  contractions_per_passage
  6.  question_marks_per_passage
  7.  exclamations_per_passage
  8.  commas_per_passage
  9.  sentence_length_variance
  10. flesch_reading_ease
  11. flesch_kincaid_grade_level
"""

import re
from typing import Dict, List

import numpy as np

try:
    import textstat
    _HAS_TEXTSTAT = True
except ImportError:
    _HAS_TEXTSTAT = False
    print("WARNING: textstat not installed — readability scores will be 0.0")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

# Common English contractions pattern
_CONTRACTION_RE = re.compile(
    r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|"
    r"he's|he'd|she's|she'd|it's|we're|we've|we'll|we'd|"
    r"they're|they've|they'll|they'd|that's|that'll|"
    r"isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|"
    r"don't|doesn't|didn't|won't|wouldn't|can't|couldn't|"
    r"shouldn't|mustn't|mightn't|needn't|there's|here's|"
    r"what's|who's|where's|when's|how's|let's|n't)\b",
    re.IGNORECASE,
)


def _tokenize_words(text: str) -> List[str]:
    if _HAS_NLTK:
        return word_tokenize(text)
    return re.findall(r"\b\w+\b", text)


def _tokenize_sentences(text: str) -> List[str]:
    if _HAS_NLTK:
        sents = sent_tokenize(text)
    else:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s.strip()]


def compute_markers(text: str) -> Dict[str, float]:
    """Return all 11 linguistic markers for *text*."""
    text = text.strip()
    if not text:
        return {k: 0.0 for k in _MARKER_KEYS}

    words = _tokenize_words(text)
    alpha_words = [w for w in words if re.search(r"[a-zA-Z]", w)]
    sentences = _tokenize_sentences(text)

    word_count = len(alpha_words)
    sentence_count = len(sentences)

    # Avg word length (alphabetic words only)
    avg_word_len = (sum(len(w) for w in alpha_words) / word_count) if word_count else 0.0

    # Lexical diversity (type-token ratio on lower-cased alpha words)
    lower_words = [w.lower() for w in alpha_words]
    lexical_diversity = (len(set(lower_words)) / len(lower_words)) if lower_words else 0.0

    # Contractions
    contractions = len(_CONTRACTION_RE.findall(text))

    # Punctuation counts (raw character counts)
    question_marks = text.count("?")
    exclamations = text.count("!")
    commas = text.count(",")

    # Sentence length variance (in words)
    sent_word_counts = []
    for s in sentences:
        sw = [w for w in _tokenize_words(s) if re.search(r"[a-zA-Z]", w)]
        sent_word_counts.append(len(sw))
    sent_length_variance = float(np.var(sent_word_counts)) if sent_word_counts else 0.0

    # Readability
    if _HAS_TEXTSTAT:
        flesch_ease = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
    else:
        flesch_ease = 0.0
        flesch_grade = 0.0

    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_word_length_chars": round(avg_word_len, 4),
        "lexical_diversity": round(lexical_diversity, 4),
        "contractions_per_passage": float(contractions),
        "question_marks_per_passage": float(question_marks),
        "exclamations_per_passage": float(exclamations),
        "commas_per_passage": float(commas),
        "sentence_length_variance": round(sent_length_variance, 4),
        "flesch_reading_ease": round(flesch_ease, 4),
        "flesch_kincaid_grade_level": round(flesch_grade, 4),
    }


_MARKER_KEYS = [
    "word_count", "sentence_count", "avg_word_length_chars", "lexical_diversity",
    "contractions_per_passage", "question_marks_per_passage", "exclamations_per_passage",
    "commas_per_passage", "sentence_length_variance", "flesch_reading_ease",
    "flesch_kincaid_grade_level",
]


def compute_marker_shift(ai_markers: Dict[str, float],
                         output_markers: Dict[str, float],
                         human_markers: Dict[str, float]) -> Dict[str, float]:
    """Compute how far the model output moved toward the human distribution.

    Shift = (output - ai) / (human - ai) clamped to [-1, 2].
    A value of 1.0 means the output exactly matches the human average on that marker.
    A value of 0.0 means nothing changed.
    """
    shifts = {}
    for k in _MARKER_KEYS:
        ai_v = ai_markers[k]
        out_v = output_markers[k]
        hum_v = human_markers[k]
        denom = hum_v - ai_v
        if abs(denom) < 1e-6:
            shifts[k] = 1.0 if abs(out_v - hum_v) < 1e-6 else 0.0
        else:
            shifts[k] = max(-1.0, min(2.0, (out_v - ai_v) / denom))
    return shifts


def average_markers(marker_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of marker dicts."""
    if not marker_list:
        return {k: 0.0 for k in _MARKER_KEYS}
    avg = {}
    for k in _MARKER_KEYS:
        avg[k] = float(np.mean([m[k] for m in marker_list]))
    return avg


def print_marker_table(ai_avg: Dict, output_avg: Dict, human_avg: Dict, model_name: str = "Model"):
    """Print a paper-style marker comparison table to stdout."""
    header = f"{'Metric':<35} {'AI Input':>12} {model_name:>14} {'Human Ref':>12} {'Shift':>8}"
    print(header)
    print("-" * len(header))
    shifts = compute_marker_shift(ai_avg, output_avg, human_avg)
    for k in _MARKER_KEYS:
        print(f"  {k:<33} {ai_avg[k]:>12.3f} {output_avg[k]:>14.3f} "
              f"{human_avg[k]:>12.3f} {shifts[k]:>8.2f}")
    avg_shift = float(np.mean(list(shifts.values())))
    print(f"\n  Mean marker shift: {avg_shift:.3f}")
