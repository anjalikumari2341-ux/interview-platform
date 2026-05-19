"""
speech_analyzer.py
Pure-Python analysis of a transcript:
  - Filler word detection via local sklearn model
  - Words-per-minute (WPM) calculation
  - Score functions matching FINAL SCORE STRUCTURE document

SCORING (from spec):
  Filler Words  : 25 marks  (filler_ratio = filler_count / total_words)
  Speaking Speed: 20 marks  (WPM bands)
"""
from __future__ import annotations

import re
import string
import pickle
import numpy as np
from pathlib import Path

# ── Model paths ───────────────────────────────────────────────────────────────
_MODEL_DIR    = Path(__file__).resolve().parent.parent / "models"
_FILLER_MODEL = _MODEL_DIR / "best_filler_model.pkl"

# ── Filler word list (must match training exactly) ────────────────────────────
FILLER_WORDS: list[str] = [
    "you know", "i mean", "kind of", "sort of", "you see", "i guess", "i think",
    "um", "uh", "er", "ah", "hmm", "erm", "uhh", "umm",
    "like", "basically", "literally", "actually", "honestly",
    "obviously", "essentially", "totally", "clearly",
    "right", "okay", "so", "well", "just",
]

DEFINITE_FILLERS: frozenset[str] = frozenset(
    {"um", "uh", "er", "ah", "hmm", "erm", "uhh", "umm",
     "you know", "i mean", "kind of", "sort of"}
)


# ── Load filler model once ────────────────────────────────────────────────────
_filler_model = None

def _get_filler_model():
    global _filler_model
    if _filler_model is None:
        try:
            with open(_FILLER_MODEL, "rb") as f:
                _filler_model = pickle.load(f)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Filler model not found at {_FILLER_MODEL}.\n"
                "Expected: models/best_filler_model.pkl"
            ) from exc
    return _filler_model


# ── Feature helpers (mirrors training code exactly) ──────────────────────────
def _clean_text(text: str) -> str:
    return text.lower().translate(
        str.maketrans("", "", string.punctuation)
    ).strip()

def _count_fillers_raw(text: str) -> int:
    return sum(text.count(w) for w in FILLER_WORDS)

def _count_words_raw(text: str) -> int:
    return len(text.split())

def _filler_ratio_raw(text: str) -> float:
    words = _count_words_raw(text)
    return _count_fillers_raw(text) / words if words > 0 else 0.0

def _count_unique_fillers_raw(text: str) -> int:
    return sum(1 for w in FILLER_WORDS if w in text)


# ── Filler analysis ───────────────────────────────────────────────────────────
def analyze_filler_words(transcript: str) -> dict:
    """
    Count filler words using regex matching AND local sklearn model.

    Returns
    -------
    {
        total_fillers        : int,
        filler_ratio         : float   (filler_count / total_words  0.0–1.0),
        filler_ratio_pct     : float   (as percentage 0–100),
        filler_breakdown     : {word: count}  sorted by frequency,
        top_fillers          : [(word, count), ...]  top-5,
        unique_filler_types  : int,
        definite_filler_count: int,
        model_prediction     : str   ('Filler Detected' | 'No Filler'),
        model_used           : bool,
    }
    """
    text   = transcript.lower()
    counts: dict[str, int] = {}

    for filler in FILLER_WORDS:
        if " " in filler:
            cnt = text.count(filler)
        else:
            cnt = len(re.findall(r"\b" + re.escape(filler) + r"\b", text))
        if cnt:
            counts[filler] = cnt

    counts         = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    definite_count = sum(v for k, v in counts.items() if k in DEFINITE_FILLERS)
    total_fillers  = sum(counts.values())
    word_count     = len(transcript.strip().split())
    filler_ratio   = round(total_fillers / word_count, 4) if word_count > 0 else 0.0

    # ── ML model prediction ───────────────────────────────────────────────────
    model_prediction = "No Filler"
    model_used = False
    try:
        model    = _get_filler_model()
        clean    = _clean_text(transcript)
        fc       = _count_fillers_raw(clean)
        wc       = _count_words_raw(clean)
        fr       = _filler_ratio_raw(clean)
        uf       = _count_unique_fillers_raw(clean)
        features = np.array([[fc, wc, fr, uf]])
        pred     = model.predict(features)[0]
        model_prediction = "Filler Detected" if pred == 1 else "No Filler"
        model_used = True
    except Exception:
        model_prediction = "Filler Detected" if total_fillers > 0 else "No Filler"

    return {
        "total_fillers":         total_fillers,
        "filler_ratio":          filler_ratio,
        "filler_ratio_pct":      round(filler_ratio * 100, 2),
        "filler_breakdown":      counts,
        "top_fillers":           list(counts.items())[:5],
        "unique_filler_types":   len(counts),
        "definite_filler_count": definite_count,
        "model_prediction":      model_prediction,
        "model_used":            model_used,
    }


# ── Speech speed ──────────────────────────────────────────────────────────────
def calculate_speech_speed(transcript: str, duration_seconds: float) -> dict:
    """
    Compute WPM and classify the pace.

    Returns
    -------
    { word_count, wpm, category, emoji, feedback, color, optimal_range }
    """
    words        = transcript.strip().split()
    word_count   = len(words)
    duration_min = max(duration_seconds / 60.0, 0.01)
    wpm          = round(word_count / duration_min, 1)

    if 120 <= wpm <= 160:
        category, emoji, color = "Optimal ✓", "🎯", "#00C851"
        feedback = "Perfect pace — clear, confident, and easy to follow."
    elif (100 <= wpm < 120) or (160 < wpm <= 180):
        category, emoji, color = "Near Optimal", "🚶", "#2196F3"
        feedback = "Slightly off optimal. Minor adjustment needed."
    elif (80 <= wpm < 100) or (180 < wpm <= 200):
        category, emoji, color = "Moderate", "⚠️", "#FF8800"
        feedback = ("Too slow — pick up the pace."
                    if wpm < 100 else
                    "A bit fast — slow down so listeners can follow.")
    else:
        category, emoji, color = "Poor", "🐢" if wpm < 80 else "💨", "#FF4444"
        feedback = ("Way too slow. Speak naturally and energetically."
                    if wpm < 80 else
                    "Too fast. Slow down significantly to be understood.")

    return {
        "word_count":    word_count,
        "wpm":           wpm,
        "category":      category,
        "emoji":         emoji,
        "feedback":      feedback,
        "color":         color,
        "optimal_range": "120–160 WPM",
    }


# ── Scoring (per FINAL SCORE STRUCTURE doc) ───────────────────────────────────

def calculate_filler_score(total_fillers: int, word_count: int) -> int:
    """
    Score filler-word usage out of 25.
    Uses filler_ratio = filler_count / total_words (as a percentage).

    Spec:
      0  – 2%  → 25  (Excellent)
      2  – 5%  → 20
      5  – 10% → 15
      10 – 20% → 8
      > 20%    → 3   (Poor)
    """
    if word_count == 0:
        return 25
    ratio_pct = (total_fillers / word_count) * 100.0

    if ratio_pct <= 2:   return 25
    if ratio_pct <= 5:   return 20
    if ratio_pct <= 10:  return 15
    if ratio_pct <= 20:  return 8
    return 3


def calculate_speed_score(wpm: float) -> int:
    """
    Score speech speed out of 20.

    Spec:
      120 – 160           → 20
      100–120 or 160–180  → 15
      80–100  or 180–200  → 10
      < 80    or > 200    → 5
    """
    if 120 <= wpm <= 160:                        return 20
    if (100 <= wpm < 120) or (160 < wpm <= 180): return 15
    if (80  <= wpm < 100) or (180 < wpm <= 200): return 10
    return 5
