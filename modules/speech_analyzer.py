"""
speech_analyzer.py
Pure-Python analysis of a transcript:
  - Filler word detection and counting
  - Words-per-minute (WPM) calculation
  - Score functions for each metric
"""
from __future__ import annotations

import re

# ── Filler word dictionary ────────────────────────────────────────────────────
# Ordered so that multi-word phrases are matched before single tokens.
FILLER_WORDS: list[str] = [
    # Multi-word fillers (checked as substrings)
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "you see",
    "i guess",
    "i think",       # often used as a filler hedge
    # Single-word fillers (checked with word boundaries)
    "um",
    "uh",
    "er",
    "ah",
    "hmm",
    "erm",
    "uhh",
    "umm",
    "like",
    "basically",
    "literally",
    "actually",
    "honestly",
    "obviously",
    "essentially",
    "totally",
    "clearly",
    "right",
    "okay",
    "so",
    "well",
    "just",
]

# Set of definite vocal fillers (used only for flagging, not filtering)
DEFINITE_FILLERS: frozenset[str] = frozenset(
    {"um", "uh", "er", "ah", "hmm", "erm", "uhh", "umm", "you know", "i mean",
     "kind of", "sort of"}
)


# ── Filler analysis ───────────────────────────────────────────────────────────

def analyze_filler_words(transcript: str) -> dict:
    """
    Count every filler word / phrase in *transcript*.

    Returns
    -------
    {
        total_fillers       : int,
        filler_breakdown    : {word: count, ...}  sorted by frequency,
        top_fillers         : [(word, count), ...]  top-5,
        unique_filler_types : int,
        definite_filler_count: int,
    }
    """
    text = transcript.lower()
    counts: dict[str, int] = {}

    for filler in FILLER_WORDS:
        if " " in filler:
            # Phrase: simple substring count
            cnt = text.count(filler)
        else:
            # Single word: require word boundary
            cnt = len(re.findall(r"\b" + re.escape(filler) + r"\b", text))
        if cnt:
            counts[filler] = cnt

    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    definite_count = sum(
        v for k, v in counts.items() if k in DEFINITE_FILLERS
    )

    return {
        "total_fillers": sum(counts.values()),
        "filler_breakdown": counts,
        "top_fillers": list(counts.items())[:5],
        "unique_filler_types": len(counts),
        "definite_filler_count": definite_count,
    }


# ── Speech speed ──────────────────────────────────────────────────────────────

def calculate_speech_speed(transcript: str, duration_seconds: float) -> dict:
    """
    Compute words-per-minute and classify the pace.

    Returns
    -------
    {
        word_count   : int,
        wpm          : float,
        category     : str,
        emoji        : str,
        feedback     : str,
        color        : str  (hex),
        optimal_range: str,
    }
    """
    words = transcript.strip().split()
    word_count = len(words)
    duration_min = max(duration_seconds / 60.0, 0.01)
    wpm = round(word_count / duration_min, 1)

    if wpm < 80:
        category, emoji, color = "Very Slow", "🐢", "#FF4444"
        feedback = "Way too slow. Speak more naturally and energetically."
    elif wpm < 120:
        category, emoji, color = "Slow", "🚶", "#FF8800"
        feedback = "Slightly slow. Pick up the pace for a more confident delivery."
    elif wpm <= 160:
        category, emoji, color = "Optimal ✓", "🎯", "#00C851"
        feedback = "Perfect pace — clear, confident, and easy to follow."
    elif wpm <= 200:
        category, emoji, color = "Fast", "🏃", "#FF8800"
        feedback = "A bit fast. Slow down slightly to give listeners time to absorb your points."
    else:
        category, emoji, color = "Very Fast", "💨", "#FF4444"
        feedback = "Too fast. Slow down significantly so you can be understood clearly."

    return {
        "word_count": word_count,
        "wpm": wpm,
        "category": category,
        "emoji": emoji,
        "feedback": feedback,
        "color": color,
        "optimal_range": "120–160 WPM",
    }


# ── Scoring functions ─────────────────────────────────────────────────────────

def calculate_filler_score(total_fillers: int, duration_seconds: float) -> int:
    """
    Score filler-word usage out of 25.
    Fewer fillers per minute → higher score.
    """
    duration_min = max(duration_seconds / 60.0, 0.01)
    rate = total_fillers / duration_min   # fillers per minute

    if rate <= 2:   return 25
    if rate <= 4:   return 22
    if rate <= 7:   return 18
    if rate <= 10:  return 14
    if rate <= 15:  return 9
    if rate <= 20:  return 5
    return 2


def calculate_speed_score(wpm: float) -> int:
    """
    Score speech speed out of 25.
    Optimal range (120–160 WPM) scores highest.
    """
    if 120 <= wpm <= 160:               return 25
    if 110 <= wpm < 120 or 160 < wpm <= 170: return 22
    if 100 <= wpm < 110 or 170 < wpm <= 185: return 18
    if  85 <= wpm < 100 or 185 < wpm <= 200: return 14
    if  70 <= wpm <  85 or 200 < wpm <= 220: return 9
    return 5
