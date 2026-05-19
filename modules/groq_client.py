"""
groq_client.py
Groq is used ONLY for:
  1. Speech-to-text  (Whisper large-v3-turbo)
  2. Content quality + Overall Fluency analysis (LLaMA 3)

SCORING (from FINAL SCORE STRUCTURE doc):
  Overall Fluency: 10 marks  (derived from LLaMA 3 fluency assessment)
    Smooth               → 10
    Minor breaks         → 7
    Noticeable pauses    → 5
    Disconnected speech  → 2
"""
from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_groq_client(api_key: str | None = None) -> Groq:
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return Groq(api_key=api_key)


# ── Speech-to-Text ────────────────────────────────────────────────────────────

def transcribe_audio(client: Groq, audio_path: str, language: str = "en") -> dict:
    """Transcribe audio using Groq Whisper large-v3-turbo."""
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    result = client.audio.transcriptions.create(
        file=(os.path.basename(audio_path), audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        language=language,
    )

    return {
        "text":     result.text,
        "duration": getattr(result, "duration", None),
    }


# ── Content + Fluency Analysis ────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are an expert interview coach evaluating a job candidate's interview response.

**Interview Duration:** {duration}

**Transcript:**
{transcript}

Evaluate on:
1. Relevance and specificity
2. Use of concrete examples (STAR method)
3. Communication clarity and structure
4. Confidence and professionalism
5. Technical depth

Also evaluate OVERALL FLUENCY based on:
- Sentence flow and coherence
- Grammar consistency
- Smoothness of delivery (smooth / minor breaks / noticeable pauses / disconnected speech)

Respond ONLY with a valid JSON object in this exact schema (no markdown):
{{
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"],
  "assessment": "<2-3 sentence overall assessment>",
  "fluency_level": "<one of: smooth | minor_breaks | noticeable_pauses | disconnected>",
  "fluency_feedback": "<one sentence on speech flow and grammar>"
}}
"""


def analyze_content(
    client:           Groq,
    transcript:       str,
    duration_seconds: float,
) -> dict:
    """
    Ask Groq LLaMA 3 to evaluate interview content and fluency.

    Returns
    -------
    {
        strengths       : list[str],
        improvements    : list[str],
        recommendations : list[str],
        assessment      : str,
        fluency_level   : str   ('smooth'|'minor_breaks'|'noticeable_pauses'|'disconnected'),
        fluency_score   : int   (10 | 7 | 5 | 2),
        fluency_feedback: str,
    }
    """
    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)

    prompt = _ANALYSIS_PROMPT.format(
        duration=f"{mins}m {secs}s",
        transcript=transcript,
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw  = response.choices[0].message.content
    data = json.loads(raw)

    for key in ("strengths", "improvements", "recommendations"):
        if key not in data or not isinstance(data[key], list):
            data[key] = []
    if "assessment" not in data:
        data["assessment"] = ""

    # ── Fluency score (per spec) ──────────────────────────────────────────────
    fluency_map = {
        "smooth":             10,
        "minor_breaks":       7,
        "noticeable_pauses":  5,
        "disconnected":       2,
    }
    fluency_level = data.get("fluency_level", "minor_breaks").lower().replace(" ", "_")
    data["fluency_level"] = fluency_level
    data["fluency_score"] = fluency_map.get(fluency_level, 7)
    if "fluency_feedback" not in data:
        data["fluency_feedback"] = ""

    return data
