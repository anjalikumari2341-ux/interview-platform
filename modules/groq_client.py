"""
groq_client.py
Thin wrappers around the Groq SDK:
  - transcribe_audio   : Whisper large-v3-turbo  → transcript text
  - analyze_content    : LLaMA 3 8B              → structured interview feedback
"""
from __future__ import annotations

import json
import os
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()


def get_groq_client(api_key: str | None = None) -> Groq:
    """Return an authenticated Groq client."""
    
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    return Groq(api_key=api_key)


# ── Speech-to-Text ────────────────────────────────────────────────────────────

def transcribe_audio(client: Groq, audio_path: str, language: str = "en") -> dict:
    """
    Transcribe *audio_path* using Groq Whisper large-v3-turbo.

    Returns
    -------
    {"text": str, "duration": float | None}
    """
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    filename = os.path.basename(audio_path)

    result = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        language=language,
    )

    return {
        "text":     result.text,
        "duration": getattr(result, "duration", None),
    }


# ── Content Quality Analysis ──────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are an expert interview coach and senior HR professional evaluating a job candidate's interview response.

**Interview Duration:** {duration}

**Transcript:**
{transcript}

Carefully evaluate this response on:
1. Relevance and specificity (does it actually answer the question?)
2. Use of concrete examples or the STAR method
3. Communication clarity, structure, and flow
4. Confidence and professionalism conveyed through word choice
5. Technical depth appropriate for a role

Respond ONLY with a valid JSON object — no markdown, no extra text — in this exact schema:
{{
  "content_score": <integer 0-25>,
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", "<recommendation 3>"],
  "assessment": "<2-3 sentence overall assessment>"
}}
"""


def analyze_content(
    client:           Groq,
    transcript:       str,
    duration_seconds: float,
) -> dict:
    """
    Ask Groq LLaMA 3 to evaluate interview content quality.

    Returns
    -------
    {
        content_score  : int   (0-25),
        strengths      : list[str],
        improvements   : list[str],
        recommendations: list[str],
        assessment     : str,
    }
    """
    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    duration_str = f"{mins}m {secs}s"

    prompt = _ANALYSIS_PROMPT.format(
        duration=duration_str,
        transcript=transcript,
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Clamp score to valid range
    data["content_score"] = max(0, min(25, int(data.get("content_score", 15))))

    # Ensure required keys exist
    for key in ("strengths", "improvements", "recommendations"):
        if key not in data or not isinstance(data[key], list):
            data[key] = []
    if "assessment" not in data:
        data["assessment"] = ""

    return data
