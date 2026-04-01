"""
report_generator.py
Combines all analysis results into a single structured report dict
and can serialise it as a plain-text download.
"""
from __future__ import annotations

# ── Grade scale (score → grade, rating, hex colour) ──────────────────────────
_GRADES: list[tuple[int, str, str, str]] = [
    (90, "A+", "Outstanding",    "#00C851"),
    (80, "A",  "Excellent",      "#007E33"),
    (70, "B+", "Very Good",      "#2196F3"),
    (60, "B",  "Good",           "#1976D2"),
    (50, "C",  "Average",        "#FF8800"),
    (40, "D",  "Below Average",  "#FF4444"),
    (0,  "F",  "Needs Work",     "#CC0000"),
]


def get_grade(score: int) -> tuple[str, str, str]:
    """Return *(grade, rating, hex_color)* for *score* out of 100."""
    for threshold, grade, rating, color in _GRADES:
        if score >= threshold:
            return grade, rating, color
    return "F", "Needs Work", "#CC0000"


# ── Report compilation ────────────────────────────────────────────────────────

def compile_report(
    *,
    transcript:       str,
    filler_analysis:  dict,
    speed_analysis:   dict,
    video_analysis:   dict,
    content_analysis: dict,
    duration_seconds: float,
    filler_score:     int,
    speed_score:      int,
) -> dict:
    """
    Combine all individual results into one report dict.

    Scores
    ------
    Each sub-category is /25; total is /100.
    """
    content_score = content_analysis.get("content_score", 15)
    video_score   = video_analysis.get("video_confidence_score", 0)
    total_score   = filler_score + speed_score + content_score + video_score

    grade, rating, color = get_grade(total_score)

    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)

    return {
        # ── Overall ──────────────────────────────────────────────────────────
        "total_score":        total_score,
        "grade":              grade,
        "rating":             rating,
        "color":              color,
        "duration_formatted": f"{mins}m {secs}s",
        "duration_seconds":   duration_seconds,

        # ── Sub-scores ───────────────────────────────────────────────────────
        "scores": {
            "filler_words":    filler_score,
            "speech_speed":    speed_score,
            "content_quality": content_score,
            "video_confidence":video_score,
        },

        # ── Raw data ─────────────────────────────────────────────────────────
        "transcript":       transcript,
        "filler_analysis":  filler_analysis,
        "speed_analysis":   speed_analysis,
        "video_analysis":   video_analysis,
        "content_analysis": content_analysis,
    }


# ── Text report ───────────────────────────────────────────────────────────────

def generate_text_report(report: dict) -> str:
    """Render *report* as a human-readable plain-text string for download."""
    sc  = report["scores"]
    fa  = report["filler_analysis"]
    sa  = report["speed_analysis"]
    va  = report["video_analysis"]
    ca  = report["content_analysis"]
    SEP = "─" * 62

    lines: list[str] = [
        "=" * 62,
        "      AI INTERVIEW INTELLIGENCE PLATFORM — REPORT",
        "=" * 62,
        f"Overall Score  : {report['total_score']} / 100",
        f"Grade          : {report['grade']}  ({report['rating']})",
        f"Duration       : {report['duration_formatted']}",
        "",
        SEP,
        "SCORE BREAKDOWN",
        SEP,
        f"  Filler Words     : {sc['filler_words']:>2} / 25",
        f"  Speech Speed     : {sc['speech_speed']:>2} / 25",
        f"  Content Quality  : {sc['content_quality']:>2} / 25",
        f"  Video Confidence : {sc['video_confidence']:>2} / 25",
        "",
        SEP,
        "SPEECH ANALYSIS",
        SEP,
        f"  Word Count       : {sa['word_count']:,} words",
        f"  Speaking Speed   : {sa['wpm']} WPM  ({sa['category']})",
        f"  Speed Feedback   : {sa['feedback']}",
        f"  Total Fillers    : {fa['total_fillers']}",
        f"  Definite Fillers : {fa.get('definite_filler_count', '—')}",
    ]

    if fa["filler_breakdown"]:
        lines.append("  Filler Breakdown :")
        for word, count in list(fa["filler_breakdown"].items())[:10]:
            lines.append(f"    • '{word}': {count}×")

    lines += [
        "",
        SEP,
        "VIDEO ANALYSIS",
        SEP,
        f"  Face Detected    : {va['face_detection_rate']}% of frames",
        f"  Head Stability   : {va['head_stability']} / 100",
        f"  Eye Contact      : {va['eye_contact_score']} / 100",
        f"  MediaPipe Used   : {'Yes' if va['mediapipe_available'] else 'No (Haar fallback)'}",
        "",
        SEP,
        "AI INSIGHTS  (Groq LLaMA 3)",
        SEP,
        "",
        "OVERALL ASSESSMENT:",
        f"  {ca.get('assessment', 'N/A')}",
        "",
        "STRENGTHS:",
    ]

    for s in ca.get("strengths", []):
        lines.append(f"  ✓  {s}")

    lines += ["", "AREAS FOR IMPROVEMENT:"]
    for i in ca.get("improvements", []):
        lines.append(f"  →  {i}")

    lines += ["", "RECOMMENDATIONS:"]
    for r in ca.get("recommendations", []):
        lines.append(f"  ★  {r}")

    lines += [
        "",
        SEP,
        "FULL TRANSCRIPT",
        SEP,
        report.get("transcript", ""),
        "",
        "=" * 62,
        "Generated by AI Interview Intelligence Platform",
        "Powered by Groq AI  |  https://console.groq.com",
        "=" * 62,
    ]

    return "\n".join(lines)
