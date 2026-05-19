"""
report_generator.py
Compiles all analysis into a structured report.

FINAL SCORE STRUCTURE:
  Filler Words       25  (filler_ratio bands)
  Speaking Speed     20  (WPM bands)
  Facial Confidence  25  (ML model, confident_prob bands)
  Voice Confidence   20  (ML model, prediction × prob)
  Overall Fluency    10  (LLaMA 3 fluency assessment)
  ─────────────────────
  TOTAL             100

GRADE TABLE:
  90–100 → Excellent   (Interview Ready)
  75–89  → Very Good   (Minor improvements needed)
  60–74  → Good        (Practice required)
  40–59  → Average     (Needs improvement)
  < 40   → Poor        (Major improvement needed)
"""
from __future__ import annotations

# ── Grade scale (per FINAL SCORE STRUCTURE doc) ───────────────────────────────
_GRADES: list[tuple[int, str, str, str, str]] = [
    (90, "A+", "Excellent",  "Interview Ready",             "#00C851"),
    (75, "A",  "Very Good",  "Minor improvements needed",   "#007E33"),
    (60, "B",  "Good",       "Practice required",           "#2196F3"),
    (40, "C",  "Average",    "Needs improvement",           "#FF8800"),
    (0,  "F",  "Poor",       "Major improvement needed",    "#FF4444"),
]


def get_grade(score: int) -> tuple[str, str, str, str]:
    """Return (grade, rating, feedback, hex_color) for score out of 100."""
    for threshold, grade, rating, feedback, color in _GRADES:
        if score >= threshold:
            return grade, rating, feedback, color
    return "F", "Poor", "Major improvement needed", "#FF4444"


# ── Report compilation ────────────────────────────────────────────────────────

def compile_report(
    *,
    transcript:        str,
    filler_analysis:   dict,
    speed_analysis:    dict,
    video_analysis:    dict,
    content_analysis:  dict,
    voice_analysis:    dict,
    duration_seconds:  float,
    filler_score:      int,   # /25
    speed_score:       int,   # /20
) -> dict:
    """
    Combine all results. Total = 25 + 20 + 25 + 20 + 10 = 100.
    """
    facial_score  = video_analysis.get("video_confidence_score", 3)   # /25
    voice_score   = voice_analysis.get("voice_score", 5)               # /20
    fluency_score = content_analysis.get("fluency_score", 7)           # /10

    total_score = filler_score + speed_score + facial_score + voice_score + fluency_score

    grade, rating, grade_feedback, color = get_grade(total_score)

    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)

    return {
        # ── Overall ──────────────────────────────────────────────────────────
        "total_score":        total_score,
        "grade":              grade,
        "rating":             rating,
        "grade_feedback":     grade_feedback,
        "color":              color,
        "duration_formatted": f"{mins}m {secs}s",
        "duration_seconds":   duration_seconds,

        # ── Sub-scores ───────────────────────────────────────────────────────
        "scores": {
            "filler_words":      {"score": filler_score,  "max": 25},
            "speaking_speed":    {"score": speed_score,   "max": 20},
            "facial_confidence": {"score": facial_score,  "max": 25},
            "voice_confidence":  {"score": voice_score,   "max": 20},
            "overall_fluency":   {"score": fluency_score, "max": 10},
        },

        # ── Raw analysis data ─────────────────────────────────────────────────
        "transcript":       transcript,
        "filler_analysis":  filler_analysis,
        "speed_analysis":   speed_analysis,
        "video_analysis":   video_analysis,
        "content_analysis": content_analysis,
        "voice_analysis":   voice_analysis,
    }


# ── Text report ───────────────────────────────────────────────────────────────

def generate_text_report(report: dict) -> str:
    sc  = report["scores"]
    fa  = report["filler_analysis"]
    sa  = report["speed_analysis"]
    va  = report["video_analysis"]
    ca  = report["content_analysis"]
    vca = report.get("voice_analysis", {})
    SEP = "─" * 62

    lines: list[str] = [
        "=" * 62,
        "      AI INTERVIEW INTELLIGENCE PLATFORM — REPORT",
        "=" * 62,
        f"Overall Score  : {report['total_score']} / 100",
        f"Grade          : {report['grade']}  ({report['rating']})",
        f"Feedback       : {report['grade_feedback']}",
        "",
        SEP,
        "SCORE BREAKDOWN                          Score  /Max",
        SEP,
        f"  1. Filler Words        :   {sc['filler_words']['score']:>2}  / {sc['filler_words']['max']}",
        f"  2. Speaking Speed      :   {sc['speaking_speed']['score']:>2}  / {sc['speaking_speed']['max']}",
        f"  3. Facial Confidence   :   {sc['facial_confidence']['score']:>2}  / {sc['facial_confidence']['max']}",
        f"  4. Voice Confidence    :   {sc['voice_confidence']['score']:>2}  / {sc['voice_confidence']['max']}",
        f"  5. Overall Fluency     :   {sc['overall_fluency']['score']:>2}  / {sc['overall_fluency']['max']}",
        f"  {'─'*42}",
        f"     TOTAL               :  {report['total_score']:>3}  / 100",
        "",
        SEP,
        "1. FILLER WORDS",
        SEP,
        f"  Filler Count    : {fa['total_fillers']}",
        f"  Filler Ratio    : {fa.get('filler_ratio_pct', 0):.1f}%",
    ]

    if fa.get("filler_breakdown"):
        lines.append("  Top Fillers     :")
        for word, count in list(fa["filler_breakdown"].items())[:8]:
            lines.append(f"    • '{word}': {count}×")

    lines += [
        "",
        SEP,
        "2. SPEAKING SPEED",
        SEP,
        f"  Word Count      : {sa['word_count']:,} words",
        f"  WPM             : {sa['wpm']}  ({sa['category']})",
        f"  Feedback        : {sa['feedback']}",
        f"  Optimal Range   : {sa['optimal_range']}",
        "",
        SEP,
        "3. FACIAL CONFIDENCE",
        SEP,
        f"  Prediction      : {va.get('facial_confidence', '—').replace('_', ' ').title()}",
        f"  Confident Prob  : {va.get('facial_confidence_pct', 0):.1f}%",
        f"  Face Detected   : {va['face_detection_rate']}% of frames",
        f"  Head Stability  : {va['head_stability']} / 100",
        f"  Eye Contact     : {va['eye_contact_score']} / 100",
        "",
        SEP,
        "4. VOICE CONFIDENCE",
        SEP,
        f"  Prediction      : {vca.get('prediction', '—').replace('_', ' ').title()}",
        f"  Quality Label   : {vca.get('quality_label', '—')}",
        f"  Confident Prob  : {vca.get('confident_prob', 0):.1%}",
    ]

    if vca.get("probabilities"):
        lines.append("  Probabilities   :")
        for label, prob in vca["probabilities"].items():
            lines.append(f"    • {label}: {prob:.1%}")

    fluency_level = ca.get("fluency_level", "—").replace("_", " ").title()
    lines += [
        "",
        SEP,
        "5. OVERALL FLUENCY",
        SEP,
        f"  Fluency Level   : {fluency_level}",
        f"  Fluency Feedback: {ca.get('fluency_feedback', '—')}",
        "",
        SEP,
        "AI CONTENT INSIGHTS",
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
    ]

    return "\n".join(lines)
