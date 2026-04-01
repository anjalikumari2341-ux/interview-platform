"""
app.py  —  AI Interview Intelligence Platform
Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
import shutil
import tempfile

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.audio_processor import extract_audio, get_file_size_mb
from modules.groq_client import analyze_content, get_groq_client, transcribe_audio
from modules.report_generator import compile_report, generate_text_report
from modules.speech_analyzer import (
    analyze_filler_words,
    calculate_filler_score,
    calculate_speech_speed,
    calculate_speed_score,
)
from modules.video_analyzer import analyze_video

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Intelligence",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Gradient title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    /* Section sub-headers */
    .section-hdr {
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 16px 0 10px 0;
    }
    /* Pill-style insight boxes */
    .insight-box {
        background: rgba(102, 126, 234, 0.10);
        border-left: 3px solid #667eea;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    /* Filler tag badges */
    .filler-tag {
        display: inline-block;
        background: rgba(255,68,68,0.15);
        border: 1px solid rgba(255,68,68,0.4);
        border-radius: 20px;
        padding: 3px 10px;
        margin: 3px;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(score: int, max_score: int = 25) -> str:
    pct = score / max_score
    if pct >= 0.84: return "#00C851"
    if pct >= 0.64: return "#2196F3"
    if pct >= 0.44: return "#FF8800"
    return "#FF4444"


def _mini_gauge(value: float, max_val: float, title: str, color: str) -> go.Figure:
    """Small gauge used in the 4-score breakdown row."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 13}},
            number={"suffix": f"/{int(max_val)}", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, max_val], "tickwidth": 1},
                "bar":  {"color": color, "thickness": 0.30},
                "steps": [
                    {"range": [0,              max_val * 0.44], "color": "rgba(255,68,68,0.20)"},
                    {"range": [max_val * 0.44, max_val * 0.64], "color": "rgba(255,136,0,0.20)"},
                    {"range": [max_val * 0.64, max_val * 0.84], "color": "rgba(33,150,243,0.20)"},
                    {"range": [max_val * 0.84, max_val],        "color": "rgba(0,200,81,0.20)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=195,
        margin=dict(l=18, r=18, t=42, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def _overall_gauge(score: int, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Overall Score", "font": {"size": 16, "color": "white"}},
            number={"suffix": "/100", "font": {"size": 44, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar":  {"color": color, "thickness": 0.32},
                "steps": [
                    {"range": [0,  50], "color": "rgba(204,0,0,0.25)"},
                    {"range": [50, 65], "color": "rgba(255,136,0,0.25)"},
                    {"range": [65, 80], "color": "rgba(33,150,243,0.25)"},
                    {"range": [80,100], "color": "rgba(0,200,81,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def _filler_bar_chart(filler_breakdown: dict) -> go.Figure:
    if not filler_breakdown:
        fig = go.Figure()
        fig.update_layout(
            title="🎉 No significant filler words detected!",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=160,
        )
        return fig

    items  = list(filler_breakdown.items())[:12]
    words  = [w for w, _ in reversed(items)]
    counts = [c for _, c in reversed(items)]
    colors = px.colors.sequential.Plasma_r[: len(words)]

    fig = go.Figure(
        go.Bar(
            y=words,
            x=counts,
            orientation="h",
            marker_color=colors,
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Filler Word Frequency",
        xaxis_title="Count",
        height=max(240, len(words) * 32 + 80),
        margin=dict(l=10, r=35, t=42, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis={"gridcolor": "rgba(255,255,255,0.08)"},
    )
    return fig


def _speed_gauge(wpm: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=wpm,
            title={"text": "Speaking Speed", "font": {"size": 14}},
            number={"suffix": " WPM", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 300]},
                "bar":  {"color": color, "thickness": 0.30},
                "steps": [
                    {"range": [0,   80], "color": "rgba(255,68,68,0.25)"},
                    {"range": [80, 120], "color": "rgba(255,136,0,0.25)"},
                    {"range": [120,160], "color": "rgba(0,200,81,0.25)"},
                    {"range": [160,200], "color": "rgba(255,136,0,0.25)"},
                    {"range": [200,300], "color": "rgba(255,68,68,0.25)"},
                ],
                "threshold": {
                    "line":      {"color": color, "width": 3},
                    "thickness": 0.80,
                    "value":     wpm,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=42, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # api_key = st.text_input(
    #     "Groq API Key",
    #     type="password",
    #     placeholder="gsk_...",
    #     help="Free key at [console.groq.com](https://console.groq.com)",
    # )

    
#     st.markdown("---")
#     st.markdown("### 📊 Analysis Pipeline")
#     st.markdown(
#         """
# 1. 🎵 **Audio Extraction** — MoviePy + FFmpeg
# 2. 📝 **Speech-to-Text** — Groq Whisper v3 Turbo
# 3. 🔍 **Filler Word Analysis** — Regex NLP
# 4. ⏱️ **Speech Speed** — WPM calculation
# 5. 🎥 **Face Confidence** — MediaPipe Face Mesh
# 6. 🤖 **Content Quality** — Groq LLaMA 3
# """
#     )

    st.markdown("---")
    st.markdown("### 🏆 Scoring System")
    st.markdown(
        "Each category is scored out of **25** (total **100**):\n"
        "- 🗣️ Filler Words — fewer = better\n"
        "- ⏱️ Speech Speed — 120–160 WPM optimal\n"
        "- 💡 Content Quality — LLM evaluation\n"
        "- 🎥 Video Confidence — face & gaze metrics"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main area — header + upload
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-title">🎤 AI Interview Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.markdown("*A real-time speech, video & content analyzer*")
st.divider()

# if not api_key:
#     st.info(
#         "👈 Paste your **Groq API Key** in the sidebar to begin.  "
#         "[Get a free key →](https://console.groq.com)"
#     )

uploaded_file = st.file_uploader(
    "📹 Upload your mock interview video",
    type=["mp4", "avi", "mov", "webm", "mkv", "m4v"],
    help="Supported: MP4, AVI, MOV, WebM, MKV — max 200 MB",
)

# if uploaded_file and not api_key:
#     st.warning("⚠️ Add your Groq API Key in the sidebar to start analysis.")

analyze_clicked = False
if uploaded_file:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        analyze_clicked = st.button(
            "🚀 Analyze Interview", type="primary", use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# Analysis pipeline (runs only when button clicked)
# ─────────────────────────────────────────────────────────────────────────────

if analyze_clicked and uploaded_file:
    tmp_dir        = tempfile.mkdtemp(prefix="interview_")
    tmp_video_path = os.path.join(tmp_dir, uploaded_file.name)

    try:
        # Persist upload to disk
        with open(tmp_video_path, "wb") as fh:
            fh.write(uploaded_file.read())

        with st.status("🔄 Analyzing your interview…", expanded=True) as status:

            # ── Step 1: Audio extraction ──────────────────────────────────
            st.write("🎵 Extracting audio from video…")
            audio_path, duration = extract_audio(tmp_video_path)

            audio_mb = get_file_size_mb(audio_path)
            if audio_mb > 24.5:
                os.unlink(audio_path)
                raise ValueError(
                    f"Extracted audio ({audio_mb:.1f} MB) exceeds Groq's 25 MB limit. "
                    "Please use a shorter video (≤ ~30 min at normal speech rate)."
                )

            # ── Step 2: Speech-to-text ────────────────────────────────────
            st.write("📝 Transcribing with Groq Whisper…")
            client        = get_groq_client()
            transcription = transcribe_audio(client, audio_path)
            transcript    = transcription["text"]
            os.unlink(audio_path)   # free disk space

            if not transcript.strip():
                raise ValueError(
                    "No speech detected in the video. "
                    "Check that the audio is clear and in English."
                )

            # ── Step 3: Speech analysis ───────────────────────────────────
            st.write("🔍 Analyzing speech patterns…")
            filler_analysis = analyze_filler_words(transcript)
            speed_analysis  = calculate_speech_speed(transcript, duration)
            filler_score    = calculate_filler_score(
                filler_analysis["total_fillers"], duration
            )
            speed_score = calculate_speed_score(speed_analysis["wpm"])

            # ── Step 4: Video analysis ────────────────────────────────────
            st.write("🎥 Analyzing video for confidence signals…")
            video_analysis = analyze_video(tmp_video_path)

            # ── Step 5: LLM content analysis ──────────────────────────────
            st.write("🤖 Generating AI insights...")
            content_analysis = analyze_content(client, transcript, duration)

            # ── Compile report ────────────────────────────────────────────
            report = compile_report(
                transcript=transcript,
                filler_analysis=filler_analysis,
                speed_analysis=speed_analysis,
                video_analysis=video_analysis,
                content_analysis=content_analysis,
                duration_seconds=duration,
                filler_score=filler_score,
                speed_score=speed_score,
            )
            st.session_state["report"] = report
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    except Exception as exc:
        st.error(f"❌ Analysis failed: {exc}")
        st.session_state.pop("report", None)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

if "report" in st.session_state:
    report = st.session_state["report"]
    scores = report["scores"]
    fa     = report["filler_analysis"]
    sa     = report["speed_analysis"]
    va     = report["video_analysis"]
    ca     = report["content_analysis"]

    st.divider()
    st.markdown("## 📊 Interview Analysis Results")

    # ── Overall score + grade ─────────────────────────────────────────────────
    col_gauge, col_grade = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(
            _overall_gauge(report["total_score"], report["color"]),
            use_container_width=True,
        )

    with col_grade:
        st.markdown(
            f"""
<div style="padding:28px 10px;">
  <span style="font-size:4.5rem;font-weight:900;color:{report['color']};">{report['grade']}</span><br>
  <span style="font-size:1.6rem;color:{report['color']};font-weight:600;">{report['rating']}</span>
  <hr style="border-color:rgba(255,255,255,0.1);margin:14px 0;">
  <table style="font-size:1rem;color:#ccc;border-spacing:0 6px;">
    <tr><td>🕐 Duration</td><td style="padding-left:14px;color:white;font-weight:600;">{report['duration_formatted']}</td></tr>
    <tr><td>💬 Words</td><td style="padding-left:14px;color:white;font-weight:600;">{sa['word_count']:,}</td></tr>
    <tr><td>🗣️ Fillers</td><td style="padding-left:14px;color:white;font-weight:600;">{fa['total_fillers']}</td></tr>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Score breakdown (4 gauges) ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📈 Score Breakdown</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    breakdown = [
        (c1, "🗣️ Filler Words",     scores["filler_words"]),
        (c2, "⏱️ Speech Speed",     scores["speech_speed"]),
        (c3, "💡 Content Quality",  scores["content_quality"]),
        (c4, "🎥 Video Confidence", scores["video_confidence"]),
    ]
    for col, label, score in breakdown:
        with col:
            color = _score_color(score)
            st.plotly_chart(
                _mini_gauge(score, 25, label, color), use_container_width=True
            )

    st.markdown("---")

    # ── Speech analysis ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🎙️ Speech Analysis</div>', unsafe_allow_html=True)

    col_filler, col_speed = st.columns([3, 2])

    with col_filler:
        st.plotly_chart(
            _filler_bar_chart(fa["filler_breakdown"]), use_container_width=True
        )
        m1, m2, m3 = st.columns(3)
        dur_min = max(report["duration_seconds"] / 60, 0.01)
        with m1: st.metric("Total Fillers",   fa["total_fillers"])
        with m2: st.metric("Filler Types",    fa["unique_filler_types"])
        with m3: st.metric("Per Minute",       round(fa["total_fillers"] / dur_min, 1))

        if fa["top_fillers"]:
            st.markdown("**Most frequent:**")
            tags = " ".join(
                f'<span class="filler-tag">"{w}" ×{c}</span>'
                for w, c in fa["top_fillers"]
            )
            st.markdown(tags, unsafe_allow_html=True)

    with col_speed:
        st.plotly_chart(
            _speed_gauge(sa["wpm"], sa["color"]), use_container_width=True
        )
        st.markdown(
            f'<div class="insight-box">'
            f'<strong>{sa["emoji"]} {sa["category"]}</strong><br>'
            f'{sa["feedback"]}<br>'
            f'<small style="color:#aaa;">Optimal: {sa["optimal_range"]}</small>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Video confidence ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🎥 Video Confidence Analysis</div>',
        unsafe_allow_html=True,
    )

    if not va["mediapipe_available"]:
        st.warning(
            "⚠️ MediaPipe not installed — using basic Haar cascade detection. "
            "Install mediapipe for full pose & eye-contact analysis."
        )

    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1: st.metric("Face Detected",  f"{va['face_detection_rate']}%",
                         help="% of sampled frames with a clear face detection")
    with vc2: st.metric("Head Stability", f"{va['head_stability']}/100",
                         help="Smoothness of head position (100 = completely still)")
    with vc3: st.metric("Eye Contact",    f"{va['eye_contact_score']}/100",
                         help="Estimated gaze toward camera (100 = looking straight)")
    with vc4: st.metric("Video Score",    f"{va['video_confidence_score']}/25")

    # Auto-generated video tips
    tips: list[str] = []
    if va["face_detection_rate"] < 70:
        tips.append("📷 Ensure your face is fully visible and well-lit throughout.")
    if va["head_stability"] < 55:
        tips.append("🧘 Minimise unnecessary head movements to appear more composed.")
    if va["eye_contact_score"] < 55:
        tips.append("👀 Look directly into the camera lens — not at your own image.")
    if not tips:
        tips.append("✅ Strong video presence! Maintain this posture and gaze in real interviews.")

    for tip in tips:
        st.markdown(f'<div class="insight-box">{tip}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── AI Insights (Groq LLaMA) ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🤖 AI Insights — Groq LLaMA 3</div>',
        unsafe_allow_html=True,
    )

    if ca.get("assessment"):
        st.info(f"**Overall Assessment:** {ca['assessment']}")

    ai1, ai2, ai3 = st.columns(3)

    with ai1:
        st.markdown("#### ✅ Strengths")
        for s in ca.get("strengths", []):
            st.markdown(
                f'<div class="insight-box">✓ {s}</div>', unsafe_allow_html=True
            )

    with ai2:
        st.markdown("#### 🎯 Areas to Improve")
        for item in ca.get("improvements", []):
            st.markdown(
                f'<div class="insight-box">→ {item}</div>', unsafe_allow_html=True
            )

    with ai3:
        st.markdown("#### 💡 Recommendations")
        for r in ca.get("recommendations", []):
            st.markdown(
                f'<div class="insight-box">★ {r}</div>', unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Transcript ────────────────────────────────────────────────────────────
    with st.expander("📄 Full Transcript", expanded=False):
        st.text_area(
            "transcript",
            value=report["transcript"],
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">⬇️ Download Report</div>', unsafe_allow_html=True
    )
    text_report = generate_text_report(report)
    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=text_report,
        file_name="interview_report.txt",
        mime="text/plain",
    )
    st.caption(
        "The report contains your transcript, all scores, filler breakdown, "
        "video metrics and AI insights."
    )

# ── Empty state ───────────────────────────────────────────────────────────────
elif not uploaded_file:
    st.markdown(
        """
<div style="text-align:center;padding:60px 0;opacity:0.65;">
  <div style="font-size:5rem;">🎤</div>
  <h2 style="margin-top:10px;">Ready to analyze your interview?</h2>
  <p>Upload a video of your mock interview above to receive AI-powered feedback on<br>
  your <strong>speech clarity</strong>, <strong>filler word usage</strong>,
  <strong>speaking speed</strong>, <strong>eye contact</strong> and <strong>answer quality</strong>.</p>
</div>
""",
        unsafe_allow_html=True,
    )
