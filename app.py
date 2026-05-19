"""
app.py  —  AI Interview Intelligence Platform
Run with:  streamlit run app.py

FINAL SCORE STRUCTURE:
  Filler Words       /25  local sklearn model
  Speaking Speed     /20  WPM calculation
  Facial Confidence  /25  local sklearn model (FER2013 + MobileNetV2)
  Voice Confidence   /20  local sklearn model (RAVDESS)
  Overall Fluency    /10  Groq LLaMA 3
  ──────────────────────
  TOTAL             /100
"""
from __future__ import annotations

import os
import shutil
import tempfile

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.audio_processor import (
    extract_audio,
    get_file_size_mb,
    analyze_voice_confidence,
)
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
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem; font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .section-hdr {
        font-size: 1.2rem; font-weight: 700;
        border-left: 4px solid #667eea;
        padding-left: 10px; margin: 16px 0 10px 0;
    }
    .insight-box {
        background: rgba(102,126,234,0.10);
        border-left: 3px solid #667eea;
        border-radius: 6px; padding: 10px 14px;
        margin: 6px 0; font-size: 0.95rem; line-height: 1.5;
    }
    .filler-tag {
        display: inline-block;
        background: rgba(255,68,68,0.15);
        border: 1px solid rgba(255,68,68,0.4);
        border-radius: 20px; padding: 3px 10px;
        margin: 3px; font-size: 0.85rem;
    }
    .model-badge {
        display: inline-block;
        background: rgba(0,200,81,0.15);
        border: 1px solid rgba(0,200,81,0.4);
        border-radius: 12px; padding: 2px 9px;
        font-size: 0.78rem; color: #00C851;
        margin-left: 6px; vertical-align: middle;
    }
    .api-badge {
        display: inline-block;
        background: rgba(33,150,243,0.15);
        border: 1px solid rgba(33,150,243,0.4);
        border-radius: 12px; padding: 2px 9px;
        font-size: 0.78rem; color: #2196F3;
        margin-left: 6px; vertical-align: middle;
    }
    .score-card {
        background: rgba(255,255,255,0.04);
        border-radius: 10px; padding: 16px 12px;
        text-align: center; border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_color(score: int, max_score: int) -> str:
    pct = score / max_score
    if pct >= 0.84: return "#00C851"
    if pct >= 0.64: return "#2196F3"
    if pct >= 0.44: return "#FF8800"
    return "#FF4444"


def _mini_gauge(value: float, max_val: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
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
    ))
    fig.update_layout(
        height=195, margin=dict(l=18, r=18, t=42, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
    )
    return fig


def _overall_gauge(score: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "Overall Score", "font": {"size": 16, "color": "white"}},
        number={"suffix": "/100", "font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar":  {"color": color, "thickness": 0.32},
            "steps": [
                {"range": [0,  40], "color": "rgba(204,0,0,0.25)"},
                {"range": [40, 60], "color": "rgba(255,136,0,0.25)"},
                {"range": [60, 75], "color": "rgba(33,150,243,0.25)"},
                {"range": [75, 90], "color": "rgba(0,130,51,0.25)"},
                {"range": [90,100], "color": "rgba(0,200,81,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=30, r=30, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
    )
    return fig


def _filler_bar_chart(filler_breakdown: dict) -> go.Figure:
    if not filler_breakdown:
        fig = go.Figure()
        fig.update_layout(
            title="🎉 No significant filler words detected!",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}, height=160,
        )
        return fig

    items  = list(filler_breakdown.items())[:12]
    words  = [w for w, _ in reversed(items)]
    counts = [c for _, c in reversed(items)]

    fig = go.Figure(go.Bar(
        y=words, x=counts, orientation="h",
        marker_color=px.colors.sequential.Plasma_r[:len(words)],
        text=counts, textposition="outside",
    ))
    fig.update_layout(
        title="Filler Word Frequency", xaxis_title="Count",
        height=max(240, len(words) * 32 + 80),
        margin=dict(l=10, r=35, t=42, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}, xaxis={"gridcolor": "rgba(255,255,255,0.08)"},
    )
    return fig


def _speed_gauge(wpm: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=wpm,
        title={"text": "Speaking Speed", "font": {"size": 14}},
        number={"suffix": " WPM", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 300]},
            "bar":  {"color": color, "thickness": 0.30},
            "steps": [
                {"range": [0,   80], "color": "rgba(255,68,68,0.25)"},
                {"range": [80, 100], "color": "rgba(255,136,0,0.25)"},
                {"range": [100,120], "color": "rgba(33,150,243,0.25)"},
                {"range": [120,160], "color": "rgba(0,200,81,0.25)"},
                {"range": [160,180], "color": "rgba(33,150,243,0.25)"},
                {"range": [180,200], "color": "rgba(255,136,0,0.25)"},
                {"range": [200,300], "color": "rgba(255,68,68,0.25)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.80, "value": wpm},
        },
    ))
    fig.update_layout(
        height=250, margin=dict(l=20, r=20, t=42, b=20),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
    )
    return fig


def _prob_bar(probs: dict, color_map: dict) -> go.Figure:
    labels = list(probs.keys())
    values = [v * 100 for v in probs.values()]
    colors = [color_map.get(l, "#2196F3") for l in labels]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        xaxis={"range": [0, 120], "gridcolor": "rgba(255,255,255,0.08)"},
        height=200, margin=dict(l=10, r=40, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("### 🏆 Score Structure")
    st.markdown("""
| Category | /Max |
|---|---|
| 🗣️ Filler Words | /25 |
| ⏱️ Speaking Speed | /20 |
| 🎥 Facial Confidence | /25 |
| 🎵 Voice Confidence | /20 |
| 📝 Overall Fluency | /10 |
| **TOTAL** | **/100** |
""")
    st.markdown("---")
    st.markdown("### 🎓 Grade Scale")
    st.markdown(
        "- **90–100** → Excellent (Interview Ready)\n"
        "- **75–89**  → Very Good\n"
        "- **60–74**  → Good (Practice required)\n"
        "- **40–59**  → Average\n"
        "- **< 40**   → Poor (Major improvement needed)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🎤 AI Interview Intelligence Platform</div>',
            unsafe_allow_html=True)
st.markdown(
    '*Real-time interview analyzer — '
    '<span class="model-badge"> Models</span> + '
    '<span class="api-badge"> (fluency + content)</span>*',
    unsafe_allow_html=True,
)
st.divider()

uploaded_file = st.file_uploader(
    "📹 Upload your mock interview video",
    type=["mp4", "avi", "mov", "webm", "mkv", "m4v"],
    help="Supported: MP4, AVI, MOV, WebM, MKV — max 200 MB",
)

analyze_clicked = False
if uploaded_file:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        analyze_clicked = st.button(
            "🚀 Analyze Interview", type="primary", use_container_width=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

if analyze_clicked and uploaded_file:
    tmp_dir        = tempfile.mkdtemp(prefix="interview_")
    tmp_video_path = os.path.join(tmp_dir, uploaded_file.name)

    try:
        with open(tmp_video_path, "wb") as fh:
            fh.write(uploaded_file.read())

        with st.status("🔄 Analyzing your interview…", expanded=True) as status:

            st.write("🎵 Extracting audio…")
            audio_path, duration = extract_audio(tmp_video_path)

            if get_file_size_mb(audio_path) > 24.5:
                os.unlink(audio_path)
                raise ValueError("Audio exceeds 25 MB limit. Use a shorter video.")

            st.write("📝 Transcribing with Groq Whisper…")
            client        = get_groq_client()
            transcription = transcribe_audio(client, audio_path)
            transcript    = transcription["text"]

            if not transcript.strip():
                os.unlink(audio_path)
                raise ValueError("No speech detected. Ensure clear English audio.")

            # Step 3: Voice confidence — local model
            st.write("🎵 Analyzing voice confidence ")
            voice_analysis = analyze_voice_confidence(audio_path)
            os.unlink(audio_path)

            # Step 4: Filler + speed — local model + WPM
            st.write("🔍 Analyzing filler words ")
            filler_analysis = analyze_filler_words(transcript)
            speed_analysis  = calculate_speech_speed(transcript, duration)

            # Score uses filler_ratio (word_count-based) per spec
            filler_score = calculate_filler_score(
                filler_analysis["total_fillers"],
                speed_analysis["word_count"],
            )
            speed_score = calculate_speed_score(speed_analysis["wpm"])

            # Step 5: Facial confidence — local model
            st.write("🎥 Analyzing facial confidence ")
            video_analysis = analyze_video(tmp_video_path)

            # Step 6: Fluency + content insights — Groq LLaMA 3
            st.write("🤖 Analyzing fluency & content ")
            content_analysis = analyze_content(client, transcript, duration)

            report = compile_report(
                transcript=transcript,
                filler_analysis=filler_analysis,
                speed_analysis=speed_analysis,
                video_analysis=video_analysis,
                content_analysis=content_analysis,
                voice_analysis=voice_analysis,
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
    vca    = report.get("voice_analysis", {})

    st.divider()
    st.markdown("## 📊 Interview Analysis Results")

    # ── Overall score ─────────────────────────────────────────────────────────
    col_gauge, col_grade = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(_overall_gauge(report["total_score"], report["color"]),
                        use_container_width=True)
    with col_grade:
        st.markdown(f"""
<div style="padding:28px 10px;">
  <span style="font-size:4.5rem;font-weight:900;color:{report['color']};">{report['grade']}</span><br>
  <span style="font-size:1.6rem;color:{report['color']};font-weight:600;">{report['rating']}</span><br>
  <span style="font-size:1rem;color:#aaa;">{report['grade_feedback']}</span>
  <hr style="border-color:rgba(255,255,255,0.1);margin:14px 0;">
  <table style="font-size:1rem;color:#ccc;border-spacing:0 6px;">
    <tr><td>💬 Words</td><td style="padding-left:14px;color:white;font-weight:600;">{sa['word_count']:,}</td></tr>
    <tr><td>🗣️ Fillers</td><td style="padding-left:14px;color:white;font-weight:600;">{fa['total_fillers']} ({fa.get('filler_ratio_pct',0):.1f}%)</td></tr>
    <tr><td>🎵 Voice</td><td style="padding-left:14px;color:white;font-weight:600;">{vca.get('quality_label','—')}</td></tr>
    <tr><td>🎥 Face</td><td style="padding-left:14px;color:white;font-weight:600;">{va.get('facial_confidence','—').replace('_',' ').title()}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Score breakdown (5 gauges) ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📈 Score Breakdown</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    breakdown = [
        (c1, "🗣️ Filler Words",      scores["filler_words"]["score"],      25),
        (c2, "⏱️ Speaking Speed",    scores["speaking_speed"]["score"],    20),
        (c3, "🎥 Facial Confidence", scores["facial_confidence"]["score"], 25),
        (c4, "🎵 Voice Confidence",  scores["voice_confidence"]["score"],  20),
        (c5, "📝 Overall Fluency",   scores["overall_fluency"]["score"],   10),
    ]
    for col, label, score, max_s in breakdown:
        with col:
            color = _score_color(score, max_s)
            st.plotly_chart(_mini_gauge(score, max_s, label, color),
                            use_container_width=True)

    st.markdown("---")

    # ── 1. Filler Words ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🗣️ 1. Filler Words /25</div>',
        unsafe_allow_html=True,
    )
    col_filler, col_speed = st.columns([3, 2])

    with col_filler:
        st.plotly_chart(_filler_bar_chart(fa["filler_breakdown"]),
                        use_container_width=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Total Fillers",  fa["total_fillers"])
        with m2: st.metric("Filler Ratio",   f"{fa.get('filler_ratio_pct', 0):.1f}%")
        with m3: st.metric("Unique Types",   fa["unique_filler_types"])

        if fa["top_fillers"]:
            st.markdown("**Most frequent:**")
            tags = " ".join(f'<span class="filler-tag">"{w}" ×{c}</span>'
                            for w, c in fa["top_fillers"])
            st.markdown(tags, unsafe_allow_html=True)

    # ── 2. Speaking Speed ─────────────────────────────────────────────────────
    with col_speed:
        st.markdown(
            '<div class="section-hdr">⏱️ 2. Speaking Speed /20</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_speed_gauge(sa["wpm"], sa["color"]),
                        use_container_width=True)
        st.markdown(
            f'<div class="insight-box">'
            f'<strong>{sa["emoji"]} {sa["category"]}</strong><br>'
            f'{sa["feedback"]}<br>'
            f'<small style="color:#aaa;">Optimal: {sa["optimal_range"]}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 3. Facial Confidence ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🎥 3. Facial Confidence /25</div>',
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    face_color = "#00C851" if va.get("facial_confidence") == "confident" else "#FF4444"
    with fc1: st.metric("Face Detected",      f"{va['face_detection_rate']}%")
    with fc2: st.metric("Head Stability",     f"{va['head_stability']}/100")
    with fc3: st.metric("Eye Contact",        f"{va['eye_contact_score']}/100")
    with fc4:
        st.markdown(
            f'<div style="text-align:center;padding-top:8px;">'
            f'<div style="font-size:0.85rem;color:#aaa;">Prediction</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{face_color};">'
            f'{va.get("facial_confidence","—").replace("_"," ").title()}</div>'
            f'<div style="font-size:0.9rem;color:#ccc;">{va.get("facial_confidence_pct",0):.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    face_tips: list[str] = []
    if va["face_detection_rate"] < 70:
        face_tips.append("📷 Ensure your face is fully visible and well-lit.")
    if va["head_stability"] < 55:
        face_tips.append("🧘 Minimise head movements to appear composed.")
    if va["eye_contact_score"] < 55:
        face_tips.append("👀 Look directly into the camera lens.")
    if va.get("facial_confidence") == "not_confident":
        face_tips.append("😊 Maintain a calm, positive expression throughout.")
    if not face_tips:
        face_tips.append("✅ Strong facial presence — keep it up!")
    for tip in face_tips:
        st.markdown(f'<div class="insight-box">{tip}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Voice Confidence ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🎵 4. Voice Confidence /20 '
        '<span class="model-badge">🤖 Local sklearn (RAVDESS)</span></div>',
        unsafe_allow_html=True,
    )

    vc1, vc2 = st.columns([1, 2])
    with vc1:
        vc_color = {"confident": "#00C851", "neutral": "#FF8800",
                    "not_confident": "#FF4444"}.get(vca.get("prediction", ""), "#2196F3")
        st.markdown(
            f"""
<div style="padding:20px;text-align:center;">
  <div style="font-size:2.4rem;font-weight:900;color:{vc_color};">
    {vca.get('prediction','—').replace('_',' ').title()}
  </div>
  <div style="font-size:1.2rem;color:#ccc;margin-top:6px;">
    {vca.get('quality_label','—')}
  </div>
  <div style="font-size:1.5rem;font-weight:700;color:{vc_color};margin-top:8px;">
    {vca.get('voice_score',0)}/20
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with vc2:
        if vca.get("probabilities"):
            fig = _prob_bar(
                vca["probabilities"],
                {"not_confident": "#FF4444", "neutral": "#FF8800", "confident": "#00C851"},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 5. Overall Fluency ────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">📝 5. Overall Fluency /10 '
        '<span class="api-badge">☁️ Groq LLaMA 3</span></div>',
        unsafe_allow_html=True,
    )

    fluency_level = ca.get("fluency_level", "—").replace("_", " ").title()
    fluency_score = scores["overall_fluency"]["score"]
    fluency_color = _score_color(fluency_score, 10)

    fl1, fl2 = st.columns([1, 3])
    with fl1:
        st.markdown(
            f"""
<div style="text-align:center;padding:20px;">
  <div style="font-size:2rem;font-weight:900;color:{fluency_color};">{fluency_level}</div>
  <div style="font-size:1.5rem;color:{fluency_color};font-weight:700;">{fluency_score}/10</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with fl2:
        if ca.get("fluency_feedback"):
            st.markdown(
                f'<div class="insight-box">📝 {ca["fluency_feedback"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── AI Content Insights ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🤖 AI Content Insights '
        '<span class="api-badge">☁️ Groq LLaMA 3</span></div>',
        unsafe_allow_html=True,
    )

    if ca.get("assessment"):
        st.info(f"**Overall Assessment:** {ca['assessment']}")

    ai1, ai2, ai3 = st.columns(3)
    with ai1:
        st.markdown("#### ✅ Strengths")
        for s in ca.get("strengths", []):
            st.markdown(f'<div class="insight-box">✓ {s}</div>', unsafe_allow_html=True)
    with ai2:
        st.markdown("#### 🎯 Areas to Improve")
        for item in ca.get("improvements", []):
            st.markdown(f'<div class="insight-box">→ {item}</div>', unsafe_allow_html=True)
    with ai3:
        st.markdown("#### 💡 Recommendations")
        for r in ca.get("recommendations", []):
            st.markdown(f'<div class="insight-box">★ {r}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Transcript ────────────────────────────────────────────────────────────
    with st.expander("📄 Full Transcript", expanded=False):
        st.text_area("transcript", value=report["transcript"],
                     height=220, disabled=True, label_visibility="collapsed")

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⬇️ Download Report</div>', unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=generate_text_report(report),
        file_name="interview_report.txt",
        mime="text/plain",
    )

# ── Empty state ───────────────────────────────────────────────────────────────
elif not uploaded_file:
    st.markdown("""
<div style="text-align:center;padding:60px 0;opacity:0.65;">
  <div style="font-size:5rem;">🎤</div>
  <h2 style="margin-top:10px;">Ready to analyze your interview?</h2>
  <p>Upload a video of your mock interview to receive AI-powered feedback.</p>
  <table style="margin:20px auto;font-size:0.95rem;border-spacing:12px 6px;">
    <tr><td>🗣️ Filler Words</td><td style="color:#00C851;">/25 — Local sklearn model</td></tr>
    <tr><td>⏱️ Speaking Speed</td><td style="color:#00C851;">/20 — WPM analysis</td></tr>
    <tr><td>🎥 Facial Confidence</td><td style="color:#00C851;">/25 — Local sklearn model</td></tr>
    <tr><td>🎵 Voice Confidence</td><td style="color:#00C851;">/20 — Local sklearn model</td></tr>
    <tr><td>📝 Overall Fluency</td><td style="color:#2196F3;">/10 — Groq LLaMA 3</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)
