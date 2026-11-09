"""AI Interviewer page (audio only). Streamlit shows it automatically in the sidebar."""
import os
import shutil
import tempfile

import plotly.graph_objects as go
import streamlit as st

from modules import ai_interviewer as ai
from modules.groq_client import get_groq_client

st.set_page_config(page_title="AI Interviewer", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
.q-card{background:#161b2e;border-left:4px solid #6C63FF;border-radius:10px;
        padding:18px 20px;font-size:1.25rem;line-height:1.5;margin-bottom:10px;}
.avatar-ring{display:inline-block;border-radius:50%;padding:5px;
        animation:pulse 2.2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(108,99,255,.55);}
        70%{box-shadow:0 0 0 18px rgba(108,99,255,0);}100%{box-shadow:0 0 0 0 rgba(108,99,255,0);}}
.name-tag{font-size:1.2rem;font-weight:700;margin-top:8px;}
.sub-tag{color:#9aa;font-size:.9rem;}
</style>""", unsafe_allow_html=True)


def avatar_svg(male: bool) -> str:
    """Original illustrated interviewer avatar (no real person's photo)."""
    if male:
        hair_back = ""
        hair_front = '<path d="M64 84 Q62 48 100 46 Q138 48 136 84 Q124 66 100 66 Q76 66 64 84 Z" fill="#1c1410"/>'
    else:
        hair_back = '<path d="M58 95 Q55 40 100 38 Q145 40 142 95 L148 150 L52 150 Z" fill="#2b1d16"/>'
        hair_front = '<path d="M64 82 Q70 50 100 50 Q132 50 136 82 Q120 62 100 64 Q80 62 64 82 Z" fill="#2b1d16"/>'
    return f"""<div class="avatar-ring"><svg viewBox="0 0 200 200" width="170" height="170" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3b4cca"/>
<stop offset="1" stop-color="#8e54e9"/></linearGradient><clipPath id="c"><circle cx="100" cy="100" r="96"/></clipPath></defs>
<circle cx="100" cy="100" r="98" fill="url(#bg)"/><g clip-path="url(#c)">{hair_back}
<path d="M20 200 Q30 150 100 145 Q170 150 180 200 Z" fill="#2d3a63"/><path d="M85 148 L100 175 L115 148 Z" fill="#fff"/>
<rect x="88" y="120" width="24" height="32" rx="10" fill="#e8b48f"/><ellipse cx="100" cy="92" rx="34" ry="40" fill="#f1c7a3"/>
{hair_front}<circle cx="86" cy="92" r="4" fill="#222"/><circle cx="114" cy="92" r="4" fill="#222"/>
<path d="M79 82 Q86 78 93 82 M107 82 Q114 78 121 82" stroke="#3a2a20" stroke-width="2.5" fill="none" stroke-linecap="round"/>
<path d="M88 110 Q100 120 112 110" stroke="#a34b4b" stroke-width="3" fill="none" stroke-linecap="round"/></g></svg></div>"""


def gauge(value: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "/100", "font": {"size": 44, "color": color}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.35},
               "bgcolor": "rgba(255,255,255,0.05)"}))
    fig.update_layout(height=260, margin=dict(l=45, r=45, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def new_interview():
    st.session_state.pop("ai", None)
    st.rerun()


st.markdown("## 🎙️ AI Interviewer")
S = st.session_state.get("ai")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────────────────────────────────────
if S is None:
    st.caption("Practise a live mock interview. The interviewer speaks each question, you answer by voice, "
               "and you get a scored report at the end. (Audio only - no camera in this mode.)")
    c1, c2 = st.columns(2)
    with c1:
        role_choice = st.selectbox("Job role", ai.JOB_ROLES)
        custom_role = ""
        if role_choice == ai.OTHER_ROLE:
            custom_role = st.text_input("Type the job role", placeholder="e.g. Cyber Security Analyst")
        level = st.selectbox("Your level", ai.LEVELS)
    with c2:
        total = st.radio("Number of questions", [3, 5, 7], horizontal=True, index=1)
        voice_name = st.selectbox("Interviewer", list(ai.VOICES))
        instant = st.checkbox("Show short feedback after each answer", value=False,
                              help="Off = like a real interview. Full feedback is always in the final report.")

    if st.button("🚀 Start interview", type="primary"):
        role = custom_role.strip() if role_choice == ai.OTHER_ROLE else role_choice
        if not role:
            st.warning("Please type the job role first.")
        else:
            try:
                with st.spinner("Preparing your first question..."):
                    client = get_groq_client()
                    q1 = ai.generate_first_question(client, role, level, total)
                voice_id, male = ai.VOICES[voice_name]
                st.session_state["ai"] = {
                    "stage": "interview", "role": role, "level": level, "total": total,
                    "voice_id": voice_id, "male": male, "name": voice_name.split(" - ")[0],
                    "instant": instant, "questions": [q1], "answers": [], "idx": 0,
                    "audio": {}, "played": -1, "replay": False, "last_feedback": "",
                }
            except Exception as exc:
                st.error(f"Could not start the interview: {exc}")
            else:
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Interview
# ─────────────────────────────────────────────────────────────────────────────
elif S["stage"] == "interview":
    idx, total = S["idx"], S["total"]
    question = S["questions"][idx]
    left, right = st.columns([1, 2.4])

    with left:
        st.markdown(avatar_svg(S["male"]), unsafe_allow_html=True)
        st.markdown(f'<div class="name-tag">{S["name"]}</div>'
                    f'<div class="sub-tag">Interviewer · {S["role"]}</div>', unsafe_allow_html=True)
        st.progress(idx / total, text=f"Question {idx + 1} of {total}")

    with right:
        st.markdown(f'<div class="q-card">{question}</div>', unsafe_allow_html=True)

        if idx not in S["audio"]:
            with st.spinner("Aria is speaking..."):
                S["audio"][idx] = ai.synthesize_speech(question, S["voice_id"])
        speech = S["audio"][idx]
        if speech:
            st.audio(speech, format="audio/mp3", autoplay=(S["played"] != idx or S["replay"]))
            S["played"], S["replay"] = idx, False
            if st.button("🔁 Hear the question again"):
                S["replay"] = True
                st.rerun()
        else:
            st.caption("🔇 Voice is unavailable (offline or edge-tts not installed) - read the question above.")

        if S["instant"] and S["last_feedback"]:
            st.info(f"💬 About your last answer: {S['last_feedback']}")

        if hasattr(st, "audio_input"):
            rec = st.audio_input("🎤 Record your answer, then press Submit", key=f"rec_{idx}")
        else:
            rec = st.file_uploader("Upload your answer (audio)", type=["wav", "mp3", "m4a", "webm"],
                                   key=f"up_{idx}")
        st.caption("Tip: speak for 30–90 seconds, in a quiet room, and take your time.")

        b1, b2 = st.columns([1, 1])
        submit = b1.button("✅ Submit answer", type="primary", disabled=rec is None,
                           use_container_width=True)
        finish = b2.button("🏁 End interview now", disabled=not S["answers"], use_container_width=True,
                           help="Get the report for the questions you have answered so far.")

        if submit and rec is not None:
            tmp = tempfile.mkdtemp(prefix="aiint_")
            done = False
            try:
                path = os.path.join(tmp, getattr(rec, "name", None) or "answer.wav")
                with open(path, "wb") as fh:
                    fh.write(rec.getvalue())
                with st.spinner("Listening to your answer..."):
                    client = get_groq_client()
                    res = ai.analyze_answer_audio(client, path)
                    remaining = total - (idx + 1)
                    ev = ai.evaluate_and_next(client, S["role"], S["level"], S["answers"],
                                              question, res["transcript"], remaining)
                S["answers"].append({"question": question, **res,
                                     **{k: v for k, v in ev.items() if k != "next_question"}})
                S["last_feedback"] = ev["feedback"]
                if remaining > 0:
                    S["questions"].append(ev["next_question"])
                    S["idx"] += 1
                else:
                    with st.spinner("Preparing your report..."):
                        S["report"] = ai.build_final_report(client, S["answers"], S["role"], S["level"])
                    S["stage"] = "done"
                done = True
            except Exception as exc:
                st.error(f"❌ {exc}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if done:
                st.rerun()

        if finish:
            try:
                with st.spinner("Preparing your report..."):
                    S["report"] = ai.build_final_report(get_groq_client(), S["answers"], S["role"], S["level"])
                S["stage"] = "done"
            except Exception as exc:
                st.error(f"❌ {exc}")
            else:
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Results
# ─────────────────────────────────────────────────────────────────────────────
else:
    r = S["report"]
    sc = r["scores"]
    st.markdown(f"### Results - {r['role']}")
    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(gauge(r["total_score"], r["color"]), use_container_width=True)
    with g2:
        st.markdown(f"<div style='padding:20px 8px;'><span style='font-size:4rem;font-weight:900;"
                    f"color:{r['color']}'>{r['grade']}</span><br><span style='font-size:1.5rem;"
                    f"color:{r['color']};font-weight:600'>{r['rating']}</span><br>"
                    f"<span style='color:#aaa'>{r['grade_feedback']}</span></div>", unsafe_allow_html=True)
    st.caption("Audio-only mode: there is no face score, so the total is rescaled to 100 "
               f"(raw {r['raw_score']}/{r['max_raw']}).")

    m = st.columns(5)
    m[0].metric("🗣️ Filler words", f"{sc['filler']}/25")
    m[0].caption(f"{r['filler']['total_fillers']} filler words")
    m[1].metric("⏱️ Speaking speed", f"{sc['speed']}/20")
    m[1].caption(f"{r['speed']['wpm']} words per minute")
    m[2].metric("🎵 Voice", f"{sc['voice']}/20" if sc["voice"] is not None else "n/a")
    m[2].caption("average over your answers")
    m[3].metric("📝 Fluency", f"{sc['fluency']}/10")
    m[3].caption("from the AI's reading")
    m[4].metric("🎯 Answer relevance",
                f"{r['avg_relevance']}/10" if r["avg_relevance"] is not None else "n/a")
    m[4].caption("not part of the total")

    ca = r["content"]
    if not r.get("feedback_complete", True):
        st.warning("The AI returned only part of its feedback, so some sections below are empty and the "
                   "fluency score may be a default value. Press the button to ask it again.")
        if st.button("🔄 Regenerate AI feedback"):
            try:
                with st.spinner("Asking the AI again..."):
                    S["report"] = ai.build_final_report(get_groq_client(), r["answers"], r["role"], r["level"])
            except Exception as exc:
                st.error(f"❌ {exc}")
            else:
                st.rerun()
    if ca.get("assessment"):
        st.info(f"**Overall assessment:** {ca['assessment']}")
    c1, c2, c3 = st.columns(3)
    for col, title, key in ((c1, "✅ Strengths", "strengths"), (c2, "🎯 Areas to improve", "improvements"),
                            (c3, "💡 Recommendations", "recommendations")):
        with col:
            st.markdown(f"**{title}**")
            for item in ca.get(key, []):
                st.markdown(f"- {item}")

    st.markdown("#### Question by question")
    for i, a in enumerate(r["answers"], 1):
        with st.expander(f"Q{i}: {a['question']}  -  relevance {a.get('relevance_score', '-')}/10"):
            st.markdown(f"**Your answer:** {a['transcript']}")
            st.caption(f"{a['wpm']} WPM · {a['fillers']} filler words · {a['duration']:.0f}s")
            st.markdown(f"**Feedback:** {a.get('feedback', '')}")
            if a.get("improvement_tip"):
                st.markdown(f"**Tip:** {a['improvement_tip']}")

    d1, d2 = st.columns(2)
    d1.download_button("📥 Download report (.txt)", ai.report_to_text(r), file_name="ai_interview_report.txt",
                       use_container_width=True)
    if d2.button("🔄 Start a new interview", use_container_width=True):
        new_interview()
