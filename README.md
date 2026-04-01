# 🎤 AI Interview Intelligence Platform

An AI-powered mock interview analyzer that records your performance, transcribes your speech, and delivers a detailed feedback report — all powered by **Groq AI**.

---

## Features

| Module | Technology | What it measures |
|---|---|---|
| Speech-to-Text | Groq Whisper v3 Turbo | Full transcript |
| Filler Word Analysis | Regex NLP | um, uh, like, basically… |
| Speech Speed | WPM calculation | Pace vs 120–160 WPM optimal |
| Face Confidence | MediaPipe Face Mesh | Detection rate, head stability, eye contact |
| Content Quality | Groq LLaMA 3 | STAR method, clarity, depth |

**Scoring** — four categories × 25 pts = **100 pts total**

```
Grade  Score
A+     90–100   Outstanding
A      80–89    Excellent
B+     70–79    Very Good
B      60–69    Good
C      50–59    Average
D      40–49    Below Average
F      < 40     Needs Work
```

---

## Prerequisites

### 1. Python 3.9+
Verify with `python --version`.

### 2. FFmpeg (required by MoviePy)
MoviePy uses FFmpeg to extract audio from video files.

**Windows** (easiest via winget):
```powershell
winget install --id=Gyan.FFmpeg -e
```
Or download from <https://ffmpeg.org/download.html> and add the `bin/` folder to your `PATH`.

Verify: `ffmpeg -version`

### 3. Groq API Key
Sign up for free at <https://console.groq.com> and create an API key (`gsk_…`).

---

## Installation

```powershell
# 1. Navigate to the project folder
cd C:\Users\Abhishek\interview_platform

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `mediapipe` requires Visual C++ Redistributable on Windows.  
> Download from <https://aka.ms/vs/17/release/vc_redist.x64.exe> if you see DLL errors.

---

## Running the App

```powershell
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. Paste your **Groq API Key** in the left sidebar.
2. Select the **target job role** for the interview context.
3. **Upload** an MP4/MOV/WebM video of your mock interview.
4. Click **🚀 Analyze Interview**.
5. Wait ~30–60 seconds while the pipeline runs (5 steps shown live).
6. Review your **scores, charts, AI insights** and download the full report.

---

## Project Structure

```
interview_platform/
├── app.py                        # Streamlit UI (main entry point)
├── requirements.txt
├── README.md
└── modules/
    ├── audio_processor.py        # Video → MP3 extraction (MoviePy)
    ├── speech_analyzer.py        # Filler words + WPM + scoring
    ├── video_analyzer.py         # MediaPipe face analysis
    ├── groq_client.py            # Groq Whisper + LLaMA wrappers
    └── report_generator.py       # Score compilation + text export
```

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| LLM model | `llama3-8b-8192` | Change in `modules/groq_client.py` |
| STT model | `whisper-large-v3-turbo` | Change in `modules/groq_client.py` |
| Video sample rate | every 10 frames | Change `sample_every` in `analyze_video()` |
| Max upload size | 200 MB (Streamlit default) | Add `[server] maxUploadSize=500` to `.streamlit/config.toml` |

---

## Troubleshooting

**`FFmpeg not found`** — Install FFmpeg and restart your terminal.

**`mediapipe` import error** — Run `pip install mediapipe` again; the app will fall back to OpenCV Haar cascade detection if it fails.

**Groq `AuthenticationError`** — Double-check your API key starts with `gsk_`.

**Audio > 25 MB** — Use a shorter video clip or reduce the recording bitrate before uploading.
