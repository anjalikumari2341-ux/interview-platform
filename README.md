# 🎤 AI Interview Intelligence Platform

A multimodal interview analysis platform. It scores a mock interview on five dimensions using locally trained machine-learning models and Groq AI, and it can also **conduct the interview itself** with a spoken AI interviewer. It works in **English and Hindi**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Two Ways to Use It](#two-ways-to-use-it)
- [Score Structure](#score-structure)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Models and Honest Results](#models-and-honest-results)
- [Languages](#languages)
- [Known Limitations](#known-limitations)
- [Technologies Used](#technologies-used)
- [Team](#team)

---

## Overview

The platform gives candidates objective, data-driven feedback on a mock interview. A video goes through a pipeline of audio extraction, speech transcription, filler-word detection, speaking-speed analysis, voice-confidence analysis, facial-confidence analysis and AI content evaluation. The result is a score out of 100, a detailed dashboard and a downloadable report.

The **AI Interviewer** adds a second mode. An interviewer called Aria (or Arjun) asks questions aloud for a chosen job role, listens to the candidate's spoken answers, chooses the next question based on what was said, and produces a scored report at the end.

---

## Features

- 🎵 **Audio extraction** from any uploaded video format (MoviePy + FFmpeg)
- 📝 **Speech-to-text** with Groq Whisper (large-v3-turbo for English, large-v3 for Hindi, with automatic fallback)
- 🗣️ **Filler-word detection** using a word list and counting (rule-based), in English and Hindi
- ⏱️ **Speaking-speed analysis** (words per minute) with feedback
- 🎥 **Facial confidence** from MobileNetV2 features, a FER2013-trained classifier, head stability and eye contact (MediaPipe)
- 🎵 **Voice confidence** from MFCC audio features and a RAVDESS-trained classifier
- 🤖 **Fluency and content evaluation** with a Groq LLM
- 🎙️ **AI Interviewer**: spoken questions, role-specific follow-ups, per-answer feedback and a final report
- 🌐 **English and Hindi** in both modes
- 📊 **Interactive dashboard** with Plotly charts
- 📥 **Downloadable report** (plain text)
- ✅ **No data stored**: analysis is session-based and temporary files are deleted

---

## Two Ways to Use It

| Mode | Where | What it does |
|---|---|---|
| **Upload a video** | Main page (`app.py`) | Analyses a recorded interview. All five scores. |
| **AI Interviewer** | "AI Interviewer" in the sidebar | Live, audio-only mock interview with a spoken AI interviewer. Four scores, rescaled to 100. |

### AI Interviewer in brief

1. **Setup:** choose the language, the job role (or **Other** to type your own), your level (fresher / intermediate / advanced), 3, 5 or 7 questions, and the interviewer voice.
2. **Interview:** the question is shown and spoken. The candidate records an answer in the browser and presses Submit. The AI rates the answer and picks the next question: at most one follow-up per topic, then a new topic.
3. **Results:** overall grade, four scores, an AI assessment with strengths, areas to improve and recommendations, a question-by-question breakdown, and a downloadable report.

---

## Score Structure

### Upload mode (out of 100)

| Category | Engine | Max Score |
|---|---|---|
| 🗣️ Filler Words | Word list + counting | /25 |
| ⏱️ Speaking Speed | WPM calculation | /20 |
| 🎥 Facial Confidence | Local sklearn + MobileNetV2 | /25 |
| 🎵 Voice Confidence | Local sklearn model | /20 |
| 📝 Overall Fluency | Groq LLM | /10 |
| **TOTAL** | | **/100** |

### AI Interviewer mode

The audio-only interviewer has no camera, so there is no face score. The remaining four parts add up to 75 points, and the total is **rescaled to 100** (`raw ÷ 75 × 100`). If the voice model cannot score an answer, that part is left out and the total is rescaled over what remains. The AI's 1–10 *answer relevance* rating is shown for information and is **not** part of the total.

### Grade Scale

| Score | Grade | Rating |
|---|---|---|
| 90–100 | A+ | Excellent — Interview ready |
| 75–89 | A | Very Good — Minor improvements needed |
| 60–74 | B | Good — Practice required |
| 40–59 | C | Average — Needs improvement |
| Below 40 | F | Poor — Major improvement needed |

---

## Project Structure

```
Ai_interview_intelligence/
├── app.py                        # Main page: upload-a-video analyzer
├── pages/
│   └── 1_AI_Interviewer.py       # AI Interviewer page (audio only)
├── modules/
│   ├── __init__.py
│   ├── config.py                 # Finds the model folder
│   ├── audio_processor.py        # Audio extraction + voice confidence
│   ├── speech_analyzer.py        # English filler detection + WPM scoring
│   ├── video_analyzer.py         # Facial confidence analysis
│   ├── groq_client.py            # Groq API (Whisper + LLM)
│   ├── ai_interviewer.py         # Interviewer logic, speech output, Hindi fillers, final report
│   └── report_generator.py       # Score compilation + report (upload mode)
├── model/                        # Trained model files (stored with Git LFS)
│   ├── best_filler_model.pkl
│   ├── filler_words.pkl
│   ├── voice_best_model.pkl
│   ├── voice_scaler.pkl
│   ├── face_best_model.pkl
│   ├── face_scaler.pkl
│   └── face_pca.pkl
├── training/                     # Training scripts (70:30 split)
│   ├── train_voice_model.py
│   ├── train_face_model.py
│   └── train_filler_model.py
├── .env                          # GROQ_API_KEY (never committed)
├── requirements.txt
├── packages.txt                  # System packages for deployment (ffmpeg, libsndfile1)
├── Dockerfile
└── .gitattributes                # Git LFS rule for *.pkl
```

---

## Requirements

### Hardware
- Processor: Intel Core i5 or equivalent (i7 recommended)
- RAM: 8 GB minimum (16 GB recommended)
- Storage: 5 GB free space
- Internet connection (needed for Groq and for the spoken questions)
- A microphone (for the AI Interviewer)

### Software
- Python 3.11 (see `.python-version`)
- FFmpeg installed and on the system PATH
- Git and [Git LFS](https://git-lfs.com) (the model files are stored with LFS)
- A modern browser (Chrome or Edge) that allows microphone access

---

## Installation

### 1. Clone the repository

```bash
git lfs install
git clone https://github.com/pushpanjali206/interview-platform.git
cd interview-platform
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This includes `streamlit>=1.40` (needed for the microphone recorder) and `edge-tts` (spoken questions).

### 4. Install FFmpeg

- **Windows**: download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to PATH
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Linux**: `sudo apt install ffmpeg libsndfile1`

### 5. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com). Never commit this file; it is already listed in `.gitignore`.

---

## Model Setup

The trained models live in the `model/` folder and are tracked with Git LFS, so a normal `git clone` (with LFS installed) downloads them. If you retrain them, put the new files in `model/`.

| File | Description |
|---|---|
| `voice_best_model.pkl` | Voice confidence ensemble (RAVDESS), two classes |
| `voice_scaler.pkl` | StandardScaler fitted on the voice training split |
| `face_best_model.pkl` | Facial confidence ensemble (FER2013 + MobileNetV2) |
| `face_scaler.pkl`, `face_pca.pkl` | Scaler and PCA for the face model (keep the three face files together) |
| `best_filler_model.pkl`, `filler_words.pkl` | Filler word list and a reference classifier (not used for the score) |

> **Important:** a model file must be loaded with the same major versions of numpy and scikit-learn that created it. A model saved with numpy 2.x cannot be loaded with numpy 1.x. Train inside the same environment the app runs in (this project uses numpy 1.26.x).

To retrain, see the scripts in `training/` (instructions are at the top of each file). Datasets: RAVDESS (speech), FER2013, and the filler transcripts CSV.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The **AI Interviewer** appears in the left sidebar.

---

## How It Works

### Upload mode
1. Choose the interview language (English or Hindi) and upload a video (MP4, AVI, MOV, WebM, MKV).
2. Press **Analyze Interview**. The pipeline runs automatically:
   - audio extraction → transcription (Groq Whisper) → voice confidence → filler words and WPM → facial analysis → fluency and content (Groq LLM)
3. View the dashboard and download the report.

### AI Interviewer
1. The LLM writes a role-specific opening question, which is also spoken aloud (edge-tts).
2. The candidate's recorded answer is transcribed, then measured (fillers, WPM, voice).
3. The LLM rates the answer (relevance 1–10), writes feedback, and picks the next question.
4. After the last answer, a final report combines all answers.

Leading and trailing silence in a recording is trimmed before speaking speed is computed.

---

## Models and Honest Results

All models were evaluated on a **70:30 train/test split**, with a fixed random state. For the voice model the split is **by actor**, so the test speakers are never seen in training.

| Model | Data | Result on held-out 30% | Notes |
|---|---|---|---|
| Voice confidence | RAVDESS speech, 24 actors, two classes (neutral/calm/happy → confident; sad/fearful → not confident) | about **69 %** accuracy, macro F1 0.69, on 7 unseen actors | A majority-class guess would score about 56 %. The model catches not-confident clips better (recall 0.82) than confident ones (0.59). The training accuracy is 100 %, so it overfits. |
| Facial confidence | FER2013 (about 35,900 images) + MobileNetV2 features | about 72 % in a corrected 70:30 retraining experiment | The model currently deployed is the earlier one. The retrained version was not deployed because its file size was too large and it was saved with a different numpy version. |
| Filler words | Word list and counting | — | A classifier on `filler_count ≥ 1` only re-learns that rule, so its near-100 % accuracy proves nothing. The filler detector is **rule-based**. |

An earlier version of the voice model reported a higher accuracy because the same actors appeared in both the training and test sets. The speaker-independent figure above is the honest one.

---

## Languages

| Part | English | Hindi |
|---|---|---|
| Transcription | Whisper large-v3-turbo | Whisper large-v3 (falls back to turbo) |
| Interview questions and AI feedback | English | Hindi (Devanagari) |
| Spoken questions | Indian or US English voices | Hindi voices |
| Filler words | English word list | Hindi word list (Devanagari and romanised) |
| Voice model and speaking-speed targets | Designed for English | **Approximate** for Hindi |

The Hindi filler list is deliberately conservative. Very common words such as "तो" and "वो" are not counted, to avoid false positives.

---

## Known Limitations

- **Transcription errors:** Hindi transcription is less accurate than English and can garble names and technical words. This can lower the fluency and relevance scores. The AI is told to ignore spelling and recognition mistakes.
- **Hesitation sounds:** Whisper often leaves out "umm" and "uhh", so filler counts can be undercounted in both languages.
- **Voice model:** trained on 1,440 acted English clips. The mapping from emotion to "confidence" is an assumption, so treat the voice score as an indicator, not a measurement. It is less reliable for Hindi.
- **Speaking-speed targets** (120–160 WPM) are English-based.
- **AI-rated parts** (fluency, relevance, feedback) can vary slightly from run to run.
- **No face score** in the AI Interviewer mode, because it is audio only.
- **Facial analysis** depends on lighting, camera angle and video quality.

---

## Roadmap

- Live webcam analysis during the AI interview (adds the face score to that mode)
- Hinglish (mixed Hindi–English) support
- Retraining the facial model with smaller trees and a matching numpy version

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Streamlit | Web interface and multi-page app |
| Groq Whisper (large-v3, large-v3-turbo) | Speech-to-text |
| Groq LLM | Question generation, feedback, fluency and content evaluation |
| edge-tts | Spoken interviewer questions (English and Hindi) |
| scikit-learn | Voice, face and filler models |
| TensorFlow / MobileNetV2 | CNN feature extraction for facial analysis |
| librosa | Audio features (MFCC, chroma) and silence trimming |
| OpenCV | Video frame processing |
| MediaPipe | Face mesh, head stability and eye contact |
| MoviePy + FFmpeg | Audio extraction from video |
| Plotly | Gauges and charts |
| NumPy, joblib | Numerical computation and model loading |
| Git LFS | Storing the model files |

---

## Team

| Name | Roll Number |
|---|---|
| Anjali Kumari | 2315045 |
| Pushpanjali Kumari | 2315226 |
| Khushi Kumari | 2315143 |

**Guide:** Dr. Sharmistha Roy
**Institution:** Usha Martin University, Ranchi
**Degree:** Bachelor of Computer Applications

---

## License

This project is developed as an academic project for Usha Martin University. All rights reserved.
