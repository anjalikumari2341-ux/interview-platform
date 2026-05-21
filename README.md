# 🎤 AI Interview Intelligence Platform

A real-time multimodal interview analysis platform that evaluates mock interview videos across five performance dimensions using locally trained machine learning models and Groq AI APIs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Score Structure](#score-structure)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Team](#team)

---

## Overview

The AI Interview Intelligence Platform analyzes uploaded interview videos and provides candidates with objective, data-driven feedback on their performance. The system processes video through a six-stage pipeline — audio extraction, speech transcription, filler word detection, voice confidence analysis, facial confidence analysis, and content quality evaluation — producing a composite score out of 100 with detailed insights and downloadable reports.

---

## Features

- 🎵 **Audio extraction** from any uploaded video format using MoviePy and FFmpeg
- 📝 **Speech-to-text transcription** using Groq Whisper large-v3-turbo
- 🗣️ **Filler word detection** using a locally trained sklearn classifier
- ⏱️ **Speaking speed analysis** with WPM calculation and feedback
- 🎥 **Facial confidence analysis** using MobileNetV2 CNN embeddings and FER2013-trained model
- 🎵 **Voice confidence classification** using MFCC features and RAVDESS-trained model
- 🤖 **Content quality and fluency evaluation** using Groq LLaMA 3 8B Instant
- 📊 **Interactive dashboard** with Plotly gauge charts, bar charts, and probability displays
- 📥 **Downloadable full analysis report** in plain text format
- ✅ **No data stored** — all analysis is session-based

---

## Score Structure

| Category | Engine | Max Score |
|---|---|---|
| 🗣️ Filler Words | Local sklearn model | /25 |
| ⏱️ Speaking Speed | WPM calculation | /20 |
| 🎥 Facial Confidence | Local sklearn + MobileNetV2 | /25 |
| 🎵 Voice Confidence | Local sklearn model | /20 |
| 📝 Overall Fluency | Groq LLaMA 3 | /10 |
| **TOTAL** | | **/100** |

### Grade Scale

| Score | Grade | Rating |
|---|---|---|
| 90–100 | A+ | Excellent — Interview Ready |
| 75–89 | A | Very Good — Minor improvements needed |
| 60–74 | B | Good — Practice required |
| 40–59 | C | Average — Needs improvement |
| Below 40 | F | Poor — Major improvement needed |

---

## Project Structure

```
your_project/
├── app.py                      # Streamlit main application
├── .env                        # Environment variables (GROQ_API_KEY)
├── requirements.txt            # All dependencies
│
├── models/                     # Trained ML model files
│   ├── best_filler_model.pkl   # Filler word classifier
│   ├── voice_best_model.pkl    # Voice confidence model
│   ├── voice_scaler.pkl        # StandardScaler for voice model
│   ├── face_best_model.pkl     # Facial confidence model
│   ├── face_scaler.pkl         # StandardScaler for face model
│   └── face_pca.pkl            # PCA for face model
│
└── modules/                    # Core analysis modules
    ├── __init__.py
    ├── audio_processor.py      # Audio extraction + voice confidence
    ├── speech_analyzer.py      # Filler detection + WPM scoring
    ├── video_analyzer.py       # Facial confidence analysis
    ├── groq_client.py          # Groq API (Whisper + LLaMA 3)
    └── report_generator.py     # Score compilation + report
```

---

## Requirements

### Hardware
- Processor: Intel Core i5 or equivalent (i7 recommended)
- RAM: 8 GB minimum (16 GB recommended)
- Storage: 5 GB free space
- Internet connection (required for Groq API)

### Software
- Python 3.9 or higher
- FFmpeg installed and added to system PATH

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-interview-intelligence.git
cd ai-interview-intelligence
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

### 4. Install FFmpeg

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Linux**: `sudo apt install ffmpeg`

### 5. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## Model Setup

Place the following trained model files inside the `models/` folder:

| File | Description |
|---|---|
| `best_filler_model.pkl` | Filler word sklearn classifier |
| `voice_best_model.pkl` | Voice confidence ensemble model (RAVDESS) |
| `voice_scaler.pkl` | StandardScaler fitted on RAVDESS training data |
| `face_best_model.pkl` | Facial confidence ensemble model (FER2013) |
| `face_scaler.pkl` | StandardScaler fitted on FER2013 training data |
| `face_pca.pkl` | PCA fitted on FER2013 training data |

> **Note:** These model files are trained separately using the provided training notebooks. They are not included in the repository due to file size.

---

## Running the App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## How It Works

1. **Upload** a mock interview video (MP4, AVI, MOV, WebM, MKV — max 200 MB)
2. **Click** the Analyze Interview button
3. The platform runs through six stages automatically:
   - Extracts audio using MoviePy and FFmpeg
   - Transcribes speech using Groq Whisper
   - Analyzes voice confidence using MFCC features and local model
   - Detects filler words and calculates WPM
   - Analyzes facial expressions from video frames using MobileNetV2
   - Evaluates content quality and fluency using Groq LLaMA 3
4. **View** the interactive dashboard with scores, charts, and AI insights
5. **Download** the full analysis report as a `.txt` file

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Streamlit | Web interface |
| Groq Whisper large-v3-turbo | Speech-to-text transcription |
| Groq LLaMA 3 8B Instant | Content quality and fluency evaluation |
| scikit-learn | Filler, voice, and facial confidence models |
| TensorFlow / MobileNetV2 | CNN feature extraction for facial analysis |
| librosa | Audio feature extraction (MFCC, chroma, etc.) |
| OpenCV | Video frame processing |
| MediaPipe | Face mesh detection and landmark tracking |
| MoviePy + FFmpeg | Audio extraction from video |
| Plotly | Interactive charts and gauge visualizations |
| NumPy | Numerical computation |
| joblib | Model serialization and loading |

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
