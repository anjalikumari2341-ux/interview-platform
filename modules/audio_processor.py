from __future__ import annotations
"""
audio_processor.py
Extracts audio + runs local voice confidence model inference.

SCORING (from FINAL SCORE STRUCTURE doc):
  Voice Confidence: 20 marks
    Clear + stable       → 20   (confident)
    Slight variation     → 15   (neutral, high conf prob)
    Some hesitation      → 10   (neutral, low conf prob)
    Frequent breaks      → 5    (not_confident)

Supports all audio/video formats: mp4, avi, mov, webm, mkv, mp3, wav, m4a, ogg, flac
"""

import os
import tempfile
import numpy as np
from pathlib import Path

# ── Model paths ───────────────────────────────────────────────────────────────
_MODEL_DIR    = Path(__file__).resolve().parent.parent / "models"
_VOICE_MODEL  = _MODEL_DIR / "voice_best_model.pkl"
_VOICE_SCALER = _MODEL_DIR / "voice_scaler.pkl"

_VOICE_LABEL_NAMES = {0: "not_confident", 1: "neutral", 2: "confident"}

# Audio config (must match training)
_SAMPLE_RATE = 22050
_DURATION    = 3.0
_N_MFCC      = 40


# ── Safe model loader (handles corrupted / version-mismatched pickles) ────────
_voice_model = _voice_scaler = None


def _safe_load(path: Path):
    """
    Try joblib first (handles numpy version mismatches better),
    fall back to pickle. Raises RuntimeError with a clear message
    if the file is corrupted or truncated.
    """
    if not path.exists():
        raise RuntimeError(
            f"Model file not found: {path}\n"
            "Re-download from Colab and place in the models/ folder."
        )
    if path.stat().st_size < 100:
        raise RuntimeError(
            f"Model file is too small / corrupted: {path}\n"
            "The Colab download was likely cut off. Re-download the file."
        )

    # Try joblib first (best for sklearn models across numpy versions)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass

    # Fall back to pickle
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except EOFError:
        raise RuntimeError(
            f"Model file is corrupted / incomplete: {path}\n"
            "The Colab download was cut off mid-file.\n"
            "Fix: Re-run Colab training and re-download the .pkl file.\n"
            "Better: In Colab use  import joblib; joblib.dump(model, 'file.pkl')"
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load model {path}: {e}\n"
            "This is usually a numpy/sklearn version mismatch.\n"
            "Fix: In Colab add !pip install joblib, then resave with:\n"
            "  import joblib; joblib.dump(model, 'voice_best_model.pkl')\n"
            "  joblib.dump(scaler, 'voice_scaler.pkl')"
        )


def _get_voice_models():
    global _voice_model, _voice_scaler
    if _voice_model is None:
        _voice_model  = _safe_load(_VOICE_MODEL)
        _voice_scaler = _safe_load(_VOICE_SCALER)
    return _voice_model, _voice_scaler


# ── Feature extraction helpers ────────────────────────────────────────────────
def _extract_features_from_array(y: np.ndarray) -> np.ndarray:
    """268-dim feature vector from a 3-sec mono float32 array."""
    import librosa

    mfcc     = librosa.feature.mfcc(y=y, sr=_SAMPLE_RATE, n_mfcc=_N_MFCC)
    delta    = librosa.feature.delta(mfcc)
    delta2   = librosa.feature.delta(mfcc, order=2)
    chroma   = librosa.feature.chroma_stft(y=y, sr=_SAMPLE_RATE)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(y=y)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=_SAMPLE_RATE)
    centroid = librosa.feature.spectral_centroid(y=y, sr=_SAMPLE_RATE)

    return np.concatenate([
        np.mean(mfcc,     axis=1), np.std(mfcc,     axis=1),
        np.mean(delta,    axis=1), np.std(delta,    axis=1),
        np.mean(delta2,   axis=1), np.std(delta2,   axis=1),
        np.mean(chroma,   axis=1), np.std(chroma,   axis=1),
        [np.mean(zcr),      np.std(zcr)],
        [np.mean(rms),      np.std(rms)],
        [np.mean(rolloff),  np.std(rolloff)],
        [np.mean(centroid), np.std(centroid)],
    ]).astype(np.float32)


def _extract_voice_features(audio_path: str) -> np.ndarray:
    """
    Load any audio format and extract features using sliding 3-sec windows.
    Averaging across windows gives a more representative prediction than
    just using the first 3 seconds of a long interview recording.
    """
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa not installed. Run: pip install librosa") from exc

    # Load full audio at target sample rate (librosa uses ffmpeg for non-wav)
    try:
        y_full, _ = librosa.load(audio_path, sr=_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load audio from '{audio_path}': {exc}\n"
            "Ensure FFmpeg is installed and in PATH."
        ) from exc

    window = int(_SAMPLE_RATE * _DURATION)   # 3-sec window in samples

    if len(y_full) <= window:
        # Short clip — pad and extract directly
        y_pad = np.pad(y_full, (0, max(0, window - len(y_full))))[:window]
        return _extract_features_from_array(y_pad)

    # Sliding window with 1.5-sec hop — average all windows
    hop       = int(_SAMPLE_RATE * 1.5)
    feat_list = [
        _extract_features_from_array(y_full[s: s + window])
        for s in range(0, len(y_full) - window + 1, hop)
    ]
    return np.mean(feat_list, axis=0).astype(np.float32)


# ── Voice score (per FINAL SCORE STRUCTURE doc) ───────────────────────────────
def calculate_voice_score(prediction: str, confident_prob: float) -> int:
    """
    Score voice confidence out of 20.
      confident                      → 20
      neutral  (confident_prob≥0.6)  → 15
      neutral  (confident_prob< 0.6) → 10
      not_confident                  → 5
    """
    if prediction == "confident":
        return 20
    if prediction == "neutral":
        return 15 if confident_prob >= 0.6 else 10
    return 5


# ── Public inference function ─────────────────────────────────────────────────
def analyze_voice_confidence(audio_path: str) -> dict:
    """
    Run local sklearn voice confidence model on *audio_path*.
    Accepts any audio format supported by librosa/ffmpeg.

    Returns
    -------
    {
        prediction        : str
        confidence_label  : int
        probabilities     : dict
        confident_prob    : float
        voice_score       : int   (0-20)
        quality_label     : str
    }
    """
    model, scaler = _get_voice_models()

    features       = _extract_voice_features(audio_path)
    feat_sc        = scaler.transform(features.reshape(1, -1))
    probs          = model.predict_proba(feat_sc)[0]
    pred_id        = int(probs.argmax())
    prediction     = _VOICE_LABEL_NAMES[pred_id]
    confident_prob = float(probs[2]) if len(probs) > 2 else float(probs[pred_id])
    voice_score    = calculate_voice_score(prediction, confident_prob)

    quality_map = {
        20: "Clear & Stable",
        15: "Slight Variation",
        10: "Some Hesitation",
        5:  "Frequent Breaks",
    }

    return {
        "prediction":       prediction,
        "confidence_label": pred_id,
        "probabilities":    {_VOICE_LABEL_NAMES[i]: round(float(p), 3)
                             for i, p in enumerate(probs)},
        "confident_prob":   round(confident_prob, 3),
        "voice_score":      voice_score,
        "quality_label":    quality_map.get(voice_score, "—"),
    }


# ── Audio extraction from video ───────────────────────────────────────────────
def extract_audio(video_path: str) -> tuple[str, float]:
    """
    Extract audio from *video_path* (any format) and write as 64-kbps MP3.
    Returns (audio_path, duration_seconds).
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as exc:
        raise RuntimeError(
            "moviepy is not installed. Run: pip install 'moviepy<2.0'"
        ) from exc

    try:
        video = VideoFileClip(video_path)
    except Exception as exc:
        err = str(exc).lower()
        if "ffmpeg" in err or "ffprobe" in err:
            raise RuntimeError(
                "FFmpeg not found. Install from https://ffmpeg.org/download.html "
                "and add to PATH."
            ) from exc
        raise RuntimeError(f"Could not open video file: {exc}") from exc

    if video.audio is None:
        video.close()
        raise ValueError("The video has no audio track.")

    duration: float = video.duration
    audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(audio_fd)

    try:
        video.audio.write_audiofile(
            audio_path, bitrate="64k", verbose=False, logger=None,
        )
    finally:
        video.close()

    return audio_path, duration


def get_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)
 