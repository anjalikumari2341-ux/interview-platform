"""
video_analyzer.py
Analyzes candidate video for confidence signals using local sklearn model.

SCORING (from FINAL SCORE STRUCTURE doc):
  Facial Confidence: 25 marks
    Mostly confident (confident_prob > 80%)  → 25
    Moderate confidence   (60–80%)           → 20
    Mixed emotions        (40–60%)           → 15
    Nervous/stressed      (20–40%)           → 8
    Very low confidence   (< 20%)            → 3
"""
from __future__ import annotations

import cv2
import pickle
import numpy as np
from pathlib import Path

# ── Model paths ───────────────────────────────────────────────────────────────
_MODEL_DIR   = Path(__file__).resolve().parent.parent / "models"
_FACE_MODEL  = _MODEL_DIR / "face_best_model.pkl"
_FACE_SCALER = _MODEL_DIR / "face_scaler.pkl"
_FACE_PCA    = _MODEL_DIR / "face_pca.pkl"

_FACE_LABEL_NAMES = {0: "not_confident", 1: "confident"}

# MediaPipe landmark indices
_NOSE_TIP      = 4
_LEFT_EAR_OUT  = 234
_RIGHT_EAR_OUT = 454

# Image config (must match training)
_IMG_SIZE    = 48
_TARGET_SIZE = 96   # upsampled for MobileNetV2


# ── Load models once ──────────────────────────────────────────────────────────
_face_model = _face_scaler = _face_pca = None
_feature_extractor = None


def _get_face_models():
    global _face_model, _face_scaler, _face_pca
    if _face_model is None:
        try:
            with open(_FACE_MODEL,  "rb") as f: _face_model  = pickle.load(f)
            with open(_FACE_SCALER, "rb") as f: _face_scaler = pickle.load(f)
            with open(_FACE_PCA,    "rb") as f: _face_pca    = pickle.load(f)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Facial model files not found in {_MODEL_DIR}.\n"
                "Expected: face_best_model.pkl, face_scaler.pkl, face_pca.pkl"
            ) from exc
    return _face_model, _face_scaler, _face_pca


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        try:
            from tensorflow.keras.applications import MobileNetV2
            base = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=(_TARGET_SIZE, _TARGET_SIZE, 3),
                pooling="avg",
            )
            base.trainable = False
            _feature_extractor = base
        except ImportError as exc:
            raise RuntimeError(
                "tensorflow is required for facial confidence model.\n"
                "Run: pip install tensorflow"
            ) from exc
    return _feature_extractor


# ── Face preprocessing (mirrors training) ─────────────────────────────────────
def _preprocess_face(gray_crop: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_48  = cv2.resize(gray_crop.astype(np.uint8), (_IMG_SIZE, _IMG_SIZE))
    img_96  = cv2.resize(img_48, (_TARGET_SIZE, _TARGET_SIZE),
                         interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_96, cv2.COLOR_GRAY2RGB)
    return preprocess_input(img_rgb.astype(np.float32))


def _predict_face_confidence(gray_crop: np.ndarray) -> dict:
    model, scaler, pca = _get_face_models()
    extractor = _get_feature_extractor()

    processed = _preprocess_face(gray_crop)
    feat      = extractor.predict(processed[np.newaxis], verbose=0)   # (1,1280)
    feat_sc   = scaler.transform(feat)
    feat_pca  = pca.transform(feat_sc)
    probs     = model.predict_proba(feat_pca)[0]
    pred_id   = int(probs.argmax())

    return {
        "prediction":     _FACE_LABEL_NAMES[pred_id],
        "confident_prob": float(probs[1]) if len(probs) > 1 else 0.0,
    }


# ── Facial confidence score (per FINAL SCORE STRUCTURE doc) ──────────────────
def calculate_facial_confidence_score(confident_prob_pct: float) -> int:
    """
    Score facial confidence out of 25.

    Spec:
      Mostly confident  (> 80%)  → 25
      Moderate          (60–80%) → 20
      Mixed             (40–60%) → 15
      Nervous/stressed  (20–40%) → 8
      Very low          (< 20%)  → 3
    """
    if confident_prob_pct > 80:  return 25
    if confident_prob_pct > 60:  return 20
    if confident_prob_pct > 40:  return 15
    if confident_prob_pct > 20:  return 8
    return 3


# ── Main video analysis ───────────────────────────────────────────────────────
def analyze_video(video_path: str, sample_every: int = 10) -> dict:
    """
    Sample frames, run facial confidence model, compute all metrics.

    Returns
    -------
    {
        face_detection_rate    : float  (0-100 %),
        head_stability         : float  (0-100),
        eye_contact_score      : float  (0-100),
        facial_confidence      : str    ('confident' | 'not_confident'),
        facial_confidence_pct  : float  (0-100, avg confident probability),
        video_confidence_score : int    (0-25, per spec),
        analyzed_frames        : int,
        face_detected_frames   : int,
        mediapipe_available    : bool,
        model_used             : bool,
    }
    """
    try:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh
        MEDIAPIPE = True
    except Exception:
        MEDIAPIPE = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    analyzed       = 0
    face_detected  = 0
    nose_positions: list[tuple[float, float]] = []
    yaw_offsets:   list[float]               = []
    face_crops:    list[np.ndarray]          = []
    frame_idx      = 0

    face_mesh    = None
    face_cascade = None

    if MEDIAPIPE:
        face_mesh = _mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # Haar always available for crop extraction
    _haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue

            analyzed += 1

            if MEDIAPIPE and face_mesh is not None:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_detected += 1
                    lm = results.multi_face_landmarks[0].landmark

                    nose_positions.append((lm[_NOSE_TIP].x, lm[_NOSE_TIP].y))

                    left_x  = lm[_LEFT_EAR_OUT].x
                    right_x = lm[_RIGHT_EAR_OUT].x
                    face_w  = abs(right_x - left_x)
                    if face_w > 0:
                        yaw_offsets.append(
                            abs(lm[_NOSE_TIP].x - (left_x + right_x) / 2.0) / face_w
                        )

                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = _haar.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
                    if len(faces) > 0:
                        x, y_c, w, h = faces[0]
                        face_crops.append(gray[y_c:y_c + h, x:x + w])
            else:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
                )
                if len(faces) > 0:
                    face_detected += 1
                    x, y_c, w, h = faces[0]
                    face_crops.append(gray[y_c:y_c + h, x:x + w])

    finally:
        cap.release()
        if face_mesh is not None:
            face_mesh.close()

    if analyzed == 0:
        return _empty_result(MEDIAPIPE)

    face_rate = face_detected / analyzed

    # ── Head stability ────────────────────────────────────────────────────────
    if len(nose_positions) > 2:
        pts    = np.array(nose_positions)
        deltas = np.diff(pts, axis=0)
        avg_mv = float(np.mean(np.linalg.norm(deltas, axis=1)))
        stability = max(0.0, min(100.0, (1.0 - avg_mv / 0.05) * 100.0))
    else:
        stability = 60.0

    # ── Eye contact ───────────────────────────────────────────────────────────
    if yaw_offsets:
        avg_yaw     = float(np.mean(yaw_offsets))
        eye_contact = max(0.0, min(100.0, (1.0 - avg_yaw / 0.40) * 100.0))
    else:
        eye_contact = 50.0

    # ── Facial confidence ML model ────────────────────────────────────────────
    facial_confidence     = "not_confident"
    facial_confidence_pct = 0.0
    model_used            = False

    if face_crops:
        try:
            step       = max(1, len(face_crops) // 10)
            sampled    = face_crops[::step][:10]
            conf_probs = []

            for crop in sampled:
                result = _predict_face_confidence(crop)
                conf_probs.append(result["confident_prob"])

            avg_conf_prob     = float(np.mean(conf_probs))
            facial_confidence = "confident" if avg_conf_prob >= 0.5 else "not_confident"
            facial_confidence_pct = round(avg_conf_prob * 100, 1)
            model_used        = True
        except Exception:
            # Geometric fallback
            geo_score             = (stability + eye_contact) / 2.0
            facial_confidence     = "confident" if geo_score >= 55 else "not_confident"
            facial_confidence_pct = round(geo_score, 1)

    # ── Score per spec (0–25) ─────────────────────────────────────────────────
    video_confidence_score = calculate_facial_confidence_score(facial_confidence_pct)

    return {
        "face_detection_rate":    round(face_rate * 100, 1),
        "head_stability":         round(stability, 1),
        "eye_contact_score":      round(eye_contact, 1),
        "facial_confidence":      facial_confidence,
        "facial_confidence_pct":  facial_confidence_pct,
        "video_confidence_score": video_confidence_score,
        "analyzed_frames":        analyzed,
        "face_detected_frames":   face_detected,
        "mediapipe_available":    MEDIAPIPE,
        "model_used":             model_used,
    }


def _empty_result(mediapipe: bool) -> dict:
    return {
        "face_detection_rate":    0.0,
        "head_stability":         0.0,
        "eye_contact_score":      0.0,
        "facial_confidence":      "not_confident",
        "facial_confidence_pct":  0.0,
        "video_confidence_score": 3,
        "analyzed_frames":        0,
        "face_detected_frames":   0,
        "mediapipe_available":    mediapipe,
        "model_used":             False,
    }
