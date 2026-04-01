"""
video_analyzer.py
Analyzes candidate video for confidence signals:
  - Face detection rate (% frames face visible)
  - Head stability (nose-position variance over time)
  - Eye contact estimate (yaw offset from camera centre)

Primary engine  : MediaPipe Face Mesh
Fallback engine : OpenCV Haar Cascade (face detection only)
"""
from __future__ import annotations

import cv2
import numpy as np

# MediaPipe landmark indices used for pose estimation
_NOSE_TIP       = 4
_LEFT_EAR_OUT   = 234
_RIGHT_EAR_OUT  = 454


def analyze_video(video_path: str, sample_every: int = 10) -> dict:
    """
    Sample *sample_every* frames from the video and extract confidence metrics.

    Returns
    -------
    {
        face_detection_rate   : float  (0-100 %),
        head_stability        : float  (0-100),
        eye_contact_score     : float  (0-100),
        video_confidence_score: int    (0-25),
        analyzed_frames       : int,
        face_detected_frames  : int,
        mediapipe_available   : bool,
    }
    """
    # Try importing MediaPipe; fall back gracefully if missing
    try:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh
        MEDIAPIPE = True
    except Exception:
        MEDIAPIPE = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    analyzed        = 0
    face_detected   = 0
    nose_positions: list[tuple[float, float]] = []
    yaw_offsets:    list[float]               = []
    frame_idx       = 0

    # ── Initialise detector ───────────────────────────────────────────────────
    face_mesh   = None
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

    # ── Frame loop ────────────────────────────────────────────────────────────
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
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_detected += 1
                    lm = results.multi_face_landmarks[0].landmark

                    # Nose position for stability
                    nose_positions.append((lm[_NOSE_TIP].x, lm[_NOSE_TIP].y))

                    # Yaw estimate: offset of nose tip from horizontal face centre
                    left_x  = lm[_LEFT_EAR_OUT].x
                    right_x = lm[_RIGHT_EAR_OUT].x
                    face_w  = abs(right_x - left_x)
                    if face_w > 0:
                        yaw_offsets.append(
                            abs(lm[_NOSE_TIP].x - (left_x + right_x) / 2.0) / face_w
                        )

            else:
                # Haar cascade fallback (detection only — no pose)
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
                )
                if len(faces) > 0:
                    face_detected += 1

    finally:
        cap.release()
        if face_mesh is not None:
            face_mesh.close()

    # ── Compute metrics ───────────────────────────────────────────────────────
    if analyzed == 0:
        return _empty_result(MEDIAPIPE)

    face_rate = face_detected / analyzed  # 0.0–1.0

    # Head stability: lower mean movement → more stable
    if len(nose_positions) > 2:
        pts    = np.array(nose_positions)
        deltas = np.diff(pts, axis=0)
        avg_mv = float(np.mean(np.linalg.norm(deltas, axis=1)))
        # Threshold: movement ≥ 0.05 (in normalised coords) → score 0
        stability = max(0.0, min(100.0, (1.0 - avg_mv / 0.05) * 100.0))
    else:
        stability = 60.0  # Unknown / insufficient data → neutral

    # Eye contact: small yaw offset means looking toward camera
    if yaw_offsets:
        avg_yaw    = float(np.mean(yaw_offsets))
        # yaw ≤ 0.10 → great; yaw ≥ 0.40 → looking away
        eye_contact = max(0.0, min(100.0, (1.0 - avg_yaw / 0.40) * 100.0))
    else:
        eye_contact = 50.0  # Unknown → neutral

    # Composite video score (0–25)
    face_pts      = face_rate  * 10.0           # 0–10
    stability_pts = (stability  / 100.0) * 8.0  # 0–8
    eye_pts       = (eye_contact / 100.0) * 7.0 # 0–7
    video_score   = min(25, round(face_pts + stability_pts + eye_pts))

    return {
        "face_detection_rate":    round(face_rate * 100, 1),
        "head_stability":         round(stability, 1),
        "eye_contact_score":      round(eye_contact, 1),
        "video_confidence_score": video_score,
        "analyzed_frames":        analyzed,
        "face_detected_frames":   face_detected,
        "mediapipe_available":    MEDIAPIPE,
    }


def _empty_result(mediapipe: bool) -> dict:
    return {
        "face_detection_rate":    0.0,
        "head_stability":         0.0,
        "eye_contact_score":      0.0,
        "video_confidence_score": 0,
        "analyzed_frames":        0,
        "face_detected_frames":   0,
        "mediapipe_available":    mediapipe,
    }
