"""
audio_processor.py
Extracts audio track from a video file and returns the path + duration.
Requires ffmpeg to be installed and in PATH.
"""
from __future__ import annotations

import os
import tempfile


def extract_audio(video_path: str) -> tuple[str, float]:
    """
    Extract audio from *video_path* and write it as a 64-kbps MP3.

    Returns
    -------
    (audio_path, duration_seconds)
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
                "FFmpeg not found. Install it from https://ffmpeg.org/download.html "
                "and add it to your system PATH."
            ) from exc
        raise RuntimeError(f"Could not open video file: {exc}") from exc

    if video.audio is None:
        video.close()
        raise ValueError(
            "The video has no audio track. Please upload a video that contains audio."
        )

    duration: float = video.duration

    # Write to a temp MP3 at low bitrate to stay under Groq's 25 MB limit
    audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(audio_fd)

    try:
        video.audio.write_audiofile(
            audio_path,
            bitrate="64k",
            verbose=False,
            logger=None,
        )
    finally:
        video.close()

    return audio_path, duration


def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)
