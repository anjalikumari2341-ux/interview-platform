"""
config.py
Shared settings for the analysis modules.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _find_model_dir() -> Path:
    """
    The README and the original code expect a folder called 'models/', but the
    GitHub repository uses 'model/'. Accept either name so the app works on a
    local machine and on Streamlit Cloud without renaming anything.
    """
    for name in ("models", "model"):
        candidate = _ROOT / name
        if candidate.is_dir():
            return candidate
    return _ROOT / "models"


MODEL_DIR: Path = _find_model_dir()
