import os
import hashlib
import tempfile
from typing import Union
from pathlib import Path


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_temp_file(content: bytes, suffix: str = "") -> str:
    """Save bytes content to a temporary file. Returns file path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def cleanup_temp_file(filepath: str) -> None:
    """Delete a temporary file safely."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass


def risk_score_to_level(score: float) -> str:
    """Convert a numeric risk score (0.0–1.0) to a human-readable level."""
    if score >= 0.85:
        return "critical"
    elif score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate long text for logging."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [{len(text) - max_length} chars truncated]"
