import os
import hashlib
import tempfile


def save_temp_file(content: bytes, suffix: str = "") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


def cleanup_temp_file(filepath: str) -> None:
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass


def risk_score_to_level(score: float) -> str:
    if score >= 0.85:
        return "critical"
    elif score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"


def truncate_text(text: str, max_length: int = 200) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [{len(text) - max_length} chars truncated]"
