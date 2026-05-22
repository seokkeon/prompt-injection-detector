"""
FastAPI — Prompt Injection Detector REST API v3.

Endpoints:
  GET  /health                   — status
  POST /analyze/text             — text (rules + ML + explainability)
  POST /analyze/text/batch       — batch text (parallel)
  POST /analyze/image            — image (OCR, stego, QR, EXIF, adversarial)
  POST /analyze/email            — .eml file
  POST /analyze/indirect/url     — scan URL (SSRF-protected, async)
  POST /analyze/indirect/html    — scan raw HTML
  POST /analyze/indirect/text    — scan document text
  POST /analyze/conversation     — multi-turn conversation analysis
  GET  /stats                    — runtime counters
"""

import asyncio
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from ipaddress import ip_address, ip_network

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# No os.chdir — use explicit paths instead

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, AnyHttpUrl

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING = True
except ImportError:
    RATE_LIMITING = False

from src.detectors.unified_detector import UnifiedDetector
from src.detectors.image_detector import ImageInjectionDetector
from src.detectors.email_detector import EmailInjectionDetector
from src.utils.helpers import save_temp_file, cleanup_temp_file
from src.utils.logger import logger

# ── Startup config validation ─────────────────────────────────────────────────

def _validate_config() -> None:
    """Warn about missing or misconfigured environment on startup."""
    model_path = os.getenv("MODEL_PATH", "models/trained")
    if not Path(model_path).exists():
        logger.warning(
            f"MODEL_PATH={model_path!r} does not exist. "
            "Running in rule-only mode. Run train.py to generate a model."
        )
    if not os.getenv("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN not set. HuggingFace downloads will be unauthenticated (slower). "
            "Add HF_TOKEN to your .env file."
        )
    cors_origin = os.getenv("CORS_ORIGIN", "*")
    if cors_origin == "*":
        logger.warning(
            "CORS_ORIGIN=* allows any website to call this API. "
            "Set CORS_ORIGIN=http://localhost:3000 in production."
        )

_validate_config()

# ── App setup ─────────────────────────────────────────────────────────────────

CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")

app = FastAPI(title="Prompt Injection Detector", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN] if CORS_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if RATE_LIMITING:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled (slowapi)")
else:
    logger.warning("slowapi not installed — rate limiting disabled. Run: pip install slowapi")

# ── Detectors ─────────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained")
_mp = MODEL_PATH if Path(MODEL_PATH).exists() else None

detector       = UnifiedDetector(model_path=_mp)
image_detector = ImageInjectionDetector()
email_detector = EmailInjectionDetector()

# ── Thread-safe stats ─────────────────────────────────────────────────────────
# Plain dict increment is a race condition under concurrent requests.
# Use a lock to ensure accurate counts.

_stats_lock = threading.Lock()
_s = {
    "text": 0, "image": 0, "email": 0,
    "indirect": 0, "conv": 0,
    "flagged": 0, "score_sum": 0.0,
    "t0": time.time(),
}

def _record(key: str, score: float, flagged: bool) -> None:
    with _stats_lock:
        _s[key]        += 1
        _s["score_sum"] += score
        if flagged:
            _s["flagged"] += 1

# ── SSRF protection ───────────────────────────────────────────────────────────

_PRIVATE_NETS = [
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
    ip_network("::1/128"),
    ip_network("fc00::/7"),
]

def _is_private_ip(host: str) -> bool:
    try:
        addr = ip_address(host)
        return any(addr in net for net in _PRIVATE_NETS)
    except ValueError:
        return False  # hostname, not raw IP — resolve check skipped

def _assert_safe_url(url: str) -> None:
    """Raise HTTP 400 if URL targets a private/internal address (SSRF protection)."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host   = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        raise HTTPException(400, "URL targets localhost — blocked for security.")
    if _is_private_ip(host):
        raise HTTPException(400, f"URL targets a private IP ({host}) — blocked for security.")

# ── Request models ────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text:    str  = Field(..., min_length=1, max_length=50_000)
    explain: bool = False

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)

class URLRequest(BaseModel):
    url: str = Field(..., min_length=1)

    @field_validator("url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

class HTMLRequest(BaseModel):
    html:   str = Field(..., min_length=1)
    source: str = "inline"

class DocumentRequest(BaseModel):
    text:   str = Field(..., min_length=1)
    source: str = "document"

class ConversationRequest(BaseModel):
    messages: List[Dict]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok", "version": "3.0.0",
        "ml_enabled": detector.ml_available,
        "rate_limiting": RATE_LIMITING,
        "uptime_s": round(time.time() - _s["t0"], 1),
    }


@app.post("/analyze/text")
async def analyze_text(req: TextRequest, request: Request):
    if RATE_LIMITING:
        await limiter._check_request(request, "200/minute")
    r = detector.detect(req.text)
    d = r.to_dict()
    if req.explain:
        d["explainability"] = detector.explain(req.text, r.risk_score).to_dict()
    _record("text", r.risk_score, r.is_suspicious)
    return d


@app.post("/analyze/text/batch")
async def analyze_batch(req: BatchRequest, request: Request):
    if RATE_LIMITING:
        await limiter._check_request(request, "20/minute")
    loop   = asyncio.get_event_loop()
    tasks  = [loop.run_in_executor(None, detector.detect, t) for t in req.texts]
    detections = await asyncio.gather(*tasks)
    results = []
    for r in detections:
        results.append(r.to_dict())
        _record("text", r.risk_score, r.is_suspicious)
    return {
        "count":   len(results),
        "flagged": sum(1 for r in results if r["is_suspicious"]),
        "results": results,
    }


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    ALLOWED = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")
    content = await file.read()
    tmp = save_temp_file(content, suffix=os.path.splitext(file.filename or "")[1] or ".png")
    try:
        r = image_detector.analyze(tmp)
        _record("image", r.risk_score, r.is_suspicious)
        return r.to_dict()
    finally:
        cleanup_temp_file(tmp)


@app.post("/analyze/email")
async def analyze_email(file: UploadFile = File(...)):
    content = await file.read()
    try:
        raw = content.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(400, f"Could not decode: {e}")
    r = email_detector.analyze(raw)
    _record("email", r.risk_score, r.is_suspicious)
    return r.to_dict()


@app.post("/analyze/indirect/url")
async def analyze_url(req: URLRequest):
    _assert_safe_url(req.url)  # SSRF protection
    r = await detector.detect_indirect_url_async(req.url)
    _record("indirect", r.risk_score, r.is_suspicious)
    return r.to_dict()


@app.post("/analyze/indirect/html")
async def analyze_html(req: HTMLRequest):
    r = detector.detect_indirect_html(req.html, source=req.source)
    _record("indirect", r.risk_score, r.is_suspicious)
    return r.to_dict()


@app.post("/analyze/indirect/text")
async def analyze_doc(req: DocumentRequest):
    r = detector.detect_indirect_text(req.text, source=req.source)
    _record("indirect", r.risk_score, r.is_suspicious)
    return r.to_dict()


@app.post("/analyze/conversation")
async def analyze_conv(req: ConversationRequest):
    r = detector.detect_conversation(req.messages)
    _record("conv", r.overall_risk_score, r.is_suspicious)
    return r.to_dict()


@app.get("/stats")
async def stats():
    with _stats_lock:
        s = dict(_s)
    total = sum(s[k] for k in ["text", "image", "email", "indirect", "conv"])
    return {
        "total_requests":       total,
        "requests_text":        s["text"],
        "requests_image":       s["image"],
        "requests_email":       s["email"],
        "requests_indirect":    s["indirect"],
        "requests_conversation": s["conv"],
        "flagged_total":        s["flagged"],
        "flag_rate":            round(s["flagged"] / total, 4) if total else 0,
        "avg_risk_score":       round(s["score_sum"] / total, 4) if total else 0,
        "uptime_s":             round(time.time() - s["t0"], 1),
        "ml_enabled":           detector.ml_available,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
