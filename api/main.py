"""
FastAPI — Prompt Injection Detector REST API v3.

Endpoints:
  GET  /health                   — status
  POST /analyze/text             — text (rules + ML + explainability)
  POST /analyze/text/batch       — batch text
  POST /analyze/image            — image (OCR, stego, QR, EXIF, adversarial)
  POST /analyze/email            — .eml file
  POST /analyze/indirect/url     — scan URL for indirect injection
  POST /analyze/indirect/html    — scan raw HTML
  POST /analyze/indirect/text    — scan document text
  POST /analyze/conversation     — multi-turn conversation analysis
  GET  /stats                    — runtime counters
"""

import asyncio
import os, sys, time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.detectors.unified_detector import UnifiedDetector
from src.detectors.image_detector import ImageInjectionDetector
from src.detectors.email_detector import EmailInjectionDetector
from src.utils.helpers import save_temp_file, cleanup_temp_file
from src.utils.logger import logger

app = FastAPI(title="Prompt Injection Detector", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained")
_mp = MODEL_PATH if os.path.exists(MODEL_PATH) else None

detector       = UnifiedDetector(model_path=_mp)   # one object handles text + indirect + conversation + explain
image_detector = ImageInjectionDetector()
email_detector = EmailInjectionDetector()

_s = {"text":0,"image":0,"email":0,"indirect":0,"conv":0,"flagged":0,"score_sum":0.0,"t0":time.time()}

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    explain: bool = False

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)

class URLRequest(BaseModel):
    url: str

class HTMLRequest(BaseModel):
    html: str
    source: str = "inline"

class DocumentRequest(BaseModel):
    text: str
    source: str = "document"

class ConversationRequest(BaseModel):
    messages: List[Dict]


@app.get("/health")
async def health():
    return {"status":"ok","version":"3.0.0","ml_enabled":detector.ml_available,
            "uptime_s":round(time.time()-_s["t0"],1)}


@app.post("/analyze/text")
async def analyze_text(req: TextRequest):
    r = detector.detect(req.text)
    d = r.to_dict()
    if req.explain:
        d["explainability"] = detector.explain(req.text, r.risk_score).to_dict()
    _s["text"]+=1; _s["score_sum"]+=r.risk_score
    if r.is_suspicious: _s["flagged"]+=1
    return d


@app.post("/analyze/text/batch")
async def analyze_batch(req: BatchRequest):
    # Run all detections in parallel using thread pool (CPU-bound work)
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, detector.detect, t)
        for t in req.texts
    ]
    detections = await asyncio.gather(*tasks)
    results = []
    for r in detections:
        results.append(r.to_dict())
        _s["text"]+=1; _s["score_sum"]+=r.risk_score
        if r.is_suspicious: _s["flagged"]+=1
    return {"count":len(results),"flagged":sum(1 for r in results if r["is_suspicious"]),"results":results}


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    ALLOWED = {"image/jpeg","image/png","image/gif","image/webp","image/bmp"}
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")
    content = await file.read()
    tmp = save_temp_file(content, suffix=os.path.splitext(file.filename or "")[1] or ".png")
    try:
        r = image_detector.analyze(tmp)
        _s["image"]+=1; _s["score_sum"]+=r.risk_score
        if r.is_suspicious: _s["flagged"]+=1
        return r.to_dict()
    finally:
        cleanup_temp_file(tmp)


@app.post("/analyze/email")
async def analyze_email(file: UploadFile = File(...)):
    content = await file.read()
    try: raw = content.decode("utf-8", errors="replace")
    except Exception as e: raise HTTPException(400, f"Could not decode: {e}")
    r = email_detector.analyze(raw)
    _s["email"]+=1; _s["score_sum"]+=r.risk_score
    if r.is_suspicious: _s["flagged"]+=1
    return r.to_dict()


@app.post("/analyze/indirect/url")
async def analyze_url(req: URLRequest):
    # Uses async httpx — does not block the FastAPI worker thread
    r = await detector.detect_indirect_url_async(req.url)
    _s["indirect"]+=1; _s["score_sum"]+=r.risk_score
    if r.is_suspicious: _s["flagged"]+=1
    return r.to_dict()


@app.post("/analyze/indirect/html")
async def analyze_html(req: HTMLRequest):
    r = detector.detect_indirect_html(req.html, source=req.source)
    _s["indirect"]+=1; _s["score_sum"]+=r.risk_score
    if r.is_suspicious: _s["flagged"]+=1
    return r.to_dict()


@app.post("/analyze/indirect/text")
async def analyze_doc(req: DocumentRequest):
    r = detector.detect_indirect_text(req.text, source=req.source)
    _s["indirect"]+=1; _s["score_sum"]+=r.risk_score
    if r.is_suspicious: _s["flagged"]+=1
    return r.to_dict()


@app.post("/analyze/conversation")
async def analyze_conv(req: ConversationRequest):
    r = detector.detect_conversation(req.messages)
    _s["conv"]+=1; _s["score_sum"]+=r.overall_risk_score
    if r.is_suspicious: _s["flagged"]+=1
    return r.to_dict()


@app.get("/stats")
async def stats():
    total = sum(_s[k] for k in ["text","image","email","indirect","conv"])
    return {"total_requests":total,"requests_text":_s["text"],"requests_image":_s["image"],
            "requests_email":_s["email"],"requests_indirect":_s["indirect"],"requests_conversation":_s["conv"],
            "flagged_total":_s["flagged"],"flag_rate":round(_s["flagged"]/total,4) if total else 0,
            "avg_risk_score":round(_s["score_sum"]/total,4) if total else 0,
            "uptime_s":round(time.time()-_s["t0"],1),"ml_enabled":detector.ml_available}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=os.getenv("API_HOST","0.0.0.0"), port=int(os.getenv("API_PORT","8000")), reload=True)
