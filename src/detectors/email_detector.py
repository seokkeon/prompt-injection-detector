import email
import os
from dataclasses import dataclass, field
from email import policy
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
from src.detectors.unified_detector import UnifiedDetector as TextInjectionDetector  # merged
from src.detectors.image_detector import ImageInjectionDetector
from src.utils.helpers import risk_score_to_level, save_temp_file, cleanup_temp_file
from src.utils.logger import logger


@dataclass
class EmailDetectionResult:
    is_suspicious: bool
    risk_level: str
    risk_score: float
    subject: str = ""
    sender: str = ""
    text_analyses: List[Dict] = field(default_factory=list)
    image_analyses: List[Dict] = field(default_factory=list)
    attachment_count: int = 0
    flagged_parts: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "is_suspicious": self.is_suspicious, "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3), "subject": self.subject,
            "sender": self.sender, "text_analyses": self.text_analyses,
            "image_analyses": self.image_analyses, "attachment_count": self.attachment_count,
            "flagged_parts": self.flagged_parts, "explanation": self.explanation,
        }


class EmailInjectionDetector:
    IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}
    IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    def __init__(self):
        self.text_det  = TextInjectionDetector()
        self.image_det = ImageInjectionDetector()

    def analyze(self, raw_email: str) -> EmailDetectionResult:
        msg = email.message_from_string(raw_email, policy=policy.default)
        return self._run(msg)

    def analyze_file(self, filepath: str) -> EmailDetectionResult:
        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        return self._run(msg)

    def _run(self, msg) -> EmailDetectionResult:
        flagged, text_results, image_results = [], [], []
        attachment_count = 0

        for h in ["subject", "from", "reply-to"]:
            val = str(msg.get(h, ""))
            if val:
                r = self.text_det.detect(val)
                if r.is_suspicious:
                    flagged.append(f"header:{h}")
                text_results.append({"source": f"header:{h}", **r.to_dict()})

        tmp_files = []
        try:
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    r = self.text_det.detect(part.get_content())
                    if r.is_suspicious: flagged.append("text/plain body")
                    text_results.append({"source": "text/plain", **r.to_dict()})
                elif ct == "text/html":
                    soup = BeautifulSoup(part.get_content(), "lxml")
                    for tag in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "")):
                        tag.decompose()
                    r = self.text_det.detect(soup.get_text(separator="\n"))
                    if r.is_suspicious: flagged.append("text/html body")
                    text_results.append({"source": "text/html", **r.to_dict()})
                elif ct in self.IMAGE_TYPES or os.path.splitext(part.get_filename() or "")[1].lower() in self.IMAGE_EXTS:
                    attachment_count += 1
                    payload = part.get_payload(decode=True)
                    if payload:
                        ext = os.path.splitext(part.get_filename() or "")[1] or ".png"
                        tmp = save_temp_file(payload, suffix=ext)
                        tmp_files.append(tmp)
                        ir = self.image_det.analyze(tmp)
                        if ir.is_suspicious: flagged.append(f"image:{part.get_filename() or 'inline'}")
                        image_results.append({"source": f"image:{part.get_filename() or 'inline'}", **ir.to_dict()})
        finally:
            for t in tmp_files: cleanup_temp_file(t)

        all_scores = [r.get("risk_score", 0.0) for r in text_results] + \
                     [r.get("risk_score", 0.0) * 0.85 for r in image_results]
        score = min(max(all_scores) * 0.7 + (sum(all_scores) / len(all_scores)) * 0.3, 1.0) if all_scores else 0.0
        level = risk_score_to_level(score)
        explanation = (
            f"Injection indicators in: {', '.join(flagged)}. Score: {score:.2f}. "
            f"Analyzed {len(text_results)} text part(s) and {len(image_results)} image(s)."
            if flagged else
            f"No indicators found. Analyzed {len(text_results)} text part(s) and {len(image_results)} image(s)."
        )
        return EmailDetectionResult(
            is_suspicious=score >= 0.35, risk_level=level, risk_score=score,
            subject=str(msg.get("subject", "")), sender=str(msg.get("from", "")),
            text_analyses=text_results, image_analyses=image_results,
            attachment_count=attachment_count, flagged_parts=flagged, explanation=explanation,
        )
