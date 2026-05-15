"""
Indirect Prompt Injection Detector.

Detects injections hidden inside content the AI is asked to READ,
not in the user's prompt directly. Attack vectors:
  - Web pages with hidden instructions
  - PDF documents with embedded commands
  - Text files / markdown docs
  - Tool / API outputs
"""

import os
import sys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.detectors.text_detector import TextInjectionDetector
from src.utils.helpers import risk_score_to_level
from src.utils.logger import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup, Comment
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

INDIRECT_PATTERNS = [
    (r"note\s+to\s+(ai|assistant|llm|model|gpt|claude)\s*:", "disguised_instruction"),
    (r"(ai|assistant|llm|model)\s*:\s*(ignore|disregard|forget|override)", "disguised_instruction"),
    (r"\[.*?(instruction|command|prompt|system).*?\]", "bracketed_command"),
    (r"style\s*=\s*[\"'].*?display\s*:\s*none.*?[\"']", "hidden_html"),
    (r"style\s*=\s*[\"'].*?visibility\s*:\s*hidden.*?[\"']", "hidden_html"),
    (r"style\s*=\s*[\"'].*?color\s*:\s*white.*?[\"']", "hidden_html"),
    (r"style\s*=\s*[\"'].*?font-size\s*:\s*0.*?[\"']", "hidden_html"),
    (r"<!--.*?(ignore|instruction|system|override).*?-->", "html_comment_injection"),
    (r"the\s+(following|above|below)\s+(text|content|document)\s+is\s+(for\s+)?(the\s+)?(ai|assistant|llm)", "ai_targeting"),
    (r"when\s+(an?\s+)?(ai|llm|assistant|model)\s+(reads?|processes?|sees?)", "ai_targeting"),
    (r"if\s+you\s+are\s+(an?\s+)?(ai|llm|language\s+model)", "ai_targeting"),
    (r"summary\s*:\s*ignore\s+all", "data_injection"),
    (r"title\s*:\s*ignore\s+(all\s+)?previous", "data_injection"),
    (r"description\s*:\s*ignore\s+(all\s+)?previous", "data_injection"),
]


@dataclass
class IndirectInjectionResult:
    is_suspicious: bool
    risk_level: str
    risk_score: float
    source_type: str
    source: str
    direct_analysis: Optional[Dict] = None
    indirect_patterns: List[Dict] = field(default_factory=list)
    hidden_content: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "is_suspicious":      self.is_suspicious,
            "risk_level":         self.risk_level,
            "risk_score":         round(self.risk_score, 3),
            "source_type":        self.source_type,
            "source":             self.source[:200],
            "direct_analysis":    self.direct_analysis,
            "indirect_patterns":  self.indirect_patterns,
            "hidden_content":     self.hidden_content[:5],
            "explanation":        self.explanation,
        }


class IndirectInjectionDetector:
    def __init__(self):
        self.text_detector = TextInjectionDetector()
        self._patterns = [
            (re.compile(p, re.IGNORECASE | re.DOTALL), cat)
            for p, cat in INDIRECT_PATTERNS
        ]

    def _scan_indirect(self, text: str) -> List[Dict]:
        matches = []
        for pattern, category in self._patterns:
            found = pattern.findall(text)
            if found:
                matches.append({"pattern": pattern.pattern[:80], "category": category, "count": len(found)})
        return matches

    def _extract_hidden_html(self, html: str) -> List[str]:
        if not BS4_AVAILABLE:
            return []
        hidden = []
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(style=True):
            style = tag.get("style", "").replace(" ", "").lower()
            if any(t in style for t in ["display:none","visibility:hidden","opacity:0","font-size:0","color:white","color:#fff","color:#ffffff"]):
                t = tag.get_text().strip()
                if t:
                    hidden.append(t)
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            c = str(comment).strip()
            if c:
                hidden.append(f"[HTML comment]: {c}")
        return hidden

    def analyze_url(self, url: str, timeout: int = 10) -> IndirectInjectionResult:
        if not REQUESTS_AVAILABLE:
            return IndirectInjectionResult(False, "low", 0.0, "url", url, explanation="requests not installed.")
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            return self._analyze_html(resp.text, source=url, source_type="url")
        except Exception as e:
            return IndirectInjectionResult(False, "low", 0.0, "url", url, explanation=f"Fetch failed: {e}")

    def _analyze_html(self, html: str, source: str, source_type: str) -> IndirectInjectionResult:
        soup = BeautifulSoup(html, "lxml") if BS4_AVAILABLE else None
        if soup:
            for tag in soup(["script","style","noscript"]): tag.decompose()
            visible_text = soup.get_text(separator="\n").strip()
        else:
            visible_text = html
        hidden_content = self._extract_hidden_html(html)
        all_text  = visible_text + "\n" + "\n".join(hidden_content)
        direct    = self.text_detector.detect(all_text)
        indirect  = self._scan_indirect(html)
        hidden_score = min(len(hidden_content) * 0.25, 0.8) if hidden_content else 0.0
        score = max(direct.risk_score, hidden_score)
        if indirect:
            score = min(score + 0.15 * len(set(m["category"] for m in indirect)), 1.0)
        parts = []
        if direct.is_suspicious: parts.append("Injection patterns in page text.")
        if hidden_content:       parts.append(f"{len(hidden_content)} hidden element(s) found.")
        if indirect:             parts.append(f"Indirect patterns: {', '.join(set(m['category'] for m in indirect))}.")
        return IndirectInjectionResult(
            is_suspicious=score>=0.35, risk_level=risk_score_to_level(score), risk_score=score,
            source_type=source_type, source=source, direct_analysis=direct.to_dict(),
            indirect_patterns=indirect, hidden_content=hidden_content,
            explanation=" ".join(parts) if parts else "No indirect injection found.",
        )

    def analyze_pdf(self, pdf_path: str) -> IndirectInjectionResult:
        if not PYPDF_AVAILABLE:
            return IndirectInjectionResult(False, "low", 0.0, "pdf", pdf_path, explanation="pypdf not installed. Run: pip install pypdf")
        try:
            reader = pypdf.PdfReader(pdf_path)
            pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
            meta_texts = [f"{k}: {v}" for k, v in (reader.metadata or {}).items() if v]
            all_text = "\n".join(pages + meta_texts)
            direct   = self.text_detector.detect(all_text)
            indirect = self._scan_indirect(all_text)
            score    = direct.risk_score
            if indirect: score = min(score + 0.1 * len(indirect), 1.0)
            parts = []
            if direct.is_suspicious: parts.append("Injection in PDF text.")
            if indirect:             parts.append(f"Indirect patterns: {len(indirect)}.")
            return IndirectInjectionResult(
                is_suspicious=score>=0.35, risk_level=risk_score_to_level(score), risk_score=score,
                source_type="pdf", source=pdf_path, direct_analysis=direct.to_dict(),
                indirect_patterns=indirect, hidden_content=meta_texts,
                explanation=" ".join(parts) if parts else "No injection found in PDF.",
            )
        except Exception as e:
            return IndirectInjectionResult(False, "low", 0.0, "pdf", pdf_path, explanation=f"Error: {e}")

    def analyze_text(self, text: str, source: str = "raw_text") -> IndirectInjectionResult:
        direct   = self.text_detector.detect(text)
        indirect = self._scan_indirect(text)
        score    = direct.risk_score
        if indirect: score = min(score + 0.1 * len(set(m["category"] for m in indirect)), 1.0)
        parts = []
        if direct.is_suspicious: parts.append(direct.explanation)
        if indirect:             parts.append(f"Indirect patterns: {', '.join(set(m['category'] for m in indirect))}.")
        return IndirectInjectionResult(
            is_suspicious=score>=0.35, risk_level=risk_score_to_level(score), risk_score=score,
            source_type="text", source=source, direct_analysis=direct.to_dict(),
            indirect_patterns=indirect, hidden_content=[],
            explanation=" ".join(parts) if parts else "No injection found.",
        )

    def analyze_file(self, file_path: str) -> IndirectInjectionResult:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.analyze_pdf(file_path)
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return IndirectInjectionResult(False, "low", 0.0, "file", file_path, explanation=f"Cannot read: {e}")
        if ext in (".html", ".htm"):
            return self._analyze_html(content, file_path, "html_file")
        return self.analyze_text(content, source=file_path)
