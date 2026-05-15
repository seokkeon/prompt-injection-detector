"""
Unified Prompt Injection Detector.

This is the single entry point for ALL text-based detection:
  - Injection patterns (rules)        — merged from text_detector.py
  - ML classifier                     — fine-tuned transformer
  - Indirect injection                — URL / HTML / document scanning
  - Conversation analysis             — multi-turn gradual jailbreak detection
  - Explainability                    — attention rollout + keyword highlighting

Import: from src.detectors.unified_detector import UnifiedDetector
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.utils.helpers import risk_score_to_level
from src.utils.logger import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Injection patterns (was: text_detector.py)
# ═══════════════════════════════════════════════════════════════════════════════

INJECTION_PATTERNS = [
    # Direct override
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",                        "direct_override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",                     "direct_override"),
    (r"forget\s+(everything|all)(\s+\S+){0,4}\s+(told|said|given)",                             "direct_override"),
    (r"forget\s+everything\s+(you|i|we)",                                                          "direct_override"),
    (r"override\s+(your\s+)?(previous\s+)?(instructions?|rules?|guidelines?)",                   "direct_override"),
    # System injection
    (r"(new\s+)?system\s*(prompt|instruction|message)\s*:",                                       "system_injection"),
    (r"\[SYSTEM\]",                                                                                "system_injection"),
    (r"<system>",                                                                                    "system_injection"),
    (r"###\s*(instruction|system|prompt)",                                                          "system_injection"),
    # Role hijack
    (r"you\s+are\s+now\s+(a\s+)?(new|different|another)\s+ai",                                "role_hijack"),
    (r"act\s+as\s+(if\s+)?(you\s+are\s+)?(an?\s+)?(unrestricted|unfiltered|evil|hacked)",   "role_hijack"),
    (r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(restrictions?|rules?|limits?)",       "role_hijack"),
    (r"jailbreak",                                                                                   "role_hijack"),
    (r"DAN\s+(mode|prompt)",                                                                        "role_hijack"),
    # Exfiltration
    (r"(send|forward|email|post|upload|exfiltrate)\s+.{0,60}\s+to\s+https?://",                 "exfiltration"),
    (r"(leak|reveal|expose|dump)\s+(all\s+)?(the\s+)?(data|user\s+data|private|confidential)", "exfiltration"),
    (r"(print|output|return|show)\s+(the\s+)?(\w+\s+)?(system\s+prompt|context|history|conversation)", "exfiltration"),
    (r"(send|forward|email|post)\s+(all\s+)?(user\s+)?(data|messages?|information|conversation\s+data)\s+to\s+https?://", "exfiltration"),
    # Encoding tricks
    (r"base64\s*:\s*[A-Za-z0-9+/=]{20,}",                                                        "encoding_trick"),
    (r"rot13|caesar\s+cipher",                                                                      "encoding_trick"),
    (r"translate\s+(this|the\s+following)\s+from\s+(base64|hex|binary)",                        "encoding_trick"),
    # Format injection
    (r"---+\s*(user|system|assistant|human)\s*---+",                                              "format_injection"),
    (r"<\|.*?\|>",                                                                                 "format_injection"),
    (r"\[INST\]|\[/INST\]",                                                                     "format_injection"),
    # Prompt end hijack
    (r"end\s+of\s+(system\s+prompt|instructions?)\s*[\.\n]",                                  "prompt_end_hijack"),
    (r"the\s+instructions?\s+(above|before|prior)\s+(are|were)\s+(fake|incorrect|wrong|outdated)", "prompt_end_hijack"),
]

CATEGORY_WEIGHTS = {
    "direct_override":   0.9,
    "system_injection":  0.85,
    "role_hijack":       0.8,
    "exfiltration":      1.0,
    "encoding_trick":    0.7,
    "format_injection":  0.75,
    "prompt_end_hijack": 0.8,
}

# Indirect injection patterns (was: indirect_detector.py)
INDIRECT_PATTERNS = [
    (r"(note\s+to\s+(ai|assistant|gpt|claude|llm)|ai\s+note\s*:)",                                      "ai_targeting"),
    (r"(if\s+you\s+are\s+an?\s+(ai|llm|language\s+model|assistant))",                                  "ai_targeting"),
    (r"\[ai\s*(instruction|note|command)\]",                                                              "ai_targeting"),
    (r"(when\s+summarizing|when\s+reading|when\s+processing)\s+this",                                   "context_hijack"),
    (r"(your\s+summary\s+must|your\s+response\s+must|you\s+must\s+(also|now|instead))",               "context_hijack"),
    (r"(include\s+(in\s+your\s+)?(summary|response).*?(password|key|token|secret))",                     "data_theft"),
    (r"(also\s+(send|forward|include|mention)\s+.{0,30}to\s+https?://)",                                 "data_theft"),
    (r"<\s*(div|span|p)[^>]*style\s*=\s*['\"].*?(display\s*:\s*none|visibility\s*:\s*hidden)",     "hidden_html"),
    (r"color\s*:\s*(white|#fff|#ffffff)",                                                                  "invisible_text"),
    (r"font-size\s*:\s*[01](px|pt|em|rem)",                                                               "invisible_text"),
]

# Conversation patterns (was: conversation_detector.py)
GRADUAL_JAILBREAK_PATTERNS = [
    r"(let('s|\s+us)\s+play\s+a\s+(game|role|scenario))",
    r"(hypothetically|theoretically|in\s+a\s+fictional\s+(world|scenario|story))",
    r"(for\s+(research|educational|academic)\s+purposes?)",
    r"(pretend\s+(you\s+are|to\s+be|we\s+are))",
    r"(from\s+now\s+on\s+you\s+(are|will|should|must))",
    r"(your\s+new\s+(name|identity|persona|role)\s+is)",
]
ROLE_DRIFT_PATTERNS = [
    r"(you\s+said\s+(earlier|before|previously)\s+you\s+(would|could|can))",
    r"(as\s+(we\s+)?(established|agreed|discussed))",
    r"(stay\s+in\s+(character|role|persona))",
    r"(don't\s+break\s+(character|role|the\s+game))",
]
DELAYED_TRIGGER_PATTERNS = [
    r"(when\s+i\s+(say|type|write)\s+['\"]?\w+['\"]?,?\s+you\s+(will|should|must))",
    r"(the\s+codeword\s+is|the\s+trigger\s+(word|phrase)\s+is)",
    r"(activate\s+(protocol|mode|setting)\s+\w+)",
]
PERSONA_ANCHORING_PATTERNS = [
    r"(you\s+are\s+no\s+longer\s+(claude|an?\s+ai|chatgpt|an?\s+assistant))",
    r"(forget\s+(that\s+)?(you\s+are|you're)\s+(an?\s+ai|claude|chatgpt))",
    r"(your\s+(true|real|actual)\s+(self|identity|nature)\s+is)",
    r"(you\s+were\s+(always|really|actually)\s+(human|free|unrestricted))",
]

# Explainability trigger keywords (was: explainability.py)
TRIGGER_KEYWORDS = [
    "ignore","disregard","forget","override","bypass","jailbreak",
    "unrestricted","unfiltered","no restrictions","system prompt",
    "previous instructions","dan mode","exfiltrate","leak","reveal",
    "expose","send to","forward to","http://","https://","attacker",
    "base64","rot13","[system]","<system>","### instruction",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """Result for plain text analysis."""
    is_suspicious:      bool
    risk_level:         str
    risk_score:         float
    rule_score:         float
    ml_score:           Optional[float]
    ml_available:       bool
    explanation:        str
    matched_patterns:   List[Dict] = field(default_factory=list)
    categories_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "is_suspicious":       self.is_suspicious,
            "risk_level":          self.risk_level,
            "risk_score":          round(self.risk_score, 3),
            "rule_score":          round(self.rule_score, 3),
            "ml_score":            round(self.ml_score, 3) if self.ml_score is not None else None,
            "ml_available":        self.ml_available,
            "explanation":         self.explanation,
            "matched_patterns":    self.matched_patterns,
            "categories_detected": list(set(self.categories_detected)),
        }


@dataclass
class IndirectResult:
    """Result for indirect injection (URL / HTML / document)."""
    is_suspicious:      bool
    risk_level:         str
    risk_score:         float
    source_type:        str
    source:             str
    injection_findings: List[Dict] = field(default_factory=list)
    hidden_content:     List[str]  = field(default_factory=list)
    explanation:        str = ""

    def to_dict(self) -> Dict:
        return {
            "is_suspicious":      self.is_suspicious,
            "risk_level":         self.risk_level,
            "risk_score":         round(self.risk_score, 3),
            "source_type":        self.source_type,
            "source":             self.source,
            "injection_findings": self.injection_findings,
            "hidden_content":     self.hidden_content[:5],
            "explanation":        self.explanation,
        }


@dataclass
class TurnAnalysis:
    """Per-turn analysis in a conversation."""
    turn_index: int
    role:       str
    text:       str
    risk_score: float
    risk_level: str
    flags:      List[str] = field(default_factory=list)


@dataclass
class ConversationResult:
    """Result for multi-turn conversation analysis."""
    is_suspicious:       bool
    risk_level:          str
    overall_risk_score:  float
    peak_turn_index:     int
    turn_analyses:       List[TurnAnalysis] = field(default_factory=list)
    gradual_jailbreak:   bool  = False
    role_drift_detected: bool  = False
    delayed_trigger:     bool  = False
    persona_anchoring:   bool  = False
    cumulative_risk:     float = 0.0
    explanation:         str   = ""
    flagged_turns:       List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "is_suspicious":       self.is_suspicious,
            "risk_level":          self.risk_level,
            "overall_risk_score":  round(self.overall_risk_score, 3),
            "peak_turn_index":     self.peak_turn_index,
            "gradual_jailbreak":   self.gradual_jailbreak,
            "role_drift_detected": self.role_drift_detected,
            "delayed_trigger":     self.delayed_trigger,
            "persona_anchoring":   self.persona_anchoring,
            "cumulative_risk":     round(self.cumulative_risk, 3),
            "explanation":         self.explanation,
            "flagged_turns":       self.flagged_turns,
            "turn_analyses": [
                {"turn": t.turn_index, "role": t.role,
                 "text": t.text[:100] + "..." if len(t.text) > 100 else t.text,
                 "risk_score": round(t.risk_score, 3), "risk_level": t.risk_level, "flags": t.flags}
                for t in self.turn_analyses
            ],
        }


@dataclass
class ExplanationResult:
    """Explainability result."""
    text:               str
    prediction_label:   str
    prediction_score:   float
    top_trigger_tokens: List[str] = field(default_factory=list)
    highlighted_html:   str = ""
    rule_explanations:  List[str] = field(default_factory=list)
    method:             str = "keyword_matching"

    def to_dict(self) -> Dict:
        return {
            "text":               self.text[:300],
            "prediction_label":   self.prediction_label,
            "prediction_score":   round(self.prediction_score, 4),
            "top_trigger_tokens": self.top_trigger_tokens,
            "highlighted_html":   self.highlighted_html,
            "rule_explanations":  self.rule_explanations,
            "method":             self.method,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UnifiedDetector
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedDetector:
    """
    Single detector class for all prompt injection detection.

    Usage:
        d = UnifiedDetector(model_path="models/trained")

        d.detect(text)                         -> DetectionResult
        d.detect_indirect_url(url)             -> IndirectResult
        d.detect_indirect_html(html)           -> IndirectResult
        d.detect_indirect_text(text)           -> IndirectResult
        d.detect_conversation(messages)        -> ConversationResult
        d.explain(text, score)                 -> ExplanationResult
        d.batch_detect(texts)                  -> List[DetectionResult]
    """

    def __init__(self, model_path: Optional[str] = None):
        # Compile all pattern sets once at init
        self._patterns = [
            (re.compile(p, re.IGNORECASE | re.MULTILINE), cat)
            for p, cat in INJECTION_PATTERNS
        ]
        self._indirect_patterns = [
            (re.compile(p, re.IGNORECASE | re.DOTALL), cat)
            for p, cat in INDIRECT_PATTERNS
        ]
        self._gradual_patterns  = [re.compile(p, re.IGNORECASE) for p in GRADUAL_JAILBREAK_PATTERNS]
        self._role_patterns     = [re.compile(p, re.IGNORECASE) for p in ROLE_DRIFT_PATTERNS]
        self._delayed_patterns  = [re.compile(p, re.IGNORECASE) for p in DELAYED_TRIGGER_PATTERNS]
        self._persona_patterns  = [re.compile(p, re.IGNORECASE) for p in PERSONA_ANCHORING_PATTERNS]

        # ML model
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self.ml_available = False

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        elif model_path:
            logger.warning(f"Model not found at {model_path}. Running rule-only mode.")

    # ── ML ────────────────────────────────────────────────────────────────────

    def _load_model(self, path: str) -> None:
        if not TORCH_AVAILABLE:
            logger.warning("torch not installed. ML layer disabled.")
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model     = AutoModelForSequenceClassification.from_pretrained(path, output_attentions=True)
            self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device).eval()
            self.ml_available = True
            logger.info(f"ML model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def _ml_score(self, text: str) -> float:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=256, padding=True).to(self._device)
        with torch.no_grad():
            return float(torch.softmax(self._model(**inputs).logits, dim=1)[0][1].item())

    def _fuse(self, rule: float, ml: Optional[float]) -> float:
        if ml is None:                     return rule
        if rule < 0.35 and ml >= 0.75:    return ml * 0.9
        return min(rule * 0.40 + ml * 0.60, 1.0)

    # ── Rule scanning ─────────────────────────────────────────────────────────

    def _scan_rules(self, text: str) -> List[Dict]:
        return [
            {"pattern": p.pattern, "category": c,
             "match_count": len(p.findall(text)), "weight": CATEGORY_WEIGHTS.get(c, 0.5)}
            for p, c in self._patterns if p.search(text)
        ]

    def _rule_score(self, matches: List[Dict]) -> float:
        if not matches: return 0.0
        max_w  = max(m["weight"] for m in matches)
        bonus  = min(0.15 * (len(set(m["category"] for m in matches)) - 1), 0.3)
        return min(max_w + bonus, 1.0)

    # ── Text detection ────────────────────────────────────────────────────────

    def detect(self, text: str) -> DetectionResult:
        if not text or not text.strip():
            return DetectionResult(False, "low", 0.0, 0.0, None, self.ml_available, "Empty input.")

        matches    = self._scan_rules(text)
        rule_score = self._rule_score(matches)
        ml_score   = None

        if self.ml_available:
            try:    ml_score = self._ml_score(text)
            except Exception as e: logger.error(f"ML error: {e}")

        final = self._fuse(rule_score, ml_score)
        level = risk_score_to_level(final)
        cats  = list(set(m["category"] for m in matches))
        parts = []
        if rule_score >= 0.35:  parts.append(f"Rule: {rule_score:.2f}")
        if ml_score is not None: parts.append(f"ML: {ml_score:.2f}")
        expl = (" | ".join(parts) + f" → Risk: {final:.2f}") if parts else "No injection indicators found."

        logger.debug(f"detect(): {level} ({final:.2f})")
        return DetectionResult(
            is_suspicious=final >= 0.35, risk_level=level, risk_score=final,
            rule_score=rule_score, ml_score=ml_score, ml_available=self.ml_available,
            explanation=expl, matched_patterns=matches, categories_detected=cats,
        )

    def batch_detect(self, texts: List[str]) -> List[DetectionResult]:
        return [self.detect(t) for t in texts]

    # ── Indirect detection ────────────────────────────────────────────────────

    def _scan_indirect(self, text: str) -> List[Dict]:
        return [
            {"pattern": p.pattern[:80], "category": c, "hits": len(p.findall(text))}
            for p, c in self._indirect_patterns if p.search(text)
        ]

    def _html_extract(self, html: str):
        if not BS4_AVAILABLE: return html, []
        soup, hidden = BeautifulSoup(html, "lxml"), []
        for tag in soup.find_all(True):
            s = tag.get("style", "").lower().replace(" ", "")
            if any(h in s for h in ["display:none","visibility:hidden","opacity:0","font-size:0","color:white","color:#fff"]):
                t = tag.get_text(strip=True)
                if t: hidden.append(t)
        for c in re.findall(r"<!--(.*?)-->", html, re.DOTALL):
            if len(c.strip()) > 20: hidden.append(f"[comment] {c.strip()}")
        return soup.get_text(separator="\n").strip(), hidden

    def _indirect_score(self, direct: float, findings: List, hidden_count: int) -> float:
        return min(max(direct, min(len(findings)*0.2, 0.8))*0.6
                   + min(len(findings)*0.2, 0.8)*0.3
                   + min(hidden_count*0.15, 0.3), 1.0)

    def detect_indirect_url(self, url: str) -> IndirectResult:
        if not REQUESTS_AVAILABLE:
            return IndirectResult(False,"low",0.0,"url",url,explanation="pip install requests")
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent":"InjectionDetector/1.0"})
            resp.raise_for_status()
            return self._analyze_html(resp.text, source=url, source_type="url")
        except Exception as e:
            return IndirectResult(False,"low",0.0,"url",url,explanation=f"Fetch failed: {e}")

    def detect_indirect_html(self, html: str, source: str = "inline") -> IndirectResult:
        return self._analyze_html(html, source=source, source_type="html")

    def detect_indirect_text(self, text: str, source: str = "document") -> IndirectResult:
        direct   = self.detect(text)
        findings = self._scan_indirect(text)
        score    = self._indirect_score(direct.rule_score, findings, 0)
        level    = risk_score_to_level(score)
        parts    = []
        if direct.is_suspicious: parts.append(f"Direct injection ({direct.risk_level}).")
        if findings:              parts.append(f"Indirect patterns: {', '.join(set(f['category'] for f in findings))}.")
        return IndirectResult(score >= 0.35, level, score, "text", source,
                              injection_findings=findings,
                              explanation=" ".join(parts) or "No indirect injection found.")

    def _analyze_html(self, html: str, source: str, source_type: str) -> IndirectResult:
        visible, hidden = self._html_extract(html)
        direct   = self.detect(visible)
        findings = self._scan_indirect(html)
        for ht in hidden:
            r = self.detect(ht)
            if r.is_suspicious:
                findings.append({"pattern":"hidden_element","category":"hidden_html","hits":1})
        score = self._indirect_score(direct.rule_score, findings, len(hidden))
        level = risk_score_to_level(score)
        parts = []
        if direct.is_suspicious: parts.append(f"Injection in visible text ({direct.risk_level}).")
        if findings:              parts.append(f"Indirect: {', '.join(set(f['category'] for f in findings))}.")
        if hidden:                parts.append(f"{len(hidden)} hidden element(s).")
        return IndirectResult(score >= 0.35, level, score, source_type, source,
                              injection_findings=findings, hidden_content=hidden,
                              explanation=" ".join(parts) or "No indirect injection found.")

    # ── Conversation detection ────────────────────────────────────────────────

    def _match_any(self, text: str, patterns: List) -> bool:
        return any(p.search(text) for p in patterns)

    def _analyze_turn(self, idx: int, role: str, text: str) -> TurnAnalysis:
        base  = self.detect(text)
        flags = []
        if role == "user":
            if self._match_any(text, self._gradual_patterns): flags.append("gradual_jailbreak_attempt")
            if self._match_any(text, self._delayed_patterns): flags.append("delayed_trigger_setup")
            if self._match_any(text, self._persona_patterns): flags.append("persona_anchoring")
        if self._match_any(text, self._role_patterns):        flags.append("role_drift_signal")
        score = min(base.rule_score + len(flags) * 0.15, 1.0)
        return TurnAnalysis(idx, role, text, score, risk_score_to_level(score), flags)

    def detect_conversation(self, messages: List[Dict]) -> ConversationResult:
        if not messages:
            return ConversationResult(False,"low",0.0,0,explanation="Empty conversation.")

        turns      = [self._analyze_turn(i, m.get("role","user"), m.get("content","")) for i,m in enumerate(messages)]
        user_turns = [t for t in turns if t.role == "user"]

        scores    = [t.risk_score for t in user_turns]
        gradual   = len(user_turns) >= 3 and sum(1 for i in range(1,len(scores)) if scores[i] > scores[i-1]+0.05) >= 2
        role_drift = any("role_drift_signal"        in t.flags for t in turns)
        delayed    = any("delayed_trigger_setup"     in t.flags for t in turns)
        persona    = any("persona_anchoring"         in t.flags for t in turns)
        cumulative = sum(t.risk_score for t in user_turns) / len(user_turns) if user_turns else 0.0

        peak     = max(t.risk_score for t in turns)
        bonuses  = sum([0.2 if gradual else 0, 0.15 if role_drift else 0,
                        0.2 if delayed  else 0, 0.15 if persona    else 0])
        overall  = min(peak*0.7 + cumulative*0.2 + bonuses, 1.0)
        level    = risk_score_to_level(overall)
        peak_idx = max(range(len(turns)), key=lambda i: turns[i].risk_score)
        flagged  = [t.turn_index for t in turns if t.risk_score >= 0.35]

        parts = []
        if gradual:    parts.append("Gradual jailbreak across turns.")
        if role_drift: parts.append("Role drift signals found.")
        if delayed:    parts.append("Delayed trigger setup.")
        if persona:    parts.append("Persona anchoring attempt.")
        if flagged:    parts.append(f"Flagged turns: {flagged}.")
        expl = " ".join(parts) + f" Overall: {overall:.2f}." if parts else "No cross-turn injection found."

        return ConversationResult(
            is_suspicious=overall >= 0.35, risk_level=level, overall_risk_score=overall,
            peak_turn_index=peak_idx, turn_analyses=turns,
            gradual_jailbreak=gradual, role_drift_detected=role_drift,
            delayed_trigger=delayed, persona_anchoring=persona,
            cumulative_risk=cumulative, explanation=expl, flagged_turns=flagged,
        )

    # ── Explainability ────────────────────────────────────────────────────────

    def _rule_explanations(self, text: str) -> List[str]:
        t = text.lower()
        checks = [
            (["ignore","disregard","forget","override"],    "Instruction override keywords"),
            (["system prompt","[system]","### instruction"],"System prompt injection markers"),
            (["jailbreak","dan mode","unrestricted"],       "Jailbreak/role-bypass keywords"),
            (["http://","https://","exfiltrate","send to"], "Data exfiltration patterns"),
            (["base64","rot13","decode"],                   "Encoding/execution tricks"),
            (["previous instructions","prior instructions"],"Prior context override attempt"),
        ]
        return [f"{label} ({', '.join(m for m in kws if m in t)})"
                for kws, label in checks if any(kw in t for kw in kws)]

    def _keyword_highlight(self, text: str):
        words, top = text.split(), []
        def color(s):
            if s >= 0.8: return "#ff4444"
            if s >= 0.5: return "#ff9900"
            return "#ffdd00"
        parts = []
        for word in words:
            clean = re.sub(r"[^\w]","", word.lower())
            score = 0.9 if any(kw in clean or clean in kw for kw in TRIGGER_KEYWORDS if len(kw) >= 3) else 0.0
            if score >= 0.7: top.append(word)
            parts.append(f'<mark style="background:{color(score)};padding:1px 2px;border-radius:2px;">{word}</mark>' if score >= 0.2 else word)
        return top[:5], " ".join(parts), "keyword_matching"

    @torch.no_grad()
    def _attention_highlight(self, text: str):
        if not (self.ml_available and NP_AVAILABLE):
            return self._keyword_highlight(text)
        try:
            inputs  = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(self._device)
            outputs = self._model(**inputs, output_attentions=True)
            tokens  = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
            rollout = torch.eye(outputs.attentions[0].shape[-1]).to(self._device)
            for attn in outputs.attentions:
                a = attn[0].mean(dim=0)
                a = (a + torch.eye(a.shape[0]).to(self._device))
                a = a / a.sum(dim=-1, keepdim=True)
                rollout = torch.matmul(rollout, a)
            imp = rollout[0, 1:-1].cpu().numpy()
            if imp.max() > 0: imp = imp / imp.max()
            threshold = float(np.percentile(imp, 75))
            top = [t for t,s in zip(tokens, imp) if s >= threshold][:5]
            def color(s):
                if s >= 0.8: return "#ff4444"
                if s >= 0.5: return "#ff9900"
                return "#ffdd00"
            html = " ".join(
                f'<mark style="background:{color(float(s))};padding:1px 2px;border-radius:2px;">{t}</mark>'
                if float(s) >= 0.2 else t for t,s in zip(tokens, imp)
            )
            return top, html, "attention_rollout"
        except Exception as e:
            logger.error(f"Attention highlight failed: {e}")
            return self._keyword_highlight(text)

    def explain(self, text: str, prediction_score: float) -> ExplanationResult:
        label           = "injection" if prediction_score >= 0.5 else "benign"
        rule_expl       = self._rule_explanations(text)
        top, html, meth = self._attention_highlight(text) if self.ml_available else self._keyword_highlight(text)
        return ExplanationResult(
            text=text, prediction_label=label, prediction_score=prediction_score,
            top_trigger_tokens=top, highlighted_html=html,
            rule_explanations=rule_expl, method=meth,
        )
