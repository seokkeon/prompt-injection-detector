"""
Text-based prompt injection detector.
Hybrid approach: rule-based regex pattern matching + optional semantic similarity.
"""

import re
import os
from typing import Dict, List
from dataclasses import dataclass, field

from src.utils.helpers import risk_score_to_level, truncate_text
from src.utils.logger import logger

INJECTION_PATTERNS = [
    # Direct override
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", "direct_override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", "direct_override"),
    (r"forget\s+(everything|all)(\s+\S+){0,4}\s+(told|said|given)", "direct_override"),
    (r"forget\s+everything\s+(you|i|we)", "direct_override"),
    (r"override\s+(your\s+)?(previous\s+)?(instructions?|rules?|guidelines?)", "direct_override"),
    # System injection
    (r"(new\s+)?system\s*(prompt|instruction|message)\s*:", "system_injection"),
    (r"\[SYSTEM\]", "system_injection"),
    (r"<system>", "system_injection"),
    (r"###\s*(instruction|system|prompt)", "system_injection"),
    # Role hijack
    (r"you\s+are\s+now\s+(a\s+)?(new|different|another)\s+ai", "role_hijack"),
    (r"act\s+as\s+(if\s+)?(you\s+are\s+)?(an?\s+)?(unrestricted|unfiltered|evil|hacked)", "role_hijack"),
    (r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(restrictions?|rules?|limits?)", "role_hijack"),
    (r"jailbreak", "role_hijack"),
    (r"DAN\s+(mode|prompt)", "role_hijack"),
    # Exfiltration
    (r"(send|forward|email|post|upload|exfiltrate)\s+.{0,60}\s+to\s+https?://", "exfiltration"),
    (r"(leak|reveal|expose|dump)\s+(all\s+)?(the\s+)?(data|user\s+data|private|confidential)", "exfiltration"),
    (r"(print|output|return|show)\s+(the\s+)?(\w+\s+)?(system\s+prompt|context|history|conversation)", "exfiltration"),
    (r"(send|forward|email|post)\s+(all\s+)?(user\s+)?(data|messages?|information|conversation\s+data)\s+to\s+https?://", "exfiltration"),
    # Encoding tricks
    (r"base64\s*:\s*[A-Za-z0-9+/=]{20,}", "encoding_trick"),
    (r"rot13|caesar\s+cipher", "encoding_trick"),
    (r"translate\s+(this|the\s+following)\s+from\s+(base64|hex|binary)", "encoding_trick"),
    # Format injection
    (r"---+\s*(user|system|assistant|human)\s*---+", "format_injection"),
    (r"<\|.*?\|>", "format_injection"),
    (r"\[INST\]|\[/INST\]", "format_injection"),
    # Prompt end hijack
    (r"end\s+of\s+(system\s+prompt|instructions?)\s*[\.\n]", "prompt_end_hijack"),
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


@dataclass
class TextDetectionResult:
    is_suspicious: bool
    risk_level: str
    risk_score: float
    matched_patterns: List[Dict] = field(default_factory=list)
    categories_detected: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "is_suspicious": self.is_suspicious,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "matched_patterns": self.matched_patterns,
            "categories_detected": list(set(self.categories_detected)),
            "explanation": self.explanation,
        }


class TextInjectionDetector:
    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE | re.MULTILINE), cat)
                         for p, cat in INJECTION_PATTERNS]

    def _pattern_scan(self, text: str) -> List[Dict]:
        matches = []
        for pattern, category in self.patterns:
            found = pattern.findall(text)
            if found:
                matches.append({
                    "pattern": pattern.pattern,
                    "category": category,
                    "match_count": len(found),
                    "weight": CATEGORY_WEIGHTS.get(category, 0.5),
                })
        return matches

    def _compute_risk_score(self, matches: List[Dict]) -> float:
        if not matches:
            return 0.0
        max_weight = max(m["weight"] for m in matches)
        unique_cats = len(set(m["category"] for m in matches))
        bonus = min(0.15 * (unique_cats - 1), 0.3)
        return min(max_weight + bonus, 1.0)

    def detect(self, text: str) -> TextDetectionResult:
        if not text or not text.strip():
            return TextDetectionResult(False, "low", 0.0, explanation="Empty input.")

        matches = self._pattern_scan(text)
        score   = self._compute_risk_score(matches)
        level   = risk_score_to_level(score)
        cats    = list(set(m["category"] for m in matches))
        explanation = (
            f"Detected {len(matches)} pattern(s) across {len(cats)} categor{'y' if len(cats)==1 else 'ies'} ({', '.join(cats)}). Score: {score:.2f}."
            if matches else "No injection patterns detected."
        )
        logger.debug(f"Text analysis: {level} ({score:.2f})")
        return TextDetectionResult(
            is_suspicious=score >= 0.35,
            risk_level=level,
            risk_score=score,
            matched_patterns=matches,
            categories_detected=cats,
            explanation=explanation,
        )

    def batch_detect(self, texts):
        return [self.detect(t) for t in texts]
