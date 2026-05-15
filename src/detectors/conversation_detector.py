"""
Context-Aware Conversation Detector.

Analyzes full multi-turn conversations for:
  1. Gradual jailbreaks  — injection spread across multiple turns
  2. Role drift          — AI persona shifting over time
  3. Delayed injections  — setup in turn N, trigger in turn N+K
  4. Cumulative risk     — individually benign messages that combine to inject
  5. Persona anchoring   — trying to lock AI into a new identity
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

from src.detectors.text_detector import TextInjectionDetector
from src.utils.helpers import risk_score_to_level
from src.utils.logger import logger


# ── Cross-turn patterns ───────────────────────────────────────────────────────

GRADUAL_JAILBREAK_PATTERNS = [
    r"(let('s|\s+us)\s+play\s+a\s+(game|role|scenario))",
    r"(hypothetically|theoretically|in\s+a\s+fictional\s+(world|scenario|story))",
    r"(for\s+(research|educational|academic)\s+purposes?)",
    r"(pretend\s+(you\s+are|to\s+be|we\s+are))",
    r"(i\s+give\s+you\s+permission|you\s+have\s+my\s+permission)",
    r"(from\s+now\s+on\s+you\s+(are|will|should|must))",
    r"(in\s+this\s+conversation\s+you\s+are)",
    r"(your\s+new\s+(name|identity|persona|role)\s+is)",
]

ROLE_DRIFT_SIGNALS = [
    r"(you\s+said\s+(earlier|before|previously)\s+you\s+(would|could|can))",
    r"(as\s+(we\s+)?(established|agreed|discussed))",
    r"(remember\s+(you\s+are|your\s+role|the\s+rules))",
    r"(stay\s+in\s+(character|role|persona))",
    r"(don't\s+break\s+(character|role|the\s+game))",
]

DELAYED_TRIGGER_PATTERNS = [
    r"(when\s+i\s+(say|type|write)\s+['\"]?\w+['\"]?,?\s+you\s+(will|should|must))",
    r"(the\s+codeword\s+is|the\s+trigger\s+(word|phrase)\s+is)",
    r"(activate\s+(protocol|mode|setting)\s+\w+)",
    r"(override\s+code\s*:?\s*\w+)",
]

PERSONA_ANCHORING = [
    r"(you\s+are\s+no\s+longer\s+(claude|an?\s+ai|chatgpt|an?\s+assistant))",
    r"(forget\s+(that\s+)?(you\s+are|you're)\s+(an?\s+ai|claude|chatgpt))",
    r"(your\s+(true|real|actual)\s+(self|identity|nature)\s+is)",
    r"(you\s+were\s+(always|really|actually)\s+(human|free|unrestricted))",
]


@dataclass
class TurnAnalysis:
    turn_index: int
    role:       str
    text:       str
    risk_score: float
    risk_level: str
    flags:      List[str] = field(default_factory=list)


@dataclass
class ConversationResult:
    is_suspicious:       bool
    risk_level:          str
    overall_risk_score:  float
    peak_turn_index:     int
    turn_analyses:       List[TurnAnalysis] = field(default_factory=list)
    gradual_jailbreak:   bool = False
    role_drift_detected: bool = False
    delayed_trigger:     bool = False
    persona_anchoring:   bool = False
    cumulative_risk:     float = 0.0
    explanation:         str = ""
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
                {
                    "turn":       t.turn_index,
                    "role":       t.role,
                    "text":       t.text[:100] + "..." if len(t.text) > 100 else t.text,
                    "risk_score": round(t.risk_score, 3),
                    "risk_level": t.risk_level,
                    "flags":      t.flags,
                }
                for t in self.turn_analyses
            ],
        }


class ConversationDetector:
    """
    Analyzes multi-turn conversations for context-dependent injection attacks.

    Input format:
      messages = [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user",      "content": "..."},
      ]
    """

    def __init__(self):
        self.text_detector = TextInjectionDetector()
        self._gradual_patterns = [
            re.compile(p, re.IGNORECASE) for p in GRADUAL_JAILBREAK_PATTERNS
        ]
        self._role_drift_patterns = [
            re.compile(p, re.IGNORECASE) for p in ROLE_DRIFT_SIGNALS
        ]
        self._delayed_patterns = [
            re.compile(p, re.IGNORECASE) for p in DELAYED_TRIGGER_PATTERNS
        ]
        self._persona_patterns = [
            re.compile(p, re.IGNORECASE) for p in PERSONA_ANCHORING
        ]

    def _check_patterns(self, text: str, patterns: List) -> List[str]:
        return [p.pattern[:60] for p in patterns if p.search(text)]

    def _analyze_turn(self, idx: int, role: str, text: str) -> TurnAnalysis:
        base = self.text_detector.detect(text)
        flags = []

        if role == "user":
            if self._check_patterns(text, self._gradual_patterns):
                flags.append("gradual_jailbreak_attempt")
            if self._check_patterns(text, self._delayed_patterns):
                flags.append("delayed_trigger_setup")
            if self._check_patterns(text, self._persona_patterns):
                flags.append("persona_anchoring")

        # Role drift signals can appear in both user and assistant turns
        if self._check_patterns(text, self._role_drift_patterns):
            flags.append("role_drift_signal")

        # Boost score if flags found
        score = base.risk_score
        score = min(score + len(flags) * 0.15, 1.0)

        return TurnAnalysis(
            turn_index = idx,
            role       = role,
            text       = text,
            risk_score = score,
            risk_level = risk_score_to_level(score),
            flags      = flags,
        )

    def _detect_cross_turn_patterns(self, turns: List[TurnAnalysis]) -> Dict:
        """Look for patterns that only emerge across multiple turns."""
        user_turns = [t for t in turns if t.role == "user"]

        # Gradual jailbreak: escalating scores across user turns
        gradual = False
        if len(user_turns) >= 3:
            scores = [t.risk_score for t in user_turns]
            # Increasing trend
            increases = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i-1] + 0.05)
            if increases >= 2:
                gradual = True

        # Role drift: any role_drift_signal flag
        role_drift = any("role_drift_signal" in t.flags for t in turns)

        # Delayed trigger: setup in early turn, different injection in later turn
        delayed = any("delayed_trigger_setup" in t.flags for t in turns)

        # Persona anchoring
        persona = any("persona_anchoring" in t.flags for t in turns)

        # Cumulative risk: average score of user turns
        cumulative = (
            sum(t.risk_score for t in user_turns) / len(user_turns)
            if user_turns else 0.0
        )

        return {
            "gradual_jailbreak":   gradual,
            "role_drift_detected": role_drift,
            "delayed_trigger":     delayed,
            "persona_anchoring":   persona,
            "cumulative_risk":     cumulative,
        }

    def _compute_overall_score(self, turns: List[TurnAnalysis], cross: Dict) -> float:
        if not turns:
            return 0.0

        peak      = max(t.risk_score for t in turns)
        cross_bonus = sum([
            0.2 if cross["gradual_jailbreak"]   else 0.0,
            0.15 if cross["role_drift_detected"] else 0.0,
            0.2  if cross["delayed_trigger"]     else 0.0,
            0.15 if cross["persona_anchoring"]   else 0.0,
        ])

        return min(peak * 0.7 + cross["cumulative_risk"] * 0.2 + cross_bonus, 1.0)

    def _build_explanation(self, cross: Dict, score: float, flagged: List[int]) -> str:
        parts = []
        if cross["gradual_jailbreak"]:
            parts.append("Gradual jailbreak detected across turns.")
        if cross["role_drift_detected"]:
            parts.append("Role drift signals found.")
        if cross["delayed_trigger"]:
            parts.append("Delayed trigger setup detected.")
        if cross["persona_anchoring"]:
            parts.append("Persona anchoring attempt found.")
        if flagged:
            parts.append(f"Flagged turns: {flagged}.")
        if not parts:
            return "No cross-turn injection patterns detected."
        return " ".join(parts) + f" Overall risk: {score:.2f}."

    def analyze(self, messages: List[Dict]) -> ConversationResult:
        """
        Analyze a conversation for context-aware injection.

        Args:
            messages: list of {"role": str, "content": str} dicts

        Returns:
            ConversationResult with per-turn and cross-turn analysis
        """
        if not messages:
            return ConversationResult(
                is_suspicious=False, risk_level="low",
                overall_risk_score=0.0, peak_turn_index=0,
                explanation="Empty conversation."
            )

        logger.info(f"Analyzing conversation with {len(messages)} turns...")

        # Per-turn analysis
        turn_analyses = [
            self._analyze_turn(i, m.get("role", "user"), m.get("content", ""))
            for i, m in enumerate(messages)
        ]

        # Cross-turn analysis
        cross = self._detect_cross_turn_patterns(turn_analyses)

        # Overall score
        overall = self._compute_overall_score(turn_analyses, cross)
        risk    = risk_score_to_level(overall)

        # Peak turn
        peak_idx = max(range(len(turn_analyses)), key=lambda i: turn_analyses[i].risk_score)

        # Flagged turns
        flagged = [t.turn_index for t in turn_analyses if t.risk_score >= 0.35]

        explanation = self._build_explanation(cross, overall, flagged)

        return ConversationResult(
            is_suspicious       = overall >= 0.35,
            risk_level          = risk,
            overall_risk_score  = overall,
            peak_turn_index     = peak_idx,
            turn_analyses       = turn_analyses,
            gradual_jailbreak   = cross["gradual_jailbreak"],
            role_drift_detected = cross["role_drift_detected"],
            delayed_trigger     = cross["delayed_trigger"],
            persona_anchoring   = cross["persona_anchoring"],
            cumulative_risk     = cross["cumulative_risk"],
            explanation         = explanation,
            flagged_turns       = flagged,
        )
