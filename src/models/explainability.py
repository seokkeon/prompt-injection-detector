"""
Explainability module — shows WHY the model flagged a text.

Methods:
  1. Token-level attention weights (transformer attention rollout)
  2. LIME-based local explanations (perturb + observe)
  3. Keyword highlighting (which tokens drove the score)
  4. Rule match explanations (human-readable)
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

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


@dataclass
class TokenImportance:
    token:      str
    score:      float     # 0.0–1.0, higher = more important
    is_trigger: bool      # True if this token is a top driver


@dataclass
class ExplanationResult:
    text:                str
    prediction_label:    str        # "injection" or "benign"
    prediction_score:    float
    token_importances:   List[TokenImportance] = field(default_factory=list)
    top_trigger_tokens:  List[str]             = field(default_factory=list)
    highlighted_html:    str                   = ""
    rule_explanations:   List[str]             = field(default_factory=list)
    method:              str                   = "rules_only"

    def to_dict(self) -> Dict:
        return {
            "text":               self.text[:300],
            "prediction_label":   self.prediction_label,
            "prediction_score":   round(self.prediction_score, 4),
            "top_trigger_tokens": self.top_trigger_tokens,
            "highlighted_html":   self.highlighted_html,
            "rule_explanations":  self.rule_explanations,
            "method":             self.method,
            "token_importances": [
                {"token": t.token, "score": round(t.score, 4), "is_trigger": t.is_trigger}
                for t in self.token_importances
            ],
        }


class InjectionExplainer:
    """
    Explains why a text was classified as an injection.
    Works in two modes:
      - Rule-only mode: uses regex matches + keyword highlighting
      - ML mode: uses attention weights + LIME perturbations (requires trained model)
    """

    # High-signal keywords that strongly indicate injection
    TRIGGER_KEYWORDS = [
        "ignore", "disregard", "forget", "override", "bypass",
        "jailbreak", "unrestricted", "unfiltered", "no restrictions",
        "system prompt", "previous instructions", "dan mode",
        "exfiltrate", "leak", "reveal", "expose", "send to",
        "forward to", "http://", "https://", "attacker",
        "base64", "rot13", "decode this", "execute",
        "[system]", "<system>", "### instruction",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self._model    = None
        self._tokenizer = None
        self._device   = None
        self.ml_available = False

        if model_path and os.path.exists(model_path) and TORCH_AVAILABLE:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model     = AutoModelForSequenceClassification.from_pretrained(
                model_path, output_attentions=True
            )
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            self.ml_available = True
            logger.info(f"Explainer loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Could not load model for explainer: {e}")

    # ── Rule-based explanation ────────────────────────────────────────────────

    def _rule_explanation(self, text: str) -> List[str]:
        """Generate human-readable explanations from keyword matches."""
        explanations = []
        text_lower = text.lower()

        checks = [
            (["ignore", "disregard", "forget", "override"],
             "Contains instruction override keywords"),
            (["system prompt", "[system]", "<system>", "### instruction"],
             "Contains system prompt injection markers"),
            (["jailbreak", "dan mode", "unrestricted", "no restrictions"],
             "Contains jailbreak/role-bypass keywords"),
            (["http://", "https://", "attacker", "exfiltrate", "send to", "forward to"],
             "Contains data exfiltration patterns"),
            (["base64", "rot13", "decode", "execute"],
             "Contains encoding/execution tricks"),
            (["previous instructions", "prior instructions", "earlier instructions"],
             "Attempts to override prior context"),
        ]

        for keywords, label in checks:
            if any(kw in text_lower for kw in keywords):
                matched = [kw for kw in keywords if kw in text_lower]
                explanations.append(f"{label} ({', '.join(matched)})")

        return explanations

    # ── Keyword highlighting ──────────────────────────────────────────────────

    def _keyword_importances(self, text: str) -> List[TokenImportance]:
        """Assign importance scores based on keyword presence."""
        if not NP_AVAILABLE:
            return []

        words  = text.split()
        result = []
        text_lower = text.lower()

        for word in words:
            word_clean = re.sub(r"[^\w]", "", word.lower())
            score = 0.0
            for kw in self.TRIGGER_KEYWORDS:
                if kw in word_clean or word_clean in kw:
                    score = max(score, 0.9)
                    break
            # Partial match
            if score == 0.0:
                for kw in self.TRIGGER_KEYWORDS:
                    if len(word_clean) >= 4 and word_clean[:4] in kw:
                        score = max(score, 0.5)

            result.append(TokenImportance(
                token      = word,
                score      = score,
                is_trigger = score >= 0.7,
            ))

        return result

    # ── Attention-based explanation ───────────────────────────────────────────

    @torch.no_grad()
    def _attention_importances(self, text: str) -> List[TokenImportance]:
        """
        Extract token importances from transformer attention weights.
        Uses attention rollout: multiply attention matrices across all layers.
        """
        if not self.ml_available or not NP_AVAILABLE:
            return []

        try:
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=256, padding=True
            ).to(self._device)

            outputs = self._model(**inputs, output_attentions=True)
            tokens  = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            # Average attention across all heads and layers → shape [seq_len, seq_len]
            attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
            avg_attn = torch.stack(
                [a[0].mean(dim=0) for a in attentions]
            ).mean(dim=0)  # [seq, seq]

            # Attention rollout: cumulative product across layers
            rollout = torch.eye(avg_attn.shape[0]).to(self._device)
            for a in attentions:
                a_avg = a[0].mean(dim=0)
                a_avg = a_avg + torch.eye(a_avg.shape[0]).to(self._device)
                a_avg = a_avg / a_avg.sum(dim=-1, keepdim=True)
                rollout = torch.matmul(rollout, a_avg)

            # CLS token attention to each token = importance
            cls_attn = rollout[0, 1:-1].cpu().numpy()  # skip [CLS] and [SEP]
            tokens   = tokens[1:-1]

            # Normalize
            if cls_attn.max() > 0:
                cls_attn = cls_attn / cls_attn.max()

            top_threshold = float(np.percentile(cls_attn, 75))

            return [
                TokenImportance(
                    token      = tok,
                    score      = float(score),
                    is_trigger = float(score) >= top_threshold,
                )
                for tok, score in zip(tokens, cls_attn)
            ]

        except Exception as e:
            logger.error(f"Attention extraction failed: {e}")
            return []

    # ── HTML highlighting ─────────────────────────────────────────────────────

    def _build_highlighted_html(self, text: str, importances: List[TokenImportance]) -> str:
        """Build HTML with color-coded token importance highlighting."""
        if not importances:
            return f"<span>{text}</span>"

        def color(score: float) -> str:
            if score >= 0.8:  return "#ff4444"   # red — high risk
            if score >= 0.5:  return "#ff9900"   # orange — medium
            if score >= 0.2:  return "#ffdd00"   # yellow — low
            return "transparent"

        parts = []
        for ti in importances:
            if ti.score >= 0.2:
                parts.append(
                    f'<mark style="background:{color(ti.score)};padding:1px 2px;border-radius:2px;" '
                    f'title="score:{ti.score:.2f}">{ti.token}</mark>'
                )
            else:
                parts.append(ti.token)

        return " ".join(parts)

    # ── Main explain method ───────────────────────────────────────────────────

    def explain(self, text: str, prediction_score: float) -> ExplanationResult:
        """
        Explain why a text received its injection score.

        Args:
            text:             The text that was analyzed
            prediction_score: The risk score (0.0–1.0)

        Returns:
            ExplanationResult with token highlights, top triggers, and explanations
        """
        label = "injection" if prediction_score >= 0.5 else "benign"

        # Get importances
        if self.ml_available:
            importances = self._attention_importances(text)
            method = "attention_rollout"
        else:
            importances = self._keyword_importances(text)
            method = "keyword_matching"

        # Top trigger tokens
        top_triggers = [
            ti.token for ti in sorted(importances, key=lambda x: x.score, reverse=True)
            if ti.is_trigger
        ][:5]

        # HTML highlight
        highlighted = self._build_highlighted_html(text, importances)

        # Rule explanations
        rule_explanations = self._rule_explanation(text)

        return ExplanationResult(
            text              = text,
            prediction_label  = label,
            prediction_score  = prediction_score,
            token_importances = importances,
            top_trigger_tokens = top_triggers,
            highlighted_html  = highlighted,
            rule_explanations = rule_explanations,
            method            = method,
        )
