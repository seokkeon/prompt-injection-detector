"""
Image-based prompt injection detector.
Combines OCR text extraction, steganography analysis, and text injection detection.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.analyzers.ocr import OCRAnalyzer
from src.analyzers.steganography import SteganographyAnalyzer
from src.detectors.text_detector import TextInjectionDetector, TextDetectionResult
from src.utils.helpers import risk_score_to_level
from src.utils.logger import logger


@dataclass
class ImageDetectionResult:
    is_suspicious: bool
    risk_level: str
    risk_score: float
    extracted_text: str
    text_analysis: Optional[Dict] = None
    steganography_analysis: Optional[Dict] = None
    hidden_text_detected: bool = False
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "is_suspicious": self.is_suspicious,
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "extracted_text": self.extracted_text[:500] if self.extracted_text else "",
            "text_analysis": self.text_analysis,
            "steganography_analysis": self.steganography_analysis,
            "hidden_text_detected": self.hidden_text_detected,
            "explanation": self.explanation,
        }


class ImageInjectionDetector:
    """
    Full pipeline for detecting prompt injections inside images.

    Steps:
      1. Extract all visible and hidden text via OCR
      2. Run text injection detector on extracted text
      3. Run steganography analysis on raw pixel data
      4. Aggregate results into a combined risk score
    """

    def __init__(self, use_easyocr: bool = False):
        self.ocr = OCRAnalyzer(use_easyocr=use_easyocr)
        self.stego = SteganographyAnalyzer()
        self.text_detector = TextInjectionDetector()

    def _combine_scores(
        self,
        text_score: float,
        stego_score: float,
        hidden_text_detected: bool,
    ) -> float:
        """
        Combine individual component scores into an overall risk score.

        Weights:
          - Text injection score: 60%
          - Steganography probability: 25%
          - Hidden text bonus: +0.15 if detected
        """
        combined = (text_score * 0.60) + (stego_score * 0.25)
        if hidden_text_detected:
            combined = min(combined + 0.15, 1.0)
        return combined

    def _build_explanation(
        self,
        text_result: Optional[TextDetectionResult],
        stego_result: Dict,
        hidden_text: bool,
        score: float,
    ) -> str:
        parts = []

        if text_result and text_result.is_suspicious:
            parts.append(
                f"Injection patterns found in image text ({text_result.risk_level} risk)."
            )

        if stego_result.get("is_suspicious"):
            prob = stego_result.get("overall_probability", 0)
            parts.append(f"Steganography detected (probability: {prob:.2f}).")

        if hidden_text:
            parts.append("Hidden/low-contrast text discovered via multi-threshold OCR.")

        if not parts:
            return "No injection indicators found in image."

        return " ".join(parts) + f" Overall risk score: {score:.2f}."

    def analyze(self, image_path: str) -> ImageDetectionResult:
        """
        Analyze an image for prompt injection content.

        Args:
            image_path: Path to image file (PNG, JPEG, etc.)

        Returns:
            ImageDetectionResult with full analysis.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return ImageDetectionResult(
                is_suspicious=False,
                risk_level="low",
                risk_score=0.0,
                extracted_text="",
                explanation=f"File not found: {image_path}",
            )

        logger.info(f"Analyzing image: {image_path}")

        # ── Step 1: Extract text (visible + hidden)
        ocr_result = self.ocr.extract_hidden_text(image_path)
        visible_text = ocr_result.get("visible_text", "")
        hidden_texts = ocr_result.get("hidden_texts", [])
        hidden_detected = ocr_result.get("hidden_text_detected", False)

        all_text = visible_text + "\n" + "\n".join(hidden_texts)

        # ── Step 2: Text injection analysis
        text_result = None
        text_score = 0.0
        if all_text.strip():
            text_result = self.text_detector.detect(all_text)
            text_score = text_result.risk_score

        # ── Step 3: Steganography analysis
        stego_result = self.stego.analyze(image_path)
        stego_score = stego_result.get("overall_probability", 0.0)

        # ── Step 4: Combine
        overall_score = self._combine_scores(text_score, stego_score, hidden_detected)
        risk_level = risk_score_to_level(overall_score)
        explanation = self._build_explanation(text_result, stego_result, hidden_detected, overall_score)

        result = ImageDetectionResult(
            is_suspicious=overall_score >= 0.35,
            risk_level=risk_level,
            risk_score=overall_score,
            extracted_text=all_text.strip(),
            text_analysis=text_result.to_dict() if text_result else None,
            steganography_analysis=stego_result,
            hidden_text_detected=hidden_detected,
            explanation=explanation,
        )

        logger.info(f"Image analysis complete — risk: {risk_level} ({overall_score:.2f})")
        return result


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.detectors.image_detector <image_path>")
        sys.exit(1)

    detector = ImageInjectionDetector()
    result = detector.analyze(sys.argv[1])
    print(f"\nRisk: {result.risk_level.upper()} ({result.risk_score:.2f})")
    print(f"Explanation: {result.explanation}")
    if result.extracted_text:
        print(f"Extracted text:\n{result.extracted_text[:300]}")
