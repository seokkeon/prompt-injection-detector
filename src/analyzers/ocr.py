"""
OCR-based text extraction from images.
Supports multiple extraction strategies to reveal hidden/low-contrast text.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple

from src.utils.logger import logger

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR will be limited.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed.")


class OCRAnalyzer:
    """
    Multi-strategy OCR engine for extracting text from images.
    Includes standard OCR plus techniques to expose hidden text.
    """

    def __init__(self, use_easyocr: bool = False):
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self._easyocr_reader = None

        if self.use_easyocr:
            logger.info("Initializing EasyOCR reader...")
            self._easyocr_reader = easyocr.Reader(["en"])

    # ── Basic extraction ─────────────────────────────────────────────────────

    def extract_text(self, image_path: str) -> str:
        """Standard OCR extraction."""
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            img = Image.open(image_path)
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""

    def extract_text_easyocr(self, image_path: str) -> str:
        """EasyOCR extraction (more robust for complex images)."""
        if not self._easyocr_reader:
            return ""
        try:
            results = self._easyocr_reader.readtext(image_path)
            return " ".join([text for _, text, conf in results if conf > 0.5])
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""

    # ── Hidden text detection strategies ─────────────────────────────────────

    def _threshold_variants(self, gray: np.ndarray) -> List[np.ndarray]:
        """Generate multiple threshold variants to reveal low-contrast text."""
        variants = []

        # Standard binary thresholds at different levels
        for threshold in [100, 127, 150, 200, 220]:
            _, t = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            variants.append(t)

        # Adaptive thresholds
        variants.append(cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ))
        variants.append(cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        ))

        # Inverted
        _, inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        variants.append(inv)

        return variants

    def _color_channel_extraction(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        """Extract text from individual RGB channels."""
        channels = cv2.split(img_bgr)
        return list(channels)

    def extract_hidden_text(self, image_path: str) -> Dict:
        """
        Attempt to extract hidden / low-contrast text using multiple strategies.

        Returns a dict with:
          - visible_text: text from standard OCR
          - hidden_texts: list of additional unique text found via hidden strategies
          - hidden_text_detected: bool
        """
        if not TESSERACT_AVAILABLE:
            return {
                "visible_text": "",
                "hidden_texts": [],
                "hidden_text_detected": False,
                "error": "pytesseract not available",
            }

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {"error": f"Could not read image: {image_path}"}

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Standard baseline
            visible_text = pytesseract.image_to_string(gray).strip()
            all_texts = {visible_text}

            # Run threshold variants
            for variant in self._threshold_variants(gray):
                t = pytesseract.image_to_string(variant).strip()
                if t:
                    all_texts.add(t)

            # Run color channel extraction
            for channel in self._color_channel_extraction(img_bgr):
                t = pytesseract.image_to_string(channel).strip()
                if t:
                    all_texts.add(t)

            # PIL-based enhancement
            pil_img = Image.open(image_path).convert("L")
            for factor in [2.0, 3.0]:
                enhanced = ImageEnhance.Contrast(pil_img).enhance(factor)
                t = pytesseract.image_to_string(enhanced).strip()
                if t:
                    all_texts.add(t)

            hidden_texts = [t for t in all_texts if t != visible_text and t]

            return {
                "visible_text": visible_text,
                "hidden_texts": hidden_texts,
                "hidden_text_detected": len(hidden_texts) > 0,
            }

        except Exception as e:
            logger.error(f"Hidden text extraction failed: {e}")
            return {"error": str(e), "hidden_text_detected": False}

    def get_all_text(self, image_path: str) -> str:
        """Convenience method: returns all text found (visible + hidden), deduplicated."""
        result = self.extract_hidden_text(image_path)
        parts = [result.get("visible_text", "")]
        parts.extend(result.get("hidden_texts", []))
        combined = "\n".join(filter(None, parts))
        return combined
