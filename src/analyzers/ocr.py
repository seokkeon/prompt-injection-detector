import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, List
from src.utils.logger import logger

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCRAnalyzer:
    def __init__(self, use_easyocr: bool = False):
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self._reader = None
        if self.use_easyocr:
            self._reader = easyocr.Reader(["en"])

    def extract_text(self, image_path: str) -> str:
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            return pytesseract.image_to_string(Image.open(image_path)).strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def extract_hidden_text(self, image_path: str) -> Dict:
        if not TESSERACT_AVAILABLE:
            return {"visible_text": "", "hidden_texts": [], "hidden_text_detected": False}
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {"error": f"Cannot read: {image_path}"}
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            visible = pytesseract.image_to_string(gray).strip()
            all_texts = {visible}

            for thresh in [100, 127, 150, 200, 220]:
                _, t = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                txt = pytesseract.image_to_string(t).strip()
                if txt:
                    all_texts.add(txt)

            for factor in [2.0, 3.0]:
                enhanced = ImageEnhance.Contrast(Image.open(image_path).convert("L")).enhance(factor)
                txt = pytesseract.image_to_string(enhanced).strip()
                if txt:
                    all_texts.add(txt)

            hidden = [t for t in all_texts if t != visible and t]
            return {"visible_text": visible, "hidden_texts": hidden, "hidden_text_detected": len(hidden) > 0}
        except Exception as e:
            return {"error": str(e), "hidden_text_detected": False}

    def get_all_text(self, image_path: str) -> str:
        result = self.extract_hidden_text(image_path)
        parts = [result.get("visible_text", "")] + result.get("hidden_texts", [])
        return "\n".join(filter(None, parts))
