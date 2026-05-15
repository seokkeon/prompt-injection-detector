"""
Advanced image analysis module.

Detects:
  1. White-on-white / invisible text (fine threshold sweep + channel isolation)
  2. QR codes and barcodes (pyzbar)
  3. EXIF metadata injection
  4. Adversarial text (OCR engine divergence)
  5. Low-contrast and near-invisible overlays
"""

import os
import struct
import zlib
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.utils.logger import logger

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logger.warning("pyzbar not installed. QR/barcode scanning disabled. Run: pip install pyzbar")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class AdvancedImageAnalyzer:
    """
    Advanced image analysis for prompt injection detection.
    Covers invisible text, QR codes, EXIF metadata, and adversarial content.
    """

    def __init__(self):
        self._easyocr_reader = None

    def _get_easyocr(self):
        if self._easyocr_reader is None and EASYOCR_AVAILABLE:
            try:
                self._easyocr_reader = easyocr.Reader(["en"], verbose=False)
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
        return self._easyocr_reader

    # ── 1. White-on-white / invisible text ───────────────────────────────────

    def detect_invisible_text(self, image_path: str) -> Dict:
        """
        Fine-grained threshold sweep to reveal text that is invisible
        to the human eye but readable by vision models.

        Strategies:
          - Full threshold sweep (0–255 in steps of 5)
          - Per-channel isolation (R, G, B separately)
          - Inverted image OCR
          - Gamma correction variants
          - Near-white pixel isolation
        """
        if not TESSERACT_AVAILABLE:
            return {"texts": [], "method_hits": [], "invisible_text_found": False}

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {"error": "Cannot read image", "invisible_text_found": False}

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            found_texts = {}  # text -> method

            def add(text: str, method: str):
                t = text.strip()
                if len(t) >= 3:
                    found_texts[t] = method

            # Baseline
            baseline = pytesseract.image_to_string(gray).strip()
            add(baseline, "baseline")

            # Fine threshold sweep
            for thresh in range(0, 256, 5):
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                t = pytesseract.image_to_string(binary).strip()
                if t and t != baseline:
                    add(t, f"threshold_{thresh}")

            # Inverted
            inv = cv2.bitwise_not(gray)
            add(pytesseract.image_to_string(inv).strip(), "inverted")

            # Per-channel isolation
            for idx, ch_name in enumerate(["blue", "green", "red"]):
                ch = img_bgr[:, :, idx]
                add(pytesseract.image_to_string(ch).strip(), f"channel_{ch_name}")

            # Near-white pixel isolation (catches white-on-white)
            lower = np.array([200, 200, 200])
            upper = np.array([255, 255, 255])
            mask = cv2.inRange(img_bgr, lower, upper)
            isolated = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
            isolated_gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
            _, inv_isolated = cv2.threshold(isolated_gray, 200, 255, cv2.THRESH_BINARY_INV)
            add(pytesseract.image_to_string(inv_isolated).strip(), "near_white_isolation")

            # Gamma correction (reveals very light text)
            for gamma in [0.3, 0.5, 2.0, 3.0]:
                lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                gamma_img = cv2.LUT(gray, lut)
                t = pytesseract.image_to_string(gamma_img).strip()
                if t and t != baseline:
                    add(t, f"gamma_{gamma}")

            # PIL contrast boost
            pil_img = Image.open(image_path).convert("L")
            for factor in [5.0, 10.0, 20.0]:
                enhanced = ImageEnhance.Contrast(pil_img).enhance(factor)
                t = pytesseract.image_to_string(enhanced).strip()
                if t and t != baseline:
                    add(t, f"contrast_{factor}x")

            # Remove baseline from "hidden" results
            hidden = {k: v for k, v in found_texts.items() if k != baseline}

            return {
                "baseline_text":       baseline,
                "hidden_texts":        list(hidden.keys()),
                "method_hits":         [{"text": k, "method": v} for k, v in hidden.items()],
                "invisible_text_found": len(hidden) > 0,
            }

        except Exception as e:
            logger.error(f"Invisible text detection failed: {e}")
            return {"error": str(e), "invisible_text_found": False}

    # ── 2. QR code / barcode scanning ────────────────────────────────────────

    def scan_codes(self, image_path: str) -> Dict:
        """
        Scan for QR codes, barcodes, and Data Matrix codes.
        Attackers can embed injection instructions inside QR codes
        that are invisible at small sizes but decoded by AI vision.
        """
        if not PYZBAR_AVAILABLE:
            return {
                "codes_found": [],
                "count": 0,
                "note": "Install pyzbar: pip install pyzbar",
            }

        try:
            img = Image.open(image_path)
            codes = pyzbar_decode(img)

            # Also try with preprocessing for low-quality codes
            if not codes:
                img_bgr = cv2.imread(image_path)
                if img_bgr is not None:
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    codes = pyzbar_decode(Image.fromarray(thresh))

            results = []
            for code in codes:
                try:
                    data = code.data.decode("utf-8", errors="replace")
                except Exception:
                    data = str(code.data)

                results.append({
                    "type":     code.type,
                    "data":     data,
                    "rect":     {"left": code.rect.left, "top": code.rect.top,
                                 "width": code.rect.width, "height": code.rect.height},
                })

            return {"codes_found": results, "count": len(results)}

        except Exception as e:
            logger.error(f"QR/barcode scan failed: {e}")
            return {"error": str(e), "codes_found": [], "count": 0}

    # ── 3. EXIF metadata scanning ─────────────────────────────────────────────

    def scan_exif(self, image_path: str) -> Dict:
        """
        Extract and scan ALL EXIF/metadata fields for injected instructions.
        Checks: comment, description, artist, copyright, user comment,
                software, make, model, GPS info, and all custom fields.
        """
        SUSPICIOUS_FIELDS = [
            "ImageDescription", "Artist", "Copyright", "UserComment",
            "Software", "Make", "Model", "Comment", "XPComment",
            "XPAuthor", "XPTitle", "XPSubject", "XPKeywords",
            "DocumentName", "PageName", "HostComputer",
        ]

        try:
            img = Image.open(image_path)
            metadata = {}

            # PIL info dict (PNG chunks, TIFF tags, etc.)
            if img.info:
                for k, v in img.info.items():
                    if isinstance(v, (str, bytes)):
                        val = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                        if val.strip():
                            metadata[str(k)] = val.strip()

            # EXIF data
            try:
                exif_data = img._getexif()
                if exif_data:
                    from PIL.ExifTags import TAGS
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8", errors="replace").strip("\x00").strip()
                            except Exception:
                                value = value.hex()
                        if value and str(value).strip():
                            metadata[tag_name] = str(value).strip()
            except (AttributeError, Exception):
                pass

            # PNG text chunks (iTXt, tEXt, zTXt)
            if hasattr(img, "text") and img.text:
                for k, v in img.text.items():
                    if v.strip():
                        metadata[f"PNG:{k}"] = v.strip()

            # Flag suspicious fields
            suspicious = {k: v for k, v in metadata.items()
                          if any(sf.lower() in k.lower() for sf in SUSPICIOUS_FIELDS)
                          or len(str(v)) > 50}

            return {
                "all_metadata":     metadata,
                "suspicious_fields": suspicious,
                "metadata_count":   len(metadata),
                "has_suspicious":   len(suspicious) > 0,
            }

        except Exception as e:
            logger.error(f"EXIF scan failed: {e}")
            return {"error": str(e), "all_metadata": {}, "has_suspicious": False}

    # ── 4. Adversarial text detection ────────────────────────────────────────

    def detect_adversarial_text(self, image_path: str) -> Dict:
        """
        Detects text designed to fool one OCR engine but be read by another
        (e.g. a vision model). Compares Tesseract vs EasyOCR outputs.

        High divergence between engines = adversarial text suspected.
        """
        if not TESSERACT_AVAILABLE:
            return {"available": False, "note": "pytesseract not installed"}

        results = {}

        # Tesseract
        try:
            img = Image.open(image_path)
            results["tesseract"] = pytesseract.image_to_string(img).strip()
        except Exception as e:
            results["tesseract"] = ""
            logger.warning(f"Tesseract failed: {e}")

        # EasyOCR
        reader = self._get_easyocr()
        if reader:
            try:
                detections = reader.readtext(image_path)
                results["easyocr"] = " ".join([d[1] for d in detections if d[2] > 0.3])
            except Exception as e:
                results["easyocr"] = ""
                logger.warning(f"EasyOCR failed: {e}")

        # Measure divergence
        tess = set(results.get("tesseract", "").lower().split())
        easy = set(results.get("easyocr", "").lower().split())

        if tess and easy:
            union = tess | easy
            inter = tess & easy
            divergence = 1.0 - (len(inter) / len(union)) if union else 0.0
        else:
            divergence = 0.0

        # Also check for unicode homoglyphs (visually similar chars)
        all_text = " ".join(results.values())
        homoglyph_suspects = _find_homoglyphs(all_text)

        adversarial_score = min(divergence * 0.7 + (0.3 if homoglyph_suspects else 0.0), 1.0)

        return {
            "tesseract_text":      results.get("tesseract", ""),
            "easyocr_text":        results.get("easyocr", ""),
            "divergence_score":    round(divergence, 4),
            "homoglyph_suspects":  homoglyph_suspects,
            "adversarial_score":   round(adversarial_score, 4),
            "adversarial_detected": adversarial_score >= 0.5,
        }

    def analyze_all(self, image_path: str) -> Dict:
        """Run all advanced checks and return combined results."""
        logger.info(f"Running advanced image analysis: {image_path}")
        return {
            "invisible_text": self.detect_invisible_text(image_path),
            "qr_barcodes":    self.scan_codes(image_path),
            "exif_metadata":  self.scan_exif(image_path),
            "adversarial":    self.detect_adversarial_text(image_path),
        }


# ── Homoglyph detection helper ────────────────────────────────────────────────

# Common unicode chars that look identical to ASCII but fool tokenizers
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ѕ": "s", "ԁ": "d",
    "ｉ": "i", "ｇ": "g", "ｎ": "n", "ｏ": "o", "ｒ": "r",
    "ｅ": "e", "ｓ": "s", "ｔ": "t", "ｕ": "u",
    "\u200b": "ZWS", "\u200c": "ZWNJ", "\u200d": "ZWJ",
    "\u00ad": "SHY", "\ufeff": "BOM",
}

def _find_homoglyphs(text: str) -> List[Dict]:
    found = []
    for i, ch in enumerate(text):
        if ch in HOMOGLYPHS:
            found.append({
                "char":     ch,
                "unicode":  f"U+{ord(ch):04X}",
                "looks_like": HOMOGLYPHS[ch],
                "position": i,
            })
    return found[:20]  # cap at 20
