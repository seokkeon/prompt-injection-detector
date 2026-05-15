"""
Image-based prompt injection detector.

All detection logic is contained in this single file:
  - Basic OCR (visible + multi-threshold hidden text)
  - Steganography (LSB, chi-square, entropy)
  - White-on-white / invisible text detection
  - QR code / barcode scanning
  - EXIF metadata scanning
  - Adversarial text detection (OCR engine divergence)
  - Homoglyph detection
"""

import os
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import chisquare

from src.detectors.unified_detector import UnifiedDetector as TextInjectionDetector  # merged
from src.utils.helpers import risk_score_to_level
from src.utils.logger import logger

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR disabled.")

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

# Unicode homoglyphs that look like ASCII but fool tokenizers
HOMOGLYPHS = {
    "а":"a","е":"e","о":"o","р":"p","с":"c","у":"y","х":"x","і":"i",
    "ｉ":"i","ｇ":"g","ｎ":"n","ｏ":"o","ｒ":"r","ｅ":"e","ｓ":"s",
    "\u200b":"ZWS","\u200c":"ZWNJ","\u200d":"ZWJ","\u00ad":"SHY","\ufeff":"BOM",
}


@dataclass
class ImageDetectionResult:
    is_suspicious:          bool
    risk_level:             str
    risk_score:             float
    extracted_text:         str
    text_analysis:          Optional[Dict] = None
    steganography_analysis: Optional[Dict] = None
    hidden_text_detected:   bool = False
    qr_codes_found:         List[Dict] = field(default_factory=list)
    exif_suspicious:        bool = False
    adversarial_detected:   bool = False
    invisible_text_found:   bool = False
    homoglyphs_found:       List[Dict] = field(default_factory=list)
    explanation:            str = ""

    def to_dict(self) -> Dict:
        return {
            "is_suspicious":        self.is_suspicious,
            "risk_level":           self.risk_level,
            "risk_score":           round(self.risk_score, 3),
            "extracted_text":       self.extracted_text[:500] if self.extracted_text else "",
            "text_analysis":        self.text_analysis,
            "steganography_analysis": self.steganography_analysis,
            "hidden_text_detected": self.hidden_text_detected,
            "qr_codes_found":       self.qr_codes_found,
            "exif_suspicious":      self.exif_suspicious,
            "adversarial_detected": self.adversarial_detected,
            "invisible_text_found": self.invisible_text_found,
            "homoglyphs_found":     self.homoglyphs_found,
            "explanation":          self.explanation,
        }


class ImageInjectionDetector:
    """
    Full image injection detection pipeline.
    All detection methods are on this class — no external analyzer files needed.
    """

    def __init__(self, use_easyocr: bool = False):
        self.text_detector   = TextInjectionDetector()
        self._easyocr_reader = None
        self._use_easyocr    = use_easyocr and EASYOCR_AVAILABLE

    def _easyocr(self):
        if self._easyocr_reader is None and self._use_easyocr:
            try:
                self._easyocr_reader = easyocr.Reader(["en"], verbose=False)
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
        return self._easyocr_reader

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _extract_text(self, image_path: str) -> Dict:
        """Extract visible + hidden text via multi-strategy OCR."""
        if not TESSERACT_AVAILABLE:
            return {"visible_text": "", "hidden_texts": [], "hidden_text_detected": False}
        try:
            img_bgr  = cv2.imread(image_path)
            if img_bgr is None:
                return {"error": "Cannot read image", "visible_text": "", "hidden_texts": []}
            gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            baseline = pytesseract.image_to_string(gray).strip()
            all_texts = {baseline}

            for thresh in range(0, 256, 5):
                _, t = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                txt = pytesseract.image_to_string(t).strip()
                if txt: all_texts.add(txt)

            for ch in cv2.split(img_bgr):
                txt = pytesseract.image_to_string(ch).strip()
                if txt: all_texts.add(txt)

            inv = cv2.bitwise_not(gray)
            all_texts.add(pytesseract.image_to_string(inv).strip())

            # Near-white isolation (white-on-white text)
            mask = cv2.inRange(img_bgr, np.array([200,200,200]), np.array([255,255,255]))
            isolated = cv2.cvtColor(cv2.bitwise_and(img_bgr, img_bgr, mask=mask), cv2.COLOR_BGR2GRAY)
            _, inv_iso = cv2.threshold(isolated, 200, 255, cv2.THRESH_BINARY_INV)
            all_texts.add(pytesseract.image_to_string(inv_iso).strip())

            # Gamma correction
            pil = Image.open(image_path).convert("L")
            for factor in [5.0, 10.0, 20.0]:
                enhanced = ImageEnhance.Contrast(pil).enhance(factor)
                all_texts.add(pytesseract.image_to_string(enhanced).strip())

            hidden = [t for t in all_texts if t and t != baseline]
            return {
                "visible_text":         baseline,
                "hidden_texts":         hidden,
                "hidden_text_detected": bool(hidden),
                "invisible_text_found": bool(hidden),
            }
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {"visible_text": "", "hidden_texts": [], "hidden_text_detected": False}

    # ── Steganography ─────────────────────────────────────────────────────────

    def _analyze_steganography(self, image_path: str) -> Dict:
        """LSB, chi-square, entropy, and noise analysis."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": "Cannot read image", "is_suspicious": False}

            flat = img.flatten().astype(np.uint8)

            # LSB analysis
            lsbs       = flat & 1
            deviation  = abs(np.sum(lsbs == 0) / len(lsbs) - 0.5)
            lsb_score  = max(0.0, 1.0 - deviation * 10)

            # Chi-square
            hist = np.bincount(flat, minlength=256)
            obs  = np.array([hist[i]   for i in range(0,255,2)], dtype=float)
            exp  = np.array([(hist[i]+hist[i+1])/2 for i in range(0,255,2)], dtype=float)
            mask = exp > 0
            _, p = chisquare(obs[mask], f_exp=exp[mask])
            chi_score = float(p)

            # Entropy
            probs   = hist / hist.sum()
            probs   = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            ent_score = max(0.0, (entropy/8.0 - 0.85) / 0.15)

            # Noise
            blurred    = cv2.GaussianBlur(img, (5,5), 0)
            noise_std  = float(np.std(img.astype(float) - blurred.astype(float)))
            noise_score = 0.6 if noise_std < 2.0 else 0.0

            overall = min(lsb_score*0.35 + chi_score*0.35 + ent_score*0.2 + noise_score*0.1, 1.0)
            return {
                "is_suspicious":    overall >= 0.6,
                "overall_probability": round(overall, 4),
                "lsb_score":        round(lsb_score, 4),
                "chi_score":        round(chi_score, 4),
                "entropy_score":    round(min(ent_score,1.0), 4),
            }
        except Exception as e:
            return {"error": str(e), "is_suspicious": False, "overall_probability": 0.0}

    # ── QR / barcode ──────────────────────────────────────────────────────────

    def _scan_codes(self, image_path: str) -> Dict:
        """Scan for QR codes and barcodes."""
        if not PYZBAR_AVAILABLE:
            return {"codes_found": [], "count": 0, "note": "pip install pyzbar"}
        try:
            img   = Image.open(image_path)
            codes = pyzbar_decode(img)
            if not codes:
                img_bgr = cv2.imread(image_path)
                if img_bgr is not None:
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    codes = pyzbar_decode(Image.fromarray(t))
            results = [{"type": c.type, "data": c.data.decode("utf-8", errors="replace")} for c in codes]
            return {"codes_found": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e), "codes_found": [], "count": 0}

    # ── EXIF metadata ─────────────────────────────────────────────────────────

    def _scan_exif(self, image_path: str) -> Dict:
        """Extract and flag suspicious EXIF metadata fields."""
        SUSPICIOUS = ["ImageDescription","Artist","Copyright","UserComment",
                      "Software","Comment","XPComment","XPAuthor","DocumentName"]
        try:
            img      = Image.open(image_path)
            metadata = {}

            if img.info:
                for k, v in img.info.items():
                    if isinstance(v, (str, bytes)):
                        val = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                        if val.strip(): metadata[str(k)] = val.strip()

            try:
                exif_data = img._getexif()
                if exif_data:
                    from PIL.ExifTags import TAGS
                    for tid, val in exif_data.items():
                        tag = TAGS.get(tid, str(tid))
                        if isinstance(val, bytes):
                            val = val.decode("utf-8", errors="replace").strip("\x00").strip()
                        if val and str(val).strip():
                            metadata[tag] = str(val).strip()
            except Exception:
                pass

            if hasattr(img, "text") and img.text:
                for k, v in img.text.items():
                    if v.strip(): metadata[f"PNG:{k}"] = v.strip()

            suspicious = {k: v for k, v in metadata.items()
                          if any(s.lower() in k.lower() for s in SUSPICIOUS) or len(str(v)) > 50}
            return {"all_metadata": metadata, "suspicious_fields": suspicious,
                    "has_suspicious": bool(suspicious)}
        except Exception as e:
            return {"error": str(e), "all_metadata": {}, "has_suspicious": False}

    # ── Adversarial text ──────────────────────────────────────────────────────

    def _detect_adversarial(self, image_path: str) -> Dict:
        """Compare Tesseract vs EasyOCR — high divergence = adversarial text suspected."""
        if not TESSERACT_AVAILABLE:
            return {"adversarial_detected": False, "adversarial_score": 0.0}
        tess, easy = "", ""
        try:
            tess = pytesseract.image_to_string(Image.open(image_path)).strip()
        except Exception:
            pass
        reader = self._easyocr()
        if reader:
            try:
                easy = " ".join([d[1] for d in reader.readtext(image_path) if d[2] > 0.3])
            except Exception:
                pass

        tess_set = set(tess.lower().split())
        easy_set = set(easy.lower().split())
        union    = tess_set | easy_set
        inter    = tess_set & easy_set
        divergence = 1.0 - (len(inter) / len(union)) if union else 0.0

        all_text       = tess + " " + easy
        homoglyphs     = [{"char": ch, "unicode": f"U+{ord(ch):04X}", "looks_like": HOMOGLYPHS[ch]}
                          for ch in all_text if ch in HOMOGLYPHS][:20]
        adv_score      = min(divergence * 0.7 + (0.3 if homoglyphs else 0.0), 1.0)

        return {
            "tesseract_text":      tess,
            "easyocr_text":        easy,
            "divergence_score":    round(divergence, 4),
            "homoglyphs_found":    homoglyphs,
            "adversarial_score":   round(adv_score, 4),
            "adversarial_detected": adv_score >= 0.5,
        }

    # ── Score combination ─────────────────────────────────────────────────────

    def _combine_scores(self, text_score, stego_score, hidden, qr_count, exif_sus, adv_score):
        score = (text_score * 0.40) + (stego_score * 0.20) + (adv_score * 0.15)
        if hidden:       score = min(score + 0.15, 1.0)
        if qr_count > 0: score = min(score + 0.20, 1.0)
        if exif_sus:     score = min(score + 0.10, 1.0)
        return min(score, 1.0)

    def _build_explanation(self, text_score, stego, adv, hidden, qr_count, exif_sus, score):
        parts = []
        if text_score >= 0.35:              parts.append(f"Injection text found ({text_score:.2f}).")
        if hidden:                          parts.append("Hidden/invisible text detected.")
        if qr_count > 0:                   parts.append(f"{qr_count} QR/barcode(s) found.")
        if exif_sus:                        parts.append("Suspicious EXIF metadata.")
        if stego.get("is_suspicious"):      parts.append(f"Steganography detected ({stego.get('overall_probability',0):.2f}).")
        if adv.get("adversarial_detected"): parts.append("Adversarial text suspected.")
        return (" ".join(parts) + f" Overall: {score:.2f}.") if parts else "No injection indicators found in image."

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyze(self, image_path: str) -> ImageDetectionResult:
        if not os.path.exists(image_path):
            return ImageDetectionResult(False, "low", 0.0, "", explanation=f"File not found: {image_path}")

        logger.info(f"Analyzing image: {image_path}")

        ocr          = self._extract_text(image_path)
        visible      = ocr.get("visible_text", "")
        hidden_texts = ocr.get("hidden_texts", [])
        all_text     = (visible + "\n" + "\n".join(hidden_texts)).strip()

        stego  = self._analyze_steganography(image_path)
        qr     = self._scan_codes(image_path)
        exif   = self._scan_exif(image_path)
        adv    = self._detect_adversarial(image_path)

        # Add QR content to text scan
        qr_texts = [f"[QR:{c.get('type','')}] {c.get('data','')}" for c in qr.get("codes_found", [])]
        full_text = (all_text + "\n" + "\n".join(qr_texts)).strip()

        text_result = self.text_detector.detect(full_text) if full_text else None
        text_score  = text_result.risk_score if text_result else 0.0

        hidden_detected = ocr.get("hidden_text_detected", False)
        qr_count        = qr.get("count", 0)
        exif_suspicious = exif.get("has_suspicious", False)
        adv_score       = adv.get("adversarial_score", 0.0)

        overall     = self._combine_scores(text_score, stego.get("overall_probability",0), hidden_detected, qr_count, exif_suspicious, adv_score)
        risk_level  = risk_score_to_level(overall)
        explanation = self._build_explanation(text_score, stego, adv, hidden_detected, qr_count, exif_suspicious, overall)

        logger.info(f"Image analysis done — {risk_level} ({overall:.2f})")
        return ImageDetectionResult(
            is_suspicious          = overall >= 0.35,
            risk_level             = risk_level,
            risk_score             = overall,
            extracted_text         = full_text,
            text_analysis          = text_result.to_dict() if text_result else None,
            steganography_analysis = stego,
            hidden_text_detected   = hidden_detected,
            qr_codes_found         = qr.get("codes_found", []),
            exif_suspicious        = exif_suspicious,
            adversarial_detected   = adv.get("adversarial_detected", False),
            invisible_text_found   = ocr.get("invisible_text_found", False),
            homoglyphs_found       = adv.get("homoglyphs_found", []),
            explanation            = explanation,
        )
