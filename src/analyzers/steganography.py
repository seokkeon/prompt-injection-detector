import numpy as np
import cv2
from typing import Dict
from scipy.stats import chisquare
from src.utils.logger import logger


class SteganographyAnalyzer:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def _lsb_analysis(self, img: np.ndarray) -> Dict:
        flat = img.flatten().astype(np.uint8)
        lsbs = flat & 1
        zero_ratio = np.sum(lsbs == 0) / len(lsbs)
        deviation = abs(zero_ratio - 0.5)
        suspicion = max(0.0, 1.0 - (deviation * 10))
        return {"zero_ratio": round(float(zero_ratio), 4), "deviation": round(float(deviation), 4), "suspicion_score": round(float(suspicion), 4)}

    def _chi_square_attack(self, img: np.ndarray) -> Dict:
        flat = img.flatten().astype(np.uint8)
        hist = np.bincount(flat, minlength=256)
        observed, expected = [], []
        for i in range(0, 255, 2):
            pair_sum = hist[i] + hist[i + 1]
            observed.append(hist[i])
            expected.append(pair_sum / 2)
        observed, expected = np.array(observed, dtype=float), np.array(expected, dtype=float)
        mask = expected > 0
        if mask.sum() < 2:
            return {"chi_square_p_value": 1.0, "suspicion_score": 0.0}
        _, p_value = chisquare(observed[mask], f_exp=expected[mask])
        return {"chi_square_p_value": round(float(p_value), 6), "suspicion_score": round(float(p_value), 4)}

    def _entropy_analysis(self, img: np.ndarray) -> Dict:
        flat = img.flatten().astype(np.uint8)
        hist = np.bincount(flat, minlength=256)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        normalized = entropy / 8.0
        suspicion = max(0.0, (normalized - 0.85) / 0.15)
        return {"entropy": round(float(entropy), 4), "normalized": round(float(normalized), 4), "suspicion_score": round(float(min(suspicion, 1.0)), 4)}

    def _noise_analysis(self, img: np.ndarray) -> Dict:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        noise = img.astype(float) - blurred.astype(float)
        noise_std = float(np.std(noise))
        suspicion = 0.6 if noise_std < 2.0 else 0.0
        return {"noise_std": round(noise_std, 4), "suspicion_score": round(suspicion, 4)}

    def analyze(self, image_path: str) -> Dict:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": f"Cannot read: {image_path}", "is_suspicious": False}
            lsb   = self._lsb_analysis(img)
            chi   = self._chi_square_attack(img)
            ent   = self._entropy_analysis(img)
            noise = self._noise_analysis(img)
            scores  = [lsb["suspicion_score"], chi["suspicion_score"], ent["suspicion_score"], noise["suspicion_score"]]
            weights = [0.35, 0.35, 0.2, 0.1]
            overall = float(np.average(scores, weights=weights))
            return {
                "is_suspicious": overall >= self.threshold,
                "overall_probability": round(overall, 4),
                "threshold": self.threshold,
                "lsb_analysis": lsb,
                "chi_square_analysis": chi,
                "entropy_analysis": ent,
                "noise_analysis": noise,
                "verdict": "suspicious" if overall >= self.threshold else "clean",
            }
        except Exception as e:
            logger.error(f"Steganography error: {e}")
            return {"error": str(e), "is_suspicious": False}
