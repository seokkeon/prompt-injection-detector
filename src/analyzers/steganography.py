"""
Steganography detection module.
Uses statistical analysis (LSB, chi-square, entropy) to detect
hidden data embedded in images.
"""

import os
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from scipy.stats import chisquare

from src.utils.logger import logger


class SteganographyAnalyzer:
    """
    Detects hidden data in images via statistical analysis.

    Methods:
      - LSB (Least Significant Bit) analysis
      - Chi-square attack
      - Entropy analysis
      - RS (Regular-Singular) analysis (simplified)
    """

    def __init__(self, threshold: float = 0.6):
        """
        Args:
            threshold: Probability above which an image is flagged as suspicious.
        """
        self.threshold = threshold

    # ── LSB Analysis ──────────────────────────────────────────────────────────

    def _lsb_analysis(self, img: np.ndarray) -> Dict:
        """
        Analyze the Least Significant Bits of pixel values.
        Random-looking LSBs suggest data is hidden.
        """
        flat = img.flatten().astype(np.uint8)
        lsbs = flat & 1  # Extract LSB of every byte

        # Expected: ~50% 0s and 50% 1s if data is hidden
        zero_ratio = np.sum(lsbs == 0) / len(lsbs)
        one_ratio = np.sum(lsbs == 1) / len(lsbs)

        # Deviation from 50/50 indicates no steganography
        # Perfect 50/50 is suspicious
        deviation = abs(zero_ratio - 0.5)
        lsb_suspicion = max(0.0, 1.0 - (deviation * 10))

        return {
            "zero_ratio": round(float(zero_ratio), 4),
            "one_ratio": round(float(one_ratio), 4),
            "deviation_from_random": round(float(deviation), 4),
            "suspicion_score": round(float(lsb_suspicion), 4),
        }

    # ── Chi-Square Attack ─────────────────────────────────────────────────────

    def _chi_square_attack(self, img: np.ndarray) -> Dict:
        """
        Chi-square statistical test on pixel value pairs.
        Steganographic embedding tends to equalize frequencies of value pairs (2k, 2k+1).
        """
        flat = img.flatten().astype(np.uint8)
        hist = np.bincount(flat, minlength=256)

        # Compare pairs (0,1), (2,3), ..., (254,255)
        observed = []
        expected = []
        for i in range(0, 255, 2):
            pair_sum = hist[i] + hist[i + 1]
            observed.append(hist[i])
            expected.append(pair_sum / 2)

        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)

        # Avoid division by zero
        mask = expected > 0
        observed = observed[mask]
        expected = expected[mask]

        if len(observed) < 2:
            return {"chi_square_p_value": 1.0, "suspicion_score": 0.0}

        _, p_value = chisquare(observed, f_exp=expected)

        # Low p-value = pixel pairs are NOT equalized = likely clean
        # High p-value (near 1.0) = pairs are very equal = suspicious
        suspicion = float(p_value)

        return {
            "chi_square_p_value": round(float(p_value), 6),
            "suspicion_score": round(suspicion, 4),
        }

    # ── Entropy Analysis ──────────────────────────────────────────────────────

    def _entropy_analysis(self, img: np.ndarray) -> Dict:
        """
        Compute Shannon entropy of the image.
        Very high entropy can indicate hidden compressed/encrypted data.
        """
        flat = img.flatten().astype(np.uint8)
        hist = np.bincount(flat, minlength=256)
        probabilities = hist / hist.sum()
        probabilities = probabilities[probabilities > 0]

        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = 8.0  # Maximum for 8-bit values

        # Normalize; very high entropy = more suspicious
        normalized = entropy / max_entropy
        suspicion = max(0.0, (normalized - 0.85) / 0.15)  # Flags above 85% max

        return {
            "entropy": round(float(entropy), 4),
            "normalized_entropy": round(float(normalized), 4),
            "suspicion_score": round(float(min(suspicion, 1.0)), 4),
        }

    # ── Noise Floor Analysis ──────────────────────────────────────────────────

    def _noise_analysis(self, img: np.ndarray) -> Dict:
        """
        Compare image noise level to a Gaussian-filtered version.
        Unusually regular noise patterns may indicate LSB embedding.
        """
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        noise = img.astype(float) - blurred.astype(float)

        noise_std = float(np.std(noise))
        noise_mean = float(np.mean(np.abs(noise)))

        # Regular, low-variance noise is suspicious
        suspicion = 0.0
        if noise_std < 2.0 and noise_mean < 1.0:
            suspicion = 0.6  # Suspiciously smooth noise = possible embedding

        return {
            "noise_std": round(noise_std, 4),
            "noise_mean": round(noise_mean, 4),
            "suspicion_score": round(suspicion, 4),
        }

    # ── Main Analysis ─────────────────────────────────────────────────────────

    def analyze(self, image_path: str) -> Dict:
        """
        Run full steganography analysis on an image.

        Returns:
            Dict with per-method results and overall probability + verdict.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": f"Cannot read image: {image_path}"}

            lsb = self._lsb_analysis(img)
            chi = self._chi_square_attack(img)
            ent = self._entropy_analysis(img)
            noise = self._noise_analysis(img)

            scores = [
                lsb["suspicion_score"],
                chi["suspicion_score"],
                ent["suspicion_score"],
                noise["suspicion_score"],
            ]
            # Weighted average: LSB and chi are most reliable
            weights = [0.35, 0.35, 0.2, 0.1]
            overall_score = float(np.average(scores, weights=weights))

            is_suspicious = overall_score >= self.threshold

            return {
                "is_suspicious": is_suspicious,
                "overall_probability": round(overall_score, 4),
                "threshold": self.threshold,
                "lsb_analysis": lsb,
                "chi_square_analysis": chi,
                "entropy_analysis": ent,
                "noise_analysis": noise,
                "verdict": "suspicious" if is_suspicious else "clean",
            }

        except Exception as e:
            logger.error(f"Steganography analysis failed: {e}")
            return {"error": str(e), "is_suspicious": False}
