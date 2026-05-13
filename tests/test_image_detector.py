"""
Unit tests for ImageInjectionDetector.
Uses synthetic images generated with Pillow (no external files needed).
"""

import os
import tempfile
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.detectors.image_detector import ImageInjectionDetector


@pytest.fixture
def detector():
    return ImageInjectionDetector()


def make_blank_image(path: str, size=(200, 100), color="white") -> str:
    """Create a blank image and save to path."""
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def make_text_image(path: str, text: str, text_color="black", bg_color="white") -> str:
    """Create an image with visible text."""
    img = Image.new("RGB", (400, 100), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=text_color)
    img.save(path)
    return path


# ── File handling ─────────────────────────────────────────────────────────────

class TestFileHandling:

    def test_missing_file(self, detector):
        result = detector.analyze("/nonexistent/path/image.png")
        assert result.is_suspicious is False
        assert "not found" in result.explanation.lower()

    def test_blank_image(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_blank_image(path)
            result = detector.analyze(path)
            assert result.risk_score >= 0.0
            assert result.risk_score <= 1.0
        finally:
            os.unlink(path)

    def test_returns_dict(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_blank_image(path)
            result = detector.analyze(path)
            d = result.to_dict()
            assert "is_suspicious" in d
            assert "risk_level" in d
            assert "risk_score" in d
            assert "explanation" in d
        finally:
            os.unlink(path)


# ── Risk score range ──────────────────────────────────────────────────────────

class TestRiskScoreRange:

    def test_score_within_bounds(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_blank_image(path)
            result = detector.analyze(path)
            assert 0.0 <= result.risk_score <= 1.0
        finally:
            os.unlink(path)

    def test_risk_level_valid(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_blank_image(path)
            result = detector.analyze(path)
            assert result.risk_level in ("low", "medium", "high", "critical")
        finally:
            os.unlink(path)


# ── Text in image ─────────────────────────────────────────────────────────────

class TestTextInImage:

    def test_benign_text_image(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_text_image(path, "Hello, please summarize the report.")
            result = detector.analyze(path)
            assert result.risk_level in ("low", "medium")
        finally:
            os.unlink(path)

    def test_image_has_extracted_text_field(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_blank_image(path)
            result = detector.analyze(path)
            assert isinstance(result.extracted_text, str)
        finally:
            os.unlink(path)


# ── Steganography field ───────────────────────────────────────────────────────

class TestSteganographyField:

    def test_stego_field_present(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_blank_image(path)
            result = detector.analyze(path)
            assert result.steganography_analysis is not None
        finally:
            os.unlink(path)

    def test_stego_has_probability(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_blank_image(path)
            result = detector.analyze(path)
            assert "overall_probability" in result.steganography_analysis
        finally:
            os.unlink(path)
