import os
import tempfile
import pytest
from PIL import Image, ImageDraw

from src.detectors.image_detector import ImageInjectionDetector

@pytest.fixture
def detector():
    return ImageInjectionDetector()

def blank_image(path, size=(200,100), color="white"):
    Image.new("RGB", size, color).save(path)
    return path

def text_image(path, text, color="black"):
    img = Image.new("RGB", (400,100), "white")
    ImageDraw.Draw(img).text((10,10), text, fill=color)
    img.save(path)
    return path

class TestFileHandling:
    def test_missing_file(self, detector):
        r = detector.analyze("/nonexistent/image.png")
        assert r.is_suspicious is False
        assert "not found" in r.explanation.lower()

    def test_blank_image(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            blank_image(path)
            r = detector.analyze(path)
            assert 0.0 <= r.risk_score <= 1.0
        finally:
            os.unlink(path)

    def test_returns_dict(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            blank_image(path)
            d = detector.analyze(path).to_dict()
            for key in ["is_suspicious","risk_level","risk_score","explanation"]:
                assert key in d
        finally:
            os.unlink(path)

class TestRiskScoreRange:
    def test_score_within_bounds(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            blank_image(path)
            r = detector.analyze(path)
            assert 0.0 <= r.risk_score <= 1.0
            assert r.risk_level in ("low","medium","high","critical")
        finally:
            os.unlink(path)

class TestSteganographyField:
    def test_stego_field_present(self, detector):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            blank_image(path)
            r = detector.analyze(path)
            assert r.steganography_analysis is not None
            assert "overall_probability" in r.steganography_analysis
        finally:
            os.unlink(path)
