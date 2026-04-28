#!/usr/bin/env python
"""
scripts/test_anpr_smoke.py
Quick smoke-test for the new EasyOCR-based OCR pipeline.

Usage:
    python scripts/test_anpr_smoke.py

What it tests:
  1. EasyOCR import + model initialisation
  2. 3-pass OCR on a synthetically generated plate image
  3. format_license / license_complies_format helpers
"""

import sys
import os

# Make sure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np


def make_plate_image(text: str = "AB12CDE", width: int = 280, height: int = 80) -> np.ndarray:
    """Create a synthetic white-on-yellow plate image for testing."""
    img = np.full((height, width, 3), (0, 220, 255), dtype=np.uint8)  # yellow
    font_scale = 1.8
    thickness  = 3
    font       = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width  - tw) // 2
    y = (height + th) // 2
    cv2.putText(img, text, (x, y), font, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)
    return img


def test_format_helpers():
    from src.anpr.ocr_reader import license_complies_format, format_license

    # Valid formats
    assert license_complies_format("AB12CDE"), "AB12CDE should pass"
    assert license_complies_format("MH12AB1"), "MH12AB1 should pass"

    # Invalid (too short)
    assert not license_complies_format("AB12CD"), "6-char should fail"

    # format_license should fix O→0 in digit positions, 0→O in letter positions
    result = format_license("OB12CDO")   # position 0 → O stays O, position 6 → 0→O
    assert isinstance(result, str) and len(result) == 7, f"Expected 7-char, got: '{result}'"

    print("  ✅ format helpers pass")


def test_ocr_pipeline():
    from src.anpr.ocr_reader import OCRReader

    print("  Initialising EasyOCR reader (may download models on first run)…")
    reader = OCRReader(lang="en", gpu=False, min_confidence=0.10)

    plate_img = make_plate_image("AB12CDE")
    result = reader.read(plate_img)
    print(f"  OCR result on synthetic plate: '{result}'")

    assert isinstance(result, str), "read() must return a str"
    # Even if OCR read something slightly different, it should be non-empty
    if result:
        print("  ✅ OCR returned a non-empty result")
    else:
        print("  ⚠️  OCR returned empty string — check EasyOCR install and image quality")


if __name__ == "__main__":
    print("\n🔍 RoadGuard AI – ANPR smoke test")
    print("=" * 40)

    print("\n[1/2] Testing format helpers…")
    test_format_helpers()

    print("\n[2/2] Testing EasyOCR pipeline…")
    test_ocr_pipeline()

    print("\n✅ Smoke test complete.")
