# src/anpr/ocr_reader.py
"""
EasyOCR-based license plate text extractor.

Performance design:
  • Lazy pass execution  – only runs the next preprocessing pass if the
    previous one didn't find a format-compliant result.
  • Image-hash cache    – if the same plate crop appears in multiple frames,
    the OCR result is returned instantly without calling EasyOCR again.
  • Pass order          – CLAHE (best quality) → raw crop (fast fallback).
    Threshold pass is only tried when both others fail.

Character-correction (reference: computervisioneng ANPR sample):
  O↔0, I↔1, J↔3, A↔4, G↔6, S↔5
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import string
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Minimum parameters ───────────────────────────────────────────────────────
_MIN_CONFIDENCE  = 0.35
_MIN_PLATE_W_OCR = 200     # upscale plates narrower than this

# ── Character-correction tables ──────────────────────────────────────────────
_CHAR_TO_INT = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
_INT_TO_CHAR = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}

# ── Result cache size (number of unique plate crops remembered) ───────────────
_CACHE_MAXSIZE = 512


def license_complies_format(text: str) -> bool:
    """
    True if text matches a known plate format:
      - 7-char international  : AB12CDE
      - 10-char Indian format : MH12AB1234  (state + district + series + number)
      - 9/10-char flexible    : allows spaces removed
    """
    clean = text.replace(" ", "").upper()

    # Indian format: 2 letters + 2 digits + 1-2 letters + 4 digits  (9 or 10 chars)
    if len(clean) in (9, 10):
        if (clean[:2].isalpha() and clean[2:4].isdigit()
                and clean[4:-4].isalpha() and clean[-4:].isdigit()):
            return True

    # International 7-char: AB12CDE
    if len(clean) == 7:
        checks = [
            (clean[0], string.ascii_uppercase, _INT_TO_CHAR),
            (clean[1], string.ascii_uppercase, _INT_TO_CHAR),
            (clean[2], "0123456789",           _CHAR_TO_INT),
            (clean[3], "0123456789",           _CHAR_TO_INT),
            (clean[4], string.ascii_uppercase, _INT_TO_CHAR),
            (clean[5], string.ascii_uppercase, _INT_TO_CHAR),
            (clean[6], string.ascii_uppercase, _INT_TO_CHAR),
        ]
        return all(ch in valid or ch in alt for (ch, valid, alt) in checks)

    return False


def format_license(text: str) -> str:
    """
    Apply character-correction to a plate string.
    Supports both 7-char international and 9/10-char Indian formats.
    """
    clean = text.replace(" ", "").upper()

    # Indian format: positions 0-1=alpha, 2-3=digit, 4-5=alpha, 6-9=digit
    if len(clean) in (9, 10) and clean[:2].isalpha():
        result = []
        for i, ch in enumerate(clean):
            if i < 2:         # State code (letters)
                result.append(_INT_TO_CHAR.get(ch, ch))
            elif i < 4:       # District (digits)
                result.append(_CHAR_TO_INT.get(ch, ch))
            elif i < len(clean) - 4:  # Series (letters)
                result.append(_INT_TO_CHAR.get(ch, ch))
            else:             # Registration number (digits)
                result.append(_CHAR_TO_INT.get(ch, ch))
        return "".join(result)

    # 7-char international
    if len(clean) == 7:
        mapping = {
            0: _INT_TO_CHAR, 1: _INT_TO_CHAR,
            2: _CHAR_TO_INT, 3: _CHAR_TO_INT,
            4: _INT_TO_CHAR, 5: _INT_TO_CHAR, 6: _INT_TO_CHAR,
        }
        return "".join(mapping[i].get(ch, ch) for i, ch in enumerate(clean))

    return clean  # no correction for unknown formats


def _img_hash(img: np.ndarray) -> str:
    """Fast perceptual hash for small images (used as cache key)."""
    # Resize to 16×8 grayscale → md5 of pixel bytes — very fast
    thumb = cv2.resize(img, (16, 8), interpolation=cv2.INTER_AREA)
    if thumb.ndim == 3:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(thumb.tobytes()).hexdigest()


class OCRReader:
    """
    Extract text from a license plate crop using EasyOCR.

    Parameters
    ----------
    lang : str          Language code (default "en").
    min_confidence      Minimum per-detection confidence.
    gpu : bool          Enable GPU for EasyOCR.
    cache_size : int    Max unique plate results cached in memory.
    """

    def __init__(
        self,
        lang: str = "en",
        min_confidence: float = _MIN_CONFIDENCE,
        gpu: bool = False,
        cache_size: int = _CACHE_MAXSIZE,
    ) -> None:
        import easyocr
        use_gpu = bool(gpu)
        if use_gpu:
            try:
                import torch

                use_gpu = bool(torch.cuda.is_available())
                if not use_gpu:
                    logger.warning("EasyOCR GPU requested but CUDA is unavailable; falling back to CPU.")
            except Exception as exc:
                logger.warning("EasyOCR GPU check failed (%s); falling back to CPU.", exc)
                use_gpu = False

        self._reader        = easyocr.Reader([lang], gpu=use_gpu)
        self.min_confidence = min_confidence
        self._cache: dict[str, str] = {}   # img_hash → plate text
        self._cache_size            = cache_size
        logger.info(
            "EasyOCR initialised (lang=%s, gpu=%s, min_conf=%.2f, cache=%d)",
            lang, use_gpu, min_confidence, cache_size,
        )

    # ------------------------------------------------------------------
    # Private: preprocessing
    # ------------------------------------------------------------------

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale if width < _MIN_PLATE_W_OCR."""
        h, w = img.shape[:2]
        if w < _MIN_PLATE_W_OCR:
            scale = _MIN_PLATE_W_OCR / max(w, 1)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        return img

    def _preprocess_clahe(self, img: np.ndarray) -> np.ndarray:
        img = self._upscale(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    def _preprocess_thresh(self, img: np.ndarray) -> np.ndarray:
        img = self._upscale(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Private: EasyOCR call
    # ------------------------------------------------------------------

    def _ocr_image(self, img: np.ndarray) -> list[tuple[str, float]]:
        """Run EasyOCR, return (cleaned_text, conf) sorted best-first."""
        try:
            detections = self._reader.readtext(img)
        except Exception as exc:
            logger.debug("EasyOCR readtext error: %s", exc)
            return []

        result = []
        for (_, text, score) in detections:
            if score < self.min_confidence:
                continue
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper().strip())
            if cleaned:
                result.append((cleaned, score))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _pick_best(self, candidates: list[tuple[str, float]]) -> tuple[str, bool]:
        """
        From a list of (text, conf), return (best_text, is_format_compliant).
        Prefers format-compliant results; falls back to longest alphanumeric.
        """
        best_fallback = ("", 0.0)
        for text, score in candidates:
            if license_complies_format(text):
                return format_license(text), True
            if len(text) >= 4 and score > best_fallback[1]:
                best_fallback = (text, score)
        return best_fallback[0], False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, plate_img: np.ndarray) -> str:
        """
        Extract plate text from a crop using lazy 3-pass OCR.

        Strategy (stops as soon as a format-compliant result is found):
          Pass 1 → CLAHE enhanced
          Pass 2 → Raw crop (fast, good for high-contrast plates)
          Pass 3 → Binary threshold (only tried if passes 1+2 both fail)

        Returns: plate text string, or "" if nothing found.
        """
        if plate_img is None or plate_img.size == 0:
            return ""

        # ── Cache lookup ────────────────────────────────────────────────
        img_key = _img_hash(plate_img)
        if img_key in self._cache:
            logger.debug("OCR cache hit: '%s'", self._cache[img_key])
            return self._cache[img_key]

        best_fallback = ""

        # ── Pass 1: CLAHE (most reliable for varied lighting) ───────────
        candidates_1 = self._ocr_image(self._preprocess_clahe(plate_img))
        result, compliant = self._pick_best(candidates_1)
        if compliant:
            logger.info("OCR [CLAHE] format-match → '%s'", result)
            return self._store_cache(img_key, result)
        if result:
            best_fallback = result

        # ── Pass 2: Raw crop (fast; works well on already-sharp plates) ─
        # Upscale if needed before passing raw
        raw_img      = self._upscale(plate_img)
        candidates_2 = self._ocr_image(raw_img)
        result, compliant = self._pick_best(candidates_2)
        if compliant:
            logger.info("OCR [RAW] format-match → '%s'", result)
            return self._store_cache(img_key, result)
        if result and len(result) > len(best_fallback):
            best_fallback = result

        if os.getenv("OCR_ENABLE_THRESHOLD_PASS", "0").strip().lower() not in {"1", "true", "yes"}:
            if best_fallback:
                logger.info("OCR fallback (no format match) → '%s'", best_fallback)
            return self._store_cache(img_key, best_fallback)

        # ── Pass 3: Binary threshold (only if both above failed) ────────
        candidates_3 = self._ocr_image(self._preprocess_thresh(plate_img))
        result, compliant = self._pick_best(candidates_3)
        if compliant:
            logger.info("OCR [THRESH] format-match → '%s'", result)
            return self._store_cache(img_key, result)
        if result and len(result) > len(best_fallback):
            best_fallback = result

        if best_fallback:
            logger.info("OCR fallback (no format match) → '%s'", best_fallback)
        return self._store_cache(img_key, best_fallback)

    def _store_cache(self, key: str, value: str) -> str:
        """Store result in the LRU-style cache (evict oldest if full)."""
        if len(self._cache) >= self._cache_size:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value
        return value