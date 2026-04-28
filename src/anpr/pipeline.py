# src/anpr/pipeline.py
"""ANPR pipeline: plate detection → deblurring → OCR.

GPU Acceleration:
  • PlateDetector uses CUDA + FP16 when available.
  • Deblurrer uses TensorFlow GPU with memory growth.
  • OCR Reader uses GPU-accelerated EasyOCR.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .plate_detector import PlateDetector
from .deblurrer      import Deblurrer
from .ocr_reader     import OCRReader

logger = logging.getLogger(__name__)

_MIN_PLATE_W = 20   # pixels – lowered to catch small/distant plates
_MIN_PLATE_H = 8    # pixels – lowered to catch wide/squashed plates


class ANPRPipeline:
    """
    End-to-end ANPR: detect plates → optionally deblur → OCR.

    Parameters
    ----------
    plate_model_path : str
        Path to the plate-detection YOLOv8 .pt model.
    deblur_weights : str
        Path to fpn_inception.h5.
    ocr_lang : str
        PaddleOCR language code.
    device : str
        Force device for plate detector ('cuda:0', 'cpu', '' = auto).
    half : bool | None
        FP16 for plate detector. None = auto.
    imgsz : int
        Inference image size for plate detector.
    deblur_gpu : bool
        Use GPU for DeblurGAN model.
    """

    def __init__(
        self,
        plate_model_path: str,
        deblur_weights: str = "",
        ocr_lang: str = "en",
        plate_conf: float = 0.25,
        ocr_gpu: bool = True,
        device: str = "",
        half: bool | None = None,
        imgsz: int = 640,
        deblur_gpu: bool = True,
    ) -> None:
        self.detector   = PlateDetector(
            plate_model_path,
            conf_threshold=plate_conf,
            device=device or None,
            half=half,
            imgsz=imgsz,
        )
        self.deblurrer  = Deblurrer(deblur_weights, use_gpu=deblur_gpu) if deblur_weights else None
        self.reader     = OCRReader(ocr_lang, gpu=ocr_gpu)
        self.enable_fullframe_fallback = os.getenv("ANPR_FULLFRAME_FALLBACK", "0") == "1"

    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Run the full pipeline on a single frame.

        Returns
        -------
        list of dict:
            'bbox'      : [x1,y1,x2,y2]
            'text'      : str (plate number, may be empty)
            'crop'      : np.ndarray (enhanced plate crop)
        """
        plates  = self.detector.detect(frame)
        results = []

        for (x1, y1, x2, y2) in plates:
            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w < _MIN_PLATE_W or crop_h < _MIN_PLATE_H:
                logger.debug("Skipping tiny plate crop (%dx%d)", crop_w, crop_h)
                continue

            plate_crop = frame[y1:y2, x1:x2]
            enhanced   = self.deblurrer.enhance(plate_crop) if self.deblurrer else plate_crop
            plate_text = self.reader.read(enhanced)

            logger.info("Plate detected at [%d,%d,%d,%d] → '%s'", x1, y1, x2, y2, plate_text or "(no text)")
            results.append({
                "bbox":  [x1, y1, x2, y2],
                "text":  plate_text,
                "crop":  enhanced,
            })

        # Fallback: if YOLO found no plates, try OCR on the whole frame/crop
        # (helps when plate is visible but detector confidence is too low)
        if self.enable_fullframe_fallback and not results:
            h, w = frame.shape[:2]
            if w >= _MIN_PLATE_W and h >= _MIN_PLATE_H:
                enhanced   = self.deblurrer.enhance(frame) if self.deblurrer else frame
                plate_text = self.reader.read(enhanced)
                if plate_text:
                    logger.info("Fallback whole-frame OCR → '%s'", plate_text)
                    results.append({
                        "bbox":  [0, 0, w, h],
                        "text":  plate_text,
                        "crop":  enhanced,
                    })

        return results