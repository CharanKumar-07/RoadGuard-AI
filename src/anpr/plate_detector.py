# src/anpr/plate_detector.py
"""YOLOv8-based license plate localiser with GPU acceleration."""

from __future__ import annotations

import logging

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Return 'cuda:0' if CUDA is available, else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class PlateDetector:
    """
    Detect license plate bounding boxes in an image.

    Parameters
    ----------
    model_path : str
        Path to a YOLOv8 .pt model trained for plate detection.
    conf_threshold : float
        Minimum confidence for a plate detection.
    device : str | None
        Force a device ('cuda:0', 'cpu'). None = auto-detect.
    half : bool | None
        Use FP16 inference. None = auto (True on CUDA).
    imgsz : int
        Inference image size. Default 640.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.15,
        device: str | None = None,
        half: bool | None = None,
        imgsz: int = 640,
    ) -> None:
        self.device = device or _select_device()
        self.use_half = half if half is not None else self.device.startswith("cuda")
        self.imgsz = imgsz
        self.conf = conf_threshold

        logger.info(
            "Loading plate detector from %s → device=%s half=%s imgsz=%d",
            model_path, self.device, self.use_half, self.imgsz,
        )
        self._model = YOLO(model_path)
        self._model.to(self.device)

        # Fuse layers for faster inference
        try:
            self._model.fuse()
        except Exception:
            pass

        # Warm up on GPU
        if self.device.startswith("cuda"):
            try:
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                self._model(dummy, conf=self.conf, verbose=False,
                            device=self.device, half=self.use_half, imgsz=self.imgsz)
                logger.info("Plate detector GPU warm-up complete.")
            except Exception:
                pass

    def detect(self, frame: np.ndarray) -> list[list[int]]:
        """
        Returns
        -------
        list of [x1, y1, x2, y2] bounding boxes (int pixels).
        """
        results = self._model(
            frame,
            conf=self.conf,
            verbose=False,
            device=self.device,
            half=self.use_half,
            imgsz=self.imgsz,
        )[0]
        plates  = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Clamp to image bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                plates.append([x1, y1, x2, y2])
        return plates