# src/detection/yolo_detector.py
"""YOLO-based object detector wrapper for RoadGuard AI.

GPU Acceleration & Performance:
  • Auto-detects CUDA and forces the model onto GPU 0 when available.
  • Uses FP16 (half-precision) inference for ~2× speed-up on NVIDIA GPUs.
  • Configurable inference image size (smaller = faster, default 640).
  • Fuses Conv+BN layers at load time for reduced latency.
"""

import logging
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def _select_device() -> str:
    """Return 'cuda:0' if CUDA is available, else 'cpu'."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA GPU detected: %s", gpu_name)
        return "cuda:0"
    logger.warning("CUDA not available — falling back to CPU inference.")
    return "cpu"


class YOLODetector:
    """
    Generic YOLOv8 detector with GPU acceleration.

    Parameters
    ----------
    model_path : str
        Path to the .pt model file.
    conf_threshold : float
        Minimum confidence to keep a detection.
    device : str | None
        Force a device ('cuda:0', 'cpu'). None = auto-detect.
    half : bool | None
        Use FP16 inference. None = auto (True on CUDA, False on CPU).
    imgsz : int
        Inference image size (pixels). Smaller = faster. Default 640.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        device: str | None = None,
        half: bool | None = None,
        imgsz: int = 640,
    ):
        self.device = device or _select_device()
        self.use_half = half if half is not None else (self.device.startswith("cuda"))
        self.imgsz = imgsz
        self.conf = conf_threshold

        logger.info(
            "Loading YOLO model from %s → device=%s half=%s imgsz=%d",
            model_path, self.device, self.use_half, self.imgsz,
        )
        self.model = YOLO(model_path)

        # Move model to the target device
        self.model.to(self.device)

        # Fuse Conv+BatchNorm layers for faster inference
        try:
            self.model.fuse()
            logger.info("Model layers fused for optimised inference.")
        except Exception:
            pass  # some model architectures don't support fuse()

        # Warm up the model with a dummy tensor to pre-allocate GPU memory
        if self.device.startswith("cuda"):
            self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference pass to warm up CUDA kernels and memory allocation."""
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model(
                dummy,
                conf=self.conf,
                verbose=False,
                device=self.device,
                half=self.use_half,
                imgsz=self.imgsz,
            )
            logger.info("GPU warm-up complete for %s", self.device)
        except Exception as exc:
            logger.debug("Warm-up pass failed (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a BGR frame.

        Returns
        -------
        list of dict with keys:
            'bbox'       : [x1, y1, x2, y2] (int)
            'confidence' : float
            'class'      : int
            'label'      : str
        """
        results = self.model(
            frame,
            conf=self.conf,
            verbose=False,
            device=self.device,
            half=self.use_half,
            imgsz=self.imgsz,
        )[0]
        detections = []
        names = results.names  # class-id → label mapping

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = names.get(cls, str(cls))
            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": conf,
                "class":      cls,
                "label":      label,
            })
        return detections

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
        """
        Laplacian-variance blurriness test.

        Returns True if the image is blurry (variance < threshold).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold