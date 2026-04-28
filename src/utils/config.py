# src/utils/config.py
"""Configuration loader for cameras and calibration YAML files."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    id:          str
    source:      str               # file path, webcam index ("0"), or RTSP URL
    fps:         int  = 25
    width:       int  = 1280
    height:      int  = 720
    enabled:     bool = True
    description: str  = ""


def load_cameras(path: str = "config/cameras.yaml") -> list[CameraConfig]:
    """Load camera list from YAML.  Returns empty list if file not found."""
    if not os.path.exists(path):
        logger.warning("Camera config not found: %s — using empty list.", path)
        return []

    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    cameras = []
    for item in data.get("cameras", []):
        cameras.append(CameraConfig(**{k: v for k, v in item.items() if k in CameraConfig.__dataclass_fields__}))
    logger.info("Loaded %d camera(s) from %s", len(cameras), path)
    return cameras


# ---------------------------------------------------------------------------
# Calibration configuration
# ---------------------------------------------------------------------------

@dataclass
class LaneConfig:
    id:      str
    vector:  list[float]          # [dx, dy] allowed direction
    polygon: list[list[float]]    # [[x,y], ...] lane boundary


@dataclass
class CalibrationConfig:
    camera_id: str
    lanes:     list[LaneConfig] = field(default_factory=list)


def load_calibration(path: str = "config/calibration.yaml") -> dict[str, CalibrationConfig]:
    """
    Load calibration data.

    Returns
    -------
    dict mapping camera_id → CalibrationConfig.
    """
    if not os.path.exists(path):
        logger.warning("Calibration config not found: %s — wrong-way detection disabled.", path)
        return {}

    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    result = {}
    for item in data.get("calibrations", []):
        cam_id = item["camera_id"]
        lanes  = [LaneConfig(**ln) for ln in item.get("lanes", [])]
        result[cam_id] = CalibrationConfig(camera_id=cam_id, lanes=lanes)
    logger.info("Loaded calibration for %d camera(s) from %s", len(result), path)
    return result


# ---------------------------------------------------------------------------
# General app settings
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    backend_url:            str   = "http://localhost:8000"
    backend_health_timeout: float = 3.0
    backend_startup_wait:   float = 15.0
    backend_health_poll_interval: float = 1.0
    accident_model:         str   = "models/acc_detect.pt"
    accident_model_2:       str   = "models/epoch61.pt"
    vehicle_model:          str   = "models/yolov8n.pt"
    plate_model:            str   = "models/yolo26n.pt"
    deblur_model:           str   = "models/fpn_inception.h5"
    accident_conf:          float = 0.50
    vehicle_conf:           float = 0.40
    plate_conf:             float = 0.25
    anpr_max_plates_per_frame: int = 1
    anpr_track_ocr_cooldown:    int = 15
    ocr_gpu:                 bool  = True
    evidence_dir:           str   = "evidence"
    hit_and_run_timeout:    float = 10.0
    hit_and_run_radius:     int   = 200
    api_retry_attempts:     int   = 3
    api_retry_delay:        float = 2.0
    # ── GPU / Performance ─────────────────────────────────────────────
    device:                 str   = ""       # '' = auto, 'cuda:0', 'cpu'
    half_precision:         bool  = True      # FP16 on GPU for ~2× speed
    infer_imgsz:            int   = 640       # YOLO inference size (px)
    deblur_gpu:             bool  = True      # Use GPU for DeblurGAN


def load_app_config() -> AppConfig:
    """Build AppConfig from environment variables (with sensible defaults)."""
    env = os.environ.get
    return AppConfig(
        backend_url          = env("BACKEND_URL",            "http://localhost:8000"),
        backend_health_timeout = float(env("BACKEND_HEALTH_TIMEOUT", "3.0")),
        backend_startup_wait = float(env("BACKEND_STARTUP_WAIT", "15.0")),
        backend_health_poll_interval = float(env("BACKEND_HEALTH_POLL_INTERVAL", "1.0")),
        accident_model       = env("ACCIDENT_MODEL",         "models/acc_detect.pt"),
        accident_model_2     = env("ACCIDENT_MODEL_2",       "models/epoch61.pt"),
        vehicle_model        = env("VEHICLE_MODEL",          "models/yolov8n.pt"),
        plate_model          = env("PLATE_MODEL",            "models/yolo26n.pt"),
        deblur_model         = env("DEBLUR_MODEL",           "models/fpn_inception.h5"),
        accident_conf        = float(env("ACCIDENT_CONF",    "0.50")),
        vehicle_conf         = float(env("VEHICLE_CONF",     "0.40")),
        plate_conf           = float(env("PLATE_CONF",       "0.25")),
        anpr_max_plates_per_frame = int(env("ANPR_MAX_PLATES_PER_FRAME", "1")),
        anpr_track_ocr_cooldown   = int(env("ANPR_TRACK_OCR_COOLDOWN", "15")),
        ocr_gpu               = env("OCR_GPU", "1").strip().lower() not in {"0", "false", "no"},
        evidence_dir         = env("EVIDENCE_DIR",           "evidence"),
        hit_and_run_timeout  = float(env("HIT_RUN_TIMEOUT",  "10.0")),
        hit_and_run_radius   = int(env("HIT_RUN_RADIUS",     "200")),
        api_retry_attempts   = int(env("API_RETRY_ATTEMPTS", "3")),
        api_retry_delay      = float(env("API_RETRY_DELAY",  "2.0")),
        # GPU / Performance
        device               = env("DEVICE",                 ""),
        half_precision       = env("HALF_PRECISION", "1").strip().lower() not in {"0", "false", "no"},
        infer_imgsz          = int(env("INFER_IMGSZ",        "640")),
        deblur_gpu           = env("DEBLUR_GPU", "1").strip().lower() not in {"0", "false", "no"},
    )
