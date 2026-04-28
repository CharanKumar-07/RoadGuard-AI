# src/processor.py
"""
RoadGuard AI – Main Pipeline Orchestrator
==========================================

Spawns one thread per camera, running the full detection → ANPR →
violation → backend-POST pipeline on each video stream.

Usage (programmatic):
    from src.processor import ProcessorManager
    mgr = ProcessorManager()
    mgr.start()
    # … runs until mgr.stop() is called or process is killed

Usage (CLI):
    python -m src.processor --cameras config/cameras.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

# ── Internal imports ────────────────────────────────────────────────────────
from src.detection.yolo_detector    import YOLODetector
from src.tracking.tracker           import SimpleTracker
from src.anpr.pipeline              import ANPRPipeline
from src.violations.wrong_way       import WrongWayDetector
from src.violations.hit_and_run     import HitAndRunMonitor
from src.utils.video_reader         import VideoReader
from src.utils.config               import load_cameras, load_calibration, load_app_config
from src.utils.calibration          import CalibrationHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Globals ──────────────────────────────────────────────────────────────────

_VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck (COCO ids)


def _laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _enhance_for_accident_detection(frame: np.ndarray) -> np.ndarray:
    """Create a sharpened, contrast-equalized view for blurry accident frames."""
    den = cv2.fastNlMeansDenoisingColored(frame, None, 2, 2, 7, 21)
    blur = cv2.GaussianBlur(den, (0, 0), 1.1)
    sharp = cv2.addWeighted(den, 1.4, blur, -0.4, 0)

    ycrcb = cv2.cvtColor(sharp, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)


def _merge_dets_by_iou(dets: list[dict], iou_thr: float = 0.60) -> list[dict]:
    merged: list[dict] = []
    for det in sorted(dets, key=lambda d: d.get("confidence", 0.0), reverse=True):
        bbox = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        keep = True
        for m in merged:
            mb = [int(v) for v in m.get("bbox", [0, 0, 0, 0])]
            ix1 = max(bbox[0], mb[0]); iy1 = max(bbox[1], mb[1])
            ix2 = min(bbox[2], mb[2]); iy2 = min(bbox[3], mb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            a1 = max(1, (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))
            a2 = max(1, (mb[2]-mb[0]) * (mb[3]-mb[1]))
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou >= iou_thr:
                keep = False
                break
        if keep:
            merged.append({**det, "bbox": bbox})
    return merged


def _wait_for_backend(cfg) -> None:
    """Wait briefly for the backend API to come up before processing starts."""
    health_url = f"{cfg.backend_url.rstrip('/')}/health"
    timeout = max(0.1, float(getattr(cfg, "backend_health_timeout", 3.0)))
    poll_interval = max(0.2, float(getattr(cfg, "backend_health_poll_interval", 1.0)))
    deadline = time.perf_counter() + max(0.0, float(getattr(cfg, "backend_startup_wait", 15.0)))
    last_error = ""

    while True:
        try:
            response = requests.get(health_url, timeout=timeout)
            response.raise_for_status()
            logger.info("Backend is reachable at %s", cfg.backend_url)
            return
        except requests.RequestException as exc:
            last_error = str(exc)

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise RuntimeError(
                "Backend API is not reachable at "
                f"{cfg.backend_url}. Start it first with "
                "`uvicorn backend.main:app --host 0.0.0.0 --port 8000` "
                f"and retry. Last error: {last_error}"
            )

        logger.info(
            "Waiting for backend at %s (%.0fs remaining)...",
            health_url,
            remaining,
        )
        time.sleep(min(poll_interval, max(remaining, 0.2)))


# ═══════════════════════════════════════════════════════════════════════════
# Per-camera processor
# ═══════════════════════════════════════════════════════════════════════════

class CameraProcessor:
    """
    Runs the full detection pipeline on a single camera stream.

    Parameters
    ----------
    camera_id : str
    source    : str | int  (file path, webcam index, RTSP URL)
    cfg       : AppConfig
    lane_dirs : dict       (lane config for WrongWayDetector)
    """

    def __init__(self, camera_id: str, source, cfg, lane_dirs: dict) -> None:
        self.camera_id = camera_id
        self.source    = source
        self.cfg       = cfg
        self._stop     = threading.Event()

        # ── Model loading (GPU-accelerated) ──────────────────────────────
        _device = cfg.device or None
        _half   = cfg.half_precision
        _imgsz  = cfg.infer_imgsz
        logger.info(
            "[%s] Loading models … device=%s half=%s imgsz=%d",
            camera_id, _device or "auto", _half, _imgsz,
        )
        self.acc_detector     = YOLODetector(
            cfg.accident_model, cfg.accident_conf,
            device=_device, half=_half, imgsz=_imgsz,
        )
        # Only load ensemble if a separate second model is provided
        if cfg.accident_model_2 and cfg.accident_model_2 != cfg.accident_model:
            self.acc_detector_2 = YOLODetector(
                cfg.accident_model_2, cfg.accident_conf,
                device=_device, half=_half, imgsz=_imgsz,
            )
        else:
            self.acc_detector_2 = None

        self.vehicle_detector = YOLODetector(
            cfg.vehicle_model, cfg.vehicle_conf,
            device=_device, half=_half, imgsz=_imgsz,
        )
        self.anpr             = ANPRPipeline(
            cfg.plate_model,
            cfg.deblur_model,
            plate_conf=cfg.plate_conf,
            ocr_gpu=cfg.ocr_gpu,
            device=cfg.device,
            half=_half,
            imgsz=_imgsz,
            deblur_gpu=cfg.deblur_gpu,
        )

        # ── Tracking / violation logic ───────────────────────────────────
        self.tracker    = SimpleTracker()
        self.hit_run    = HitAndRunMonitor(cfg.hit_and_run_radius, cfg.hit_and_run_timeout)
        self.wrong_way  = WrongWayDetector(lane_dirs) if lane_dirs else None

        # ── Evidence storage ─────────────────────────────────────────────
        os.makedirs(cfg.evidence_dir, exist_ok=True)

        # ── Plate cache: track_id → best plate text seen (carry across frames) ──
        self._plate_cache: dict[int, str] = {}
        self._plate_last_attempt_frame: dict[int, int] = {}
        self._pending_posts: list[threading.Thread] = []

        self._frames_processed = 0
        self._accident_events = 0
        self._time_detect_ms = 0.0
        self._time_track_ms = 0.0
        self._time_anpr_ms = 0.0
        self._time_post_ms = 0.0
        self._run_started = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("[%s] Starting stream: %s", self.camera_id, self.source)
        reader = VideoReader(self.source)
        self._run_started = time.perf_counter()

        for frame in reader.stream():
            if self._stop.is_set():
                break
            self._frames_processed += 1
            self._process_frame(frame)

            if self._frames_processed % 100 == 0:
                elapsed = max(time.perf_counter() - self._run_started, 1e-6)
                logger.info(
                    "[%s] Progress frames=%d elapsed=%.2fs fps=%.2f",
                    self.camera_id,
                    self._frames_processed,
                    elapsed,
                    self._frames_processed / elapsed,
                )

        reader.release()
        for t in list(self._pending_posts):
            t.join(timeout=2)

        elapsed = max(time.perf_counter() - self._run_started, 1e-6)
        logger.info(
            "[%s] Summary frames=%d elapsed=%.2fs avg_fps=%.2f accidents=%d detect_ms=%.1f track_ms=%.1f anpr_ms=%.1f post_ms=%.1f",
            self.camera_id,
            self._frames_processed,
            elapsed,
            self._frames_processed / elapsed,
            self._accident_events,
            self._time_detect_ms,
            self._time_track_ms,
            self._time_anpr_ms,
            self._time_post_ms,
        )
        logger.info("[%s] Processor stopped.", self.camera_id)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Frame-level logic
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Full pipeline per frame:
          1. Accident detection
          2. Vehicle detection + tracking
          3. ANPR: detect plates on full frame, assign each plate to nearest vehicle
             track, OCR, cache best plate text per track
          4. On accident: look up cached plate for the closest vehicle
          5. Hit-and-run / wrong-way checks
        """
        timestamp = datetime.now(timezone.utc)

        # 1. Accident detection (blur-aware ensemble)
        t0 = time.perf_counter()
        acc_dets = self.acc_detector.detect(frame)
        if self.acc_detector_2:
            acc_dets.extend(self.acc_detector_2.detect(frame))

        # On blurry frames, run a second pass on an enhanced view.
        blur_var = _laplacian_variance(frame)
        blur_thr = float(os.getenv("ACCIDENT_BLUR_VAR_THRESHOLD", "70"))
        is_blurry = blur_var < blur_thr
        if is_blurry:
            enhanced_acc = _enhance_for_accident_detection(frame)
            acc_dets.extend(self.acc_detector.detect(enhanced_acc))
            if self.acc_detector_2:
                acc_dets.extend(self.acc_detector_2.detect(enhanced_acc))

        acc_dets = _merge_dets_by_iou(acc_dets)

        conf_clear = float(os.getenv("ACCIDENT_CONF", str(self.cfg.accident_conf)))
        conf_blurry = float(os.getenv("ACCIDENT_CONF_BLURRY", "0.32"))
        active_acc_conf = conf_blurry if is_blurry else conf_clear

        accident_this_frame = any(
            d["confidence"] >= active_acc_conf for d in acc_dets
        )
        acc_bboxes = [
            [int(v) for v in d["bbox"]]
            for d in acc_dets
            if d["confidence"] >= active_acc_conf
        ]
        self._time_detect_ms += (time.perf_counter() - t0) * 1000

        # 2. Vehicle detection + tracking (every frame)
        t1 = time.perf_counter()
        vehicle_dets = self.vehicle_detector.detect(frame)
        vehicle_dets = [d for d in vehicle_dets if d["class"] in _VEHICLE_CLASSES]
        tracks = self.tracker.update(vehicle_dets)
        self._time_track_ms += (time.perf_counter() - t1) * 1000

        # 3. ANPR – run plate detector on full frame, assign plates to tracks
        t2 = time.perf_counter()
        try:
            max_plates = max(1, int(getattr(self.cfg, "anpr_max_plates_per_frame", 2)))
            ocr_cooldown = max(0, int(getattr(self.cfg, "anpr_track_ocr_cooldown", 10)))
            plate_candidates: dict[int, tuple[list[int], object, np.ndarray, int]] = {}
            loose_candidates: list[tuple[list[int], object, np.ndarray, int]] = []

            plate_dets = self.anpr.detector.detect(frame)  # returns [[x1,y1,x2,y2], ...]
            for (px1, py1, px2, py2) in plate_dets:
                if (px2 - px1) < 20 or (py2 - py1) < 8:
                    continue
                plate_bbox = [px1, py1, px2, py2]
                matched = self._assign_plate_to_track(plate_bbox, tracks)
                plate_crop = frame[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue
                area = (px2 - px1) * (py2 - py1)

                if matched is not None:
                    tid = matched.track_id
                    last_attempt = self._plate_last_attempt_frame.get(tid, -10**9)
                    if (self._frames_processed - last_attempt) < ocr_cooldown:
                        continue
                    current = plate_candidates.get(tid)
                    if current is None or area > current[3]:
                        plate_candidates[tid] = (plate_bbox, matched, plate_crop, area)
                else:
                    loose_candidates.append((plate_bbox, matched, plate_crop, area))

            selected_candidates = sorted(plate_candidates.values(), key=lambda item: item[3], reverse=True)
            if len(selected_candidates) < max_plates and loose_candidates:
                loose_candidates.sort(key=lambda item: item[3], reverse=True)
                selected_candidates.extend(loose_candidates[: max_plates - len(selected_candidates)])

            for plate_bbox, matched, plate_crop, _area in selected_candidates[:max_plates]:
                if matched is not None:
                    self._plate_last_attempt_frame[matched.track_id] = self._frames_processed

                enhanced   = self.anpr.deblurrer.enhance(plate_crop) if self.anpr.deblurrer else plate_crop
                plate_text = self.anpr.reader.read(enhanced)
                if plate_text:
                    # Match plate to best vehicle track
                    if matched is None:
                        matched = self._assign_plate_to_track(plate_bbox, tracks)
                    if matched is not None:
                        tid      = matched.track_id
                        existing = self._plate_cache.get(tid, "")
                        if len(plate_text.replace(" ", "")) >= len(existing.replace(" ", "")):
                            self._plate_cache[tid] = plate_text
                            logger.info("[%s] Plate cached track #%d → '%s'",
                                        self.camera_id, tid, plate_text)
        except Exception as exc:
            logger.debug("[%s] ANPR error: %s", self.camera_id, exc)
        self._time_anpr_ms += (time.perf_counter() - t2) * 1000

        # 4. Handle accident
        if accident_this_frame:
            self._accident_events += 1
            logger.info("[%s] ACCIDENT detected!", self.camera_id)

            # Find plate from the closest vehicle track to the accident
            plate_text: Optional[str] = None
            if tracks and acc_bboxes:
                primary_acc_bbox = acc_bboxes[0]
                sorted_tracks = sorted(
                    tracks,
                    key=lambda t: self._bbox_iou([int(v) for v in t.bbox], primary_acc_bbox),
                    reverse=True,
                )
                for t in sorted_tracks:
                    cached = self._plate_cache.get(t.track_id)
                    if cached:
                        plate_text = cached
                        break

            # Fallback: run full ANPR on accident crop
            if not plate_text and acc_bboxes:
                try:
                    ax1, ay1, ax2, ay2 = acc_bboxes[0]
                    h_img, w_img = frame.shape[:2]
                    pad_x = int((ax2 - ax1) * 0.2)
                    pad_y = int((ay2 - ay1) * 0.2)
                    crop = frame[
                        max(0, ay1-pad_y):min(h_img, ay2+pad_y),
                        max(0, ax1-pad_x):min(w_img, ax2+pad_x)
                    ]
                    if crop.size > 0:
                        results = self.anpr.process_frame(crop)
                        if results:
                            plate_text = results[0].get("text") or None
                except Exception as exc:
                    logger.debug("[%s] Accident crop ANPR: %s", self.camera_id, exc)

            evidence_path = self._save_evidence(frame, "accident", timestamp)

            acc_cx = acc_bboxes[0][0] + (acc_bboxes[0][2] - acc_bboxes[0][0]) / 2 if acc_bboxes else frame.shape[1] / 2
            acc_cy = acc_bboxes[0][1] + (acc_bboxes[0][3] - acc_bboxes[0][1]) / 2 if acc_bboxes else frame.shape[0] / 2
            self.hit_run.register_accident(self.camera_id, (acc_cx, acc_cy), tracks)

            self._post_incident_async(
                incident_type  = "accident",
                timestamp      = timestamp,
                license_plate  = plate_text or "",
                evidence_path  = evidence_path,
            )

        # 5. Hit-and-run check
        suspects = self.hit_run.update(tracks)
        for tid in suspects:
            logger.warning("[%s] Hit-and-run! track_id=%d", self.camera_id, tid)
            evidence_path = self._save_evidence(frame, "hit_and_run", timestamp)
            plate_text    = self._plate_cache.get(tid, "")
            self._post_incident_async(
                incident_type = "hit_and_run",
                timestamp     = timestamp,
                license_plate = plate_text,
                evidence_path = evidence_path,
            )

        # 6. Wrong-way check
        if self.wrong_way:
            track_history = {t.track_id: t.history for t in tracks}
            violations    = self.wrong_way.check(track_history)
            for tid, lane_id in violations:
                logger.warning("[%s] Wrong-way! track_id=%d lane=%s", self.camera_id, tid, lane_id)
                evidence_path = self._save_evidence(frame, "wrong_way", timestamp)
                plate_text    = self._plate_cache.get(tid, "")
                self._post_incident_async(
                    incident_type = "wrong_way",
                    timestamp     = timestamp,
                    license_plate = plate_text,
                    evidence_path = evidence_path,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_evidence(self, frame: np.ndarray, incident_type: str, timestamp: datetime) -> str:
        ts    = timestamp.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.camera_id}_{ts}_{incident_type}.jpg"
        path  = os.path.join(self.cfg.evidence_dir, fname)
        cv2.imwrite(path, frame)
        logger.debug("Evidence saved: %s", path)
        return path

    @staticmethod
    def _bbox_iou(b1: list[int], b2: list[int]) -> float:
        """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
        ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def _assign_plate_to_track(self, plate_bbox: list[int], tracks: list):
        """Return the vehicle Track best matching a plate bbox, or None."""
        best_track = None
        best_score = -1.0
        for t in tracks:
            vbbox = [int(v) for v in t.bbox]
            # containment bonus
            contained = (vbbox[0] <= plate_bbox[0] and vbbox[1] <= plate_bbox[1]
                         and vbbox[2] >= plate_bbox[2] and vbbox[3] >= plate_bbox[3])
            iou   = self._bbox_iou(vbbox, plate_bbox)
            score = (1.0 + iou) if contained else iou
            if score > best_score:
                best_score = score
                best_track = t
        # fallback: nearest centroid
        if best_score < 0.01 and tracks:
            def _dist(t):
                cx1 = (plate_bbox[0]+plate_bbox[2])/2; cy1 = (plate_bbox[1]+plate_bbox[3])/2
                vb  = [int(v) for v in t.bbox]
                cx2 = (vb[0]+vb[2])/2;                cy2 = (vb[1]+vb[3])/2
                return math.sqrt((cx1-cx2)**2+(cy1-cy2)**2)
            best_track = min(tracks, key=_dist)
        return best_track

    def _post_incident(
        self,
        incident_type: str,
        timestamp:     datetime,
        license_plate: str,
        evidence_path: str,
    ) -> None:
        """POST incident data to the backend with retry logic."""
        url     = f"{self.cfg.backend_url}/incidents/"
        payload = {
            "incident_type": incident_type,
            "camera_id":     self.camera_id,
            "license_plate": license_plate,
            "timestamp":     timestamp.isoformat(),
        }

        for attempt in range(1, self.cfg.api_retry_attempts + 1):
            try:
                with open(evidence_path, "rb") as img_fh:
                    resp = requests.post(
                        url,
                        data=payload,
                        files={"file": (os.path.basename(evidence_path), img_fh, "image/jpeg")},
                        timeout=10,
                    )
                resp.raise_for_status()
                logger.info(
                    "[%s] Incident posted OK (type=%s plate='%s') → id=%s",
                    self.camera_id, incident_type, license_plate,
                    resp.json().get("id", "?"),
                )
                return
            except Exception as exc:
                logger.warning(
                    "[%s] POST attempt %d/%d failed: %s",
                    self.camera_id, attempt, self.cfg.api_retry_attempts, exc,
                )
                if attempt < self.cfg.api_retry_attempts:
                    time.sleep(self.cfg.api_retry_delay * attempt)

        logger.error("[%s] All retry attempts exhausted for %s incident.", self.camera_id, incident_type)

    def _post_incident_async(
        self,
        incident_type: str,
        timestamp: datetime,
        license_plate: str,
        evidence_path: str,
    ) -> None:
        t = threading.Thread(
            target=self._timed_post_incident,
            args=(incident_type, timestamp, license_plate, evidence_path),
            daemon=True,
        )
        t.start()
        self._pending_posts.append(t)

    def _timed_post_incident(
        self,
        incident_type: str,
        timestamp: datetime,
        license_plate: str,
        evidence_path: str,
    ) -> None:
        t0 = time.perf_counter()
        self._post_incident(incident_type, timestamp, license_plate, evidence_path)
        self._time_post_ms += (time.perf_counter() - t0) * 1000


# ═══════════════════════════════════════════════════════════════════════════
# Multi-camera manager
# ═══════════════════════════════════════════════════════════════════════════

class ProcessorManager:
    """
    Loads camera and calibration config then starts one thread per camera.
    """

    def __init__(
        self,
        cameras_path:     str = "config/cameras.yaml",
        calibration_path: str = "config/calibration.yaml",
    ) -> None:
        self.cfg          = load_app_config()
        self.cameras      = load_cameras(cameras_path)
        self.calibrations = load_calibration(calibration_path)
        self._threads:    list[threading.Thread]    = []
        self._processors: list[CameraProcessor]     = []

    def start(self) -> None:
        if not self.cameras:
            logger.warning("No cameras configured — nothing to process.")
            return

        _wait_for_backend(self.cfg)

        for cam in self.cameras:
            if not cam.enabled:
                logger.info("Camera %s disabled — skipping.", cam.id)
                continue

            lane_dirs = {}
            if cam.id in self.calibrations:
                calib = self.calibrations[cam.id]
                lane_dirs = {
                    ln.id: {"vector": ln.vector, "polygon": ln.polygon}
                    for ln in calib.lanes
                }

            proc = CameraProcessor(cam.id, cam.source, self.cfg, lane_dirs)
            self._processors.append(proc)

            t = threading.Thread(
                target=proc.run,
                name=f"cam-{cam.id}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
            logger.info("Started processor thread for camera '%s'.", cam.id)

    def stop(self) -> None:
        for proc in self._processors:
            proc.stop()
        for t in self._threads:
            t.join(timeout=5)
        logger.info("All camera processors stopped.")

    def join(self) -> None:
        """Block until all camera threads finish."""
        for t in self._threads:
            t.join()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoadGuard AI – Multi-camera Processor")
    p.add_argument("--cameras",     default="config/cameras.yaml",    help="Path to cameras YAML config")
    p.add_argument("--calibration", default="config/calibration.yaml", help="Path to calibration YAML config")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manager = ProcessorManager(args.cameras, args.calibration)
    try:
        manager.start()
        manager.join()
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down …")
        manager.stop()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
