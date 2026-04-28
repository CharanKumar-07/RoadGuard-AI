# run_video.py
"""
RoadGuard AI – GPU-Accelerated Video Analysis Script
=====================================================
Processes a video through the full detection pipeline:
  1. Vehicle detection + tracking  (yolov8n.pt on GPU)
  2. Accident detection            (acc_detect.pt on GPU)
  3. License plate detection       (yolo26n.pt on GPU)
  4. Plate deblurring              (fpn_inception.h5)
  5. Plate OCR                     (EasyOCR on GPU)

Outputs an annotated video with bounding boxes, labels,
and plate text overlays.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch

# -- GPU Info --
print("=" * 60)
if torch.cuda.is_available():
    print(f"  [GPU]    {torch.cuda.get_device_name(0)}")
    print(f"  CUDA:    {torch.version.cuda}")
    print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    DEVICE = "cuda:0"
    HALF = True
else:
    print("  [WARN] No CUDA GPU -- running on CPU")
    DEVICE = "cpu"
    HALF = False
print(f"  Device:  {DEVICE}")
print(f"  FP16:    {HALF}")
print("=" * 60)

# ── Config ───────────────────────────────────────────────────────────────
VIDEO_PATH     = r"kling_20260420_VIDEO_Realistic__3560_0.mp4"
OUTPUT_PATH    = r"output_analysed.mp4"
IMGSZ          = 640
VEHICLE_CONF   = 0.40
ACCIDENT_CONF  = 0.50
PLATE_CONF     = 0.15

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO IDs)

# ── Colour palette ───────────────────────────────────────────────────────
COLOURS = {
    "accident": (0,   0,   220),   # red
    "vehicle":  (0,  200,   80),   # green
    "plate":    (0,  255,  200),   # cyan
}


def draw_box(img, x1, y1, x2, y2, label, category="vehicle"):
    colour = COLOURS.get(category, (200, 200, 200))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bg_y1 = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(img, label, (x1 + 3, max(y1 - 4, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_plate_banner(img, text, vx1, vy1, vx2, vy2):
    """Draw a floating license plate banner above the vehicle box."""
    if not text:
        return
    h_img, w_img = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 0.75, 2
    pad_x, pad_y = 12, 6

    label = text.upper().strip()
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    plate_w = tw + pad_x * 2
    plate_h = th + pad_y * 2 + baseline

    cx = (vx1 + vx2) // 2
    px1 = max(4, cx - plate_w // 2)
    px2 = min(w_img - 4, px1 + plate_w)
    py2 = max(plate_h + 10, vy1 - 10)
    py1 = py2 - plate_h

    # Yellow plate background
    cv2.rectangle(img, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), (30, 30, 30), -1)
    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 220, 255), -1)
    # Text
    text_x = px1 + pad_x
    text_y = py2 - pad_y - baseline
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)


def main():
    # ── Load models ──────────────────────────────────────────────────────
    from src.detection.yolo_detector import YOLODetector
    from src.tracking.tracker import SimpleTracker
    from src.anpr.pipeline import ANPRPipeline

    print("\n[*] Loading models on GPU...")
    t_load = time.perf_counter()

    vehicle_det = YOLODetector("models/yolov8n.pt", VEHICLE_CONF,
                               device=DEVICE, half=HALF, imgsz=IMGSZ)
    accident_det = YOLODetector("models/acc_detect.pt", ACCIDENT_CONF,
                                device=DEVICE, half=HALF, imgsz=IMGSZ)

    anpr = ANPRPipeline(
        plate_model_path="models/yolo26n.pt",
        deblur_weights="models/fpn_inception.h5",
        plate_conf=PLATE_CONF,
        device=DEVICE,
        half=HALF,
        imgsz=IMGSZ,
        deblur_gpu=DEVICE.startswith("cuda"),
        ocr_gpu=DEVICE.startswith("cuda"),
    )

    tracker = SimpleTracker()
    load_time = time.perf_counter() - t_load
    print(f"[OK] Models loaded in {load_time:.1f}s\n")

    # ── Open video ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[VIDEO] {VIDEO_PATH}")
    print(f"   {w}x{h} @ {fps:.0f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    # ── Output video writer ──────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    # ── State ────────────────────────────────────────────────────────────
    plate_cache: dict[int, str] = {}     # track_id → best plate text
    frame_idx = 0
    total_vehicles = 0
    total_accidents = 0
    total_plates = 0
    t_start = time.perf_counter()

    print(f"\n[*] Analysing frames...\n")

    # ── Frame loop ───────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated = frame.copy()

        # ── 1. Vehicle detection + tracking ──────────────────────────────
        veh_dets = vehicle_det.detect(frame)
        veh_dets = [d for d in veh_dets if d["class"] in VEHICLE_CLASSES]
        tracks = tracker.update(veh_dets)
        total_vehicles = max(total_vehicles, len(tracks))

        # ── 2. Accident detection ────────────────────────────────────────
        acc_dets = accident_det.detect(frame)
        acc_dets = [d for d in acc_dets if d["confidence"] >= ACCIDENT_CONF]
        if acc_dets:
            total_accidents += 1

        for det in acc_dets:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]
            draw_box(annotated, x1, y1, x2, y2,
                     f"ACCIDENT {conf:.0%}", "accident")

        # ── 3. ANPR (plate detection + OCR) ──────────────────────────────
        try:
            plate_dets = anpr.detector.detect(frame)
            for (px1, py1, px2, py2) in plate_dets:
                if (px2 - px1) < 20 or (py2 - py1) < 8:
                    continue
                plate_crop = frame[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue

                enhanced = (anpr.deblurrer.enhance(plate_crop)
                            if anpr.deblurrer else plate_crop)
                plate_text = anpr.reader.read(enhanced)

                # Assign plate to nearest vehicle track
                best_track = None
                best_dist = float("inf")
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                for t in tracks:
                    tcx = (t.bbox[0] + t.bbox[2]) / 2
                    tcy = (t.bbox[1] + t.bbox[3]) / 2
                    dist = ((pcx - tcx)**2 + (pcy - tcy)**2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_track = t

                if plate_text and best_track:
                    existing = plate_cache.get(best_track.track_id, "")
                    if len(plate_text) >= len(existing):
                        plate_cache[best_track.track_id] = plate_text
                        total_plates += 1

                draw_box(annotated, px1, py1, px2, py2,
                         plate_text or "plate", "plate")
        except Exception as e:
            pass  # ANPR errors are non-fatal

        # ── 4. Draw vehicle boxes + plate banners ────────────────────────
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cached_plate = plate_cache.get(t.track_id, "")
            draw_box(annotated, x1, y1, x2, y2,
                     f"#{t.track_id} {t.label}", "vehicle")
            if cached_plate:
                draw_plate_banner(annotated, cached_plate, x1, y1, x2, y2)

        # ── 5. HUD overlay ───────────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        current_fps = frame_idx / max(elapsed, 0.001)
        progress = int(frame_idx * 100 / total_frames)

        hud_lines = [
            f"Frame: {frame_idx}/{total_frames} ({progress}%)",
            f"FPS: {current_fps:.1f}",
            f"Vehicles: {len(tracks)}",
            f"Accidents: {len(acc_dets)}",
            f"Plates: {len(plate_cache)}",
            f"GPU: {DEVICE}",
        ]
        for i, line in enumerate(hud_lines):
            y_pos = 30 + i * 28
            cv2.putText(annotated, line, (12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, line, (12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

        writer.write(annotated)

        # Progress bar
        bar_len = 30
        filled = int(bar_len * frame_idx / total_frames)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {progress:3d}% | frame {frame_idx}/{total_frames} | "
              f"{current_fps:.1f} fps | vehicles={len(tracks)} acc={len(acc_dets)} plates={len(plate_cache)}",
              end="", flush=True)

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    total_time = time.perf_counter() - t_start

    print(f"\n\n{'='*60}")
    print(f"  [DONE] Analysis Complete!")
    print(f"  {'─'*56}")
    print(f"  Frames processed:  {frame_idx}")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Average FPS:       {frame_idx / total_time:.1f}")
    print(f"  Max vehicles:      {total_vehicles}")
    print(f"  Accident frames:   {total_accidents}")
    print(f"  Plates detected:   {total_plates}")
    print(f"  Unique plates:     {len(plate_cache)}")
    if plate_cache:
        print(f"  {'─'*56}")
        print(f"  Plate readings:")
        for tid, txt in plate_cache.items():
            print(f"    Track #{tid}: {txt}")
    print(f"  {'─'*56}")
    print(f"  Output saved to:   {os.path.abspath(OUTPUT_PATH)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
