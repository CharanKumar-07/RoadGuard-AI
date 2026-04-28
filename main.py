# main.py
"""
RoadGuard AI – Quick-start entry point with GPU acceleration.

This script demonstrates direct model usage. For production,
use the ProcessorManager from src/processor.py instead.
"""
import os
import sys

import cv2
import torch

from src.detection.yolo_detector import YOLODetector
from src.tracking.tracker import SimpleTracker
from src.anpr.pipeline import ANPRPipeline


def main():
    # ── GPU Detection ────────────────────────────────────────────────────
    device = os.getenv("DEVICE", "")
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    use_half = device.startswith("cuda")
    imgsz = int(os.getenv("INFER_IMGSZ", "640"))

    if device.startswith("cuda"):
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   FP16 (half precision): {use_half}")
        print(f"   Inference size: {imgsz}px")
    else:
        print("⚠️  No CUDA GPU detected — running on CPU (slower)")

    # ── Load models (all GPU-accelerated) ────────────────────────────────
    vehicle_model = YOLODetector(
        'models/yolov8n.pt',
        device=device, half=use_half, imgsz=imgsz,
    )
    accident_model = YOLODetector(
        'models/acc_detect.pt',
        device=device, half=use_half, imgsz=imgsz,
    )

    anpr = ANPRPipeline(
        plate_model_path='models/yolo26n.pt',
        deblur_weights='models/fpn_inception.h5',
        device=device,
        half=use_half,
        imgsz=imgsz,
        deblur_gpu=device.startswith("cuda"),
        ocr_gpu=device.startswith("cuda"),
    )

    # Video source
    cap = cv2.VideoCapture(0)  # or RTSP URL

    tracker = SimpleTracker()

    _VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detections (all on GPU if available)
        vehicles = vehicle_model.detect(frame)
        vehicles = [d for d in vehicles if d["class"] in _VEHICLE_CLASSES]
        accidents = accident_model.detect(frame)

        # Update tracker with vehicle detections
        tracked_objects = tracker.update(vehicles)

        # For each accident detection, trigger alert and ANPR
        if len(accidents) > 0:
            # Find nearby vehicles, get their plates
            # Save frame evidence
            # Send alert
            pass

        # For each plate detected via ANPR pipeline
        plate_results = anpr.process_frame(frame)
        for plate in plate_results:
            plate_text = plate.get("text", "")
            if plate_text:
                x1, y1, x2, y2 = plate["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        # Display or save output
        cv2.imshow('RoadGuard AI', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()