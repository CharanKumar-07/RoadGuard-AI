#!/usr/bin/env python3
"""
scripts/calibrate_camera.py
===========================
Interactive tool to define lane directions by clicking two points on a
video frame.  Saves results to config/calibration.yaml.

Usage:
    python scripts/calibrate_camera.py --source 0 --camera-id cam_01

Controls:
    Click point 1  → start of allowed direction vector
    Click point 2  → end of allowed direction vector
    Press  s       → save calibration and exit
    Press  r       → reset / redo this lane
    Press  q       → quit without saving
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from src.utils.calibration import CalibrationHelper


points:     list[tuple[int, int]] = []
frame_copy: np.ndarray | None     = None


def mouse_callback(event, x, y, flags, param) -> None:
    global points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(frame_copy, (x, y), 6, (0, 255, 0), -1)
        if len(points) == 2:
            cv2.arrowedLine(frame_copy, points[0], points[1], (0, 0, 255), 2, tipLength=0.3)
        cv2.imshow("Calibrate", frame_copy)


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    global points, frame_copy

    parser = argparse.ArgumentParser(description="RoadGuard AI – Camera Lane Calibration")
    parser.add_argument("--source",            default="0",                    help="Camera source (index or RTSP URL or file path)")
    parser.add_argument("--camera-id",         default="cam_01",               help="Camera ID to save under")
    parser.add_argument("--lane-id",           default="lane_1",               help="Lane ID")
    parser.add_argument("--calibration-path",  default="config/calibration.yaml")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: cannot open source '{args.source}'")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read a frame.")
        sys.exit(1)

    frame_copy = frame.copy()
    cv2.putText(frame_copy,
                "Click 2 points to define lane direction. Press S to save, R to reset, Q to quit.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", mouse_callback)
    cv2.imshow("Calibrate", frame_copy)

    h, w = frame.shape[:2]

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            print("Quit without saving.")
            break

        elif key == ord("r"):
            points = []
            frame_copy = frame.copy()
            cv2.putText(frame_copy,
                        "Click 2 points to define lane direction. S=save  R=reset  Q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Calibrate", frame_copy)

        elif key == ord("s") and len(points) == 2:
            p1, p2 = points
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            polygon = [[0, 0], [w, 0], [w, h], [0, h]]

            helper = CalibrationHelper(args.calibration_path)
            helper.set_lane(
                camera_id = args.camera_id,
                lane_id   = args.lane_id,
                vector    = [float(dx), float(dy)],
                polygon   = polygon,
            )
            helper.save()
            print(f"Saved lane '{args.lane_id}' for camera '{args.camera_id}' → vector [{dx}, {dy}]")
            print(f"  Calibration written to: {args.calibration_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
