# src/utils/calibration.py
"""
Camera lane calibration helper.

Loads/saves lane direction vectors and polygons to/from YAML.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class CalibrationHelper:
    """
    Read and write lane calibration data for a specific camera.

    Format expected in calibration.yaml:
    
    calibrations:
      - camera_id: cam_01
        lanes:
          - id: lane_1
            vector: [1, 0]          # dx, dy of allowed travel direction
            polygon: [[0,0],[640,0],[640,720],[0,720]]
    """

    def __init__(self, calibration_path: str = "config/calibration.yaml") -> None:
        self.path = calibration_path
        self._data: dict[str, Any] = {"calibrations": []}
        if os.path.exists(calibration_path):
            self._load()

    # ------------------------------------------------------------------

    def _load(self) -> None:
        with open(self.path, "r") as fh:
            loaded = yaml.safe_load(fh) or {}
        self._data = loaded
        logger.info("Calibration loaded from %s", self.path)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            yaml.dump(self._data, fh, default_flow_style=False)
        logger.info("Calibration saved to %s", self.path)

    # ------------------------------------------------------------------

    def get_lane_directions(self, camera_id: str) -> dict:
        """
        Return lane_directions dict suitable for WrongWayDetector.

        dict structure: lane_id → {'vector': [dx,dy], 'polygon': [[x,y],...]}
        """
        for entry in self._data.get("calibrations", []):
            if entry.get("camera_id") == camera_id:
                return {
                    lane["id"]: {
                        "vector":  lane.get("vector", [1, 0]),
                        "polygon": lane.get("polygon", []),
                    }
                    for lane in entry.get("lanes", [])
                }
        return {}

    def set_lane(
        self,
        camera_id: str,
        lane_id:   str,
        vector:    list[float],
        polygon:   list[list[float]],
    ) -> None:
        """Add or update a lane entry."""
        calibrations = self._data.setdefault("calibrations", [])
        cam_entry    = next((e for e in calibrations if e["camera_id"] == camera_id), None)
        if cam_entry is None:
            cam_entry = {"camera_id": camera_id, "lanes": []}
            calibrations.append(cam_entry)

        lanes     = cam_entry.setdefault("lanes", [])
        lane_entry = next((ln for ln in lanes if ln["id"] == lane_id), None)
        if lane_entry is None:
            lane_entry = {"id": lane_id}
            lanes.append(lane_entry)

        lane_entry["vector"]  = vector
        lane_entry["polygon"] = polygon
