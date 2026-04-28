# src/violations/wrong_way.py
"""Wrong-way driving detector using per-lane direction vectors."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class WrongWayDetector:
    """
    Detects vehicles travelling in the wrong direction for their lane.

    Parameters
    ----------
    lane_directions : dict
        Mapping of lane_id (str) → dict with keys:
            'polygon' : list of [x,y] points defining the lane region
            'vector'  : [dx, dy] allowed movement direction
    """

    def __init__(self, lane_directions: dict) -> None:
        self.lane_dirs = lane_directions

    # ------------------------------------------------------------------

    def check(self, track_history: dict[int, list[tuple[float, float]]]) -> list[tuple]:
        """
        Parameters
        ----------
        track_history : dict
            track_id → list of (x, y) centroid positions (oldest first).

        Returns
        -------
        list of (track_id, lane_id) tuples for violators.
        """
        violations = []
        for track_id, positions in track_history.items():
            if len(positions) < 5:
                continue

            # Movement vector: average of last 3 positions minus average of first 3
            vec = (
                np.mean(positions[-3:], axis=0) - np.mean(positions[:3], axis=0)
            )
            if np.linalg.norm(vec) < 10:  # not moving
                continue

            lane_id = self._get_lane(positions[-1])
            if lane_id is None:
                continue

            allowed = np.array(self.lane_dirs[lane_id]["vector"], dtype=float)
            if np.linalg.norm(allowed) == 0:
                continue

            # Negative dot product → opposite direction → wrong way
            if np.dot(vec, allowed) < 0:
                logger.warning("Wrong-way vehicle: track %d in lane %s", track_id, lane_id)
                violations.append((track_id, lane_id))

        return violations

    # ------------------------------------------------------------------

    def _get_lane(self, point: tuple[float, float]) -> str | None:
        """
        Determine which lane a point belongs to using polygon containment.

        Returns the lane_id or None if the point is not in any lane.
        """
        import cv2  # local import — only needed for pointPolygonTest

        px, py = float(point[0]), float(point[1])
        for lane_id, cfg in self.lane_dirs.items():
            polygon = np.array(cfg.get("polygon", []), dtype=np.float32)
            if polygon.shape[0] < 3:
                continue
            result = cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), (px, py), False)
            if result >= 0:
                return lane_id
        return None