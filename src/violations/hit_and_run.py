# src/violations/hit_and_run.py
"""
Hit-and-run detection.

After an accident event, monitors tracked vehicles in the scene.
If a vehicle that was present at the accident zone leaves within
a configurable timeout, it is flagged as a hit-and-run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AccidentEvent:
    """Snapshot of the scene when an accident was detected."""
    camera_id:     str
    timestamp:     float                              # time.time()
    location:      tuple[float, float]               # accident centroid (x, y) in pixels
    track_ids:     set[int] = field(default_factory=set)  # tracks present at the scene


class HitAndRunMonitor:
    """
    Monitors tracked vehicles after an accident and flags those that leave
    the accident zone prematurely.

    Parameters
    ----------
    accident_radius : int
        Pixel radius around the accident centroid defining the "scene zone".
    timeout_sec : float
        Seconds after an accident before hit-and-run monitoring expires.
    """

    def __init__(
        self,
        accident_radius: int = 200,
        timeout_sec: float = 10.0,
    ) -> None:
        self.accident_radius = accident_radius
        self.timeout_sec     = timeout_sec
        self._accidents: list[AccidentEvent] = []
        self._already_flagged: set[int] = set()

    # ------------------------------------------------------------------

    def register_accident(
        self,
        camera_id: str,
        location: tuple[float, float],
        tracks_in_scene: list,
        accident_bbox: list[int] | None = None,
    ) -> None:
        """
        Call this immediately after an accident is detected.

        Only registers vehicles whose bounding box overlaps with the accident
        zone (IoU > 0.15 or within radius).  This prevents normal traffic that
        happens to be in the frame from being falsely flagged as hit-and-run.

        Parameters
        ----------
        camera_id : str
        location : (x, y) centroid of the accident
        tracks_in_scene : list of Track objects present in the frame
        accident_bbox : optional [x1,y1,x2,y2] of the accident detection
        """
        ax, ay = location

        # Filter: only include vehicles that are near the accident zone
        involved_ids: set[int] = set()
        for t in tracks_in_scene:
            cx, cy = t.centroid
            dist = np.hypot(cx - ax, cy - ay)
            if dist <= self.accident_radius:
                # Vehicle is within the accident radius
                involved_ids.add(t.track_id)
            elif accident_bbox is not None:
                # Check bbox overlap with accident bbox
                vbbox = [int(v) for v in t.bbox]
                iou = self._bbox_iou(vbbox, accident_bbox)
                if iou > 0.15:
                    involved_ids.add(t.track_id)

        if not involved_ids:
            logger.info("Accident registered but no vehicles overlap the accident zone — skipping hit-and-run monitoring.")
            return

        event = AccidentEvent(
            camera_id=camera_id,
            timestamp=time.time(),
            location=location,
            track_ids=involved_ids,
        )
        self._accidents.append(event)
        logger.info(
            "Accident registered at (%.0f, %.0f) — monitoring %d involved vehicles (of %d in scene)",
            location[0], location[1], len(involved_ids), len(tracks_in_scene),
        )

    @staticmethod
    def _bbox_iou(b1: list[int], b2: list[int]) -> float:
        """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, current_tracks: list) -> list[int]:
        """
        Call every frame with the current list of Track objects.

        Returns
        -------
        list of track_ids that are new hit-and-run suspects.
        """
        suspects: list[int] = []
        now = time.time()

        # Expire old accident events
        self._accidents = [
            ev for ev in self._accidents
            if (now - ev.timestamp) <= self.timeout_sec
        ]

        current_ids = {t.track_id: t for t in current_tracks}

        for ev in self._accidents:
            for tid in ev.track_ids:
                if tid in self._already_flagged:
                    continue
                if tid not in current_ids:
                    # Vehicle has left the frame / scene
                    logger.warning(
                        "Hit-and-run suspect: track %d fled scene (camera %s)",
                        tid, ev.camera_id,
                    )
                    suspects.append(tid)
                    self._already_flagged.add(tid)
                else:
                    # Check if vehicle moved far from the accident zone
                    track = current_ids[tid]
                    cx, cy = track.centroid
                    ax, ay = ev.location
                    dist = np.hypot(cx - ax, cy - ay)
                    if dist > self.accident_radius:
                        logger.warning(
                            "Hit-and-run suspect: track %d moved %.0fpx from accident (camera %s)",
                            tid, dist, ev.camera_id,
                        )
                        suspects.append(tid)
                        self._already_flagged.add(tid)

        return suspects
