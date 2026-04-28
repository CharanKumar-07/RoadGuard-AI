# src/tracking/tracker.py
"""
Lightweight IoU-based multi-object tracker (ByteTrack-inspired).

Keeps stable integer IDs across frames without requiring external dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class Track:
    track_id: int
    bbox: list[int]
    label: str
    confidence: float
    missed: int = 0                           # consecutive frames without match
    history: list[tuple[float, float]] = field(default_factory=list)  # centroid log

    @property
    def centroid(self) -> tuple[float, float]:
        return _centroid(self.bbox)


class SimpleTracker:
    """
    IoU-based tracker.  Call ``update()`` every frame with detector output.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to associate a detection with an existing track.
    max_missed : int
        Frames without match before a track is removed.
    max_history : int
        Max centroid history entries kept per track.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 15,
        max_history: int = 50,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed    = max_missed
        self.max_history   = max_history
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    # ------------------------------------------------------------------

    def update(self, detections: list[dict]) -> list[Track]:
        """
        Parameters
        ----------
        detections : list of dict
            Each dict has keys: 'bbox' [x1,y1,x2,y2], 'confidence', 'label'.

        Returns
        -------
        list[Track]  — currently active tracks.
        """
        if not detections:
            self._age_tracks()
            return list(self._tracks.values())

        track_ids   = list(self._tracks.keys())
        track_boxes = [self._tracks[t].bbox for t in track_ids]
        det_boxes   = [d["bbox"] for d in detections]

        matched_tracks: set[int] = set()
        matched_dets:   set[int] = set()

        # Build IoU cost matrix
        if track_ids:
            iou_matrix = np.zeros((len(track_ids), len(det_boxes)), dtype=float)
            for i, tb in enumerate(track_boxes):
                for j, db in enumerate(det_boxes):
                    iou_matrix[i, j] = _iou(tb, db)

            # Greedy matching (highest IoU first)
            ranked = np.dstack(np.unravel_index(
                np.argsort(-iou_matrix, axis=None), iou_matrix.shape
            ))[0]
            for ti, di in ranked:
                if iou_matrix[ti, di] < self.iou_threshold:
                    break
                if ti in matched_tracks or di in matched_dets:
                    continue
                tid = track_ids[ti]
                t   = self._tracks[tid]
                det = detections[di]
                t.bbox       = det["bbox"]
                t.confidence = det["confidence"]
                t.label      = det.get("label", t.label)
                t.missed     = 0
                t.history.append(_centroid(t.bbox))
                if len(t.history) > self.max_history:
                    t.history.pop(0)
                matched_tracks.add(ti)
                matched_dets.add(di)

        # Age unmatched existing tracks
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[tid].missed += 1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_track = Track(
                    track_id=self._next_id,
                    bbox=det["bbox"],
                    label=det.get("label", "unknown"),
                    confidence=det["confidence"],
                )
                new_track.history.append(_centroid(new_track.bbox))
                self._tracks[self._next_id] = new_track
                self._next_id += 1

        # Remove stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.missed > self.max_missed]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    def _age_tracks(self) -> None:
        for t in self._tracks.values():
            t.missed += 1
        stale = [tid for tid, t in self._tracks.items() if t.missed > self.max_missed]
        for tid in stale:
            del self._tracks[tid]

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1