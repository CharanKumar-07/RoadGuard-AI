# src/utils/video_reader.py
"""
Multi-source video reader with auto-reconnect for RTSP streams.
"""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Opens a video source (file, webcam index, or RTSP URL) and yields frames.

    Parameters
    ----------
    source : str | int
        File path, camera index (0, 1, …), or RTSP URL.
    retry_delay : float
        Seconds to wait before reconnecting after a lost stream.
    max_retries : int
        Maximum consecutive reconnect attempts before giving up (0 = infinite).
    width, height : int
        Desired capture resolution (webcam / RTSP only).
    """

    def __init__(
        self,
        source,
        retry_delay: float = 2.0,
        max_retries: int   = 10,
        width:  int = 0,
        height: int = 0,
    ) -> None:
        self.source       = source
        self.retry_delay  = retry_delay
        self.max_retries  = max_retries
        self.width        = width
        self.height       = height
        self._cap: cv2.VideoCapture | None = None
        self._is_file_source = isinstance(source, str) and os.path.exists(source)

    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the video source.  Returns True on success."""
        src = int(self.source) if str(self.source).isdigit() else self.source
        self._cap = cv2.VideoCapture(src)
        if self.width and self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        ok = self._cap.isOpened()
        if ok:
            logger.info("Opened video source: %s", self.source)
        else:
            logger.error("Cannot open video source: %s", self.source)
        return ok

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame.  Returns (success, frame)."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame

    # ------------------------------------------------------------------

    def stream(self):
        """
        Generator that yields frames.  Automatically reconnects on failure.

        Yields
        ------
        np.ndarray  BGR frames
        """
        retries = 0
        self.open()

        while True:
            ret, frame = self.read()
            if ret:
                retries = 0
                yield frame
            else:
                if self._is_file_source:
                    logger.info("Reached end of file for source: %s", self.source)
                    break

                retries += 1
                logger.warning(
                    "Stream read failed for %s (attempt %d/%s) — retrying in %.1fs …",
                    self.source, retries, self.max_retries or "∞", self.retry_delay,
                )
                self.release()

                if self.max_retries and retries > self.max_retries:
                    logger.error("Max retries exceeded for %s — stopping.", self.source)
                    break

                time.sleep(self.retry_delay)
                self.open()

    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        return 25.0
