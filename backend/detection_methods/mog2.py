"""
detection_methods/mog2.py
MOG2 Background Subtraction.

Uses OpenCV's cv2.createBackgroundSubtractorMOG2 to model
the background adaptively. Suitable for scenes with slow lighting changes.
"""

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from config import settings


class MOG2Detector(BaseDetector):
    name = "mog2"

    def __init__(self, camera_id: int = 0):
        super().__init__(camera_id)
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True,
        )
        self._warmup_frames = 30  # let the model stabilise
        self._frame_count = 0

    # ──────────────────────────────────────────
    def initialize(self, frame: np.ndarray) -> None:
        """Prime the MOG2 model with the initial frame."""
        self._subtractor.apply(frame)
        self._frame_count = 1

    # ──────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> DetectionResult:
        fg_mask = self._subtractor.apply(frame)
        self._frame_count += 1

        # Shadow pixels (127) → background; remove them
        fg_mask[fg_mask == 127] = 0

        morph = self.apply_morphology(fg_mask)

        # Suppress detections during warm-up period
        if self._frame_count < self._warmup_frames:
            return DetectionResult(prediction=0, latency_ms=0.0, mask=None)

        mask, detected = self.build_filled_mask(morph)
        return DetectionResult(prediction=int(detected), latency_ms=0.0, mask=mask)
