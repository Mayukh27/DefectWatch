"""
detection_methods/running_avg.py
Running Average Background Subtraction.

Maintains a floating-point accumulation of past frames as the
background model. Slower to adapt than MOG2 but smoother.
"""

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from config import settings


class RunningAverageDetector(BaseDetector):
    name = "running_avg"

    def __init__(self, camera_id: int = 0, alpha: float = 0.05):
        super().__init__(camera_id)
        self._bg_float: np.ndarray = None
        # Alpha controls update speed: small = slower adaptation
        self.alpha = alpha

    # ──────────────────────────────────────────
    def initialize(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._bg_float = gray.astype("float32")

    # ──────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> DetectionResult:
        if self._bg_float is None:
            self.initialize(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype("float32")

        diff = cv2.absdiff(self._bg_float.astype("uint8"), gray)
        _, thresh = cv2.threshold(
            diff, settings.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        morph = self.apply_morphology(thresh)
        mask, detected = self.build_filled_mask(morph)

        # Update running average with current frame
        cv2.accumulateWeighted(gray_f, self._bg_float, self.alpha)

        return DetectionResult(prediction=int(detected), latency_ms=0.0, mask=mask)
