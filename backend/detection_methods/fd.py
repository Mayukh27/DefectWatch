"""
detection_methods/fd.py
Frame Differencing (FD) — simplest baseline.

Algorithm:
  1. Convert both reference and current frames to grayscale + Gaussian blur
  2. Absolute difference → threshold → morphology
  3. Filled contour mask → prediction
"""

import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from config import settings


class FrameDifferencingDetector(BaseDetector):
    name = "fd"

    def __init__(self, camera_id: int = 0):
        super().__init__(camera_id)
        self._reference_gray: np.ndarray = None

    # ──────────────────────────────────────────
    def initialize(self, frame: np.ndarray) -> None:
        """Store blurred grayscale reference."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._reference_gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # ──────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> DetectionResult:
        if self._reference_gray is None:
            self.initialize(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        diff = cv2.absdiff(self._reference_gray, blurred)
        _, thresh = cv2.threshold(
            diff, settings.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        morph = self.apply_morphology(thresh)
        mask, detected = self.build_filled_mask(morph)

        return DetectionResult(
            prediction=int(detected),
            latency_ms=0.0,  # measured externally in base.process()
            mask=mask,
        )
