"""
Base class for all detection methods.
Each method must inherit this and implement initialize() and _detect().

CSV logging is handled by roi_processor._log(), NOT here.
BaseDetector.__init__ no longer touches the filesystem.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np

from config import settings


@dataclass
class DetectionResult:
    prediction: int    # 0 = no defect, 1 = defect
    latency_ms: float  # inference time in milliseconds
    mask: np.ndarray = field(default=None, repr=False)


class BaseDetector(ABC):
    """
    Abstract base for all detection methods.

    Subclasses must implement:
        initialize(frame)  — set up reference state from first frame
        _detect(frame)     — run detection, return DetectionResult

    CSV logging is intentionally absent here.
    roi_processor calls _detect() directly and handles all logging,
    so creating a detector instance has zero filesystem side-effects.
    """

    name: str = "base"

    def __init__(self, camera_id: int = 0):
        self.camera_id   = camera_id
        self.frame_index = 0
        # No _init_csv() — CSV creation was the root cause of junk empty files
        # being created on every detector instantiation.

    # ── Must override ──────────────────────────────────────────────

    @abstractmethod
    def initialize(self, frame: np.ndarray) -> None:
        """Set up reference state from the first frame."""
        ...

    @abstractmethod
    def _detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a frame.
        Return a DetectionResult (prediction + latency + optional mask).
        Called directly by roi_processor — do NOT log to CSV here.
        """
        ...

    # ── Shared CV utilities ────────────────────────────────────────

    @staticmethod
    def apply_morphology(binary: np.ndarray) -> np.ndarray:
        """Morphological open → dilate to remove noise and fill gaps."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
        dilated = cv2.dilate(opened, kernel, iterations=2)
        return dilated

    @staticmethod
    def build_filled_mask(
        morph: np.ndarray,
        min_area: int = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Find contours on morphed binary map.
        Returns (filled_mask, detected).
        """
        if min_area is None:
            min_area = settings.MIN_CONTOUR_AREA

        filled   = np.zeros_like(morph)
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detected = False
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                detected = True
                cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

        return filled, detected

    @staticmethod
    def normalize_gray(img: np.ndarray) -> np.ndarray:
        img = img.astype("float32")
        lo, hi = img.min(), img.max()
        if hi - lo < 1e-6:
            return np.zeros_like(img, dtype="uint8")
        return ((img - lo) / (hi - lo) * 255).astype("uint8")