"""
Base class for all detection methods.
Each method must inherit this and implement initialize() and process().
"""

import csv
import os
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
    mask: np.ndarray = field(default=None, repr=False)  # filled contour mask (optional)


class BaseDetector(ABC):
    """
    Abstract base for all detection methods.

    Subclasses must implement:
        initialize(frame)  — set up reference state
        process(frame)     — run detection, return (annotated_frame, prediction)
    """

    name: str = "base"

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.frame_index = 0
        self._csv_path = os.path.join(settings.PREDICTIONS_DIR, f"{self.name}.csv")
        self._init_csv()

    # ──────────────────────────────────────────
    # Must override
    # ──────────────────────────────────────────

    @abstractmethod
    def initialize(self, frame: np.ndarray) -> None:
        """Set up reference state from the first frame."""
        ...

    @abstractmethod
    def _detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a frame.
        Return a DetectionResult (prediction + latency + optional mask).
        """
        ...

    # ──────────────────────────────────────────
    # Public API (called by app.py worker)
    # ──────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Run detection and return (annotated_frame, prediction).
        Logs result to CSV automatically.
        """
        t0 = time.perf_counter()
        result = self._detect(frame)
        latency_ms = (time.perf_counter() - t0) * 1000

        self._log(self.frame_index, result.prediction, latency_ms)
        self.frame_index += 1

        annotated = self._annotate(frame.copy(), result)
        return annotated, result.prediction

    # ──────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Overlay filled contour mask on frame.
        Red semi-transparent fill (research requirement §10).
        """
        if result.mask is not None and result.prediction == 1:
            overlay = frame.copy()
            # Filled contour mask
            colored_mask = np.zeros_like(frame)
            colored_mask[result.mask > 0] = (0, 0, 220)  # BGR red fill
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.45, 0)

        # Status label
        label = "DEFECT" if result.prediction == 1 else "OK"
        color = (0, 0, 255) if result.prediction == 1 else (0, 200, 80)
        cv2.putText(
            frame, f"[{self.name.upper()}] {label}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
        )
        return frame

    # ──────────────────────────────────────────
    # Shared CV utilities
    # ──────────────────────────────────────────

    @staticmethod
    def apply_morphology(binary: np.ndarray) -> np.ndarray:
        """
        Morphological open → dilate to remove noise and fill gaps.
        Used by all classical methods.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        dilated = cv2.dilate(opened, kernel, iterations=2)
        return dilated

    @staticmethod
    def build_filled_mask(
        morph: np.ndarray,
        min_area: int = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Find contours on morphed binary map.
        Returns:
            filled_mask — uint8 image with filled contour shapes
            detected    — True if any contour exceeds min_area
        """
        if min_area is None:
            min_area = settings.MIN_CONTOUR_AREA

        filled = np.zeros_like(morph)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # ──────────────────────────────────────────
    # CSV logging
    # ──────────────────────────────────────────
    def _init_csv(self):
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["frame", "prediction", "latency_ms"])

    def _log(self, frame_idx: int, prediction: int, latency_ms: float):
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([frame_idx, prediction, round(latency_ms, 3)])