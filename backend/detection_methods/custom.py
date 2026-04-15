"""
detection_methods/custom.py
Custom LAB-Colour Defect Detector — original by Mayukh Ghosh.

Changes from original (minimal, surgical):
  1. L-channel diff added and combined with A-B diff via max()
     → now detects dark wires, cracks (luminance change) in addition to
       colour changes that A-B alone already caught
  2. Per-ROI threshold + min_area read from self._roi_threshold / self._roi_min_area
     (set by roi_processor from dashboard sliders; falls back to settings defaults)
  3. SSIM fusion changed from bitwise_and → bitwise_or so SSIM amplifies rather
     than gates detection (bitwise_and was blocking valid detections when only
     ~14% of SSIM pixels fired)

Everything else is identical to the original:
  ✓ LAB conversion + GaussianBlur(7,7)
  ✓ A-B channel magnitude difference
  ✓ SSIM structural similarity layer (optional)
  ✓ MORPH_OPEN + dilate (ellipse 5×5, iter=2)
  ✓ Filled contour overlay
  ✓ Debug image saving (01_original … 06_final_overlay)
"""

import os
import time
import cv2
import numpy as np

from .base import BaseDetector, DetectionResult
from config import settings

try:
    from skimage.metrics import structural_similarity as ssim
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False


class CustomDefectDetector(BaseDetector):
    """
    LAB colour-space defect detector with:
      - Gaussian pre-blur
      - A-B channel magnitude difference  (colour changes)
      - L-channel difference              (wire/crack/luminance changes)  ← NEW
      - Combined diff = max(AB, L×1.5)                                    ← NEW
      - Optional SSIM fusion (OR mode — amplifies, doesn't gate)
      - Morphological open + dilate
      - Filled contour overlay (not bounding boxes)
      - Per-ROI threshold + min_area from dashboard sliders               ← NEW
    """

    name = "custom"

    def __init__(self, camera_id: int = 0, use_ssim: bool = True):
        super().__init__(camera_id)
        self._reference_lab:  np.ndarray = None
        self._reference_gray: np.ndarray = None
        self.use_ssim = use_ssim and _SSIM_AVAILABLE
        self._save_intermediates = True   # Save debug images like original code

    # ──────────────────────────────────────────
    def initialize(self, frame: np.ndarray) -> None:
        """Convert first frame to LAB and store as reference."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        self._reference_lab  = cv2.GaussianBlur(lab, (7, 7), 0)
        self._reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ──────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> DetectionResult:
        if self._reference_lab is None:
            self.initialize(frame)
            return DetectionResult(prediction=0, latency_ms=0.0, mask=None)

        # Per-ROI sensitivity (set by roi_processor from dashboard sliders)
        thr      = getattr(self, "_roi_threshold", settings.BINARY_THRESHOLD)
        min_area = getattr(self, "_roi_min_area",  settings.MIN_CONTOUR_AREA)

        # ── LAB conversion (original) ──────────────────────────────
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab = cv2.GaussianBlur(lab, (7, 7), 0)

        # ── A-B magnitude difference (original) ───────────────────
        diff_ab  = cv2.absdiff(
            self._reference_lab[:, :, 1:].astype("float32"),
            lab[:, :, 1:].astype("float32"),
        )
        diff_mag = np.sqrt(diff_ab[:, :, 0] ** 2 + diff_ab[:, :, 1] ** 2)

        # ── L-channel difference (NEW — detects wires, cracks) ────
        # A-B channels only capture colour shifts. Dark wires and cracks
        # have the same hue as the background but are much darker → L diff.
        l_diff = cv2.absdiff(
            self._reference_lab[:, :, 0].astype("float32"),
            lab[:, :, 0].astype("float32"),
        )

        # ── Combine: take max of colour diff and luminance diff (NEW)
        # L values range 0-255 like A-B magnitude, ×1.5 compensates for
        # the fact that luminance differences are typically smaller per-pixel.
        diff_combined = np.maximum(diff_mag, l_diff * 1.5)
        diff_uint8    = np.clip(diff_combined, 0, 255).astype("uint8")

        # ── Threshold (original used Otsu; now uses per-ROI value) ─
        _, thresh = cv2.threshold(diff_uint8, thr, 255, cv2.THRESH_BINARY)

        # ── Optional SSIM layer (original, fusion mode changed) ────
        if self.use_ssim:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score, ssim_diff = ssim(
                self._reference_gray, gray, full=True, data_range=255
            )
            ssim_map = ((1 - ssim_diff) * 127.5).clip(0, 255).astype("uint8")
            _, ssim_thresh = cv2.threshold(ssim_map, 30, 255, cv2.THRESH_BINARY)
            # OR instead of AND: SSIM adds evidence rather than gating.
            # Original AND suppressed ~86% of valid pixels when only part of
            # the defect region had high SSIM disagreement.
            thresh = cv2.bitwise_or(thresh, ssim_thresh)

        # ── Morphology (original: open + dilate, ellipse 5×5 iter=2)
        morph = self.apply_morphology(thresh)

        # ── Filled contour mask (original) ────────────────────────
        # Filter by per-ROI min_area instead of hardcoded value
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid    = [c for c in cnts if cv2.contourArea(c) > min_area]
        detected = bool(valid)

        mask = np.zeros(frame.shape[:2], dtype="uint8")
        if detected:
            cv2.drawContours(mask, valid, -1, 255, thickness=cv2.FILLED)

        # ── Save intermediates (original feature) ─────────────────
        if detected and self._save_intermediates:
            self._save_debug_images(frame, diff_combined, thresh, morph, mask)

        return DetectionResult(prediction=int(detected), latency_ms=0.0, mask=mask)

    # ──────────────────────────────────────────
    def _save_debug_images(
        self,
        original: np.ndarray,
        diff_mag: np.ndarray,
        thresh:   np.ndarray,
        morph:    np.ndarray,
        mask:     np.ndarray,
    ) -> None:
        """
        Save intermediate processing images identical to the original project's
        pipeline visualisation (01_original … 06_final_overlay).
        """
        folder = os.path.join(
            settings.BASE_DIR,
            "static", "captured", "processing",
            time.strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(folder, exist_ok=True)

        # Build final overlay using filled mask (not bounding boxes)
        overlay      = original.copy()
        colored_mask = np.zeros_like(original)
        colored_mask[mask > 0] = (0, 0, 220)
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.50, 0)

        # Reconstruct LAB AB magnitude for visualisation (column 02)
        lab    = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        ab_mag = np.sqrt(
            lab[:, :, 1].astype("float32") ** 2 +
            lab[:, :, 2].astype("float32") ** 2
        )

        cv2.imwrite(f"{folder}/01_original.png",      original)
        cv2.imwrite(f"{folder}/02_lab_ab_mag.png",    self.normalize_gray(ab_mag))
        cv2.imwrite(f"{folder}/03_diff_mag.png",      self.normalize_gray(diff_mag))
        cv2.imwrite(f"{folder}/04_threshold.png",     thresh)
        cv2.imwrite(f"{folder}/05_morph.png",         morph)
        cv2.imwrite(f"{folder}/06_final_overlay.png", overlay)