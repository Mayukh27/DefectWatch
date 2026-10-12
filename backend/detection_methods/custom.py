"""
detection_methods/custom.py  (v2)
Custom LAB-Colour Defect Detector — original by Mayukh Ghosh.

v1 -> v2 changes
  1. L-channel difference combined with A-B distance via max()      (v1)
  2. Per-ROI threshold / min_area from dashboard sliders             (v1)
  3. SSIM fusion changed AND -> OR                                   (v1)
  4. NEW  illumination normalisation (optional, settings.ILLUM_NORM):
        the multiplicative change between reference and current lightness is
        estimated as a robust, low-frequency ratio field (median-filtered,
        heavily smoothed, computed at 1/4 resolution) and divided out of the
        current L channel before differencing. Defects smaller than the
        smoothing scale are NOT absorbed, global gain and soft shadows are.
  5. NEW  ablation switches: use_luminance, use_ssim, illum_norm
  6. NEW  continuous threshold-free `raw_score` (99th percentile of the
        combined difference / 255) for proper ROC / PR analysis
  7. NEW  config_dict() so every logged run records its operating point
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
except ImportError:                                    # pragma: no cover
    _SSIM_AVAILABLE = False

L_WEIGHT = 1.5          # luminance difference weight in max(AB, L_WEIGHT*L)
SSIM_THRESHOLD = 30     # on the map 127.5*(1-SSIM)


def normalise_luminance(l_ref: np.ndarray, l_cur: np.ndarray) -> np.ndarray:
    """Divide the smooth multiplicative illumination change out of l_cur.

    l_ref, l_cur: blurred float32 L channels (same shape).
    Robust ratio field: ratio = (l_cur+1)/(l_ref+1) at 1/4 resolution ->
    5x5 median (rejects small defects) -> Gaussian smoothing with
    sigma ~ max(side)/5 (keeps only global gain + soft shading).
    """
    h, w = l_cur.shape
    sw, sh = max(8, w // 4), max(8, h // 4)
    ratio = (l_cur + 1.0) / (l_ref + 1.0)
    small = cv2.resize(ratio, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small.astype("float32"), 5)
    sigma = max(sw, sh) / 5.0
    small = cv2.GaussianBlur(small, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    shade = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    shade = np.clip(shade, 0.5, 2.0)
    return np.clip((l_cur + 1.0) / shade - 1.0, 0, 255)


class CustomDefectDetector(BaseDetector):
    name = "custom"

    def __init__(self, camera_id: int = 0, use_ssim: bool = True,
                 use_luminance: bool = True, illum_norm: bool = None):
        super().__init__(camera_id)
        self._reference_lab = None
        self._reference_gray = None
        self.use_ssim = use_ssim and _SSIM_AVAILABLE
        self.use_luminance = use_luminance
        self.illum_norm = settings.ILLUM_NORM if illum_norm is None else illum_norm
        self._save_intermediates = True

    def initialize(self, frame: np.ndarray) -> None:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        self._reference_lab = cv2.GaussianBlur(lab, (7, 7), 0)
        self._reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def config_dict(self) -> dict:
        return {"method": self.name,
                "threshold": int(getattr(self, "_roi_threshold", settings.BINARY_THRESHOLD)),
                "min_area": int(getattr(self, "_roi_min_area", settings.MIN_CONTOUR_AREA)),
                "use_luminance": self.use_luminance, "use_ssim": self.use_ssim,
                "illum_norm": self.illum_norm, "l_weight": L_WEIGHT,
                "ssim_threshold": SSIM_THRESHOLD}

    def _detect(self, frame: np.ndarray) -> DetectionResult:
        if self._reference_lab is None:
            self.initialize(frame)
            return DetectionResult(prediction=0, latency_ms=0.0, mask=None)

        thr = getattr(self, "_roi_threshold", settings.BINARY_THRESHOLD)
        min_area = getattr(self, "_roi_min_area", settings.MIN_CONTOUR_AREA)

        lab = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), (7, 7), 0)
        ref = self._reference_lab

        d_ab = cv2.absdiff(ref[:, :, 1:].astype("float32"), lab[:, :, 1:].astype("float32"))
        diff_mag = np.sqrt(d_ab[:, :, 0] ** 2 + d_ab[:, :, 1] ** 2)
        diff_combined = diff_mag

        if self.use_luminance:
            l_ref = ref[:, :, 0].astype("float32")
            l_cur = lab[:, :, 0].astype("float32")
            if self.illum_norm:
                l_cur = normalise_luminance(l_ref, l_cur)
            l_diff = cv2.absdiff(l_ref, l_cur)
            diff_combined = np.maximum(diff_mag, l_diff * L_WEIGHT)

        diff_uint8 = np.clip(diff_combined, 0, 255).astype("uint8")
        _, thresh = cv2.threshold(diff_uint8, thr, 255, cv2.THRESH_BINARY)

        if self.use_ssim:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, ssim_diff = ssim(self._reference_gray, gray, full=True, data_range=255)
            ssim_map = ((1 - ssim_diff) * 127.5).clip(0, 255).astype("uint8")
            _, ssim_thresh = cv2.threshold(ssim_map, SSIM_THRESHOLD, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_or(thresh, ssim_thresh)       # OR: amplify, don't gate

        morph = self.apply_morphology(thresh)
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        detected = bool(valid)

        mask = np.zeros(frame.shape[:2], dtype="uint8")
        if detected:
            cv2.drawContours(mask, valid, -1, 255, thickness=cv2.FILLED)

        if detected and self._save_intermediates:
            self._save_debug_images(frame, diff_combined, thresh, morph, mask)

        raw = float(np.percentile(diff_combined, 99)) / 255.0
        return DetectionResult(prediction=int(detected), latency_ms=0.0, mask=mask, raw_score=raw)

    def _save_debug_images(self, original, diff_mag, thresh, morph, mask) -> None:
        folder = os.path.join(settings.BASE_DIR, "static", "captured", "processing",
                              time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        overlay = original.copy()
        colored = np.zeros_like(original)
        colored[mask > 0] = (0, 0, 220)
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.50, 0)
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        ab_mag = np.sqrt(lab[:, :, 1].astype("float32") ** 2 + lab[:, :, 2].astype("float32") ** 2)
        cv2.imwrite(f"{folder}/01_original.png", original)
        cv2.imwrite(f"{folder}/02_lab_ab_mag.png", self.normalize_gray(ab_mag))
        cv2.imwrite(f"{folder}/03_diff_mag.png", self.normalize_gray(diff_mag))
        cv2.imwrite(f"{folder}/04_threshold.png", thresh)
        cv2.imwrite(f"{folder}/05_morph.png", morph)
        cv2.imwrite(f"{folder}/06_final_overlay.png", overlay)
