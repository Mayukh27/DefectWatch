"""
roi_processor.py
ROI-based detection coordinator.

Architecture
------------
Each (camera_id, roi_id, method) gets its OWN detector instance from
detection_methods/{method}.py. This means:

  detection_methods/
    custom.py       → CustomDefectDetector   (LAB + L-channel, original algorithm)
    fd.py           → FrameDifferencingDetector
    mog2.py         → MOG2Detector
    running_avg.py  → RunningAverageDetector
    dl_model.py     → DLModelDetector        (inactive)

This file ONLY:
  • Maintains per-ROI detector instances
  • Crops the ROI from each frame
  • Calls detector.process(crop) or detector._detect(crop)
  • Calls defect_marker to draw the result on the frame
  • Logs predictions to predictions/{method}_{camera_id}.csv
  • Saves reference/current crops for the Before/After popup
  • Shows "SELECT ROI" overlay when no ROI is configured

No detection algorithm lives here.

CSV OUTPUT FORMAT
-----------------
File:    predictions/{method}_{camera_id}.csv
Columns: frame, camera_id, roi_id, prediction, score, latency_ms
Example: predictions/custom_0.csv
         predictions/fd_1.csv

One file per (method, camera_id) pair. Frame index starts at 0 for
each camera independently — no cross-camera offset is applied.
The legacy single-file format (custom.csv, fd.csv ...) is never written.
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings

# -- Detection method imports (each algorithm in its own file) -----
from detection_methods.custom      import CustomDefectDetector
from detection_methods.fd          import FrameDifferencingDetector
from detection_methods.mog2        import MOG2Detector
from detection_methods.running_avg import RunningAverageDetector
from detection_methods.dl_model    import DLModelDetector

from defect_marker import overlay_defect_on_frame, draw_roi_border

# -- ROI result dataclass ------------------------------------------
@dataclass
class ROIDefectResult:
    roi_id:   str
    detected: bool
    contours: list                 = field(default_factory=list, repr=False)
    mask:     Optional[np.ndarray] = field(default=None,        repr=False)
    x: int = 0; y: int = 0; w: int = 0; h: int = 0


# -- Method registry (key -> class) --------------------------------
METHOD_CLASSES = {
    "custom":      CustomDefectDetector,
    "fd":          FrameDifferencingDetector,
    "mog2":        MOG2Detector,
    "running_avg": RunningAverageDetector,
    "dl":          DLModelDetector,
}

# Methods to run silently on every frame for evaluation logging.
# All process the same ROI crop with the same frame number -> comparable CSVs.
# Only the selected method (default: custom) is used for display output.
EVAL_METHODS = ["custom", "fd", "mog2", "running_avg"]


# -- Per-camera state ----------------------------------------------
_camera_rois: Dict[str, List[dict]] = {}
_detectors:   Dict[str, Dict[str, Dict[str, object]]] = {}
_snapshots:   Dict[str, dict] = {}


# -- Prediction CSV logging ----------------------------------------
_PRED_DIR = Path(settings.PREDICTIONS_DIR)

# Per-(method, camera_id) frame counter — each camera index starts at 0.
_frame_counters: Dict[Tuple[str, str], int] = {}

# Track which (method, camera_id) files have had their header written.
_csv_headers_written: set = set()


def _log(
    method:     str,
    camera_id:  str,
    roi_id:     str,
    prediction: int,
    score:      float,
    latency_ms: float,
) -> None:
    """
    Append one prediction row to predictions/{method}_{camera_id}.csv.

    CSV columns (always present, always with header on first write):
        frame, camera_id, roi_id, prediction, score, latency_ms

    Key design decisions:
      - One file per (method, camera_id): custom_0.csv, fd_1.csv, etc.
      - frame index is independent per (method, camera_id) — starts at 0
        for every camera, no cross-camera offset applied.
      - camera_id and roi_id are written as columns so evaluator can merge
        on (frame, camera_id, roi_id) without ambiguity.
      - score is the defect-area fraction (0.0-1.0) used for ROC curves.
      - The old legacy files (custom.csv, fd.csv ...) are NEVER created.
    """
    _PRED_DIR.mkdir(parents=True, exist_ok=True)

    file_key = (method, camera_id)
    csv_path = _PRED_DIR / f"{method}_{camera_id}.csv"

    _frame_counters[file_key] = _frame_counters.get(file_key, -1) + 1
    idx = _frame_counters[file_key]

    write_header = file_key not in _csv_headers_written and not csv_path.exists()
    _csv_headers_written.add(file_key)

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["frame", "camera_id", "roi_id", "prediction", "score", "latency_ms"])
        w.writerow([idx, camera_id, roi_id, prediction, round(score, 5), round(latency_ms, 3)])


# -- Public API ----------------------------------------------------

def set_rois(camera_id: str, rois: List[dict]) -> None:
    """Register ROI zones for a camera. Clears all existing detector state."""
    _camera_rois[camera_id] = rois
    _detectors.pop(camera_id, None)
    _snapshots.pop(camera_id, None)


def get_rois(camera_id: str) -> List[dict]:
    return _camera_rois.get(camera_id, [])


def has_rois(camera_id: str) -> bool:
    return bool(_camera_rois.get(camera_id))


def remove_camera_rois(camera_id: str) -> None:
    _camera_rois.pop(camera_id, None)
    _detectors.pop(camera_id, None)
    _snapshots.pop(camera_id, None)


def get_snapshot(camera_id: str) -> Optional[dict]:
    return _snapshots.get(camera_id)


def _clear_state(camera_id: str) -> None:
    _detectors.pop(camera_id, None)
    _snapshots.pop(camera_id, None)


def reset_all_references(camera_id: str) -> None:
    """Force all detectors for this camera to re-initialize on next frame."""
    _clear_state(camera_id)


def reset_reference_for_roi(camera_id: str, roi_id: str, frame: np.ndarray) -> bool:
    """Reset detector for one specific ROI."""
    cam_dets = _detectors.get(camera_id, {})
    if roi_id in cam_dets:
        cam_dets.pop(roi_id)
    _snapshots.get(camera_id, {}).pop(roi_id + "_ref", None)
    _snapshots.get(camera_id, {}).pop(roi_id + "_cur", None)
    return True


def _get_detector(camera_id: str, roi_id: str, method: str,
                  threshold: int, min_area: int) -> object:
    """
    Return the detector for (camera_id, roi_id, method).
    Creates a new instance if one does not exist yet.
    """
    cam = _detectors.setdefault(camera_id, {})
    roi = cam.setdefault(roi_id, {})

    if method not in roi:
        DetCls = METHOD_CLASSES.get(method, CustomDefectDetector)
        det = DetCls(camera_id=0)
        det._roi_threshold = threshold
        det._roi_min_area  = min_area
        roi[method] = det
    else:
        roi[method]._roi_threshold = threshold
        roi[method]._roi_min_area  = min_area

    return roi[method]


# -- "SELECT ROI" overlay ------------------------------------------

def _draw_roi_required(frame: np.ndarray) -> np.ndarray:
    """Dimmed overlay with instruction text when no ROI is configured."""
    out = frame.copy()
    h, w = out.shape[:2]
    dark = out.copy()
    cv2.rectangle(dark, (0, 0), (w, h), (8, 8, 10), cv2.FILLED)
    out = cv2.addWeighted(out, 0.5, dark, 0.5, 0)

    for text, y_off, scale, colour in [
        ("SELECT ROI TO START DETECTION", -10, 0.55, (200, 200, 200)),
        ("Click  Set ROI zones  below",    18, 0.38, (100, 100, 100)),
    ]:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(out, text, ((w - tw) // 2, h // 2 + y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    return out


# -- Main entry point ----------------------------------------------

def process_frame(
    frame:     np.ndarray,
    camera_id: str,
    rois:      Optional[List[dict]] = None,
    method:    str = "custom",
) -> Tuple[np.ndarray, bool, List[ROIDefectResult]]:
    """
    Process one frame: run ALL eval methods inside each ROI, draw
    the selected method's result, and log every method to its CSV.

    Args:
        frame     -- full BGR camera frame
        camera_id -- string camera id (becomes part of the CSV filename
                     AND is written into the camera_id column so the
                     evaluator can merge on it)
        rois      -- ROI list; if None uses registry
        method    -- which detection method drives the display output

    Returns:
        (annotated_frame, any_defect_detected, [ROIDefectResult, ...])
    """
    if rois is None:
        rois = get_rois(camera_id)

    if not rois:
        return _draw_roi_required(frame), False, []

    # DL is inactive -- draw borders only
    if method == "dl":
        output = frame.copy()
        for i, roi in enumerate(rois):
            x, y, w, h = int(roi.get("x",0)), int(roi.get("y",0)), int(roi.get("w",0)), int(roi.get("h",0))
            draw_roi_border(output, x, y, w, h, label=roi.get("roi_id", f"roi_{i}"))
        return output, False, []

    output     = frame.copy()
    any_defect = False
    results: List[ROIDefectResult] = []

    for i, roi in enumerate(rois):
        roi_id    = roi.get("roi_id", f"roi_{i}")
        x         = int(roi.get("x",         0))
        y         = int(roi.get("y",         0))
        w         = int(roi.get("w",         0))
        h         = int(roi.get("h",         0))
        threshold = int(roi.get("threshold", 8))
        min_area  = int(roi.get("min_area",  150))

        fh, fw = frame.shape[:2]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > fw or y + h > fh:
            continue

        crop = frame[y:y+h, x:x+w]

        # -- Run ALL eval methods on the same crop -----------------
        # Every method sees the same frame crop at the same frame index,
        # keeping all prediction CSVs frame-aligned for valid comparison.
        display_detected = False
        display_mask     = None

        for eval_method in EVAL_METHODS:
            det = _get_detector(camera_id, roi_id, eval_method, threshold, min_area)

            t0 = time.perf_counter()

            # First frame: initialise reference background
            if not hasattr(det, "_initialized_"):
                det.initialize(crop)
                det._initialized_ = True
                lat = (time.perf_counter() - t0) * 1000

                # Log initialisation frame as clean (prediction=0, score=0)
                _log(eval_method, camera_id, roi_id, 0, 0.0, lat)

                if eval_method == method:
                    _snapshots.setdefault(camera_id, {})[roi_id + "_ref"] = crop.copy()
                    _snapshots[camera_id]["latest_roi_id"]   = roi_id
                    _snapshots[camera_id]["latest_roi_type"] = method
                continue

            # Subsequent frames: detect
            result   = det._detect(crop)
            lat      = (time.perf_counter() - t0) * 1000
            det_flag = bool(result.prediction)

            if result.mask is not None:

                roi_area = result.mask.size
                changed_pixels = np.sum(result.mask > 0)

                area_ratio = changed_pixels / float(roi_area)

                contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                max_contour_area = 0.0
                if contours:
                    max_contour_area = max(cv2.contourArea(c) for c in contours)

                contour_ratio = max_contour_area / float(roi_area)

                # 🔥 NEW: penalize noise
                if area_ratio < 0.01:
                    area_ratio *= 0.3

                # 🔥 NEW: weighted nonlinear scoring
                score = (0.6 * area_ratio + 0.4 * contour_ratio)

                # 🔥 CRITICAL: expand distribution
                score = score ** 1.5   # spreads values

                score = max(0.0, min(1.0, score))

            else:
                score = 0.0
            
            _log(eval_method, camera_id, roi_id, int(det_flag), score, lat)

            if eval_method == method:
                display_detected = det_flag
                display_mask     = result.mask

        # Skip draw on first frame (detector not yet initialised)
        display_det = _detectors.get(camera_id, {}).get(roi_id, {}).get(method)
        if display_det is not None and not hasattr(display_det, "_initialized_"):
            draw_roi_border(output, x, y, w, h, label=roi_id)
            results.append(ROIDefectResult(roi_id=roi_id, detected=False,
                                           x=x, y=y, w=w, h=h))
            continue

        detected = display_detected
        mask     = display_mask

        if detected and mask is not None:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
            if valid_cnts:
                overlay_defect_on_frame(output, mask, valid_cnts, x, y, alpha=0.35)
                any_defect = True
                _snapshots.setdefault(camera_id, {})[roi_id + "_cur"] = crop.copy()
                _snapshots[camera_id]["latest_roi_id"]   = roi_id
                _snapshots[camera_id]["latest_roi_type"] = method
                _snapshots[camera_id]["latest_ts"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                detected = False

        draw_roi_border(output, x, y, w, h, label=f"{roi_id} [{method}]")

        results.append(ROIDefectResult(
            roi_id=roi_id, detected=detected,
            contours=[], mask=mask, x=x, y=y, w=w, h=h,
        ))

    return output, any_defect, results