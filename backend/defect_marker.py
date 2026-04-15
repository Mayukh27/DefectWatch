"""
defect_marker.py
Detection using LAB + Luminance combined difference.

Key fix: original code only used A-B (colour) channels.
Mouse wires, cracks, dark lines are luminance changes (L channel),
not colour changes — they were invisible to pure A-B diff.

New pipeline:
  1. LAB conversion + GaussianBlur (original)
  2. L-channel diff  (catches dark/light objects — wires, cracks)
  3. A-B magnitude diff (catches colour changes — original method)
  4. Combine: max(L_diff, AB_diff) — catches BOTH types of change
  5. Threshold → morphology → contours (original logic preserved)
  6. Draw: red bounding rect + semi-transparent filled contour
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

RED_BGR      = (0,   0,  220)
RED_FILL_BGR = (30,  30, 200)
ROI_BORDER   = (70, 100,  70)


@dataclass
class ROIDefectResult:
    roi_id:   str
    detected: bool
    contours: list                 = field(default_factory=list, repr=False)
    mask:     Optional[np.ndarray] = field(default=None,        repr=False)
    x: int = 0; y: int = 0; w: int = 0; h: int = 0


def compute_lab_defect(
    reference_lab: np.ndarray,   # blurred LAB reference crop
    current_bgr:   np.ndarray,   # current BGR crop
    threshold:     int = 5,
    min_area:      int = 300,
) -> Tuple[np.ndarray, List, bool]:
    """
    Combined L + AB difference detection.
    Detects BOTH:
      - Colour changes  (A-B channels) → coloured objects, stickers, paper
      - Luminance changes (L channel)  → dark wires, cracks, thin lines
    """
    # LAB conversion + blur (original)
    lab = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (7, 7), 0)

    # A-B magnitude difference (original — detects colour changes)
    ref_ab   = reference_lab[:, :, 1:].astype(np.float32)
    cur_ab   = lab[:, :, 1:].astype(np.float32)
    diff_ab  = cv2.absdiff(ref_ab, cur_ab)
    diff_mag_ab = np.sqrt(diff_ab[:, :, 0] ** 2 + diff_ab[:, :, 1] ** 2)

    # L-channel difference (NEW — detects luminance changes: wires, cracks, dark lines)
    diff_l = cv2.absdiff(
        reference_lab[:, :, 0].astype(np.float32),
        lab[:, :, 0].astype(np.float32)
    )

    # Combine: take the maximum of colour and luminance diffs
    # Multiply L diff by 1.5 to compensate — L values are 0-255, AB are smaller
    combined = np.maximum(diff_mag_ab, diff_l * 1.5)
    combined = np.clip(combined, 0, 255).astype("uint8")

    # Threshold (original: 5)
    _, thresh = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)

    # Morphology (original: open + dilate, ellipse 5x5, iter=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morph  = cv2.dilate(morph, kernel, iterations=2)

    # Contours + area filter (original: > 300)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid    = [c for c in contours if cv2.contourArea(c) > min_area]
    detected = bool(valid)

    mask = np.zeros(current_bgr.shape[:2], dtype=np.uint8)
    if detected:
        cv2.drawContours(mask, valid, -1, 255, thickness=cv2.FILLED)

    return mask, valid, detected


def overlay_defect_on_frame(
    frame:      np.ndarray,
    roi_mask:   np.ndarray,
    valid_cnts: list,
    roi_x:      int,
    roi_y:      int,
    alpha:      float = 0.35,
) -> np.ndarray:
    """Semi-transparent filled contour + red bounding rect per contour."""
    h, w      = roi_mask.shape[:2]
    roi_slice = frame[roi_y:roi_y+h, roi_x:roi_x+w]

    colour_layer = np.zeros_like(roi_slice)
    colour_layer[roi_mask > 0] = RED_FILL_BGR
    blended = cv2.addWeighted(roi_slice, 1.0, colour_layer, alpha, 0)

    for cnt in valid_cnts:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        cv2.rectangle(blended, (cx, cy), (cx + cw, cy + ch), RED_BGR, 2)

    frame[roi_y:roi_y+h, roi_x:roi_x+w] = blended
    return frame


def draw_roi_border(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str = "",
) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), ROI_BORDER, 1)
    if label:
        fs = 0.30
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        lx, ly = x + 2, max(y - 2, th + 2)
        cv2.rectangle(frame, (lx-1, ly-th-1), (lx+tw+1, ly+1), (10, 10, 10), cv2.FILLED)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, ROI_BORDER, 1, cv2.LINE_AA)