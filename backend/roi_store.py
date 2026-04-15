"""
roi_store.py
Per-camera, per-ROI frame storage.

Maintains two frame buffers per (camera_id, roi_id):
    reference_frames  — fixed on first frame, only updated manually
    prev_frames       — updated every frame (used for movement detection)

ROI type controls which buffer is used for comparison:
    "crack"    → compare with reference_frames (static background diff)
    "movement" → compare with prev_frames      (temporal diff)
    "general"  → hybrid: primary=reference, secondary=prev

Thread-safe: all writes go through a single lock per camera.
"""

import threading
from typing import Dict, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────────────────────────
# FrameStore[camera_id][roi_id] = np.ndarray (grayscale crop)
FrameStore = Dict[str, Dict[str, np.ndarray]]


class ROIFrameStore:
    """
    Centralised storage for reference and previous frames,
    scoped to (camera_id, roi_id).

    Usage:
        store = ROIFrameStore()

        # On first frame for a ROI:
        store.init_roi(camera_id, roi_id, gray_crop)

        # Every subsequent frame:
        ref   = store.get_reference(camera_id, roi_id)
        prev  = store.get_previous(camera_id, roi_id)
        store.update_previous(camera_id, roi_id, gray_crop)

        # Manual reference reset (e.g. POST /reset_reference):
        store.reset_reference(camera_id, roi_id, gray_crop)
    """

    def __init__(self):
        self._reference: FrameStore = {}   # fixed; never auto-updated
        self._previous:  FrameStore = {}   # updated every frame
        self._lock = threading.Lock()

    # ──────────────────────────────────────────
    # Initialise
    # ──────────────────────────────────────────

    def init_roi(
        self,
        camera_id: str,
        roi_id: str,
        gray_crop: np.ndarray,
    ) -> None:
        """
        Call on the FIRST frame for this (camera_id, roi_id).
        Sets BOTH reference and previous to the same initial frame.
        Idempotent: no-op if already initialised.
        """
        with self._lock:
            cam_ref = self._reference.setdefault(camera_id, {})
            cam_prev = self._previous.setdefault(camera_id, {})

            if roi_id not in cam_ref:               # only set once
                cam_ref[roi_id]  = gray_crop.copy()
                cam_prev[roi_id] = gray_crop.copy()

    def is_initialised(self, camera_id: str, roi_id: str) -> bool:
        return (
            camera_id in self._reference
            and roi_id in self._reference[camera_id]
        )

    # ──────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────

    def get_reference(self, camera_id: str, roi_id: str) -> Optional[np.ndarray]:
        """Return fixed reference frame, or None if not initialised."""
        return self._reference.get(camera_id, {}).get(roi_id)

    def get_previous(self, camera_id: str, roi_id: str) -> Optional[np.ndarray]:
        """Return previous frame (last seen), or None."""
        return self._previous.get(camera_id, {}).get(roi_id)

    def get_comparison_frame(
        self,
        camera_id: str,
        roi_id: str,
        roi_type: str,
    ) -> Optional[np.ndarray]:
        """
        Return the correct comparison frame based on roi_type:
            "crack"    → reference (static diff)
            "movement" → previous  (temporal diff)
            "general"  → reference (default; fallback to previous)
        """
        if roi_type == "movement":
            return self.get_previous(camera_id, roi_id)
        # crack / general / unknown → reference
        frame = self.get_reference(camera_id, roi_id)
        if frame is None:
            frame = self.get_previous(camera_id, roi_id)
        return frame

    # ──────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────

    def update_previous(
        self,
        camera_id: str,
        roi_id: str,
        gray_crop: np.ndarray,
    ) -> None:
        """
        Update previous frame after processing.
        Called every frame — reference is NEVER touched here.
        """
        with self._lock:
            self._previous.setdefault(camera_id, {})[roi_id] = gray_crop.copy()

    def reset_reference(
        self,
        camera_id: str,
        roi_id: str,
        gray_crop: np.ndarray,
    ) -> None:
        """
        Manually reset reference frame (e.g. triggered by API).
        Also resets previous so both are in sync after reset.
        """
        with self._lock:
            self._reference.setdefault(camera_id, {})[roi_id] = gray_crop.copy()
            self._previous.setdefault(camera_id, {})[roi_id]  = gray_crop.copy()

    def reset_all_references(self, camera_id: str) -> None:
        """Reset ALL ROIs for a camera (e.g. on camera restart)."""
        with self._lock:
            self._reference.pop(camera_id, None)
            self._previous.pop(camera_id, None)

    def remove_camera(self, camera_id: str) -> None:
        """Clean up on camera deregister."""
        with self._lock:
            self._reference.pop(camera_id, None)
            self._previous.pop(camera_id, None)


# ─────────────────────────────────────────────────────────────────
# Singleton — shared across all workers
# ─────────────────────────────────────────────────────────────────
roi_store = ROIFrameStore()
