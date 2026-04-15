"""
camera_registry.py
Unified camera source abstraction layer.

Tracks ALL cameras regardless of type (rtsp / client / local).
Existing code (camera_states, camera_caps, camera_threads in app.py) is untouched.
This module adds the parallel registry needed for RTSP + client-push sources.

Structure of each registry entry:
    active_cameras[camera_id] = {
        "camera_id" : str,
        "type"      : "rtsp" | "client" | "local",
        "source"    : rtsp_url | "push" | int,
        "active"    : bool,
        "defect"    : bool,
        "method"    : str,
        "fps"       : float,
        "last_frame": np.ndarray | None,
        "last_ts"   : str | None,
        "detector"  : BaseDetector | None,
    }
"""

import threading
import time
from datetime import datetime
from typing import Dict, Optional, Any

import cv2
import numpy as np

from config import settings
from detection_methods.fd import FrameDifferencingDetector
from detection_methods.mog2 import MOG2Detector
from detection_methods.running_avg import RunningAverageDetector
from detection_methods.custom import CustomDefectDetector
from detection_methods.dl_model import DLModelDetector
from roi_processor import process_frame as _roi_process_frame, get_rois, remove_camera_rois

# ─────────────────────────────────────────────────────────────────
# Registry globals
# ─────────────────────────────────────────────────────────────────

#: Unified camera registry  { camera_id:str → state:dict }
active_cameras: Dict[str, dict] = {}

#: Latest raw/processed frame per camera  { camera_id:str → np.ndarray }
frames: Dict[str, np.ndarray] = {}

#: Background threads  { camera_id:str → Thread }
_threads: Dict[str, threading.Thread] = {}

_registry_lock = threading.Lock()

METHOD_MAP = {
    "fd":          FrameDifferencingDetector,
    "mog2":        MOG2Detector,
    "running_avg": RunningAverageDetector,
    "custom":      CustomDefectDetector,
    "dl":          DLModelDetector,
}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_state(
    camera_id: str,
    cam_type: str,
    source: Any,
    method: str,
) -> dict:
    return {
        "camera_id":  camera_id,
        "type":       cam_type,       # "rtsp" | "client" | "local"
        "source":     source,
        "active":     False,
        "defect":     False,
        "method":     method,
        "fps":        0.0,
        "last_frame": None,
        "last_ts":    None,
        "detector":   None,
        "_fps_count": 0,
        "_fps_start": time.time(),
    }


def _init_detector(state: dict) -> None:
    """Instantiate detector for the camera and attach to state."""
    DetCls = METHOD_MAP.get(state["method"], CustomDefectDetector)
    state["detector"] = DetCls(camera_id=0)   # camera_id=0 placeholder for string ids


def _run_detection(state: dict, frame: np.ndarray) -> np.ndarray:
    """
    Run detection on one frame using the selected method inside any configured ROIs.
    No ROI set → "SELECT ROI" overlay, defect=False.
    """
    camera_id = state["camera_id"]
    method    = state.get("method", "custom")

    processed, any_defect, _results = _roi_process_frame(frame, camera_id, method=method)
    state["defect"]  = any_defect
    state["last_ts"] = datetime.now().isoformat()
    _tick_fps(state)
    return processed


def _tick_fps(state: dict) -> None:
    """Shared FPS counter update."""
    state["_fps_count"] += 1
    elapsed = time.time() - state["_fps_start"]
    if elapsed >= 1.0:
        state["fps"]        = round(state["_fps_count"] / elapsed, 1)
        state["_fps_count"] = 0
        state["_fps_start"] = time.time()


# ─────────────────────────────────────────────────────────────────
# RTSP worker thread
# ─────────────────────────────────────────────────────────────────

def _rtsp_worker(camera_id: str) -> None:
    """
    Continuously reads from an RTSP stream, runs detection,
    stores processed frame in frames[camera_id].
    Reconnects on disconnect with exponential back-off.
    """
    state = active_cameras[camera_id]
    rtsp_url = state["source"]

    backoff = 1.0
    MAX_BACKOFF = 30.0

    while state["active"]:
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            state["last_frame"] = _offline_frame(camera_id, "RTSP unreachable")
            time.sleep(min(backoff, MAX_BACKOFF))
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        backoff = 1.0  # reset on successful connect

        # Initialise detector with first frame
        ret, init_frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(1.0)
            continue

        if state["detector"] is None:
            _init_detector(state)
            state["detector"].initialize(init_frame)

        while state["active"]:
            ret, frame = cap.read()
            if not ret:
                break  # reconnect

            processed = _run_detection(state, frame)
            state["last_frame"] = processed
            frames[camera_id] = processed

        cap.release()
        if state["active"]:
            # Stream dropped — reconnect after short pause
            time.sleep(2.0)

    # Thread exits cleanly
    frames.pop(camera_id, None)


# ─────────────────────────────────────────────────────────────────
# Client-push worker  (runs detection on frames pushed via HTTP)
# ─────────────────────────────────────────────────────────────────

def _client_detection_worker(camera_id: str) -> None:
    """
    Watches frames[camera_id] for new frames pushed by the client endpoint.
    Runs detection and writes the processed frame back.
    """
    state = active_cameras[camera_id]
    last_processed_id = -1

    while state["active"]:
        raw = frames.get(camera_id)
        if raw is None:
            time.sleep(0.02)
            continue

        frame_id = id(raw)
        if frame_id == last_processed_id:
            time.sleep(0.01)
            continue

        if state["detector"] is None:
            _init_detector(state)
            state["detector"].initialize(raw)

        processed = _run_detection(state, raw)
        state["last_frame"] = processed
        last_processed_id = frame_id


# ─────────────────────────────────────────────────────────────────
# Public API — called from routes_camera.py
# ─────────────────────────────────────────────────────────────────

def register_rtsp(camera_id: str, rtsp_url: str, method: str = None) -> dict:
    """
    Register and start an RTSP camera.
    Idempotent: stops existing instance first if already registered.
    """
    method = method or settings.DEFAULT_METHOD
    with _registry_lock:
        # Stop existing if already running
        if camera_id in active_cameras:
            stop_camera(camera_id)

        state = _make_state(camera_id, "rtsp", rtsp_url, method)
        state["active"] = True
        active_cameras[camera_id] = state

        t = threading.Thread(
            target=_rtsp_worker,
            args=(camera_id,),
            daemon=True,
            name=f"rtsp-{camera_id}",
        )
        _threads[camera_id] = t
        t.start()

    return {"camera_id": camera_id, "type": "rtsp", "method": method, "status": "started"}


def register_client(camera_id: str, method: str = None) -> dict:
    """
    Register a client-push camera (HTTP frame upload).
    The detection worker starts and waits for frames to appear in frames[].
    """
    method = method or settings.DEFAULT_METHOD
    with _registry_lock:
        if camera_id in active_cameras and active_cameras[camera_id]["active"]:
            return {"camera_id": camera_id, "status": "already_active"}

        state = _make_state(camera_id, "client", "push", method)
        state["active"] = True
        active_cameras[camera_id] = state

        t = threading.Thread(
            target=_client_detection_worker,
            args=(camera_id,),
            daemon=True,
            name=f"client-{camera_id}",
        )
        _threads[camera_id] = t
        t.start()

    return {"camera_id": camera_id, "type": "client", "method": method, "status": "registered"}


def push_client_frame(camera_id: str, frame: np.ndarray) -> None:
    """
    Called by the HTTP frame-upload endpoint.
    Stores raw frame for the detection worker to pick up.
    """
    frames[camera_id] = frame


def stop_camera(camera_id: str) -> dict:
    """Stop and deregister a camera (RTSP or client)."""
    with _registry_lock:
        state = active_cameras.get(camera_id)
        if not state:
            return {"camera_id": camera_id, "status": "not_found"}
        state["active"] = False

    t = _threads.pop(camera_id, None)
    if t and t.is_alive():
        t.join(timeout=4.0)

    active_cameras.pop(camera_id, None)
    frames.pop(camera_id, None)
    remove_camera_rois(camera_id)
    return {"camera_id": camera_id, "status": "stopped"}


def get_status_all() -> list:
    """Return status of all registered cameras."""
    from roi_processor import has_rois as _has_rois
    return [
        {
            "camera_id":     s["camera_id"],
            "type":          s["type"],
            "source":        s["source"] if s["type"] == "rtsp" else "push",
            "active":        s["active"],
            "defect":        s["defect"],
            "method":        s["method"],
            "fps":           s["fps"],
            "timestamp":     s["last_ts"],
            "roi_configured": _has_rois(s["camera_id"]),
        }
        for s in active_cameras.values()
    ]


def get_latest_frame(camera_id: str) -> Optional[np.ndarray]:
    """Return the latest processed frame, or None."""
    state = active_cameras.get(camera_id)
    if state:
        return state.get("last_frame")
    return None


# ─────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────

def _offline_frame(camera_id: str, msg: str = "Offline") -> np.ndarray:
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype="uint8")
    cv2.putText(frame, f"[{camera_id}] {msg}", (20, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    return frame