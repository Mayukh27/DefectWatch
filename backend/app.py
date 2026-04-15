"""
Real-Time Defect & Change Detection System — FastAPI Backend
Upgraded from original Flask + OpenCV project by Mayukh Ghosh
"""

import cv2
import numpy as np
import os
import time
import threading
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from detection_methods.fd import FrameDifferencingDetector
from detection_methods.mog2 import MOG2Detector
from detection_methods.running_avg import RunningAverageDetector
from detection_methods.custom import CustomDefectDetector
from detection_methods.dl_model import DLModelDetector
from database import Database

# ── v2: RTSP + client-push camera support (additive — does not modify existing routes)
from routes_camera import camera_router

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Defect Detection System API",
    description="Research-grade real-time defect and change detection backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── v2 camera routes (RTSP + client-push)
app.include_router(camera_router)

# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
db = Database()

# Per-camera state
camera_states: dict[int, dict] = {}
camera_threads: dict[int, threading.Thread] = {}
camera_caps: dict[int, cv2.VideoCapture] = {}
camera_lock = threading.Lock()

METHOD_MAP = {
    "custom":      CustomDefectDetector,
    "fd":          FrameDifferencingDetector,
    "mog2":        MOG2Detector,
    "running_avg": RunningAverageDetector,
    "dl":          DLModelDetector,
}


def _get_or_init_state(camera_id: int) -> dict:
    if camera_id not in camera_states:
        camera_states[camera_id] = {
            "active":       False,
            "defect":       False,
            "method":       settings.DEFAULT_METHOD,
            "detector":     None,
            "last_frame":   None,
            "last_ts":      None,
            "frame_count":  0,
            "fps":          0.0,
            "fps_timer":    time.time(),
        }
    return camera_states[camera_id]


# ─────────────────────────────────────────────
# Camera Worker Thread
# ─────────────────────────────────────────────
def _camera_worker(camera_id: int):
    from roi_processor import process_frame as _roi_process

    state      = _get_or_init_state(camera_id)
    cap        = camera_caps.get(camera_id)
    cam_id_str = str(camera_id)

    if cap is None or not cap.isOpened():
        return

    fps_counter = 0
    fps_start   = time.time()

    while state["active"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Store raw frame BEFORE processing (used by ROI editor snapshot)
        state["raw_frame"] = frame

        # Route through roi_processor using the selected method.
        method = state.get("method", "custom")
        processed_frame, any_defect, _ = _roi_process(frame, cam_id_str, method=method)

        state["defect"]     = any_defect
        state["last_frame"] = processed_frame
        state["last_ts"]    = datetime.now().isoformat()

        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            state["fps"]   = fps_counter / elapsed
            fps_counter    = 0
            fps_start      = time.time()

    cap.release()
    camera_caps.pop(camera_id, None)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.post("/start")
def start_detection(camera_id: int = 0, method: Optional[str] = None):
    """Start detection on a camera."""
    with camera_lock:
        state = _get_or_init_state(camera_id)
        if state["active"]:
            return {"status": "already_running", "camera_id": camera_id}

        if method:
            if method not in METHOD_MAP:
                raise HTTPException(400, f"Unknown method '{method}'. Valid: {list(METHOD_MAP)}")
            state["method"] = method

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            # Fallback: open video file (for demo / testing without physical camera)
            demo_path = settings.DEMO_VIDEO_PATH
            if demo_path and os.path.exists(demo_path):
                cap = cv2.VideoCapture(demo_path)
            else:
                raise HTTPException(503, f"Cannot open camera {camera_id}")

        camera_caps[camera_id] = cap
        state["active"] = True

        t = threading.Thread(
            target=_camera_worker,
            args=(camera_id,),
            daemon=True,
            name=f"cam-worker-{camera_id}"
        )
        camera_threads[camera_id] = t
        t.start()

    return {
        "status": "started",
        "camera_id": camera_id,
        "method": state["method"]
    }


@app.post("/stop")
def stop_detection(camera_id: int = 0):
    """Stop detection on a camera."""
    with camera_lock:
        state = camera_states.get(camera_id)
        if not state or not state["active"]:
            return {"status": "not_running", "camera_id": camera_id}
        state["active"] = False

    t = camera_threads.pop(camera_id, None)
    if t:
        t.join(timeout=3.0)

    return {"status": "stopped", "camera_id": camera_id}


@app.get("/status/{camera_id}")
def get_status(camera_id: int):
    """Get current status of a camera."""
    state = camera_states.get(camera_id)
    if not state:
        return {"active": False, "defect": False, "camera_id": camera_id}
    return {
        "active":     state["active"],
        "defect":     state["defect"],
        "method":     state["method"],
        "fps":        round(state["fps"], 1),
        "timestamp":  state["last_ts"],
        "camera_id":  camera_id,
    }


@app.get("/status")
def get_all_status():
    """Get status for all cameras — legacy (int id) + v2 (string id, RTSP/client)."""
    from roi_processor import has_rois as _has_rois
    # Legacy cameras (integer ids, local/USB)
    legacy = [
        {
            "camera_id":      cam_id,
            "type":           "local",
            "active":         s["active"],
            "defect":         s["defect"],
            "method":         s["method"],
            "fps":            round(s["fps"], 1),
            "timestamp":      s["last_ts"],
            "roi_configured": _has_rois(str(cam_id)),
        }
        for cam_id, s in camera_states.items()
    ]
    # v2 cameras (string ids, RTSP + client-push)
    from camera_registry import get_status_all as _v2_status
    v2 = _v2_status()
    return legacy + v2


@app.get("/stream/{camera_id}")
def stream_video(camera_id: int):
    """MJPEG video stream for the given camera."""
    def generate():
        while True:
            state = camera_states.get(camera_id)
            if not state or not state["active"] or state["last_frame"] is None:
                time.sleep(0.05)
                # Yield a black placeholder frame when camera is inactive
                placeholder = _black_frame()
                _, buf = cv2.imencode(".jpg", placeholder)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                continue

            frame = state["last_frame"]
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(0.033)  # ~30 FPS cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def _black_frame(width: int = 640, height: int = 480):
    placeholder = np.zeros((height, width, 3), dtype="uint8")
    cv2.putText(
        placeholder, "Camera Offline", (160, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2
    )
    return placeholder


@app.get("/evaluate")
def evaluate(camera_id: int = 0):
    """
    Run evaluation pipeline and return metrics + graph URLs for the dashboard.
    Graphs are served via /static/graphs/{name}.png
    """
    try:
        from evaluator import run_full_evaluation
        result = run_full_evaluation()

        # Convert file paths → URL paths served by FastAPI static mount
        graph_urls = [
            "/static/graphs/" + os.path.basename(g)
            for g in result["graphs"]
        ]

        # Comparison table as list of dicts
        comparison = []
        if result.get("comparison_df") is not None:
            df = result["comparison_df"].reset_index()
            comparison = df.fillna("—").to_dict(orient="records")

        metrics = {m: v for m, v in result["metrics"].items() if "error" not in v}

        return {
            "status":     "ok",
            "metrics":    metrics,
            "graphs":     graph_urls,
            "comparison": comparison,
        }
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/compare")
def compare_methods():
    """Return comparison table across all methods + literature baselines."""
    try:
        from evaluator import build_comparison_table, compare_with_baselines, compute_all_metrics
        df      = build_comparison_table()
        metrics = compute_all_metrics()
        lit_df  = compare_with_baselines(metrics)
        return {
            "status":     "ok",
            "comparison": df.reset_index().to_dict(orient="records"),
            "literature": lit_df.to_dict(orient="records"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/methods")
def list_methods():
    """List available detection methods."""
    return {
        "methods": list(METHOD_MAP.keys()),
        "active":  settings.USE_DL,
        "dl_note": "DL method is currently inactive (no dataset). Set USE_DL=True in config to enable."
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ─────────────────────────────────────────────
# Camera configuration endpoint
# ─────────────────────────────────────────────

@app.get("/cameras")
def list_cameras():
    """Return configured cameras from cameras.json."""
    import json
    cfg_path = os.path.join(os.path.dirname(__file__), "cameras.json")
    if not os.path.exists(cfg_path):
        return {"cameras": []}
    with open(cfg_path) as f:
        return {"cameras": json.load(f)}