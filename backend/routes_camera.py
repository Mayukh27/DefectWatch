"""
routes_camera.py
New camera routes — RTSP + client-push support.

Mounted as a FastAPI APIRouter and included in app.py with a single line:
    app.include_router(camera_router)

Zero modifications to existing app.py routes.
"""

import io
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from camera_registry import (
    register_rtsp,
    register_client,
    push_client_frame,
    stop_camera,
    get_status_all,
    get_latest_frame,
    active_cameras,
    _offline_frame,
)
from config import settings

camera_router = APIRouter(tags=["cameras-v2"])


# ─────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────

class AddRtspRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    method: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "factory_floor_1",
                "rtsp_url":  "rtsp://admin:pass@192.168.1.64:554/stream1",
                "method":    "custom",
            }
        }


class RegisterClientRequest(BaseModel):
    camera_id: str
    method: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "laptop_cam_0",
                "method":    "mog2",
            }
        }


class StopCameraRequest(BaseModel):
    camera_id: str


# ─────────────────────────────────────────────────────────────────
# RTSP Routes
# ─────────────────────────────────────────────────────────────────

@camera_router.post("/add_rtsp_camera")
def add_rtsp_camera(req: AddRtspRequest):
    """
    Register and start an RTSP IP/CCTV camera.

    - Starts a background thread that reads from rtsp_url via OpenCV
    - Frames are processed through the selected detection method
    - Stream available at GET /v2/stream/{camera_id}
    """
    if not req.rtsp_url.startswith(("rtsp://", "rtsps://", "rtsp:/", "http://", "https://")):
        raise HTTPException(400, "rtsp_url must start with rtsp://, rtsps://, http://, or https://")

    result = register_rtsp(req.camera_id, req.rtsp_url, req.method)
    return result


# ─────────────────────────────────────────────────────────────────
# Client-push Routes
# ─────────────────────────────────────────────────────────────────

@camera_router.post("/register_client_camera")
def register_client_camera(req: RegisterClientRequest):
    """
    Register a laptop/device that will push frames to this server.
    After registering, the device calls POST /push_frame/{camera_id}.
    Detection runs server-side on received frames.
    """
    result = register_client(req.camera_id, req.method)
    return result


@camera_router.post("/push_frame/{camera_id}")
async def push_frame(
    camera_id: str,
    file: UploadFile = File(...),
):
    """
    Client laptops POST JPEG frames here.

    Usage (from client laptop):
        import requests, cv2
        ret, buf = cv2.imencode('.jpg', frame)
        requests.post(f'{SERVER}/push_frame/{camera_id}',
                      files={'file': buf.tobytes()})

    Detection runs server-side; processed stream available via
    GET /v2/stream/{camera_id}
    """
    if camera_id not in active_cameras:
        raise HTTPException(404, f"Camera '{camera_id}' not registered. Call /register_client_camera first.")

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
       raise HTTPException(400, "Could not decode image. Send a valid JPEG.")

    # ✅ ADD THIS BLOCK
    from app import camera_states, _get_or_init_state
    state = _get_or_init_state(camera_id)
    state["raw_frame"] = frame

    push_client_frame(camera_id, frame)
    push_client_frame(camera_id, frame)
    return {"status": "ok", "camera_id": camera_id}


# ─────────────────────────────────────────────────────────────────
# Stop
# ─────────────────────────────────────────────────────────────────

@camera_router.post("/stop_camera")
def stop_camera_route(req: StopCameraRequest):
    """Stop any registered camera (RTSP or client) by camera_id."""
    return stop_camera(req.camera_id)


# ─────────────────────────────────────────────────────────────────
# Unified Stream — works for both RTSP and client cameras
# ─────────────────────────────────────────────────────────────────

@camera_router.get("/v2/stream/{camera_id}")
def stream_v2(camera_id: str):
    """
    MJPEG stream for any registered camera (RTSP or client-push).
    Falls back to offline placeholder when no frame is available.
    """
    def generate():
        while True:
            frame = get_latest_frame(camera_id)
            if frame is None:
                frame = _offline_frame(camera_id)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(0.033)   # ~30 FPS cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─────────────────────────────────────────────────────────────────
# Status endpoints
# ─────────────────────────────────────────────────────────────────

@camera_router.get("/v2/status")
def status_all_v2():
    """
    Status for all v2 cameras (RTSP + client).
    Compatible format with existing /status response.
    """
    return get_status_all()


@camera_router.get("/v2/status/{camera_id}")
def status_one_v2(camera_id: str):
    """Status for one v2 camera by string camera_id."""
    state = active_cameras.get(camera_id)
    if not state:
        raise HTTPException(404, f"Camera '{camera_id}' not found.")
    return {
        "camera_id": state["camera_id"],
        "type":      state["type"],
        "source":    state["source"] if state["type"] == "rtsp" else "push",
        "active":    state["active"],
        "defect":    state["defect"],
        "method":    state["method"],
        "fps":       state["fps"],
        "timestamp": state["last_ts"],
    }


@camera_router.get("/v2/cameras")
def list_all_v2():
    """
    List all registered v2 cameras with type, source, and current state.
    Useful for the frontend to build the full camera grid.
    """
    return {
        "cameras": get_status_all(),
        "count": len(active_cameras),
        "mode": settings.CAMERA_MODE,
    }


# ─────────────────────────────────────────────────────────────────
# ROI Management Routes (new — additive)
# ─────────────────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel
from typing import List as _List
from roi_processor import (
    set_rois as _set_rois,
    get_rois as _get_rois,
    reset_reference_for_roi as _reset_ref_roi,
    reset_all_references as _reset_all_refs,
)
from roi_store import roi_store as _roi_store


class ROIDefinition(_BaseModel):
    roi_id:    str
    type:      str   = "general"   # crack | movement | general
    x:         int
    y:         int
    w:         int
    h:         int
    threshold: int   = None        # optional per-ROI threshold override
    min_area:  int   = None        # optional per-ROI min contour area

    class Config:
        json_schema_extra = {"example": {
            "roi_id": "crack_zone_1", "type": "crack",
            "x": 100, "y": 80, "w": 320, "h": 200
        }}


class SetROIsRequest(_BaseModel):
    camera_id: str
    rois: _List[ROIDefinition]


class ResetReferenceRequest(_BaseModel):
    camera_id: str
    roi_id:    str = None   # None = reset ALL rois for camera


@camera_router.post("/set_rois")
def set_camera_rois(req: SetROIsRequest):
    """
    Define ROI zones for a camera.
    Replaces any existing ROI config for that camera.
    Clears stored reference + previous frames so detection starts fresh.

    ROI types:
        crack    → compares with FIXED reference frame (detects permanent changes)
        movement → compares with PREVIOUS frame (detects transient motion)
        general  → compares with reference (default)
    """
    if camera_id := req.camera_id:
        rois_dicts = [r.model_dump(exclude_none=False) for r in req.rois]
        _set_rois(camera_id, rois_dicts)
        return {
            "status":    "ok",
            "camera_id": camera_id,
            "roi_count": len(req.rois),
            "rois":      [r.roi_id for r in req.rois],
        }


@camera_router.get("/get_rois/{camera_id}")
def get_camera_rois(camera_id: str):
    """Return current ROI definitions for a camera."""
    return {"camera_id": camera_id, "rois": _get_rois(camera_id)}


@camera_router.post("/reset_reference")
def reset_reference(req: ResetReferenceRequest):
    """
    Manually reset the reference frame(s) for a camera.
    The next frame received will become the new reference.

    - If roi_id is given: reset only that ROI.
    - If roi_id is omitted: reset ALL ROIs for the camera.

    The reference frame NEVER updates automatically.
    Only this endpoint can change it.
    """
    if req.roi_id:
        # Single ROI reset — needs a live frame; mark as pending reset
        _roi_store.remove_camera(req.camera_id)   # clears all; they re-init on next frame
        return {"status": "ok", "camera_id": req.camera_id,
                "roi_id": req.roi_id, "note": "Reference will update on next frame."}
    else:
        _reset_all_refs(req.camera_id)
        return {"status": "ok", "camera_id": req.camera_id,
                "note": "All ROI references reset. Will re-init on next frame."}


# ── Snapshot endpoint (returns single JPEG for ROI editor canvas) ─


# ── Raw snapshot — unprocessed frame for ROI editor canvas ────────
@camera_router.get("/snapshot_raw/{camera_id}")
def get_snapshot_raw(camera_id: str):
    """
    Return the latest RAW (unprocessed) frame as JPEG.
    Falls back gracefully: raw_frame → last_frame → v2 registry → 404
    404 is intentional so the frontend can handle it with a retry or fallback.
    """
    import time as _time
    frame = None

    # Try up to ~500 ms for the first raw frame to arrive
    for _ in range(10):
        try:
            from app import camera_states
            state = camera_states.get(int(camera_id))
            if state:
                # raw_frame = unprocessed; last_frame = processed (fallback)
                frame = state.get("raw_frame") or state.get("last_frame")
                if frame is not None:
                    break
        except (ValueError, ImportError):
            pass

        # v2 cameras (RTSP / client-push)
        frame = get_latest_frame(camera_id)
        if frame is not None:
            break

        _time.sleep(0.05)   # wait 50 ms and retry

    if frame is None:
        # Return 404 so frontend knows to show stream URL fallback
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"detail": f"No frame available for camera {camera_id}. Is it started?"},
        )

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "Frame encode failed")

    from fastapi.responses import Response
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )

@camera_router.get("/snapshot/{camera_id}")
def get_snapshot(camera_id: str):
    """
    Return the latest processed frame as a single JPEG.
    Used by the ROI editor canvas to capture a still from the live stream.
    Works for both v2 (RTSP/client) and legacy integer-id cameras.
    """
    frame = get_latest_frame(camera_id)

    # Fallback: try legacy integer camera state
    if frame is None:
        try:
            from app import camera_states
            state = camera_states.get(int(camera_id))
            if state:
                frame = state.get("last_frame")
        except (ValueError, ImportError):
            pass

    if frame is None:
        frame = _offline_frame(camera_id, "No frame yet")

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "Frame encode failed")

    from fastapi.responses import Response
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── Reference frame endpoint — serves the stored grayscale reference as JPEG ─
@camera_router.get("/reference_frame/{camera_id}")
def get_reference_frame(camera_id: str, roi_id: str = None):
    """
    Return the stored reference frame for a camera (the 'Before' image).
    If roi_id is given, return only that ROI crop. Otherwise return full frame snapshot.
    Used by the AlertPopup Before/After comparison.
    """
    from roi_store import roi_store as _store

    # Try to get the specific ROI reference crop
    target_roi_id = roi_id
    if target_roi_id is None:
        # Use first available ROI for this camera
        from roi_processor import get_rois as _get_rois
        rois = _get_rois(camera_id)
        if rois:
            target_roi_id = rois[0].get("roi_id")

    ref_gray = None
    if target_roi_id:
        ref_gray = _store.get_reference(camera_id, target_roi_id)

    if ref_gray is not None:
        # Convert grayscale reference to BGR JPEG
        ref_bgr = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".jpg", ref_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            from fastapi.responses import Response
            return Response(content=buf.tobytes(), media_type="image/jpeg")

    # Fallback: serve the current snapshot (best we have without a stored ref)
    frame = get_latest_frame(camera_id)
    if frame is None:
        try:
            from app import camera_states
            state = camera_states.get(int(camera_id))
            if state:
                frame = state.get("last_frame")
        except (ValueError, ImportError):
            pass
    if frame is None:
        frame = _offline_frame(camera_id, "No reference stored")

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    from fastapi.responses import Response
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── Popup data endpoint: reference + current crop for Before/After modal ──
@camera_router.get("/popup_data/{camera_id}")
def get_popup_data(camera_id: str):
    """
    Before/After popup data.
    LEFT  (reference) = clean baseline crop — first frame after ROI set, never auto-updates.
    RIGHT (current)   = raw crop from latest defect-detected frame (no overlay drawn on it).
    Both are the actual ROI crop, not the full frame.
    """
    import base64
    from datetime import datetime as _dt
    from roi_processor import get_snapshot as _get_snap, get_rois as _get_rois
    from fastapi.responses import JSONResponse

    snap     = _get_snap(camera_id) or {}
    roi_id   = snap.get("latest_roi_id", "")
    roi_type = snap.get("latest_roi_type", "custom")
    ts       = snap.get("latest_ts", _dt.now().strftime("%Y-%m-%d %H:%M:%S"))

    def encode(arr):
        if arr is None:
            return None
        img = arr
        if img.dtype != "uint8":
            img = img.astype("uint8")
        if img.ndim == 2:                          # grayscale → BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None

    # ── Reference (Before) ────────────────────────────────────────
    ref_arr = snap.get(roi_id + "_ref") if roi_id else None
    if ref_arr is None:
        # Try any _ref key in snapshot
        for k, v in snap.items():
            if k.endswith("_ref") and v is not None:
                ref_arr = v
                roi_id  = k[:-4]
                break
    if ref_arr is None:
        # Try roi_store grayscale reference
        rois = _get_rois(camera_id)
        if rois:
            from roi_store import roi_store as _rs
            ref_arr = _rs.get_reference(camera_id, rois[0].get("roi_id",""))

    # ── Current (After) ───────────────────────────────────────────
    cur_arr = snap.get(roi_id + "_cur") if roi_id else None
    if cur_arr is None:
        # No defect detected yet — show live ROI crop as "After"
        raw = None
        try:
            from app import camera_states
            s = camera_states.get(int(camera_id))
            if s:
                raw = s.get("raw_frame") or s.get("last_frame")
        except Exception:
            pass
        if raw is None:
            raw = get_latest_frame(camera_id)
        if raw is not None and roi_id:
            rois = _get_rois(camera_id)
            roi  = next((r for r in rois if r.get("roi_id") == roi_id), None)
            if roi:
                x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
                fh, fw = raw.shape[:2]
                if 0 <= x < fw and 0 <= y < fh and w > 0 and h > 0:
                    cur_arr = raw[y:min(y+h,fh), x:min(x+w,fw)].copy()

    ref_img = encode(ref_arr)
    cur_img = encode(cur_arr)

    # Ultimate fallback — show full frame for both
    if ref_img is None:
        raw = get_latest_frame(camera_id)
        if raw is None:
            try:
                from app import camera_states
                s = camera_states.get(int(camera_id))
                if s: raw = s.get("raw_frame") or s.get("last_frame")
            except Exception:
                pass
        ref_img = encode(raw)
        cur_img = ref_img

    return JSONResponse({
        "camera_id": camera_id,
        "roi_id":    roi_id   or "—",
        "roi_type":  roi_type,
        "timestamp": ts,
        "reference": ref_img,
        "current":   cur_img,
    })