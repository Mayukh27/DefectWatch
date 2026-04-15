"""
client/client_v2.py
Laptop camera client — push mode.

Captures frames from the local webcam and POSTs them to the server's
  POST /push_frame/{camera_id}
endpoint. Detection runs entirely server-side.

The original client/client.py (local Flask stream) is UNCHANGED.
Use this when you want detection to happen centrally on the server.

Usage:
    python client_v2.py --server http://192.168.1.10:8000 --id laptop_mayukh
    python client_v2.py --server http://192.168.1.10:8000 --id laptop_mayukh --cam 0 --method custom --fps 15
"""

import argparse
import sys
import time

import cv2
import requests

# ──────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DefectWatch — push client")
parser.add_argument("--server",  required=True,   help="Server base URL, e.g. http://192.168.1.10:8000")
parser.add_argument("--id",      required=True,   help="Unique camera ID for this laptop")
parser.add_argument("--cam",     type=int, default=0, help="Local webcam index (default: 0)")
parser.add_argument("--method",  default="custom",
                    choices=["fd", "mog2", "running_avg", "custom", "dl"],
                    help="Detection method to request server-side (default: custom)")
parser.add_argument("--fps",     type=float, default=60.0, help="Target push FPS (default: 60)")
parser.add_argument("--quality", type=int,   default=70,   help="JPEG quality 1-100 (default: 70)")
args = parser.parse_args()

SERVER         = args.server.rstrip("/")
CAM_ID         = args.id
CAM_IDX        = args.cam
METHOD         = args.method
TARGET_FPS     = max(1.0, args.fps)
QUALITY        = max(1, min(100, args.quality))
FRAME_INTERVAL = 1.0 / TARGET_FPS

# ──────────────────────────────────────────────────────────────────
# Register with server
# ──────────────────────────────────────────────────────────────────
print(f"[client_v2] Connecting to server: {SERVER}")
print(f"[client_v2] Camera ID : {CAM_ID}")
print(f"[client_v2] Method    : {METHOD}")
print(f"[client_v2] Target FPS: {TARGET_FPS}")

try:
    r = requests.post(
        f"{SERVER}/register_client_camera",
        json={"camera_id": CAM_ID, "method": METHOD},
        timeout=5,
    )
    r.raise_for_status()
    print(f"[client_v2] Registered: {r.json()}")
except Exception as e:
    print(f"[client_v2] ERROR registering with server: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────
# Open webcam
# ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_IDX)
if not cap.isOpened():
    print(f"[client_v2] ERROR: Cannot open webcam index {CAM_IDX}")
    sys.exit(1)

print(f"[client_v2] Webcam {CAM_IDX} opened. Streaming to server...")
print(f"[client_v2] Press Ctrl+C to stop.")

PUSH_URL      = f"{SERVER}/push_frame/{CAM_ID}"
ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]

frame_count = 0
error_count = 0
MAX_ERRORS  = 10
session     = requests.Session()

try:
    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[client_v2] WARNING: Failed to read frame")
            time.sleep(0.1)
            continue

        ok, buf = cv2.imencode(".jpg", frame, ENCODE_PARAMS)
        if not ok:
            continue

        try:
            resp = session.post(
                PUSH_URL,
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=2.0,
            )
            if resp.status_code == 200:
                error_count  = 0
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"[client_v2] Pushed {frame_count} frames")
            elif resp.status_code == 404:
                print("[client_v2] Re-registering with server...")
                requests.post(
                    f"{SERVER}/register_client_camera",
                    json={"camera_id": CAM_ID, "method": METHOD},
                    timeout=5,
                )
            else:
                error_count += 1
                print(f"[client_v2] Server returned {resp.status_code}")

        except requests.exceptions.ConnectionError:
            error_count += 1
            print(f"[client_v2] Connection error ({error_count}/{MAX_ERRORS})")
            if error_count >= MAX_ERRORS:
                print("[client_v2] Too many consecutive errors - exiting.")
                break
            time.sleep(1.0)
            continue

        elapsed   = time.perf_counter() - t0
        sleep_for = FRAME_INTERVAL - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

except KeyboardInterrupt:
    print(f"\n[client_v2] Stopped. Total frames pushed: {frame_count}")

finally:
    cap.release()
    try:
        requests.post(f"{SERVER}/stop_camera",
                      json={"camera_id": CAM_ID}, timeout=3)
        print(f"[client_v2] Camera '{CAM_ID}' deregistered from server.")
    except Exception:
        pass
