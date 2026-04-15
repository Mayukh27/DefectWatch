import cv2
import atexit
import os
from flask import Flask, Response, jsonify, send_file
from detection import DefectDetector

# =====================
# Config
# =====================
CAM_ID = 0
PORT = 5000 + CAM_ID

BASE_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(BASE_DIR, "static", "captured")
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# App & Camera
# =====================
app = Flask(__name__)
cap = cv2.VideoCapture(CAM_ID)
detector = DefectDetector(SAVE_DIR)

# =====================
# Init ROI
# =====================
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read camera")

detector.initialize_roi(frame)


# --------------------------------------------------
# Cleanup on exit
# --------------------------------------------------
@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()


# =====================
# Streaming
# =====================
def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = detector.process(frame)
        _, buf = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )

# =====================
# Routes
# =====================
@app.route("/video")
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify({
        "active": cap.isOpened(),
        "defect": detector.change_detected,
        "timestamp": detector.last_timestamp
    })


@app.route("/initial_image")
def initial_image():
    path = os.path.join(SAVE_DIR, "initial.jpg")
    return send_file(path, mimetype="image/jpeg", max_age=0) if os.path.exists(path) else ("", 404)


@app.route("/defect_image")
def defect_image():
    path = detector.latest_defect_path
    return send_file(path, mimetype="image/jpeg", max_age=0) if path and os.path.exists(path) else ("", 404)

# =====================
# Main
# =====================
if __name__ == "__main__":
    print(f"[CLIENT] Streaming on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
