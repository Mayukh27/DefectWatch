# DefectWatch — Real-Time Multi-Camera Change Detection System

> **RT-MCCDS v2.0.0** · FastAPI · React · OpenCV · Multi-ROI · Research Evaluation Pipeline

DefectWatch is a real-time industrial defect and change detection platform designed for multi-camera deployment. It detects surface defects, object placement changes, cracks, and foreign material using classical computer vision methods, with a built-in research evaluation pipeline for method comparison and publication-ready output.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Detection Methods](#detection-methods)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Camera Setup](#camera-setup)
- [ROI Configuration](#roi-configuration)
- [Research Evaluation](#research-evaluation)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Performance](#performance)

---

## Overview

DefectWatch monitors inspection surfaces in real time across multiple camera sources simultaneously. A reference frame is captured when the scene is clean; any subsequent change within a defined Region of Interest (ROI) is detected, marked with a red contour overlay, and logged.

The system supports local USB cameras, RTSP IP cameras, and laptop client cameras that push frames over HTTP — all managed from a single web dashboard.

```
Camera sources  →  FastAPI backend  →  ROI detection  →  React dashboard
  USB / RTSP         (port 8000)       per-method         (port 3000)
  Client push                          CSV logging
```

---

## Key Features

| Feature | Details |
|---|---|
| **Multi-camera** | Unlimited simultaneous cameras — USB, RTSP, client push |
| **Multi-ROI** | Multiple detection zones per camera with independent sensitivity |
| **4 detection methods** | Custom LAB+L, Frame Differencing, MOG2, Running Average |
| **Parallel evaluation** | All 4 methods run silently on every frame for fair comparison |
| **Live dashboard** | Real-time MJPEG streams, defect badges, event log |
| **Before / After popup** | Side-by-side reference vs defect frame on sidebar click |
| **Research pipeline** | Metrics (Precision/Recall/F1), 8 publication-ready graphs |
| **Literature comparison** | Auto-comparison with YOLOv4, Mask R-CNN, MOG2 GMM |
| **Per-ROI tuning** | Threshold and min-area sliders per zone in the ROI editor |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Dashboard                          │
│   CameraCard  │  ROISelectorModal  │  AlertPopup  │  Sidebar   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP / MJPEG
┌───────────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend  :8000                        │
│                                                                  │
│   app.py              routes_camera.py        evaluator.py      │
│   ├─ /start           ├─ /set_rois            ├─ metrics        │
│   ├─ /stop            ├─ /push_frame          ├─ graphs         │
│   ├─ /stream          ├─ /v2/stream           └─ literature     │
│   └─ /status          └─ /popup_data                            │
│                                                                  │
│   roi_processor.py    detection_methods/      database.py       │
│   ├─ process_frame    ├─ custom.py            └─ SQLite         │
│   ├─ _log() CSV       ├─ fd.py                                  │
│   └─ snapshots        ├─ mog2.py                                │
│                        └─ running_avg.py                        │
└─────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
    USB cameras          RTSP cameras        client_v2.py
                                             (laptop push)
```

### Detection Pipeline

For every incoming frame:

```
Frame arrives
    │
    ▼
ROI crop extracted  ←── ROI zones from dashboard
    │
    ├──► custom._detect(crop)      → log → custom_0.csv   [displayed]
    ├──► fd._detect(crop)          → log → fd_0.csv        [silent]
    ├──► mog2._detect(crop)        → log → mog2_0.csv      [silent]
    └──► running_avg._detect(crop) → log → running_avg_0.csv [silent]
                │
                ▼
        Draw overlay on frame (display method only)
        Red bounding rect + semi-transparent filled contour
```

All 4 methods process every frame with the same frame number, enabling fair comparison without requiring multiple test sessions.

---

## Detection Methods

### Custom — LAB + Luminance (default)

The primary method, based on the original algorithm by Mayukh Ghosh. Detects **both colour changes and luminance changes**:

```
LAB conversion + GaussianBlur(7,7)
    ├── A-B channel magnitude diff   → colour changes (paper, stickers)
    └── L-channel diff × 1.5         → luminance changes (wires, cracks)
Combined = max(AB_diff, L_diff × 1.5)
Threshold → MORPH_OPEN(5×5, iter=2) → dilate(iter=2) → contours
Optional SSIM structural layer (OR fusion)
```

### Frame Differencing (`fd`)

Simplest baseline. Grayscale absdiff against a fixed reference frame. Very fast (~1ms/frame), sensitive to camera shake.

### MOG2 Background Subtraction (`mog2`)

Adaptive per-pixel Gaussian mixture model (OpenCV implementation of Stauffer & Grimson, 1999). 30-frame warmup period. Adapts to slow lighting changes but drifts away from persistent defects.

### Running Average (`running_avg`)

Weighted exponential running average background model (`α = 0.05`). Slower adaptation than MOG2, smoother but with similar long-exposure drift behaviour.

### Sensitivity Tuning

| Object type | Threshold | Min area (px²) |
|---|---|---|
| Hairline crack | 6–10 | 40–80 |
| Mouse wire / thin cable | 12–18 | 80–150 |
| Thick cable | 18–25 | 200–400 |
| Paper / sticker | 22–30 | 300–600 |

---

## Project Structure

```
DefectWatch-V2/
├── backend/
│   ├── app.py                    # FastAPI app, camera workers, stream endpoints
│   ├── camera_registry.py        # RTSP / client-push camera management
│   ├── config.py                 # Settings dataclass (env-configurable)
│   ├── database.py               # SQLite evaluation log
│   ├── defect_marker.py          # Overlay drawing — bounding rect + filled contour
│   ├── evaluator.py              # Research evaluation pipeline
│   ├── roi_processor.py          # ROI coordinator, parallel method dispatch, CSV log
│   ├── roi_store.py              # Reference / previous frame storage
│   ├── routes_camera.py          # v2 camera API routes
│   ├── detection_methods/
│   │   ├── base.py               # BaseDetector abstract class
│   │   ├── custom.py             # LAB + L-channel detector (default)
│   │   ├── fd.py                 # Frame differencing
│   │   ├── mog2.py               # MOG2 adaptive background
│   │   ├── running_avg.py        # Running average background
│   │   └── dl_model.py           # YOLO stub (inactive — no dataset yet)
│   ├── predictions/              # Per-method prediction CSVs (auto-created)
│   │   ├── custom_0.csv
│   │   ├── fd_0.csv
│   │   ├── mog2_0.csv
│   │   ├── running_avg_0.csv
│   │   ├── comparison.csv
│   │   └── comparison_with_literature.csv
│   ├── static/
│   │   └── graphs/               # Evaluation graphs (auto-created)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/Dashboard.jsx   # Main layout
│   │   ├── components/
│   │   │   ├── CameraCard.jsx    # Per-camera feed + controls
│   │   │   ├── Sidebar.jsx       # Right sidebar, camera list
│   │   │   ├── ROISelectorModal.jsx  # Interactive ROI zone editor
│   │   │   ├── AlertPopup.jsx    # Before/After defect popup
│   │   │   ├── AlertLog.jsx      # Event log strip
│   │   │   ├── AddCameraModal.jsx    # RTSP / client camera registration
│   │   │   └── TopBar.jsx        # Header
│   │   ├── hooks/useDetection.js # Polling hook, camera state merge
│   │   └── utils/api.js          # All API calls with error handling
│   ├── package.json
│   └── tailwind.config.js
├── client/
│   ├── client.py                 # Original Flask stream client (unchanged)
│   └── client_v2.py              # Push-mode client (POSTs frames to server)
├── ground_truth.csv              # Frame labels for evaluation (frame, label)
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- A webcam or IP camera (or use the client push mode from another machine)

### Backend

```bash
cd backend
pip install -r requirements.txt
```

`requirements.txt` includes: `fastapi`, `uvicorn`, `opencv-python`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `scikit-image`, `requests`, `python-multipart`

### Frontend

```bash
cd frontend
npm install
```

---

## Running the System

### 1. Start the backend

```bash
cd backend
python app.py
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 2. Start the frontend

```bash
cd frontend
npm start
# Dashboard opens at http://localhost:3000
```

### 3. Add a camera and start detection

1. The dashboard shows 4 camera slots by default (CAM 00–03)
2. Click **▶ Start** on any camera
3. The stream shows **"SELECT ROI TO START DETECTION"** until a zone is drawn
4. Click **Set ROI zones** — draw a rectangle on the live frame
5. Click **Save Zones** — the first clean frame becomes the reference baseline
6. Place a defect inside the zone — red overlay appears within 1–2 frames
7. Click any camera row in the right sidebar to see **Before / After** comparison

---

## Camera Setup

### Local USB cameras (default)

CAM 00–03 in the dashboard map to webcam indices 0–3. No configuration needed.

### RTSP IP cameras

Click **+ Add Camera** in the sidebar:

```
Type:    RTSP / IP Camera
ID:      factory_floor_1        (any unique string)
URL:     rtsp://admin:pass@192.168.1.64:554/stream1
Method:  custom
```

### Laptop client (push mode)

Register from the dashboard:

```
Type:    Client Laptop (push)
ID:      laptop_mayukh
Method:  custom
```

Then run on the laptop:

```bash
python client/client_v2.py \
  --server http://192.168.1.10:8000 \
  --id laptop_mayukh \
  --cam 0 \
  --method custom \
  --fps 15
```

The laptop captures from its own webcam and pushes frames to the server. Detection runs centrally.

---

## ROI Configuration

### Drawing zones

1. Camera must be **started** before opening the ROI editor (snapshot loads from live stream)
2. Click **Set ROI zones** → drag rectangles on the frame
3. Each zone has independent **Threshold** and **Min Size** sliders:
   - **Threshold 2–40**: lower = more sensitive (detects subtle changes like cracks)
   - **Min Size 50–1000 px²**: lower = detects thinner objects (wires)
4. Click **Save Zones** — the next frame captured becomes the **clean reference baseline**

### Resetting the reference

Click **↺ Reset Reference** in the ROI editor at any time. The next frame received becomes the new baseline. Use this when lighting changes or the scene is rearranged.

> **Important:** The reference frame must be captured with a **clean scene** (no defect present). If the defect is already in view when zones are saved, it will be part of the baseline and will not be detected.

---

## Research Evaluation

### How prediction logging works

Every frame processed by the system logs a row to `predictions/{method}_{camera_id}.csv`:

```
frame, roi_id, prediction, score, latency_ms
0,     roi_a,  0,          0.0,   7.2
1,     roi_a,  0,          0.0,   6.8
42,    roi_a,  1,          1.0,   8.1
```

All 4 methods log on every frame with the same frame number, enabling fair comparison from a single test session.

### Creating ground truth

Create `ground_truth.csv` at the project root:

```csv
frame,label
0,0
1,0
42,1
43,1
44,1
50,0
```

- `label = 1` → defect physically present in this frame
- `label = 0` → clean surface
- Frame numbers correspond to row positions in the prediction CSVs

To generate a template: `python backend/evaluator.py --example-gt`

### Running evaluation

```bash
cd backend
python evaluator.py
```

**Output files:**

| File | Contents |
|---|---|
| `predictions/comparison.csv` | All methods × Precision, Recall, F1, Accuracy, FPS, Latency |
| `predictions/comparison_with_literature.csv` | Our methods + YOLOv4, Mask R-CNN, MOG2 GMM references |
| `static/graphs/metrics_comparison.png` | Precision / Recall / F1 bar chart |
| `static/graphs/fps_comparison.png` | FPS comparison with literature reference lines |
| `static/graphs/latency_per_frame.png` | Per-frame latency rolling average |
| `static/graphs/accuracy_vs_fps.png` | Accuracy vs FPS scatter (bubble size ∝ F1) |
| `static/graphs/confusion_matrices.png` | 2×2 confusion matrix grid for all methods |
| `static/graphs/detection_rate.png` | Detection rate per method |
| `static/graphs/fps_vs_cameras.png` | Scalability: FPS vs number of cameras |
| `static/graphs/comparison_with_literature.png` | Horizontal bar: ours vs literature |

### CLI options

```bash
python evaluator.py                          # full evaluation
python evaluator.py --example-gt             # write example ground_truth.csv template
python evaluator.py --gt /path/to/gt.csv     # custom ground truth path
python evaluator.py --masks /path/to/masks/  # include IoU from segmentation masks
python evaluator.py --graphs-only            # regenerate graphs without recomputing metrics
```

### Literature references

| Method | F1 | FPS | Reference |
|---|---|---|---|
| YOLOv4 (GPU) | 0.910 | 62 | Bochkovskiy et al., arXiv:2004.10934 (2020) |
| Mask R-CNN (GPU) | 0.865 | 8 | He et al., arXiv:1703.06870 (2017) |
| MOG2 / GMM (CPU) | 0.769 | 80 | Stauffer & Grimson, IEEE CVPR 1999 |
| Frame Diff (CPU) | 0.715 | 120 | Czimmermann et al., arXiv:2103.14030 (2020) |

---

## API Reference

### Camera control

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/start?camera_id=0&method=custom` | Start local USB camera |
| `POST` | `/stop?camera_id=0` | Stop local camera |
| `GET` | `/stream/{camera_id}` | MJPEG stream (legacy int id) |
| `GET` | `/status` | All cameras merged status |

### v2 cameras (RTSP / client push)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/add_rtsp_camera` | Register RTSP camera `{camera_id, rtsp_url, method}` |
| `POST` | `/register_client_camera` | Register client push camera `{camera_id, method}` |
| `POST` | `/push_frame/{camera_id}` | Upload frame from client (multipart JPEG) |
| `POST` | `/stop_camera` | Stop v2 camera `{camera_id}` |
| `GET` | `/v2/stream/{camera_id}` | MJPEG stream (string id) |
| `GET` | `/v2/cameras` | List all v2 cameras |

### ROI management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/set_rois` | Save ROI zones `{camera_id, rois:[...]}` |
| `GET` | `/get_rois/{camera_id}` | Get current ROI zones |
| `POST` | `/reset_reference` | Clear reference frames `{camera_id}` |
| `GET` | `/snapshot_raw/{camera_id}` | Unprocessed frame for ROI editor |
| `GET` | `/popup_data/{camera_id}` | Before/After base64 crops for popup |

### Evaluation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/evaluate` | Run full evaluation pipeline, return metrics + graph URLs |
| `GET` | `/compare` | Return comparison table + literature |
| `GET` | `/methods` | List available detection methods |
| `GET` | `/health` | Server health check |

---

## Configuration

All settings can be overridden with environment variables:

```bash
# Detection sensitivity
export MIN_CONTOUR_AREA=300      # minimum blob size in pixels²
export BINARY_THRESHOLD=25       # threshold for fd / running_avg methods
export DEFAULT_METHOD=custom     # default method for new cameras

# Camera mode
export CAMERA_MODE=mixed         # mixed | rtsp | client

# Paths
export DEMO_VIDEO_PATH=/path/to/demo.mp4   # use video file instead of webcam

# Deep Learning (inactive until dataset is prepared)
export USE_DL=false
export YOLO_MODEL_PATH=models/best.pt
export YOLO_CONF=0.5
```

Or edit `backend/config.py` directly.

---

## Performance

Measured on Intel Core i5, CPU-only, 640×480 frames, single ROI zone:

| Method | FPS | Latency | Notes |
|---|---|---|---|
| Custom (LAB+L+SSIM) | 37–50 | 20–27ms | Most accurate, slowest |
| Frame Differencing | 735–895 | 1–1.4ms | Fastest, sensitive to shake |
| MOG2 | 210–422 | 2.4–4.8ms | 30-frame warmup |
| Running Average | 782–1142 | 0.9–1.3ms | Drifts on persistent defects |

When all 4 methods run in parallel (default), throughput is bounded by the slowest — Custom at ~37 FPS. This is sufficient for real-time inspection at standard camera frame rates.

**Scalability:** Each additional camera adds approximately 13% latency overhead per camera due to threading context switching. Estimated FPS at N cameras:

```
FPS(N) ≈ FPS(1) / (1 + 0.13 × (N − 1))
```

---

## Acknowledgements

Original detection algorithm: **Mayukh Ghosh**  
Framework: FastAPI, React 18, OpenCV 4, scikit-learn  
Reference implementations: YOLOv4 (Bochkovskiy et al.), Mask R-CNN (He et al.), MOG2 (Stauffer & Grimson)
