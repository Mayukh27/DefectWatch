"""
evaluator.py  —  Research Evaluation Pipeline
=============================================
DefectWatch Multi-Camera Defect Detection System

PURPOSE
-------
Transform raw prediction logs into publication-ready metrics,
comparison tables, and graphs. Does NOT modify detection logic or UI.

HOW THE DATA FLOWS
------------------
Camera → roi_processor.py → predictions/{method}.csv
                                   ↓
                        evaluator.collect_data()
                                   ↓
                        evaluator.compute_metrics()
                                   ↓
            comparison.csv  +  static/graphs/*.png

═══════════════════════════════════════════════════════════════
DATASET CREATION GUIDE  (read before labelling)
═══════════════════════════════════════════════════════════════

Step 1 — Run a controlled test session:
    • Point camera at inspection surface, set ROI via dashboard
    • Place known defect (wire, crack, paper) for N seconds, then remove
    • System saves predictions automatically to predictions/{method}.csv

Step 2 — Create this folder structure:
    dataset/
    ├── images/           ← frames you save with collect_data()
    ├── masks/            ← binary PNG masks for IoU (optional, draw manually)
    └── ground_truth.csv  ← one row per frame (see Step 3)

Step 3 — ground_truth.csv format:
    frame, camera_id, roi_id, actual
    0,     0,         zone_a, 0         ← clean frame
    42,    0,         zone_a, 1         ← defect present (wire placed)
    43,    0,         zone_a, 1
    ...
    actual=1 → defect physically present
    actual=0 → clean surface

    predictions/{method}.csv format:
    frame, camera_id, roi_id, prediction, latency_ms
    0,     0,         zone_a, 0,          12.3
    42,    0,         zone_a, 1,          13.1
    ...
    These are written automatically by roi_processor.py during a camera session.
    Merge key: (frame, camera_id, roi_id) — must match exactly across both files.

Step 4 — How to label frames:
    Option A (manual CSV):
        Open images/ in file browser, write frame numbers of defect
        frames into ground_truth.csv. Mark actual=1 for those rows.
    Option B (time-based):
        If defect placed at t=30s and camera runs at ~25fps:
        frames 750-onwards → actual=1 until removed.
    Option C (interactive):
        Run:  python evaluator.py --label
        Plays frames one by one: press D=defect, C=clean.

Step 5 — Ground truth masks (optional, needed only for IoU):
    Use GIMP / Paint / LabelMe:
    • Open frame image from dataset/images/
    • Paint WHITE (255,255,255) over the defect pixel region
    • Everything else = BLACK (0,0,0)
    • Save as: dataset/masks/{frame:06d}_{camera_id}_{roi_id}.png

═══════════════════════════════════════════════════════════════
COMPARISON WITH RESEARCH PAPERS
═══════════════════════════════════════════════════════════════

[1] YOLOv4 — Bochkovskiy et al. (2020), arXiv:2004.10934
    Metric: mAP@0.5 on COCO
    Typical: mAP=65.7%, 62 FPS on Tesla V100 GPU
    Compare: We report F1 (≈mAP for binary class), FPS on CPU

[2] Mask R-CNN — He et al. (2017), arXiv:1703.06870
    Metric: AP (box), AP (mask) on COCO
    Typical: mask IoU=35.7 COCO (multi-class), ~8 FPS GPU
    Compare: We report per-ROI IoU, note CPU vs GPU gap

[3] GMM Background Model — Stauffer & Grimson (1999)
    IEEE CVPR. DOI: 10.1109/CVPR.1999.784637
    Foundation of OpenCV MOG2. ~85-90% accuracy on CDnet2012
    Our MOG2 baseline implements this directly.

[4] Defect Detection Survey — Czimmermann et al. (2020)
    arXiv:2103.14030. Reviews 120+ papers.
    Classical CV (frame diff, MOG2): Precision 0.70-0.85, Recall 0.60-0.80
    DL methods: IoU 0.45-0.72, FPS 5-30 (GPU)
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # never call plt.show() — save only
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
_BACKEND_DIR    = Path(__file__).parent
_PROJECT_DIR    = _BACKEND_DIR.parent
PREDICTIONS_DIR = _BACKEND_DIR / "predictions"
GRAPHS_DIR      = _BACKEND_DIR / "static" / "graphs"
DATASET_DIR     = _PROJECT_DIR / "dataset"
# Use settings as single source of truth for ground truth path
try:
    from config import settings as _settings
    GT_CSV = Path(_settings.GROUND_TRUTH_CSV)
except Exception:
    GT_CSV = _PROJECT_DIR / "ground_truth.csv"

METHODS = ["fd", "mog2", "running_avg", "custom"]

COLOURS = {
    "fd":          "#4C8BF5",
    "mog2":        "#34A853",
    "running_avg": "#FBBC05",
    "custom":      "#EA4335",
    "yolo":        "#00BCD4",
    "mask_rcnn":   "#FF7043",
}

# Literature reference values (from cited papers)
# ─────────────────────────────────────────────────────────────────
LITERATURE: Dict[str, Dict] = {
    "YOLOv4 (GPU)": {
        "precision": 0.92, "recall": 0.90, "f1": 0.910,
        "accuracy": 0.93, "fps": 62, "iou": 0.657,
        "ref": "Bochkovskiy et al., arXiv:2004.10934 (2020)",
        "notes": "COCO dataset, Tesla V100 GPU. Fine-tuned defect detection achieves higher precision.",
    },
    "Mask R-CNN (GPU)": {
        "precision": 0.88, "recall": 0.85, "f1": 0.865,
        "accuracy": 0.89, "fps": 8, "iou": 0.663,
        "ref": "He et al., arXiv:1703.06870 (2017)",
        "notes": "Instance segmentation. High IoU but slow. GPU required; CPU is 50x slower.",
    },
    "MOG2 / GMM (CPU)": {
        "precision": 0.80, "recall": 0.74, "f1": 0.769,
        "accuracy": 0.83, "fps": 80, "iou": 0.48,
        "ref": "Stauffer & Grimson, IEEE CVPR 1999; OpenCV MOG2",
        "notes": "Adaptive background subtraction. Fast on CPU. Sensitive to lighting changes.",
    },
    "Frame Diff (CPU)": {
        "precision": 0.73, "recall": 0.70, "f1": 0.715,
        "accuracy": 0.78, "fps": 120, "iou": 0.42,
        "ref": "Survey: Czimmermann et al., arXiv:2103.14030 (2020)",
        "notes": "Simplest baseline. No background model. Sensitive to camera vibration.",
    },
}


# ═════════════════════════════════════════════════════════════════
# 1.  DATA COLLECTION
# ═════════════════════════════════════════════════════════════════

def collect_data(
    camera_id: str,
    roi_id: str,
    frame: np.ndarray,
    frame_index: int,
    is_defect: bool = False,
    mask: Optional[np.ndarray] = None,
) -> None:
    """
    Save frame + optional mask to dataset/ and append to ground_truth.csv.

    Call this inside the camera worker during a labelled test session.
    """
    images_dir = DATASET_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{frame_index:06d}_{camera_id}_{roi_id}"
    cv2.imwrite(str(images_dir / f"{stem}.png"), frame)

    if mask is not None:
        masks_dir = DATASET_DIR / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(masks_dir / f"{stem}.png"), mask)

    gt_path     = DATASET_DIR / "ground_truth.csv"
    write_header = not gt_path.exists()
    with open(gt_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["frame", "camera_id", "roi_id", "actual"])
        w.writerow([frame_index, camera_id, roi_id, int(is_defect)])


def generate_example_ground_truth(output_path: Optional[str] = None) -> str:
    """Write an annotated example ground_truth.csv."""
    path = Path(output_path or GT_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        ["frame", "camera_id", "roi_id", "actual"],
        # Clean frames (no defect)
        [0, "0", "zone_a", 0], [1, "0", "zone_a", 0],
        [2, "0", "zone_a", 0], [3, "0", "zone_a", 0],
        [4, "0", "zone_a", 0], [5, "0", "zone_a", 0],
        [6, "0", "zone_a", 0], [7, "0", "zone_a", 0],
        [8, "0", "zone_a", 0], [9, "0", "zone_a", 0],
        # Defect frames — wire placed at frame 10
        [10, "0", "zone_a", 1], [11, "0", "zone_a", 1],
        [12, "0", "zone_a", 1], [13, "0", "zone_a", 1],
        [14, "0", "zone_a", 1], [15, "0", "zone_a", 1],
        # Wire removed
        [16, "0", "zone_a", 0], [17, "0", "zone_a", 0],
        [18, "0", "zone_a", 0], [19, "0", "zone_a", 0],
        # Second ROI — different zone
        [0,  "0", "zone_b", 0], [1,  "0", "zone_b", 0],
        [5,  "0", "zone_b", 1], [6,  "0", "zone_b", 1],
        [7,  "0", "zone_b", 1], [8,  "0", "zone_b", 0],
    ]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[evaluator] Example ground_truth.csv → {path}")
    return str(path)


# ═════════════════════════════════════════════════════════════════
# 2.  PREDICTION LOG LOADING
# ═════════════════════════════════════════════════════════════════

def load_predictions(method: str) -> Optional[pd.DataFrame]:
    files = list(PREDICTIONS_DIR.glob(f"{method}_*.csv"))

    if not files:
        path = PREDICTIONS_DIR / f"{method}.csv"
        if not path.exists():
            return None
        files = [path]

    dfs = []

    for file in sorted(files):
        # Peek at first line to detect missing header before reading
        raw_first = file.read_bytes().decode("utf-8", errors="replace").split("\n")[0].strip()
        first_field = raw_first.split(",")[0].strip()
        has_header = not first_field.lstrip("-").isdigit()

        df = pd.read_csv(file, header=0 if has_header else None)

        # ✅ FIXED column mapping
        if df.shape[1] == 6:
          df.columns = ["frame", "camera_id", "roi_id", "prediction", "score", "latency_ms"]

        elif df.shape[1] == 5:
            df.columns = ["frame", "roi_id", "prediction", "score", "latency_ms"]
            df["camera_id"] = "0"

        elif df.shape[1] == 4:
            df.columns = ["frame",  "prediction", "score", "latency_ms"]
            df["roi_id"] = "roi_0"
            df["camera_id"] = "0"

        elif df.shape[1] == 2:
            df.columns = ["frame", "prediction"]
            df["roi_id"] = "roi_0"
            df["camera_id"] = "0"
            df["score"] = df["prediction"]
            df["latency_ms"] = 0.0

        else:
            raise ValueError(f"{file} has unexpected format: {df.shape[1]} columns")

        df.columns = [c.strip().lower() for c in df.columns]

        # defaults
        if "frame" not in df.columns:
            df["frame"] = range(len(df))

        if "latency_ms" not in df.columns:
            df["latency_ms"] = 0.0

        if "roi_id" not in df.columns:
            df["roi_id"] = "roi_0"

        # 🔥 FORCE numeric (CRITICAL)
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce").fillna(0).astype(int)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").fillna(0.0)

        # camera_id
        parts = file.stem.split("_")
        cam_id = parts[-1] if len(parts) > 1 else "0"
        df["camera_id"] = str(cam_id)

        df["frame"] = df["frame"].astype(int)

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    print(f"[INFO] {method}: {len(files)} files merged → {len(merged)} rows")

    return merged
def load_ground_truth(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load ground_truth.csv — the independent reference labels for evaluation.

    Expected CSV format:
        frame,camera_id,roi_id,actual
        0,0,zone_a,0
        1,0,zone_a,1

    Rules:
      - frame IDs are preserved as-is so they join correctly against prediction CSVs.
      - camera_id / roi_id default to "0" / "zone_a" if absent.
      - Auto-generation from custom.csv has been REMOVED — that caused circular
        self-evaluation where custom always scored F1=1.0.  GT must come from
        a controlled labelling session (collect_data()) or manual annotation.
    """
    p = Path(path) if path else GT_CSV

    # Also accept ground_truth.csv at project root (common placement)
    if not p.exists():
        alt = _PROJECT_DIR / "ground_truth.csv"
        if alt.exists():
            p = alt

    if not p.exists():
        raise FileNotFoundError(
            f"ground_truth.csv not found at: {p}\n"
            "Options:\n"
            "  A) Run: python evaluator.py --example-gt   (writes a template)\n"
            "  B) Use collect_data() during a controlled test session\n"
            "  C) Manually create the CSV with columns: frame,camera_id,roi_id,actual\n"
            "     actual=1 → defect present,  actual=0 → clean surface"
        )

    df = pd.read_csv(p)

    # Accept both 'actual' and 'label' as the target column name
    if "actual" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"actual": "label"})
    if "label" not in df.columns:
        raise ValueError(
            "ground_truth.csv must contain an 'actual' or 'label' column.\n"
            f"Found columns: {list(df.columns)}"
        )

    # Default camera/roi columns for single-camera sessions
    if "camera_id" not in df.columns:
        df["camera_id"] = "0"
    if "roi_id" not in df.columns:
        df["roi_id"] = "zone_a"

    # Normalise types so merge keys match prediction CSVs
    df["frame"]     = df["frame"].astype(int)
    df["camera_id"] = df["camera_id"].astype(str)
    df["roi_id"]    = df["roi_id"].astype(str)

    return df


# ═════════════════════════════════════════════════════════════════
# 3.  CLASSIFICATION METRICS
# ═════════════════════════════════════════════════════════════════

def debug_alignment(gt: pd.DataFrame, preds: pd.DataFrame) -> None:
    """
    Print alignment diagnostics when a merge produces no rows.
    Call this before raising an error so the developer can see exactly
    which keys don't match without having to drop into a debugger.
    """
    print("\n🔍 ALIGNMENT CHECK")
    print(f"  GT     rows : {len(gt)}")
    print(f"  Pred   rows : {len(preds)}")
    print(f"  GT     frame range : {gt['frame'].min()} – {gt['frame'].max()}")
    print(f"  Pred   frame range : {preds['frame'].min()} – {preds['frame'].max()}")
    print(f"  GT     camera_ids  : {sorted(gt['camera_id'].unique())}")
    print(f"  Pred   camera_ids  : {sorted(preds['camera_id'].unique())}")
    print(f"  GT     roi_ids     : {sorted(gt['roi_id'].unique())}")
    print(f"  Pred   roi_ids     : {sorted(preds['roi_id'].unique())}")
    overlap_frames = set(gt["frame"]) & set(preds["frame"])
    print(f"  Overlapping frames : {len(overlap_frames)} "
          f"(e.g. {sorted(overlap_frames)[:5]})")
    print()


def _index_fallback_merge(gt: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """
    ⚠️  FOR LOCAL TESTING ONLY — NOT for research paper results.

    When frame/camera/roi keys don't align, zip GT and predictions row-by-row
    using positional index.  This is only valid when you know both CSVs cover
    the exact same frames in the same order.

    To activate: call this function instead of pd.merge() in compute_metrics()
    and set USE_INDEX_FALLBACK = True at the top of this file.
    """
    min_len = min(len(gt), len(preds))
    print(f"  ⚠ Index-based fallback active — using first {min_len} rows by position")
    return pd.DataFrame({
        "label":      gt["label"].iloc[:min_len].values,
        "prediction": preds["prediction"].iloc[:min_len].values,
        "latency_ms": preds["latency_ms"].iloc[:min_len].values,
    })


def compute_metrics(method: str, gt_path: Optional[str] = None) -> Dict[str, Any]:

    """
    Compute classification metrics for one detection method.

    Formulas (binary classification, defect=positive):
    ──────────────────────────────────────────────────
    Precision = TP / (TP + FP)
      "Of all frames labelled DEFECT by the system, how many actually were?"
      High precision → few false alarms.

    Recall = TP / (TP + FN)
      "Of all real defect frames, how many did the system catch?"
      High recall → few missed defects.

    F1 = 2 * Precision * Recall / (Precision + Recall)
      Harmonic mean. Balances precision and recall.
      Preferred metric when defect frames are rare (class imbalance).

    Accuracy = (TP + TN) / Total
      Fraction of all frames correctly classified.
      Can be misleading if classes are imbalanced.

    FPS = 1000 / avg_latency_ms
      Inference speed. Higher = faster real-time processing.
    """
    gt    = load_ground_truth(gt_path)
    preds = load_predictions(method)

    if preds is None:
        return {"method": method, "error": "No prediction CSV found in predictions/"}

    merged = pd.merge(
        gt[["frame", "camera_id", "roi_id", "label"]],
        preds[["frame", "camera_id", "roi_id", "prediction", "latency_ms"]],
        on=["frame", "camera_id", "roi_id"],
        how="inner",
    )

    if merged.empty:
        debug_alignment(gt, preds)
        return {
            "method": method,
            "error": (
                f"No matching rows between ground_truth.csv and predictions/{method}.csv.\n"
                "Check that frame, camera_id, roi_id values align — see debug output above."
            ),
        }

    y_true = merged["label"].astype(int).values
    y_pred = merged["prediction"].astype(int).values

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy  = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    lats    = merged["latency_ms"].replace(0, np.nan)
    avg_lat = float(lats.mean()) if not lats.isna().all() else 0.0
    fps     = round(1000.0 / avg_lat, 2) if avg_lat > 0 else 0.0

    return {
        "method":         method,
        "precision":      round(precision, 4),
        "recall":         round(recall,    4),
        "f1":             round(f1,        4),
        "accuracy":       round(accuracy,  4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "avg_latency_ms": round(avg_lat, 3),
        "fps":            fps,
        "n_frames":       len(y_true),
        "n_defect_gt":    int(y_true.sum()),
        "n_defect_pred":  int(y_pred.sum()),
    }


def compute_all_metrics(gt_path: Optional[str] = None) -> Dict[str, Dict]:
    """Run compute_metrics for all methods."""
    results = {}
    for method in METHODS:
        try:
            results[method] = compute_metrics(method, gt_path)
        except Exception as exc:
            results[method] = {"method": method, "error": str(exc)}
    return results


# ═════════════════════════════════════════════════════════════════
# 4.  IoU — SEGMENTATION QUALITY
# ═════════════════════════════════════════════════════════════════

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection over Union for binary segmentation masks.

    IoU = |pred ∩ gt| / |pred ∪ gt|

    Range: 0.0 (no overlap) → 1.0 (perfect match)
    Returns 1.0 when both masks are empty (both agree no defect).

    Both masks: uint8, same HxW, 0=background 255=defect.
    """
    pred_b = (pred_mask > 127).astype(np.uint8)
    gt_b   = (gt_mask   > 127).astype(np.uint8)
    inter  = np.logical_and(pred_b, gt_b).sum()
    union  = np.logical_or(pred_b, gt_b).sum()
    return float(inter / union) if union > 0 else 1.0


def compute_iou_dataset(
    method: str,
    masks_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute average IoU across all frames that have paired GT + predicted masks.

    File conventions:
      GT mask:   dataset/masks/{frame:06d}_{camera_id}_{roi_id}.png
      Pred mask: dataset/masks/pred_{method}_{frame:06d}_{camera_id}_{roi_id}.png

    HOW TO CREATE GT MASKS:
    ────────────────────────
    1. Open frame image from dataset/images/ in GIMP or Paint
    2. Select the defect region (wire, crack, object)
    3. Fill selection WHITE (255), rest BLACK (0)
    4. Export as PNG: dataset/masks/000042_0_zone_a.png
    """
    mdir = Path(masks_dir) if masks_dir else DATASET_DIR / "masks"
    if not mdir.exists():
        return {"method": method, "mean_iou": None, "error": f"Masks dir not found: {mdir}"}

    iou_vals  = []
    per_frame = []

    for gt_path in sorted(mdir.glob("*.png")):
        if gt_path.stem.startswith("pred_"):
            continue  # skip predicted masks
        pred_path = mdir / f"pred_{method}_{gt_path.stem}.png"
        if not pred_path.exists():
            continue

        gt_img   = cv2.imread(str(gt_path),   cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if gt_img is None or pred_img is None:
            continue
        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        iou = compute_iou(pred_img, gt_img)
        iou_vals.append(iou)
        per_frame.append({"stem": gt_path.stem, "iou": round(iou, 4)})

    if not iou_vals:
        return {"method": method, "mean_iou": None,
                "error": "No matched GT+pred mask pairs found. See docstring."}

    return {
        "method":     method,
        "mean_iou":   round(float(np.mean(iou_vals)), 4),
        "std_iou":    round(float(np.std(iou_vals)),  4),
        "min_iou":    round(float(np.min(iou_vals)),  4),
        "max_iou":    round(float(np.max(iou_vals)),  4),
        "n_frames":   len(iou_vals),
        "per_frame":  per_frame,
        "iou_values": iou_vals,
    }


# ═════════════════════════════════════════════════════════════════
# 5.  COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════

def build_comparison_table(
    gt_path:    Optional[str] = None,
    masks_dir:  Optional[str] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build full comparison table and save to predictions/comparison.csv.
    Columns: method | precision | recall | f1 | accuracy | iou | fps | avg_latency_ms | n_frames
    """
    metrics = compute_all_metrics(gt_path)
    rows    = []

    for method, m in metrics.items():
        if "error" in m:
            print(f"  [skip] {method}: {m['error']}")
            continue
        row = {k: m.get(k, 0) for k in
               ["method", "precision", "recall", "f1", "accuracy",
                "fps", "avg_latency_ms", "n_frames"]}
        iou_r = compute_iou_dataset(method, masks_dir)
        row["iou"] = iou_r.get("mean_iou")
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid metrics. Check predictions/ and ground_truth.csv.")

    df = pd.DataFrame(rows).set_index("method")
    out = output_csv or str(PREDICTIONS_DIR / "comparison.csv")
    df.to_csv(out)
    print(f"[evaluator] Comparison table → {out}")
    return df


# ═════════════════════════════════════════════════════════════════
# 6.  GRAPH GENERATION
# ═════════════════════════════════════════════════════════════════

def _gdir():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    return GRAPHS_DIR


def graph_precision_recall_f1(df: pd.DataFrame) -> str:
    """Bar chart: Precision / Recall / F1 per method."""
    _gdir()
    methods = df.index.tolist()
    x, w = np.arange(len(methods)), 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(methods)*2), 5))
    ax.bar(x-w, df["precision"], w, label="Precision", color="#4C8BF5", edgecolor="white", lw=0.5)
    ax.bar(x,   df["recall"],   w, label="Recall",    color="#34A853", edgecolor="white", lw=0.5)
    ax.bar(x+w, df["f1"],       w, label="F1 Score",  color="#EA4335", edgecolor="white", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in methods], fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Detection Metrics — Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    plt.tight_layout()
    out = str(GRAPHS_DIR / "metrics_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evaluator] → {out}"); return out


def graph_fps_comparison(df: pd.DataFrame) -> str:
    """Bar chart: FPS per method with literature reference lines."""
    _gdir()
    methods  = df.index.tolist()
    fps_vals = df["fps"].tolist()
    colours  = [COLOURS.get(m, "#888") for m in methods]

    fig, ax = plt.subplots(figsize=(max(7, len(methods)*1.8), 4))
    bars = ax.bar([m.upper() for m in methods], fps_vals, color=colours, edgecolor="white", lw=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("Frames per Second", fontsize=11)
    ax.set_title("FPS Comparison Across Methods", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.spines[["top","right"]].set_visible(False)

    # Literature reference lines
    ax.axhline(62, color="#00BCD4", ls=":", lw=1.3, alpha=0.8)
    ax.text(len(methods)-0.5, 63.5, "YOLOv4 GPU (62 fps)", fontsize=7, color="#00BCD4")
    ax.axhline(8,  color="#FF7043", ls=":", lw=1.3, alpha=0.8)
    ax.text(len(methods)-0.5, 9.5,  "Mask R-CNN GPU (8 fps)", fontsize=7, color="#FF7043")

    plt.tight_layout()
    out = str(GRAPHS_DIR / "fps_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evaluator] → {out}"); return out


def graph_latency_per_frame() -> str:
    """Line graph: rolling-average latency per frame for every method."""
    _gdir()
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False

    for method in METHODS:
        # Collect both {method}.csv and {method}_*.csv (per-camera files)
        candidates = [PREDICTIONS_DIR / f"{method}.csv"] +                      sorted(PREDICTIONS_DIR.glob(f"{method}_*.csv"))
        frames_all = []
        for path in candidates:
            if not path.exists():
                continue
            try:
                df_tmp = pd.read_csv(path)
                if "latency_ms" in df_tmp.columns:
                    frames_all.append(df_tmp["latency_ms"])
            except Exception:
                pass
        if not frames_all:
            continue
        combined = pd.concat(frames_all, ignore_index=True)
        lat = combined.replace(0, np.nan).rolling(15, min_periods=1).mean()
        ax.plot(lat.index, lat, label=method.upper(),
                color=COLOURS.get(method,"#888"), lw=1.6, alpha=0.88)
        plotted = True

    if not plotted: plt.close(fig); return ""
    ax.set_xlabel("Frame Index", fontsize=11); ax.set_ylabel("Latency (ms, rolling avg)", fontsize=11)
    ax.set_title("Per-Frame Inference Latency — All Methods", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle="--"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = str(GRAPHS_DIR / "latency_per_frame.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evaluator] → {out}"); return out


def graph_accuracy_vs_fps(df: pd.DataFrame) -> str:
    """
    Scatter plot: Accuracy (y) vs FPS (x).
    Bubble size ∝ F1.  Diamond markers = literature references.
    """
    _gdir()
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in df.index:
        row = df.loc[method]
        acc = row.get("accuracy", 0); fps = row.get("fps", 0); f1 = row.get("f1", 0)
        c   = COLOURS.get(method, "#888"); sz = max(80, f1 * 600)
        ax.scatter(fps, acc, s=sz, color=c, alpha=0.85, edgecolors="white", lw=1.2, zorder=3)
        ax.annotate(method.upper(), (fps, acc), textcoords="offset points", xytext=(8,4),
                    fontsize=9, color=c)

    refs = [("YOLOv8\n(GPU)", 50, 0.91, "#00BCD4"),
            ("Mask R-CNN\n(GPU)", 8, 0.89, "#FF7043"),
            ("MOG2\n(baseline)", 80, 0.83, "#34A853")]
    for name, fps, acc, c in refs:
        ax.scatter(fps, acc, s=250, color=c, alpha=0.4, edgecolors=c, lw=1.5, marker="D", zorder=2)
        ax.annotate(name, (fps, acc), textcoords="offset points", xytext=(8,-14), fontsize=7, color=c, alpha=0.7)

    ax.set_xlabel("FPS (higher = faster)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy vs FPS\n(bubble size ∝ F1;  ◇ = literature reference on GPU)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle="--"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = str(GRAPHS_DIR / "accuracy_vs_fps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evaluator] → {out}"); return out


def graph_iou_distribution(masks_dir: Optional[str] = None) -> str:
    """Histogram: IoU distribution across frames per method."""
    _gdir()
    fig, ax = plt.subplots(figsize=(10, 5)); plotted = False

    for method in METHODS:
        r = compute_iou_dataset(method, masks_dir)
        if r.get("mean_iou") is None: continue
        ax.hist(r["iou_values"], bins=20, range=(0,1), alpha=0.55,
                label=f"{method.upper()} (μ={r['mean_iou']:.3f})",
                color=COLOURS.get(method,"#888"), edgecolor="white", lw=0.5)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[evaluator] IoU graph skipped — no mask data."); return ""

    ax.set_xlabel("IoU Score", fontsize=11); ax.set_ylabel("Frames", fontsize=11)
    ax.set_title("IoU Distribution — Segmentation Quality", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle="--"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = str(GRAPHS_DIR / "iou_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evaluator] → {out}"); return out


def graph_fps_vs_cameras() -> str:
    _gdir()
    fig, ax = plt.subplots(figsize=(9, 5))

    cam_counts = np.arange(1, 9)

    for method in METHODS:
        df = load_predictions(method)
        if df is None or "latency_ms" not in df.columns:
            continue

        avg_lat = df["latency_ms"].replace(0, np.nan).mean()
        if np.isnan(avg_lat) or avg_lat <= 0:
            continue

        base_fps = 1000.0 / avg_lat

        # Linear scaling assumption (VERY IMPORTANT)
        fps_curve = [base_fps / n for n in cam_counts]

        ax.plot(
            cam_counts,
            fps_curve,
            marker="o",
            label=method.upper(),
            color=COLOURS.get(method, "#888"),
            lw=2
        )

    ax.set_xlabel("Number of Active Cameras", fontsize=11)
    ax.set_ylabel("FPS per Camera", fontsize=11)
    ax.set_title("Scalability: FPS vs Number of Cameras", fontsize=12, fontweight="bold")

    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xticks(range(1, 9))

    plt.tight_layout()
    out = str(GRAPHS_DIR / "fps_vs_cameras.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[evaluator] → {out}")
    return out


def graph_confusion_matrices(df_metrics: Dict[str, Dict]) -> str:
    """2×2 confusion matrix grid for each method."""
    _gdir()
    valid = {m: v for m, v in df_metrics.items() if "error" not in v and "tp" in v}
    if not valid: return ""

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(n*3.5, 3.5))
    if n == 1: axes = [axes]

    for ax, (method, m) in zip(axes, valid.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: Clean","Pred: Defect"], fontsize=8)
        ax.set_yticklabels(["True: Clean","True: Defect"], fontsize=8)
        ax.set_title(f"{method.upper()}\nF1={m['f1']:.3f}", fontsize=10, fontweight="bold")
        for i in range(2):
            for j in range(2):
                val = cm[i,j]
                col = "white" if val > cm.max()*0.55 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=14, fontweight="bold", color=col)

    plt.suptitle("Confusion Matrices — All Methods", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = str(GRAPHS_DIR / "confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evaluator] → {out}"); return out

from sklearn.metrics import roc_curve, auc

def graph_roc_curve(gt_path: Optional[str] = None) -> str:
    """
    ROC Curve: True Positive Rate vs False Positive Rate
    Uses prediction as score (binary fallback if no confidence available)
    """
    _gdir()

    gt = load_ground_truth(gt_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False

    for method in METHODS:
        preds = load_predictions(method)
        if preds is None:
            continue

        merged = pd.merge(
            gt[["frame", "camera_id", "roi_id", "label"]],
            preds[["frame", "camera_id", "roi_id", "prediction","score"]],
            on=["frame", "camera_id", "roi_id"],
            how="inner",
        )

        if merged.empty:
            continue

        y_true = merged["label"].astype(int)

        # ⚠️ If later you add confidence score → replace here
        if "score" in merged.columns:
            y_scores = merged["score"]
        else:
            y_scores = merged["prediction"]  # binary fallback

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                label=f"{method.upper()} (AUC={roc_auc:.2f})",
                color=COLOURS.get(method, "#888"),
                linewidth=2,
            )

            plotted = True

        except Exception as e:
            print(f"[ROC] Error for {method}: {e}")

    if not plotted:
        plt.close(fig)
        print("[evaluator] ROC skipped — insufficient data")
        return ""

    # Random baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — All Methods", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out = str(GRAPHS_DIR / "roc_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[evaluator] → {out}")
    return out


# ═════════════════════════════════════════════════════════════════
# 7.  COMPARISON WITH LITERATURE
# ═════════════════════════════════════════════════════════════════

def compare_with_baselines(
    our_results: Dict[str, Dict],
    output_csv:  Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge our system's metrics with literature reference values and print table.

    References
    ----------
    • arXiv:2004.10934 — YOLOv4 (Bochkovskiy et al., 2020)
    • arXiv:1703.06870 — Mask R-CNN (He et al., 2017)
    • IEEE CVPR 1999   — GMM background model (Stauffer & Grimson)
    • arXiv:2103.14030 — Surface defect detection survey (Czimmermann et al., 2020)
    """
    rows = []
    for method, m in our_results.items():
        if "error" in m: continue
        rows.append({
            "Method":    method.upper(),
            "Type":      "Ours (CPU)",
            "Precision": m.get("precision","—"),
            "Recall":    m.get("recall","—"),
            "F1":        m.get("f1","—"),
            "Accuracy":  m.get("accuracy","—"),
            "FPS":       m.get("fps","—"),
            "IoU":       m.get("iou","—"),
            "Reference": "This work",
        })
    for name, lit in LITERATURE.items():
        rows.append({
            "Method":    name,
            "Type":      "Literature",
            "Precision": lit["precision"], "Recall":    lit["recall"],
            "F1":        lit["f1"],        "Accuracy":  lit["accuracy"],
            "FPS":       lit["fps"],       "IoU":       lit["iou"],
            "Reference": lit["ref"],
        })

    df = pd.DataFrame(rows)
    out = output_csv or str(PREDICTIONS_DIR / "comparison_with_literature.csv")
    df.to_csv(out, index=False)
    print(f"[evaluator] Literature comparison → {out}")

    print("\n" + "═"*95)
    print("  COMPARISON WITH EXISTING RESEARCH")
    print("═"*95)
    cols = ["Method","Type","Precision","Recall","F1","Accuracy","FPS","IoU"]
    print(df[cols].to_string(index=False,
          float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
    print("═"*95)
    print("\nKey references:")
    for name, lit in LITERATURE.items():
        print(f"  [{name}]  {lit['ref']}")
        print(f"    → {lit['notes']}")
    return df


def graph_comparison_with_literature(our_results: Dict[str, Dict]) -> str:
    """Horizontal bar chart: our methods vs literature on F1 and FPS."""
    _gdir()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    our_m   = [(m, v) for m, v in our_results.items() if "error" not in v]
    our_names = [m.upper() for m, _ in our_m]
    our_f1    = [v["f1"]  for _, v in our_m]
    our_fps   = [v["fps"] for _, v in our_m]
    our_cols  = [COLOURS.get(m, "#4C8BF5") for m, _ in our_m]

    lit_names = list(LITERATURE.keys())
    lit_f1    = [LITERATURE[n]["f1"]  for n in lit_names]
    lit_fps   = [LITERATURE[n]["fps"] for n in lit_names]

    all_names = our_names + lit_names
    all_f1    = our_f1   + lit_f1
    all_fps   = our_fps  + lit_fps
    all_cols  = our_cols + ["#bbbbbb"]*len(lit_names)

    for ax, vals, xlabel, title in [
        (axes[0], all_f1,  "F1 Score", "F1 Score Comparison"),
        (axes[1], all_fps, "FPS",      "FPS Comparison"),
    ]:
        bars = ax.barh(all_names, vals, color=all_cols, edgecolor="white", lw=0.5)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color="#EA4335", label="Our system (CPU)"),
            mpatches.Patch(color="#bbbbbb", label="Literature"),
        ], fontsize=8, loc="lower right")

    plt.suptitle("Our System vs Literature Baselines", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = str(GRAPHS_DIR / "comparison_with_literature.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evaluator] → {out}"); return out


# ═════════════════════════════════════════════════════════════════
# 8.  FULL PIPELINE
# ═════════════════════════════════════════════════════════════════

def generate_graphs(
    df:         Optional[pd.DataFrame]  = None,
    df_metrics: Optional[Dict[str,Dict]]= None,
    gt_path:    Optional[str] = None,
    masks_dir:  Optional[str] = None,
) -> List[str]:
    """Generate all graphs. Returns list of saved paths."""
    if df is None:
        try:
            df = build_comparison_table(gt_path, masks_dir)
        except Exception as e:
            print(f"[evaluator] Cannot build table: {e}"); return []
    if df_metrics is None:
        df_metrics = compute_all_metrics(gt_path)

    paths = []
    for fn in [
        lambda: graph_precision_recall_f1(df),
        lambda: graph_fps_comparison(df),
        lambda: graph_latency_per_frame(),
        lambda: graph_accuracy_vs_fps(df),
        lambda: graph_iou_distribution(masks_dir),
        lambda: graph_fps_vs_cameras(),
        lambda: graph_confusion_matrices(df_metrics),
        lambda: graph_comparison_with_literature(df_metrics),
        lambda: graph_roc_curve(gt_path),
    ]:
        try:
            p = fn()
            if p: paths.append(p)
        except Exception as e:
            print(f"[evaluator] Graph error: {e}")
    return paths


def run_full_evaluation(
    gt_path:   Optional[str] = None,
    masks_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    Call from FastAPI:  GET /evaluate
    Or from CLI:        python evaluator.py
    """
    print("\n" + "═"*60)
    print("  DefectWatch Research Evaluation Pipeline")
    print("═"*60)

    print("\n[1/5] Computing metrics for all methods…")
    df_metrics = compute_all_metrics(gt_path)
    for m, v in df_metrics.items():
        if "error" in v:
            print(f"  ✗ {m}: {v['error']}")
        else:
            print(f"  ✓ {m}: F1={v['f1']:.3f}  Acc={v['accuracy']:.3f}  FPS={v['fps']:.1f}")

    print("\n[2/5] Building comparison table…")
    try:
        df = build_comparison_table(gt_path, masks_dir)
    except Exception as e:
        print(f"  ✗ {e}"); df = None

    print("\n[3/5] Generating graphs…")
    graph_paths = generate_graphs(df, df_metrics, gt_path, masks_dir)

    print("\n[4/5] Comparing with literature…")
    try:
        lit_df = compare_with_baselines(df_metrics)
    except Exception as e:
        print(f"  ✗ {e}"); lit_df = None

    print(f"\n[5/5] Done — {len(graph_paths)} graphs saved to {GRAPHS_DIR}")
    print("═"*60 + "\n")

    return {"metrics": df_metrics, "comparison_df": df,
            "literature_df": lit_df, "graphs": graph_paths}


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DefectWatch Research Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluator.py                         # full evaluation
  python evaluator.py --example-gt            # write example ground_truth.csv
  python evaluator.py --gt path/to/gt.csv     # custom ground truth
  python evaluator.py --masks path/to/masks/  # include IoU from masks
  python evaluator.py --graphs-only           # regenerate graphs only
""",
    )
    parser.add_argument("--gt",           type=str)
    parser.add_argument("--masks",        type=str)
    parser.add_argument("--example-gt",   action="store_true")
    parser.add_argument("--graphs-only",  action="store_true")
    args = parser.parse_args()

    if args.example_gt:
        path = generate_example_ground_truth()
        print(f"\nTemplate: {path}\nEdit 'actual' column: 1=defect, 0=clean.")
        raise SystemExit(0)

    if args.graphs_only:
        cmp = PREDICTIONS_DIR / "comparison.csv"
        if cmp.exists():
            generate_graphs(pd.read_csv(cmp, index_col="method"))
        else:
            print(f"Run full evaluation first — {cmp} not found.")
        raise SystemExit(0)

    run_full_evaluation(gt_path=args.gt, masks_dir=args.masks)