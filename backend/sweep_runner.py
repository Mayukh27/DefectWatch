"""
sweep_runner.py — threshold-matched real-data sweep (answers the 'unfair comparison' objection).

  python sweep_runner.py --frames DIR --labels labels.csv --roi X Y W H [--out sweep.csv]

DIR holds frames named <index>.png|jpg (index = frame number); labels.csv has columns
frame,actual. Frame with the smallest index is the clean reference. Every method is run at the
SAME threshold tau (2..40) and the SAME min-area, so the curves compare detectors, not operating points.
Output: one row per (method, tau) with TPR/FPR, plus AUC of each sweep curve.
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import settings
from detection_methods.custom import CustomDefectDetector
from detection_methods.fd import FrameDifferencingDetector
from detection_methods.running_avg import RunningAverageDetector


def make(method, tau, min_area, **kw):
    cls = {"custom": CustomDefectDetector, "fd": FrameDifferencingDetector,
           "running_avg": RunningAverageDetector}[method]
    det = cls(camera_id=0, **kw) if method == "custom" else cls(camera_id=0)
    if method == "custom": det._save_intermediates = False
    det._roi_threshold, det._roi_min_area = tau, min_area
    return det


def run(frames, labels, roi, taus, min_area, variants):
    x, y, w, h = roi
    idx = sorted(frames)
    crops = {i: cv2.imread(str(frames[i]))[y:y + h, x:x + w] for i in idx}
    ref = crops[idx[0]]
    lab = labels.set_index("frame")["actual"]
    ev = [i for i in idx[1:] if i in lab.index]
    yv = np.array([lab[i] for i in ev]); rows = []
    prev = settings.MATCHED_THRESHOLDS; settings.MATCHED_THRESHOLDS = True
    for name, (method, kw) in variants.items():
        for tau in taus:
            det = make(method, tau, min_area, **kw); det.initialize(ref)
            pred = np.array([int(det._detect(crops[i]).prediction) for i in ev])
            tp = int(((yv == 1) & (pred == 1)).sum()); fp = int(((yv == 0) & (pred == 1)).sum())
            rows.append(dict(variant=name, tau=tau, tpr=tp / max(1, (yv == 1).sum()), fpr=fp / max(1, (yv == 0).sum())))
    settings.MATCHED_THRESHOLDS = prev
    return pd.DataFrame(rows)


VARIANTS = {"fd_gray": ("fd", {}), "running_avg": ("running_avg", {}),
            "lab_ab_only": ("custom", dict(use_luminance=False, use_ssim=False, illum_norm=False)),
            "lab_ab_L": ("custom", dict(use_ssim=False, illum_norm=False)),
            "lab_ab_L_ssim": ("custom", dict(illum_norm=False)),
            "lab_ab_L_ssim_illum": ("custom", dict(illum_norm=True))}


def sweep_auc(df):
    out = {}
    for v, g in df.groupby("variant"):
        pts = sorted(set([(0.0, 0.0), (1.0, 1.0)] + list(zip(g.fpr, g.tpr))))
        out[v] = float(np.trapezoid([p[1] for p in pts], [p[0] for p in pts]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--roi", nargs=4, type=int, required=True)
    ap.add_argument("--min-area", type=int, default=150); ap.add_argument("--out", default="sweep.csv")
    a = ap.parse_args()
    fr = {int(p.stem): p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    df = run(fr, pd.read_csv(a.labels), a.roi, list(range(2, 41, 2)), a.min_area, VARIANTS)
    df.to_csv(a.out, index=False); print(pd.Series(sweep_auc(df)).round(3).to_string())
