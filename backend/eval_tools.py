"""
eval_tools.py — reproducible, reviewer-proof evaluation.

  python eval_tools.py --gt ../ground_truth.csv --pred predictions --camera 0
  python eval_tools.py ... --allow-positional      # UNVERIFIED alignment, loudly flagged

What it fixes compared with the original evaluator
  * STRICT merge on (frame, camera_id, roi_id). If nothing matches it stops and prints
    both ROI-id sets instead of silently aligning by row position.
  * Always-positive reference baseline (a detector that never misses and never thinks).
  * Block-bootstrap confidence intervals (resamples contiguous blocks of frames, because
    consecutive video frames are not independent).
  * Threshold-free ROC-AUC and PR-AUC from `raw_score` when present.
  * Per-ROI tables, F2 and MCC, and the operating point read from the *.meta.json sidecar.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["frame", "camera_id", "roi_id"]


def load_pred(pred_dir: Path, method: str, camera: int) -> pd.DataFrame:
    f = Path(pred_dir) / f"{method}_{camera}.csv"
    df = pd.read_csv(f)
    df["camera_id"] = df["camera_id"].astype(str)
    meta = Path(str(f) + ".meta.json")
    df.attrs["meta"] = json.loads(meta.read_text()) if meta.exists() else None
    return df


def strict_merge(gt: pd.DataFrame, pred: pd.DataFrame):
    g = gt.copy(); g["camera_id"] = g["camera_id"].astype(str)
    p = pred.copy(); p["camera_id"] = p["camera_id"].astype(str)
    m = p.merge(g[KEYS + ["actual"]], on=KEYS, how="inner")
    return m


def positional_merge(gt: pd.DataFrame, pred: pd.DataFrame, camera: int):
    """Row-order alignment. NOT verified — only for diagnosing legacy logs."""
    g = gt[gt["camera_id"].astype(str) == str(camera)].reset_index(drop=True)
    n = min(len(g), len(pred))
    m = pred.iloc[:n].reset_index(drop=True).copy()
    m["actual"] = g["actual"].iloc[:n].values
    return m


def confusion(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    return (int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum()),
            int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum()))


def metrics(y, p):
    tn, fp, fn, tp = confusion(y, p)
    d = lambda a, b: a / b if b else 0.0
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    f1 = d(2 * prec * rec, prec + rec); f2 = d(5 * prec * rec, 4 * prec + rec)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec, f1=f1, f2=f2,
                specificity=d(tn, tn + fp), accuracy=d(tp + tn, len(y)),
                mcc=d(tp * tn - fp * fn, den))


def auc_scores(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y)
    if len(set(y)) < 2 or s is None:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def block_bootstrap(y, p, block=50, n_boot=2000, seed=0, groups=None):
    """95% CI for recall, precision, F1 by resampling contiguous blocks of frames."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    starts = np.arange(0, n, block)
    blocks = [(s, min(s + block, n)) for s in starts]
    out = {"recall": [], "precision": [], "f1": []}
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(*blocks[i]) for i in rng.integers(0, len(blocks), len(blocks))])
        m = metrics(y[idx], p[idx])
        for k in out: out[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def evaluate(gt, pred_dir, methods, camera, allow_positional=False, block=50, n_boot=2000):
    rows, per_roi, notes = [], [], []
    for meth in methods:
        pr = load_pred(pred_dir, meth, camera)
        m = strict_merge(gt, pr)
        if len(m) == 0:
            msg = (f"[{meth}] 0 rows matched on {KEYS}. ROI ids in predictions: "
                   f"{sorted(pr.roi_id.unique())[:6]}; in ground truth (camera {camera}): "
                   f"{sorted(gt[gt.camera_id.astype(str) == str(camera)].roi_id.unique())[:6]}")
            if not allow_positional:
                raise SystemExit("STRICT MERGE FAILED\n" + msg +
                                 "\nFix the ROI ids, or pass --allow-positional (UNVERIFIED).")
            notes.append("WARNING positional alignment used for " + meth + " — unverified")
            m = positional_merge(gt, pr, camera)
        y, p = m["actual"].values, m["prediction"].values
        s = m["raw_score"].values if "raw_score" in m else m["score"].values
        r = metrics(y, p); r["method"] = meth; r["n"] = len(m)
        r["roc_auc"], r["pr_auc"] = auc_scores(y, s)
        ci = block_bootstrap(y, p, block, n_boot)
        for k, v in ci.items(): r[f"{k}_ci"] = f"{v[0]:.3f}-{v[1]:.3f}"
        r["config"] = json.dumps((pr.attrs.get("meta") or {}).get("detector", "no meta"))
        rows.append(r)
        if "roi_id" in m:
            for roi, g in m.groupby("roi_id"):
                q = metrics(g["actual"].values, g["prediction"].values); q.update(method=meth, roi_id=roi, n=len(g))
                per_roi.append(q)
        if meth == methods[0]:
            ap = metrics(y, np.ones_like(y)); ap.update(method="always_positive", n=len(y)); rows.append(ap)
    return pd.DataFrame(rows), pd.DataFrame(per_roi), notes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["custom", "fd", "mog2", "running_avg"])
    ap.add_argument("--allow-positional", action="store_true")
    ap.add_argument("--block", type=int, default=50); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="evaluation_report")
    a = ap.parse_args()
    gt = pd.read_csv(a.gt)
    tab, roi, notes = evaluate(gt, a.pred, a.methods, a.camera, a.allow_positional, a.block, a.boot)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
    print(tab.drop(columns=["config"]).round(3).to_string(index=False)); [print(n) for n in notes]
    tab.to_csv(a.out + "_overall.csv", index=False); roi.to_csv(a.out + "_per_roi.csv", index=False)
