import sys, os, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings
from temporal import TemporalVoter
from detection_methods.custom import CustomDefectDetector, normalise_luminance
from detection_methods.fd import FrameDifferencingDetector
import eval_tools as ev

def scene(seed=0, h=240, w=320):
    r = np.random.default_rng(seed); yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lum = 150 + 10*np.sin(xx/55) + 8*np.cos(yy/40)
    img = np.dstack([lum, lum+2, lum+5]) + r.normal(0, 2.5, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)

def test_voter():
    v = TemporalVoter(5, 3); out = [v.update("a", d) for d in [1, 1, 0, 1, 0, 0, 0]]
    assert out == [False, False, False, True, True, False, False], out
    assert TemporalVoter(1, 1).update("a", True) is True
    try: TemporalVoter(2, 3); assert False
    except ValueError: pass

def test_illum_norm_removes_gain_keeps_defect():
    ref = scene(0); cur_gain = np.clip(scene(1).astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
    d0 = CustomDefectDetector(0, illum_norm=False); d1 = CustomDefectDetector(0, illum_norm=True)
    for d in (d0, d1):
        d._save_intermediates = False; d._roi_threshold = 8; d._roi_min_area = 150; d.initialize(ref)
    assert d0._detect(cur_gain).prediction == 1      # gain alone falsely triggers without normalisation
    assert d1._detect(cur_gain).prediction == 0      # ...and is removed with it
    defect = cur_gain.copy(); cv2.rectangle(defect, (100, 80), (140, 120), (20, 20, 20), -1)
    assert d1._detect(defect).prediction == 1        # a real defect is still found

def test_matched_thresholds_flag():
    ref = scene(0); cur = ref.copy(); cv2.rectangle(cur, (50, 50), (90, 90), (120, 120, 120), -1)
    fd = FrameDifferencingDetector(0); fd.initialize(ref)
    fd._roi_threshold, fd._roi_min_area = 8, 150
    settings.MATCHED_THRESHOLDS = False; assert fd.op_threshold() == settings.BINARY_THRESHOLD
    settings.MATCHED_THRESHOLDS = True; assert fd.op_threshold() == 8 and fd.op_min_area() == 150
    settings.MATCHED_THRESHOLDS = False

def test_raw_score_monotone():
    ref = scene(0); d = CustomDefectDetector(0); d._save_intermediates = False; d._roi_threshold = 8; d._roi_min_area = 150; d.initialize(ref)
    s0 = d._detect(scene(2)).raw_score; cur = scene(2).copy(); cv2.rectangle(cur, (50, 50), (120, 120), (0, 0, 0), -1)
    assert d._detect(cur).raw_score > s0

def test_strict_merge_fails_loudly_and_positional_flagged(tmp_path=None):
    tmp = Path(tempfile.mkdtemp())
    gt = pd.DataFrame({"frame": range(10), "camera_id": 0, "roi_id": "roi_A", "actual": [0, 1] * 5})
    pr = pd.DataFrame({"frame": range(10), "camera_id": 0, "roi_id": "roi_B", "prediction": [0, 1] * 5, "score": 0.1, "latency_ms": 1})
    pr.to_csv(tmp / "custom_0.csv", index=False)
    try: ev.evaluate(gt, tmp, ["custom"], 0); assert False
    except SystemExit as e: assert "STRICT MERGE FAILED" in str(e)
    tab, _, notes = ev.evaluate(gt, tmp, ["custom"], 0, allow_positional=True, n_boot=50)
    assert any("positional" in n for n in notes)

def test_metrics_and_always_positive_and_bootstrap():
    tmp = Path(tempfile.mkdtemp()); n = 400; rng = np.random.default_rng(0)
    y = (np.arange(n) // 40 % 2).astype(int)                      # persistent segments
    err = np.repeat(rng.random(n // 20) < 0.1, 20)               # errors come in runs of 20 frames
    p = np.where(err, 1 - y, y)
    gt = pd.DataFrame({"frame": range(n), "camera_id": 0, "roi_id": "r", "actual": y})
    pd.DataFrame({"frame": range(n), "camera_id": 0, "roi_id": "r", "prediction": p, "score": p * .5, "latency_ms": 1, "raw_score": p * .5 + rng.random(n) * .1}).to_csv(tmp / "custom_0.csv", index=False)
    tab, roi, _ = ev.evaluate(gt, tmp, ["custom"], 0, n_boot=200)
    c = tab[tab.method == "custom"].iloc[0]; a = tab[tab.method == "always_positive"].iloc[0]
    assert a.recall == 1.0 and abs(a.precision - y.mean()) < 1e-9 and c.roc_auc > 0.8 and len(roi) == 1
    lo, hi = [float(x) for x in c.recall_ci.split("-")]; assert lo <= c.recall <= hi
    iid = ev.block_bootstrap(y, p, block=1, n_boot=400)["recall"]; blk = ev.block_bootstrap(y, p, block=20, n_boot=400)["recall"]
    assert (blk[1] - blk[0]) > (iid[1] - iid[0])                       # correlated errors -> block CI must be wider

def test_roi_processor_logs_new_columns():
    import roi_processor as rp
    tmp = Path(tempfile.mkdtemp()); rp._PRED_DIR = tmp
    rois = [{"roi_id": "roi_T", "x": 10, "y": 10, "w": 200, "h": 150, "threshold": 8, "min_area": 150}]
    base = cv2.resize(scene(0), (640, 480)); cams = "9"
    rp.set_rois(cams, rois)
    for k in range(4):
        f = base.copy()
        if k >= 2: cv2.rectangle(f, (60, 60), (120, 120), (10, 10, 10), -1)
        rp.process_frame(f, cams, rois)
    df = pd.read_csv(tmp / "custom_9.csv"); assert list(df.columns)[-1] == "raw_score" and len(df) == 4
    meta = json.loads((tmp / "custom_9.csv.meta.json").read_text()); assert meta["detector"]["method"] == "custom"
    assert df.prediction.tolist()[2:] == [1, 1]

if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception as e: fails += 1; print("FAIL", n, type(e).__name__, e)
    sys.exit(fails)
