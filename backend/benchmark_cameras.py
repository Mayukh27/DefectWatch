"""
benchmark_cameras.py — MEASURED multi-camera throughput (replaces the 13 %-overhead model).

  python benchmark_cameras.py --method custom --cams 1 2 4 --mode thread process --seconds 6
Reports per-camera and aggregate FPS and the logical CPU count, so numbers are comparable only
on the same hardware. Run it on the deployment machine and quote that machine in the paper.
"""
import argparse, os, sys, threading, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))


def _frames(n=24, h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 170, (h, w, 3), dtype=np.uint8)
    return [np.clip(base.astype(np.int16) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]


def _make(method):
    from detection_methods.custom import CustomDefectDetector
    from detection_methods.fd import FrameDifferencingDetector
    from detection_methods.mog2 import MOG2Detector
    d = {"custom": CustomDefectDetector, "fd": FrameDifferencingDetector, "mog2": MOG2Detector}[method](camera_id=0)
    if method == "custom": d._save_intermediates = False
    return d


def _worker(method, seconds, q=None, out=None, idx=0):
    det = _make(method); fr = _frames(seed=idx); det.initialize(fr[0]); n = 0; t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        det._detect(fr[n % len(fr)]); n += 1
    fps = n / (time.perf_counter() - t0)
    if q is not None: q.put(fps)
    else: out[idx] = fps


def run(method, n, mode, seconds):
    if mode == "thread":
        out = {}; th = [threading.Thread(target=_worker, args=(method, seconds, None, out, i)) for i in range(n)]
        [t.start() for t in th]; [t.join() for t in th]; v = list(out.values())
    else:
        q = mp.Queue(); ps = [mp.Process(target=_worker, args=(method, seconds, q, None, i)) for i in range(n)]
        [p.start() for p in ps]; v = [q.get() for _ in ps]; [p.join() for p in ps]
    return float(np.mean(v)), float(np.sum(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="custom"); ap.add_argument("--cams", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--mode", nargs="+", default=["thread", "process"]); ap.add_argument("--seconds", type=float, default=6)
    a = ap.parse_args(); print(f"logical CPUs: {os.cpu_count()}  method: {a.method}")
    print("mode,cameras,fps_per_camera,fps_aggregate")
    for mode in a.mode:
        for n in a.cams:
            pc, ag = run(a.method, n, mode, a.seconds); print(f"{mode},{n},{pc:.2f},{ag:.2f}")
