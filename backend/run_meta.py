"""run_meta.py — sidecar JSON that records the operating point of every logged run."""
import json, subprocess, time
from pathlib import Path


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_meta(csv_path: Path, method: str, detector_cfg: dict, settings) -> None:
    """Write <csv>.meta.json next to a prediction CSV (once per file per session)."""
    meta = {
        "method": method,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": _git_hash(),
        "detector": detector_cfg,
        "global": {"BINARY_THRESHOLD": settings.BINARY_THRESHOLD,
                   "MIN_CONTOUR_AREA": settings.MIN_CONTOUR_AREA,
                   "MATCHED_THRESHOLDS": settings.MATCHED_THRESHOLDS,
                   "ILLUM_NORM": settings.ILLUM_NORM,
                   "VOTE_N": settings.VOTE_N, "VOTE_K": settings.VOTE_K},
    }
    Path(str(csv_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
