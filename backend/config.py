"""
Configuration for the Defect Detection System.
Edit values here or set environment variables with the same name (uppercased).
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_env_file() -> None:
    """Load backend/.env when python-dotenv is available."""
    if load_dotenv is None:
        return
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)


def _parse_cors_origins(raw: str) -> list[str]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return vals or ["*"]


_load_env_file()


@dataclass
class Settings:
    # ──────────────────────────────────────────
    # Feature Flags
    # ──────────────────────────────────────────

    # Master switch for Deep Learning pipeline.
    # Keep False until a labelled dataset is available.
    USE_DL: bool = bool(os.getenv("USE_DL", "False").lower() in ("1", "true"))

    # ──────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────
    DEFAULT_METHOD: str = os.getenv("DEFAULT_METHOD", "custom")

    # Minimum contour area (pixels²) to register a defect event
    # Raised from 300 → 800 to reject small noise blobs
    MIN_CONTOUR_AREA: int = int(os.getenv("MIN_CONTOUR_AREA", "300"))

    # Threshold value for binary thresholding in simple methods
    # Raised from 25 → 35; Otsu adaptation in defect_marker adds further protection
    BINARY_THRESHOLD: int = int(os.getenv("BINARY_THRESHOLD", "25"))

    # ──────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    PREDICTIONS_DIR: str = os.path.join(BASE_DIR, "predictions")
    GRAPHS_DIR: str      = os.path.join(BASE_DIR, "static", "graphs")
    DB_PATH: str         = os.path.join(BASE_DIR, "detections.db")

    GROUND_TRUTH_CSV: str = os.path.join(
        os.path.dirname(BASE_DIR), "ground_truth.csv"
    )

    # Optional: path to a demo video file (used when no physical camera is available)
    DEMO_VIDEO_PATH: str = os.getenv("DEMO_VIDEO_PATH", "")

    # ──────────────────────────────────────────
    # Camera Mode
    # ──────────────────────────────────────────
    # "client"  — laptop cameras push frames via HTTP
    # "rtsp"    — server pulls from RTSP IP cameras
    # "mixed"   — both types active simultaneously (default)
    CAMERA_MODE: str = os.getenv("CAMERA_MODE", "mixed")

    # ──────────────────────────────────────────
    # DL / YOLO (inactive until dataset ready)
    # ──────────────────────────────────────────
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "models/best.pt")
    YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF", "0.5"))

    # Comma-separated list, e.g. "http://localhost:3000,http://127.0.0.1:3000"
    CORS_ORIGINS: list[str] = field(
        default_factory=lambda: _parse_cors_origins(os.getenv("CORS_ORIGINS", "*"))
    )

    def __post_init__(self):
        os.makedirs(self.PREDICTIONS_DIR, exist_ok=True)
        os.makedirs(self.GRAPHS_DIR, exist_ok=True)


settings = Settings()
