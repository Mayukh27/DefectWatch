"""
detection_methods/dl_model.py
Deep Learning Defect Detector — YOLO (ultralytics)

⚠️  CURRENTLY INACTIVE — returns default prediction = 0
    Uncomment the DL-related sections when a labelled dataset is available.

To activate:
    1. Collect and label defect images (YOLO format)
    2. Train: `yolo train data=defects.yaml model=yolov8n.pt epochs=100`
    3. Place weights at: models/best.pt  (or set YOLO_MODEL_PATH in config)
    4. Set USE_DL = True in config.py
    5. Uncomment all blocks marked  # [DL-ACTIVE]
"""

import time
import numpy as np
import cv2

from .base import BaseDetector, DetectionResult
from config import settings


# ──────────────────────────────────────────────────────────────────
# [DL-ACTIVE] Uncomment when dataset is available
# ──────────────────────────────────────────────────────────────────
# from ultralytics import YOLO
# ──────────────────────────────────────────────────────────────────


class DLModelDetector(BaseDetector):
    """
    YOLO-based defect detector.

    Currently returns prediction=0 (no defect) for every frame
    so the system can include the DL method in comparisons without
    crashing. All frame/latency logging is active.

    Once training data is available:
      • Uncomment [DL-ACTIVE] blocks
      • Set USE_DL = True
      • Retrain on your defect dataset
    """

    name = "dl"

    def __init__(self, camera_id: int = 0):
        super().__init__(camera_id)

        self._model = None  # Will hold YOLO instance when active
        self._conf  = settings.YOLO_CONF_THRESHOLD

        # ──────────────────────────────────────
        # [DL-ACTIVE] Uncomment when dataset is available
        # ──────────────────────────────────────
        # if settings.USE_DL:
        #     try:
        #         self._model = YOLO(settings.YOLO_MODEL_PATH)
        #         print(f"[DL] Loaded YOLO model from {settings.YOLO_MODEL_PATH}")
        #     except Exception as exc:
        #         print(f"[DL] WARNING — could not load model: {exc}")
        #         self._model = None
        # ──────────────────────────────────────

    # ──────────────────────────────────────────
    def initialize(self, frame: np.ndarray) -> None:
        """
        Nothing to initialize for YOLO — model is stateless per frame.
        Uncomment warm-up section below when dataset is available.
        """
        # ──────────────────────────────────────
        # [DL-ACTIVE] Optional warm-up pass
        # ──────────────────────────────────────
        # if self._model is not None:
        #     _ = self._model(frame, conf=self._conf, verbose=False)
        # ──────────────────────────────────────
        pass

    # ──────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run YOLO inference.

        Currently: inactive — returns default prediction = 0 every frame.
        Uncomment [DL-ACTIVE] block when model weights are available.
        """

        # ──────────────────────────────────────
        # [DL-ACTIVE] Uncomment when dataset is available
        # ──────────────────────────────────────
        # if self._model is not None and settings.USE_DL:
        #     results = self._model(frame, conf=self._conf, verbose=False)
        #     detected = False
        #     mask = np.zeros(frame.shape[:2], dtype="uint8")
        #
        #     for r in results:
        #         boxes = r.boxes
        #         if boxes is not None and len(boxes) > 0:
        #             detected = True
        #             for box in boxes:
        #                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        #                 # Fill detected region in mask
        #                 mask[y1:y2, x1:x2] = 255
        #
        #     return DetectionResult(
        #         prediction=int(detected),
        #         latency_ms=0.0,
        #         mask=mask if detected else None,
        #     )
        # ──────────────────────────────────────

        # ── DEFAULT (inactive) ─────────────────
        # Returns 0 until dataset + weights are provided.
        # Latency is still logged for fair comparison.
        return DetectionResult(prediction=0, latency_ms=0.0, mask=None)
