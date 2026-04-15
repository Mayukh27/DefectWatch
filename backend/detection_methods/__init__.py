from .fd import FrameDifferencingDetector
from .mog2 import MOG2Detector
from .running_avg import RunningAverageDetector
from .custom import CustomDefectDetector
from .dl_model import DLModelDetector

__all__ = [
    "FrameDifferencingDetector",
    "MOG2Detector",
    "RunningAverageDetector",
    "CustomDefectDetector",
    "DLModelDetector",
]
