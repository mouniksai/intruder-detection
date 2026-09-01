"""
Preprocessing Package
---------------------
Shared modules for face detection, facial landmark extraction, eye-level alignment,
and photometric normalization (Grayscale + CLAHE).
"""

from .detect_align import FacePreprocessor, detect_and_align_face
from .dataset_loader import DatasetManager

__all__ = ["FacePreprocessor", "detect_and_align_face", "DatasetManager"]
