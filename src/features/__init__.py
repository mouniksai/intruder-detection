"""
Features Package
----------------
Exposes the 5 independent handcrafted classical feature extractors:
- Pipeline 1: BSIF (Binarized Statistical Image Features)
- Pipeline 2: LPQ (Local Phase Quantization)
- Pipeline 3: WLD (Weber Local Descriptor)
- Pipeline 4: Gabor Wavelet Filter Bank
- Pipeline 5: Facial Landmark Geometry & Bilateral Symmetry
"""

from .bsif import BSIFExtractor, extract_features_bsif
from .lpq import LPQExtractor, extract_features_lpq
from .wld import WLDExtractor, extract_features_wld
from .gabor import GaborExtractor, extract_features_gabor
from .geometry import LandmarkGeometryExtractor, extract_features_geometry

__all__ = [
    "BSIFExtractor", "extract_features_bsif",
    "LPQExtractor", "extract_features_lpq",
    "WLDExtractor", "extract_features_wld",
    "GaborExtractor", "extract_features_gabor",
    "LandmarkGeometryExtractor", "extract_features_geometry"
]
