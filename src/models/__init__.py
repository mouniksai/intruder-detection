"""
src/models/__init__.py
======================
Deep learning architecture suite for Review 2:
1. AlexNet (2012) - Shallow, large-kernel baseline
2. VGG-16 (2014) - Deep homogeneous 3x3 filter stacking (VGGFace pedigree)
3. GoogLeNet / Inception-v1 (2014) - Multi-scale parallel Inception modules (FaceNet pedigree)
4. ResNet-50 (2015) - Residual bottleneck connections (ArcFace / InsightFace standard)
5. MobileNetV2 (2018) - Inverted residuals & linear bottlenecks for edge CCTV surveillance
"""

from .common import (
    DeepFaceDataset,
    OpenSetGatingEngine,
    train_model,
    evaluate_model,
    count_parameters,
    compute_model_size_mb,
    measure_inference_latency
)
from .alexnet import AlexNetFace
from .vgg16 import VGG16Face
from .googlenet import GoogLeNetFace
from .resnet50 import ResNet50Face
from .mobilenetv2 import MobileNetV2Face

__all__ = [
    "DeepFaceDataset",
    "OpenSetGatingEngine",
    "train_model",
    "evaluate_model",
    "count_parameters",
    "compute_model_size_mb",
    "measure_inference_latency",
    "AlexNetFace",
    "VGG16Face",
    "GoogLeNetFace",
    "ResNet50Face",
    "MobileNetV2Face",
]
