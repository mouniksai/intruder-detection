"""
src/models/mobilenetv2.py
=========================
MobileNetV2 (2018) Architecture for Face Recognition & Edge CCTV Intruder Detection.

Key architectural characteristics:
- Depthwise Separable Convolutions reducing FLOPs by 85-90%
- Inverted Residual Blocks (narrow -> wide -> narrow channel expansion)
- Linear Bottlenecks (no non-linearity after projection) preventing biometric manifold collapse
- Residual shortcut connections across bottleneck stages
- Designed specifically for real-time edge security camera deployment (CCTV, Raspberry Pi, Jetson)
- Ultra-compact footprint (~3.4M parameters, ~14 MB FP32)
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block with Linear Bottleneck.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # 1x1 Pointwise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # 3x3 Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 1x1 Pointwise linear bottleneck projection (NO non-linearity)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Face(nn.Module):
    """
    MobileNetV2 tailored for biometric face recognition and edge intruder gating.
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 512, in_channels: int = 3, width_mult: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Inverted residual setting: [t, c, n, s]
        # t: expansion factor, c: output channels, n: repeat count, s: stride
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Initial stem
        first_channel = int(32 * width_mult)
        features: List[nn.Module] = [
            nn.Conv2d(in_channels, first_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_channel),
            nn.ReLU6(inplace=True),
        ]

        # Build Inverted Residual blocks
        in_c = first_channel
        for t, c, n, s in inverted_residual_setting:
            out_c = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_c, out_c, stride=stride, expand_ratio=t))
                in_c = out_c

        # Last 1x1 stage
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        features.extend([
            nn.Conv2d(in_c, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        ])

        self.features = nn.Sequential(*features)

        # Global Average Pooling & Biometric Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc_embedding = nn.Linear(last_channel, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized biometric embedding vector e = f(x) / ||f(x)||_2.
        """
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        embeddings = self.fc_embedding(x)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits
