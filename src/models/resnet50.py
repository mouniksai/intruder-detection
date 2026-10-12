"""
src/models/resnet50.py
======================
ResNet-50 (2015) Architecture for Face Recognition & Intruder Detection.

Key architectural characteristics:
- Residual bottleneck learning blocks with identity shortcut connections: y = F(x) + x
- Gradient Highway bypassing non-linear degradations and vanishing gradients
- 50 deep layers organized into 4 residual stages [3, 4, 6, 3] with 1x1-3x3-1x1 bottlenecks
- Preservation of low-level landmark coordinates and fine alignment cues
- Universal gold standard backbone for modern face biometrics (ArcFace, CosFace, InsightFace)
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    """
    ResNet Bottleneck Residual Block with expansion factor 4.
    F(x) = Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN
    y = F(x) + shortcut(x)
    """
    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        # 1x1 dimensionality reduction
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 spatial convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 dimensionality expansion
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50Face(nn.Module):
    """
    ResNet-50 architecture adapted for biometric face recognition
    and open-set prototype verification.
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 512, in_channels: int = 3) -> None:
        super().__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 Bottleneck Stages: [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(planes=64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=3, stride=2)

        # Global Average Pooling & Biometric Embedding Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_embedding = nn.Linear(512 * BottleneckBlock.expansion, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * BottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * BottleneckBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BottleneckBlock.expansion)
            )

        layers = []
        layers.append(BottleneckBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * BottleneckBlock.expansion

        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized biometric embedding vector e = f(x) / ||f(x)||_2.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc_embedding(x)
        embeddings = self.bn_embedding(x)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits
