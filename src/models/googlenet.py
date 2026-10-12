"""
src/models/googlenet.py
=======================
GoogLeNet / Inception-v1 (2014) Architecture for Face Recognition & Intruder Detection.

Key architectural characteristics:
- Multi-scale parallel Inception modules (1x1, 3x3, 5x5 convs and 3x3 max-pooling)
- 1x1 dimensionality reduction convolutions before expensive spatial filters
- Simultaneous capture of facial micro-textures (1x1, 3x3) and macro-geometry (5x5)
- Global Average Pooling (GAP) replacing parameter-heavy fully connected layers
- Biometric pedigree: Architectural family underlying Google's FaceNet (Schroff et al., 2015)
- Highly parameter-efficient (~6.8M parameters) with strong generalization
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    Standard Inception module with 4 parallel branches and 1x1 bottleneck reductions.
    """

    def __init__(
        self,
        in_channels: int,
        n1x1: int,
        n3x3_reduce: int,
        n3x3: int,
        n5x5_reduce: int,
        n5x5: int,
        pool_proj: int
    ) -> None:
        super().__init__()

        # Branch 1: 1x1 Conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True),
        )

        # Branch 2: 1x1 Conv -> 3x3 Conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 1x1 Conv -> 5x5 Conv (parameterized as two 3x3 for spatial stability)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
        )

        # Branch 4: 3x3 MaxPool -> 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class GoogLeNetFace(nn.Module):
    """
    GoogLeNet / Inception-v1 architecture adapted for face representation
    and open-set intruder gating.
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 512, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained

        if pretrained:
            import torchvision.models as tv
            self.backbone = tv.googlenet(weights=tv.GoogLeNet_Weights.DEFAULT)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=0.4)
            self.embedding_proj = nn.Linear(1024, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_classes)
            nn.init.normal_(self.embedding_proj.weight, 0, 0.01)
            nn.init.constant_(self.embedding_proj.bias, 0)
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            nn.init.constant_(self.classifier.bias, 0)
        else:
            self.backbone = None
            # Initial stem layers
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.Conv2d(64, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

            # Inception Stages 3, 4, 5
            # Stage 3
            self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)   # 256
            self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64) # 480
            self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Stage 4
            self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)  # 512
            self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64) # 512
            self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64) # 512
            self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64) # 528
            self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128) # 832
            self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Stage 5
            self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128) # 832
            self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128) # 1024

            # Global Average Pooling & Biometric Embedding Projection
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=0.4)
            self.embedding_proj = nn.Linear(1024, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_classes)
            self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized biometric embedding vector e = f(x) / ||f(x)||_2.
        """
        if self.pretrained and self.backbone is not None:
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool1(x)
            x = self.backbone.conv2(x)
            x = self.backbone.conv3(x)
            x = self.backbone.maxpool2(x)

            x = self.backbone.inception3a(x)
            x = self.backbone.inception3b(x)
            x = self.backbone.maxpool3(x)

            x = self.backbone.inception4a(x)
            x = self.backbone.inception4b(x)
            x = self.backbone.inception4c(x)
            x = self.backbone.inception4d(x)
            x = self.backbone.inception4e(x)
            x = self.backbone.maxpool4(x)

            x = self.backbone.inception5a(x)
            x = self.backbone.inception5b(x)
            x = self.gap(x)
        else:
            x = self.stem(x)
            x = self.inception3a(x)
            x = self.inception3b(x)
            x = self.maxpool3(x)

            x = self.inception4a(x)
            x = self.inception4b(x)
            x = self.inception4c(x)
            x = self.inception4d(x)
            x = self.inception4e(x)
            x = self.maxpool4(x)

            x = self.inception5a(x)
            x = self.inception5b(x)
            x = self.gap(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        embeddings = self.embedding_proj(x)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits
