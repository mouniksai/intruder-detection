"""
src/models/vgg16.py
===================
VGGNet / VGG-16 (2014) Architecture for Face Recognition & Intruder Detection.

Key architectural characteristics:
- Homogeneous cascades of small 3x3 filters (stride 1, padding 1)
- Stacking 3x3 convolutions effectively expands receptive field (two 3x3 = 5x5, three 3x3 = 7x7)
- Rich non-linear activation depth (ReLU) capturing fine micro-textures and hierarchical facial contours
- Foundational architectural paradigm of VGGFace (Parkhi et al., 2015)
- High parameter count and memory footprint producing higher inference latency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16Face(nn.Module):
    """
    VGG-16 architecture with Batch Normalization tailored for face identification
    and open-set embedding extraction.
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 512, in_channels: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Configuration for VGG-16: 'M' = MaxPool2d
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(cfg, in_channels=in_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully Connected Layers
        self.classifier_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Linear(embedding_dim, num_classes)
        self._init_weights()

    def _make_layers(self, cfg, in_channels: int = 3) -> nn.Sequential:
        layers = []
        curr_in = in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(curr_in, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ])
                curr_in = v
        return nn.Sequential(*layers)

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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embeddings = self.classifier_block(x)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.classifier_block(x)
        logits = self.head(feat)
        return logits
