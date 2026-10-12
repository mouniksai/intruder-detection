"""
src/models/alexnet.py
=====================
AlexNet (2012) Architecture for Face Recognition & Intruder Detection.

Key architectural characteristics:
- Large convolutional receptive fields (11x11, 5x5) in early layers
- Local Response Normalization (LRN)
- Overlapping Max-Pooling (kernel 3, stride 2)
- Heavy dense fully connected layers (~60M parameters)
- Overfitting propensity on small facial datasets & rapid loss of micro-textures
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetFace(nn.Module):
    """
    AlexNet architecture adapted for biometric face recognition.
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 512, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        if pretrained:
            import torchvision.models as tv
            backbone = tv.alexnet(weights=tv.AlexNet_Weights.DEFAULT)
            self.features = backbone.features
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
            )
            self.fc1[1].weight.data.copy_(backbone.classifier[1].weight.data)
            self.fc1[1].bias.data.copy_(backbone.classifier[1].bias.data)
            self.fc2 = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(4096, embedding_dim),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Linear(embedding_dim, num_classes)
            nn.init.normal_(self.fc2[1].weight, 0, 0.01)
            nn.init.constant_(self.fc2[1].bias, 0)
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            nn.init.constant_(self.classifier.bias, 0)
        else:
            # 5 Convolutional Feature Extraction Layers
            self.features = nn.Sequential(
                # Conv 1: Large 11x11 receptive field (stride 2 for 128x128 face crops)
                nn.Conv2d(in_channels, 64, kernel_size=11, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),

                # Conv 2: 5x5 receptive field
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),

                # Conv 3: 3x3
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                # Conv 4: 3x3
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                # Conv 5: 3x3
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

            # Adaptive pooling to ensure fixed spatial dimensions
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            # Fully Connected Layers
            self.fc1 = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
            )

            self.fc2 = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(4096, embedding_dim),
                nn.ReLU(inplace=True),
            )

            # Classification Head
            self.classifier = nn.Linear(embedding_dim, num_classes)
            self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
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
        x = self.fc1(x)
        embeddings = self.fc2(x)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing classification logits.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        feat = self.fc2(x)
        logits = self.classifier(feat)
        return logits
