"""Member 2: CNN encoder + classifier trained completely from scratch (no pretrained weights).

Five convolutional blocks; each has two 3x3 conv + BatchNorm + ReLU layers and a 2x2
max-pool, so a 224px input becomes a 7x7 map with 512 channels. Global average pooling
turns that into a 512-d feature vector (the *encoder output*, reused in Milestone 2),
and a dropout + linear head maps it to the 200 species.
"""
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BirdCNN(nn.Module):
    feature_dim = 512

    def __init__(self, num_classes=200, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), block(256, 512), block(512, 512)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def features(self, x):
        """512-d encoder embedding (before dropout and the classification head)."""
        return self.classifier[1](self.classifier[0](self.encoder(x)))

    def forward(self, x):
        return self.classifier(self.encoder(x))
