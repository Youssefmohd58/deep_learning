import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BirdCNN(nn.Module):
    def __init__(self, num_classes=200, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), block(256, 512), block(512, 512)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))