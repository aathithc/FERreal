import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )


class SimpleCNN(nn.Module):
    """Custom CNN baseline for FER-2013.

    Input:  (B, 1, 48, 48) grayscale images
    Output: (B, num_classes) logits

    Architecture:
        Block 1: 1  -> 32  channels, 48x48 -> 24x24
        Block 2: 32 -> 64  channels, 24x24 -> 12x12
        Block 3: 64 -> 128 channels, 12x12 ->  6x6
        Block 4: 128-> 256 channels,  6x6  ->  3x3
        Classifier: GlobalAvgPool -> Dropout(0.5) -> Linear(256, num_classes)
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
