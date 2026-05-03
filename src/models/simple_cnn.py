import torch.nn as nn


class SimpleCNN(nn.Module):
    """Custom CNN baseline for FER-2013.

    Implement a convolutional architecture from scratch here.
    Common starting point: 3-4 conv blocks (Conv -> BN -> ReLU -> MaxPool)
    followed by a classifier head.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        raise NotImplementedError("Implement SimpleCNN architecture")

    def forward(self, x):
        raise NotImplementedError
