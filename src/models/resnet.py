import torch.nn as nn


class ResNetFER(nn.Module):
    """ResNet-50 wrapper fine-tuned for FER-2013.

    Suggested implementation:
        - Load torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        - Replace the final fc layer: nn.Linear(2048, num_classes)
        - Optionally freeze early layers for the first N epochs
        - Input expects 3-channel images at 224x224 (use channels=3 in config)
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        raise NotImplementedError("Implement ResNetFER wrapper")

    def forward(self, x):
        raise NotImplementedError
