import torch.nn as nn
import torchvision.models as models


class ResNetFER(nn.Module):
    """ResNet-50 fine-tuned for FER-2013.

    Input:  (B, 3, 224, 224) — use channels=3, image_size=224 in config
    Output: (B, num_classes) logits

    The backbone is pretrained on ImageNet. Only the final fc layer is
    randomly initialised. Set freeze_backbone=True to train only the
    classifier head for the first few epochs (useful when GPU memory is tight).
    """

    def __init__(self, num_classes: int = 7, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def unfreeze(self):
        """Unfreeze all layers (call after warm-up epochs)."""
        for param in self.model.parameters():
            param.requires_grad = True
