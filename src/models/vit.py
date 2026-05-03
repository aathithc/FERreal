import torch.nn as nn
import timm


class ViTFER(nn.Module):
    """ViT-Base/16 fine-tuned for FER-2013 via timm.

    Input:  (B, 3, 224, 224) — use channels=3, image_size=224 in config
    Output: (B, num_classes) logits

    Uses vit_base_patch16_224 pretrained on ImageNet-21k then fine-tuned on
    ImageNet-1k. The classification head is replaced with a new Linear layer.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
