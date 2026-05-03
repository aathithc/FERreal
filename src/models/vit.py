import torch.nn as nn


class ViTFER(nn.Module):
    """Vision Transformer wrapper for FER-2013.

    Suggested implementation options:
        A) timm: timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        B) HuggingFace transformers: ViTForImageClassification.from_pretrained(
               'google/vit-base-patch16-224-in21k', num_labels=num_classes)

    Input expects 3-channel images at 224x224 (use channels=3, image_size=224 in config).
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        raise NotImplementedError("Implement ViTFER wrapper")

    def forward(self, x):
        raise NotImplementedError
