import torch
import torch.nn as nn


class AttentionMapExtractor:
    """Extract and visualize attention maps from Vision Transformer models.

    Implementation notes:
        - Register hooks on the attention layers to capture attention weights.
        - For classification token rollout, average attention across heads then
          apply attention rollout (Abnar & Zuidema, 2020) to propagate attention
          through all layers.
        - Return a 2-D spatial map that can be overlaid on the input image.

    Works with timm ViT models; may need adaptation for HuggingFace ViT outputs.
    """

    def __init__(self, model: nn.Module):
        raise NotImplementedError("Implement AttentionMapExtractor")

    def generate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return attention map of shape (H, W) for the given input."""
        raise NotImplementedError
