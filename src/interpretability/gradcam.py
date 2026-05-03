import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM visualization for CNN models.

    Implementation notes:
        - Register a forward hook on the target convolutional layer to capture activations.
        - Register a backward hook to capture gradients.
        - Compute weights as global average pooled gradients, then form the weighted
          activation map and apply ReLU.
        - Upsample the heatmap to match input spatial resolution.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        raise NotImplementedError("Implement GradCAM")

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None):
        """Return a heatmap of shape (H, W) for the given input."""
        raise NotImplementedError
