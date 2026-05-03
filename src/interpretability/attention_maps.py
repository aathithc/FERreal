import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMapExtractor:
    """Extract attention rollout maps from timm Vision Transformer models.

    Implements Abnar & Zuidema (2020) attention rollout: fuse heads by averaging,
    add the identity matrix to model the residual stream, normalise by row, then
    multiply matrices across layers so information flow from every token to CLS
    accumulates.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attn_weights: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        # timm ViT stores blocks in model.blocks; each block has an attn sub-module
        # whose softmax output passes through attn_drop before being used.
        # We hook attn_drop to capture the post-softmax weights at shape
        # (B, heads, N, N) where N = num_patches + 1 (CLS token first).
        for block in model.blocks:
            h = block.attn.attn_drop.register_forward_hook(self._capture)
            self._hooks.append(h)

    def _capture(self, _module, _input, output):
        # output: (B, heads, N, N)
        self._attn_weights.append(output.detach())

    def generate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return attention rollout map of shape (H, W) for the given input."""
        self._attn_weights.clear()

        self.model.eval()
        with torch.no_grad():
            self.model(input_tensor)

        # Build rollout matrix across layers
        # Each layer: average over heads, add identity (residual), normalise rows
        device = input_tensor.device
        n_tokens = self._attn_weights[0].shape[-1]
        rollout = torch.eye(n_tokens, device=device)  # (N, N)

        for attn in self._attn_weights:
            # attn: (B, heads, N, N) — take first item in batch
            a = attn[0].mean(dim=0)          # (N, N): average over heads
            a = a + torch.eye(n_tokens, device=device)  # add residual
            a = a / a.sum(dim=-1, keepdim=True)         # row-normalise
            rollout = a @ rollout

        # CLS row (index 0) gives each patch token's contribution to CLS
        cls_row = rollout[0, 1:]  # (num_patches,)

        # Reshape to spatial grid
        num_patches = cls_row.shape[0]
        grid_size = int(math.isqrt(num_patches))
        spatial = cls_row.reshape(grid_size, grid_size)  # (G, G)

        # Upsample to input spatial resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        spatial = spatial.unsqueeze(0).unsqueeze(0)  # (1, 1, G, G)
        spatial = F.interpolate(spatial, size=(h, w), mode="bilinear", align_corners=False)
        spatial = spatial.squeeze()  # (H, W)

        # normalise to [0, 1]
        s_min, s_max = spatial.min(), spatial.max()
        if s_max > s_min:
            spatial = (spatial - s_min) / (s_max - s_min)

        return spatial

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
