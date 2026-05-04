#!/usr/bin/env python3
"""Generate saliency grid and failure analysis figures.

Produces:
    results/figures/saliency_grid.pdf   — 7 emotion classes × 4 columns
                                          (input | CNN GradCAM | ResNet GradCAM | ViT attention)
    results/figures/failures.pdf        — 2×3 grid of examples misclassified by all models

Usage:
    python scripts/generate_saliency_grid.py \
        --cnn-checkpoint    checkpoints/simple_cnn/best.pt \
        --resnet-checkpoint checkpoints/resnet/best.pt \
        --vit-checkpoint    checkpoints/vit/best.pt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image as PILImage

from src.data.dataset import FER2013Dataset
from src.data.transforms import get_eval_transforms
from src.models import MODEL_REGISTRY
from src.training.utils import load_checkpoint
from src.interpretability.gradcam import GradCAM
from src.interpretability.attention_maps import AttentionMapExtractor

plt.rcParams["font.family"] = "sans-serif"
MODEL_DISPLAY = {"simple_cnn": "SimpleCNN\nGrad-CAM", "resnet": "ResNet-50\nGrad-CAM", "vit": "ViT\nAttention"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_name: str, ckpt_path: str):
    p = Path(ckpt_path)
    if not p.exists():
        print(f"  WARNING: checkpoint not found: {p} — {model_name} will be skipped")
        return None, None, None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    model = MODEL_REGISTRY[model_name](num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["cfg"], ckpt["class_names"]


def _get_target_layer(model_name: str, model):
    """Return the Conv layer to hook for GradCAM (None for ViT)."""
    if model_name == "simple_cnn":
        return model.features[3][3]          # last Conv2d in last block → (B, 256, 6, 6)
    if model_name == "resnet":
        return model.model.layer4[-1].conv3  # last conv in last Bottleneck → (B, 2048, 7, 7)
    return None                              # ViT uses AttentionMapExtractor


def _run_all_inference(models: dict, cfgs: dict, data_root: str, device):
    """Return preds dict {model_name: np.array} and labels np.array."""
    all_preds  = {}
    all_labels = None
    for name, model in models.items():
        cfg = cfgs[name]
        tf  = get_eval_transforms(cfg["image_size"], cfg["channels"])
        ds  = FER2013Dataset(data_root, split="test", transform=tf)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        preds, labels = [], []
        model.eval()
        with torch.no_grad():
            for imgs, lbls in loader:
                preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
                labels.extend(lbls.numpy())
        all_preds[name]  = np.array(preds)
        all_labels       = np.array(labels)
        acc = (all_preds[name] == all_labels).mean() * 100
        print(f"  {name}: test acc {acc:.2f}%")
    return all_preds, all_labels


def _denorm_to_gray(tensor: torch.Tensor, channels: int) -> np.ndarray:
    """Return (H,W) float32 in [0,1] from a normalized model-input tensor."""
    img = tensor.cpu().numpy()
    if channels == 1:
        img = img[0] * 0.5 + 0.5
    else:
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        img  = np.clip(img * std + mean, 0, 1)
        img  = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return np.clip(img, 0, 1).astype(np.float32)


def _overlay_jet(gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend jet heatmap over grayscale image. Returns (H,W,3) in [0,1]."""
    h, w = gray.shape
    if heatmap.shape != gray.shape:
        hm = PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
        heatmap = np.array(hm) / 255.0
    rgb     = np.stack([gray] * 3, axis=-1)
    colored = mpl_cm.jet(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * rgb + alpha * colored, 0, 1)


def _saliency(model_name: str, model, img_tensor: torch.Tensor,
              target_layer, device) -> np.ndarray:
    """Return (H,W) saliency map for one image tensor (C,H,W)."""
    t = img_tensor.unsqueeze(0).to(device)
    if model_name == "vit":
        ext = AttentionMapExtractor(model.model)
        sal = ext.generate(t).cpu().numpy()
        ext.remove_hooks()
    else:
        gc  = GradCAM(model, target_layer)
        sal = gc.generate(t).detach().cpu().numpy()
        gc.remove_hooks()
    return sal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",          default="data/fer2013")
    parser.add_argument("--cnn-checkpoint",      default="checkpoints/simple_cnn/best.pt")
    parser.add_argument("--resnet-checkpoint",   default="checkpoints/resnet/best.pt")
    parser.add_argument("--vit-checkpoint",      default="checkpoints/vit/best.pt")
    parser.add_argument("--seed",  type=int,     default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    specs = {"simple_cnn": args.cnn_checkpoint,
             "resnet":      args.resnet_checkpoint,
             "vit":         args.vit_checkpoint}
    models, cfgs, class_names_map = {}, {}, {}
    for name, ckpt in specs.items():
        m, cfg, cnames = _load_model(name, ckpt)
        if m is None:
            continue
        models[name]          = m.to(device)
        cfgs[name]            = cfg
        class_names_map[name] = cnames
    if not models:
        raise RuntimeError("No checkpoints found. Train at least one model first.")

    class_names = list(class_names_map.values())[0]
    n_classes   = len(class_names)

    # Run inference
    print("Running inference on test split...")
    all_preds, all_labels = _run_all_inference(models, cfgs, args.data_root, device)

    # Per-class example selection: prefer all models correct, fall back to majority
    correct_mask = np.ones(len(all_labels), dtype=bool)
    for p in all_preds.values():
        correct_mask &= (p == all_labels)

    class_idx_map = {}
    for cls_i in range(n_classes):
        candidates = np.where(correct_mask & (all_labels == cls_i))[0]
        if len(candidates) == 0:
            candidates = np.where(all_labels == cls_i)[0]
        class_idx_map[cls_i] = int(rng.choice(candidates))

    # Failure examples: all models wrong
    wrong_mask = np.ones(len(all_labels), dtype=bool)
    for p in all_preds.values():
        wrong_mask &= (p != all_labels)
    fail_idx = np.where(wrong_mask)[0]
    if len(fail_idx) < 6:
        wrong_count = sum((p != all_labels).astype(int) for p in all_preds.values())
        fail_idx = np.where(wrong_count >= max(1, len(models) - 1))[0]
    fail_idx = rng.choice(fail_idx, size=min(6, len(fail_idx)), replace=False)

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_order = [n for n in ["simple_cnn", "resnet", "vit"] if n in models]
    n_cols = 1 + len(model_order)

    # Pre-load per-model test datasets (avoid reloading inside loops)
    raw_ds     = FER2013Dataset(args.data_root, split="test", transform=None)
    model_ds   = {n: FER2013Dataset(args.data_root, split="test",
                                     transform=get_eval_transforms(cfgs[n]["image_size"],
                                                                   cfgs[n]["channels"]))
                  for n in model_order}
    target_layers = {n: _get_target_layer(n, models[n]) for n in model_order}

    # -----------------------------------------------------------------------
    # Saliency grid: n_classes rows × n_cols cols
    # -----------------------------------------------------------------------
    print("Generating saliency grid...")
    fig, axes = plt.subplots(n_classes, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * n_classes),
                             squeeze=False)

    for cls_i in range(n_classes):
        idx      = class_idx_map[cls_i]
        raw_img, _ = raw_ds[idx]
        raw_np   = np.array(raw_img, dtype=np.float32) / 255.0

        # Col 0: raw input
        axes[cls_i][0].imshow(raw_np, cmap="gray", vmin=0, vmax=1)
        axes[cls_i][0].set_ylabel(class_names[cls_i], fontsize=9, fontweight="bold", rotation=0,
                                   labelpad=50, va="center")

        for col, name in enumerate(model_order, start=1):
            img_tensor, _ = model_ds[name][idx]
            sal = _saliency(name, models[name], img_tensor, target_layers[name], device)
            gry = _denorm_to_gray(img_tensor, cfgs[name]["channels"])
            axes[cls_i][col].imshow(_overlay_jet(gry, sal))

    col_titles = ["Input"] + [MODEL_DISPLAY[n] for n in model_order]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=9)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Saliency Grid — correctly classified examples", fontsize=12, y=1.005)
    fig.tight_layout()
    sal_path = out_dir / "saliency_grid.pdf"
    fig.savefig(sal_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {sal_path}")

    # -----------------------------------------------------------------------
    # Failures grid: 2×3
    # -----------------------------------------------------------------------
    print("Generating failures grid...")
    if len(fail_idx) == 0:
        print("  No failure examples found — skipping failures.pdf")
        return

    n_fail   = len(fail_idx)
    nr, nc   = 2, 3
    fig_f, axes_f = plt.subplots(nr, nc, figsize=(3.5 * nc, 3.5 * nr), squeeze=False)

    for i in range(nr * nc):
        r, c = divmod(i, nc)
        ax   = axes_f[r][c]
        if i >= n_fail:
            ax.axis("off")
            continue
        fidx        = int(fail_idx[i])
        raw_img, tl = raw_ds[fidx]
        raw_np      = np.array(raw_img, dtype=np.float32) / 255.0
        ax.imshow(raw_np, cmap="gray", vmin=0, vmax=1)
        pred_text = "\n".join(
            f"{n.replace('simple_cnn','CNN').replace('resnet','ResNet').replace('vit','ViT')}: "
            f"{class_names[all_preds[n][fidx]]}"
            for n in model_order
        )
        ax.set_title(f"True: {class_names[tl]}\n{pred_text}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig_f.suptitle("Failure cases — misclassified by all available models", fontsize=11)
    fig_f.tight_layout()
    fail_path = out_dir / "failures.pdf"
    fig_f.savefig(fail_path, bbox_inches="tight")
    plt.close(fig_f)
    print(f"  Saved {fail_path}")


if __name__ == "__main__":
    main()
