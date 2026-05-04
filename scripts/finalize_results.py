#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the held-out PrivateTest split and write metrics.json.

Usage:
    python scripts/finalize_results.py --model simple_cnn --checkpoint checkpoints/simple_cnn/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import FER2013Dataset
from src.data.transforms import get_eval_transforms
from src.evaluation.evaluate import evaluate_model, print_results
from src.models import MODEL_REGISTRY
from src.training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Finalize results on the test (PrivateTest) split")
    parser.add_argument("--model",      required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--image-size", type=int, default=None, help="Override image_size from checkpoint cfg")
    parser.add_argument("--channels",   type=int, default=None, help="Override channels from checkpoint cfg")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg         = ckpt.get("cfg", {})
    class_names = ckpt.get("class_names", [])

    # CLI overrides take precedence over checkpoint cfg
    image_size = args.image_size or cfg.get("image_size", 48)
    channels   = args.channels   or cfg.get("channels",   1)
    data_root  = cfg.get("data_root", "data/fer2013")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | image_size: {image_size} | channels: {channels}")

    model = MODEL_REGISTRY[args.model](num_classes=len(class_names))
    load_checkpoint(ckpt_path, model)
    model = model.to(device)

    tf = get_eval_transforms(image_size, channels)
    test_ds = FER2013Dataset(data_root, split="test", transform=tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    print(f"Evaluating on PrivateTest split ({len(test_ds)} samples) ...")
    results = evaluate_model(model, test_loader, device, class_names)
    print_results(results)

    num_params  = sum(p.numel() for p in model.parameters())
    macro_f1    = sum(v["f1"] for v in results["per_class"].values()) / len(class_names)

    out = {
        "accuracy":           results["accuracy"],
        "macro_f1":           macro_f1,
        "per_class_f1":       {k: v["f1"]        for k, v in results["per_class"].items()},
        "per_class_precision":{k: v["precision"]  for k, v in results["per_class"].items()},
        "per_class_recall":   {k: v["recall"]     for k, v in results["per_class"].items()},
        "confusion_matrix":   results["confusion_matrix"].tolist(),
        "num_params":         num_params,
        "image_size":         image_size,
        "channels":           channels,
        "n_test_samples":     len(test_ds),
        "class_names":        class_names,
    }

    out_dir = Path("results") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nParams:    {num_params:,}")
    print(f"Macro F1:  {macro_f1*100:.2f}%")
    print(f"Saved →    {out_path}")


if __name__ == "__main__":
    main()
