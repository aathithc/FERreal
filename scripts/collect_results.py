#!/usr/bin/env python3
"""Evaluate all trained models and collect results (accuracy, per-class F1, confusion matrices)."""

import sys
import json
from pathlib import Path

import torch
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import FER2013Dataset
from src.data.transforms import get_eval_transforms
from src.evaluation.evaluate import evaluate_model
from src.models import MODEL_REGISTRY
from torch.utils.data import DataLoader


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def evaluate_all_models():
    """Evaluate all trained models on test set."""
    
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("ERROR: No checkpoints directory found. Run training first with: python scripts/train_all_models.py")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load base config for data setup
    with open("configs/default.yaml") as f:
        base_cfg = yaml.safe_load(f)
    
    results = {}
    
    for model_name in ["simple_cnn", "resnet", "vit"]:
        checkpoint_path = checkpoint_dir / model_name / "best.pt"
        
        if not checkpoint_path.exists():
            print(f"SKIP: {model_name} - checkpoint not found at {checkpoint_path}")
            results[model_name] = {"error": "checkpoint_not_found"}
            continue
        
        print(f"Evaluating {model_name}...")
        
        try:
            # Load checkpoint
            checkpoint = load_checkpoint(checkpoint_path, device)
            cfg = checkpoint["cfg"]
            class_names = checkpoint["class_names"]
            
            # Build test dataloader
            eval_tf = get_eval_transforms(cfg["image_size"], cfg["channels"])
            test_ds = FER2013Dataset(cfg["data_root"], split="test", transform=eval_tf)
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True,
            )
            
            # Load model and weights
            model = MODEL_REGISTRY[model_name](num_classes=len(class_names)).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Evaluate
            eval_results = evaluate_model(model, test_loader, device, class_names)
            
            # Format results for JSON serialization
            results[model_name] = {
                "accuracy": eval_results["accuracy"],
                "per_class": eval_results["per_class"],
                "confusion_matrix": eval_results["confusion_matrix"].tolist(),
                "best_epoch": checkpoint.get("epoch", "unknown"),
                "best_val_acc": checkpoint.get("best_val_acc", "unknown"),
            }
            
            # Print results
            print(f"\n{model_name.upper()} Results:")
            print(f"  Accuracy: {eval_results['accuracy']*100:.2f}%")
            print(f"  Per-class F1 scores:")
            for cls, metrics in eval_results["per_class"].items():
                print(f"    {cls:15} F1={metrics['f1']:.4f} (Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f})")
            print(f"  Confusion Matrix:")
            cm = eval_results["confusion_matrix"]
            for i, row in enumerate(cm):
                print(f"    {class_names[i]:15} {row}")
            print()
            
        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Save results to JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "evaluation_results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("Next step: python scripts/visualize_results.py")
    
    return results


if __name__ == "__main__":
    evaluate_all_models()
