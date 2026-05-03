#!/usr/bin/env python3
"""Train all 3 models (SimpleCNN, ResNet-50, ViT) sequentially."""

import os
import sys
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import train


def main():
    models = ["simple_cnn", "resnet", "vit"]
    config_path = "configs/default.yaml"
    
    # Load base config
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)
    
    results_summary = {}
    
    for model_name in models:
        print("\n" + "="*80)
        print(f"Training {model_name.upper()}")
        print("="*80)
        
        # Adjust config for model
        cfg = base_cfg.copy()
        
        # Adjust image size and channels for different models
        if model_name == "simple_cnn":
            cfg["image_size"] = 48
            cfg["channels"] = 1
        elif model_name == "resnet":
            cfg["image_size"] = 224
            cfg["channels"] = 3
        elif model_name == "vit":
            cfg["image_size"] = 224
            cfg["channels"] = 3
        
        try:
            train(model_name, cfg)
            results_summary[model_name] = "✓ Completed"
        except Exception as e:
            results_summary[model_name] = f"✗ Failed: {str(e)}"
            print(f"ERROR training {model_name}: {e}")
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for model_name, status in results_summary.items():
        print(f"{model_name:15} {status}")
    
    print("\nNext step: python scripts/collect_results.py")


if __name__ == "__main__":
    main()
