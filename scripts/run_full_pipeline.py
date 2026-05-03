#!/usr/bin/env python3
"""
Master script to orchestrate the full FER pipeline:
1. Train all 3 models (SimpleCNN, ResNet-50, ViT)
2. Collect evaluation results
3. Generate visualizations
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully")
    return True


def main():
    """Execute the full pipeline."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     FER-2013 FULL TRAINING PIPELINE                        ║
║                                                                            ║
║   This script will:                                                        ║
║   1. Train SimpleCNN, ResNet-50, and ViT models                            ║
║   2. Evaluate all models on the test set                                   ║
║   3. Generate comprehensive visualizations                                 ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to start training... (This will take a while!)")
    
    steps = [
        (
            ["python", "scripts/train_all_models.py"],
            "Training all 3 models"
        ),
        (
            ["python", "scripts/collect_results.py"],
            "Collecting evaluation results"
        ),
        (
            ["python", "scripts/visualize_results.py"],
            "Generating visualizations"
        ),
    ]
    
    for cmd, description in steps:
        if not run_command(cmd, description):
            print(f"\n⚠️  Pipeline stopped at: {description}")
            sys.exit(1)
    
    print("""
    
╔════════════════════════════════════════════════════════════════════════════╗
║                         🎉 PIPELINE COMPLETE! 🎉                          ║
╚════════════════════════════════════════════════════════════════════════════╝

📍 Results Location:
   - Checkpoints:   checkpoints/
   - Results JSON:  results/evaluation_results.json
   - Summary:       results/00_summary_metrics.txt
   - Visualizations: results/visualizations/
   
📊 Generated Visualizations:
   ✓ Accuracy comparison bar chart
   ✓ Per-class F1 scores comparison
   ✓ Confusion matrices for each model (normalized)
   ✓ Metrics heatmap (Accuracy, Precision, Recall, F1)
   
📚 Next Steps:
   - Review results/00_summary_metrics.txt for a quick overview
   - Open visualizations in results/visualizations/ folder
   - Check checkpoints/ for trained model weights
    """)


if __name__ == "__main__":
    main()
