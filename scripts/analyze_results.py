#!/usr/bin/env python3
"""
Post-training analysis and insights script.
Generates detailed analysis and comparisons from trained models.
"""

import json
from pathlib import Path
import numpy as np
from typing import Dict


def analyze_per_class_performance(results: Dict):
    """Identify per-class strengths and weaknesses."""
    
    print("\n" + "="*80)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("="*80)
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        
        per_class = result["per_class"]
        
        # Find best and worst classes
        f1_scores = {cls: metrics["f1"] for cls, metrics in per_class.items()}
        
        best_class = max(f1_scores, key=f1_scores.get)
        worst_class = min(f1_scores, key=f1_scores.get)
        
        print(f"Best performing:  {best_class:12} F1={f1_scores[best_class]:.4f}")
        print(f"Worst performing: {worst_class:12} F1={f1_scores[worst_class]:.4f}")
        print(f"Performance spread: {f1_scores[best_class] - f1_scores[worst_class]:.4f}")


def analyze_confusion_patterns(results: Dict):
    """Find common confusion patterns between emotions."""
    
    print("\n" + "="*80)
    print("CONFUSION PATTERN ANALYSIS")
    print("="*80)
    
    emotion_order = None
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        if emotion_order is None:
            emotion_order = list(result["per_class"].keys())
        
        cm = np.array(result["confusion_matrix"])
        
        # Find major confusions (off-diagonal peaks)
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        
        confusions = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:
                    confusions.append((cm[i][j], i, j))
        
        # Sort by frequency
        confusions.sort(reverse=True)
        
        if confusions:
            print("Top confusion pairs (true → predicted):")
            for count, true_idx, pred_idx in confusions[:5]:
                true_emotion = emotion_order[true_idx]
                pred_emotion = emotion_order[pred_idx]
                print(f"  {true_emotion:12} → {pred_emotion:12} ({int(count):3d} times)")
        else:
            print("No confusion detected")


def compare_model_strengths(results: Dict):
    """Compare which emotions each model handles best."""
    
    print("\n" + "="*80)
    print("MODEL SPECIALIZATION")
    print("="*80)
    
    emotion_order = None
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        if emotion_order is None:
            emotion_order = list(result["per_class"].keys())
        
        per_class = result["per_class"]
        
        # Get F1 scores
        f1_scores = [per_class[emotion]["f1"] for emotion in emotion_order]
        
        best_emotion_idx = np.argmax(f1_scores)
        best_emotion = emotion_order[best_emotion_idx]
        best_f1 = f1_scores[best_emotion_idx]
        
        print(f"{model_name:15} excels at {best_emotion:12} (F1={best_f1:.4f})")


def model_ranking(results: Dict):
    """Rank models by overall performance."""
    
    print("\n" + "="*80)
    print("OVERALL MODEL RANKING")
    print("="*80)
    
    rankings = []
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        accuracy = result["accuracy"]
        per_class = result["per_class"]
        
        # Macro averages
        f1_avg = np.mean([m["f1"] for m in per_class.values()])
        precision_avg = np.mean([m["precision"] for m in per_class.values()])
        recall_avg = np.mean([m["recall"] for m in per_class.values()])
        
        rankings.append({
            "model": model_name,
            "accuracy": accuracy,
            "f1": f1_avg,
            "precision": precision_avg,
            "recall": recall_avg,
        })
    
    # Sort by accuracy
    rankings.sort(key=lambda x: x["accuracy"], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    
    for rank, r in enumerate(rankings, 1):
        print(f"{rank:<5} {r['model']:<15} {r['accuracy']*100:>10.2f}% {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")
    
    print("\nRecommendation:")
    if rankings[0]["model"] == "vit":
        print("  🏆 Vision Transformer achieves best performance but requires most compute")
    elif rankings[0]["model"] == "resnet":
        print("  🏆 ResNet-50 balances performance and efficiency")
    else:
        print("  🏆 SimpleCNN offers fastest training time")


def class_imbalance_impact(results: Dict):
    """Analyze impact of class imbalance on model performance."""
    
    print("\n" + "="*80)
    print("CLASS IMBALANCE IMPACT ANALYSIS")
    print("="*80)
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        per_class = result["per_class"]
        cm = np.array(result["confusion_matrix"])
        
        # Get support (sample count) for each class
        supports = np.array([metrics["support"] for metrics in per_class.values()])
        f1_scores = np.array([metrics["f1"] for metrics in per_class.values()])
        
        # Calculate recall (true positive rate)
        recalls = np.array([metrics["recall"] for metrics in per_class.values()])
        
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        
        # Correlation between support and F1
        if len(supports) > 1:
            correlation = np.corrcoef(supports, f1_scores)[0, 1]
            print(f"Correlation (support ↔ F1 score): {correlation:.3f}")
            if correlation > 0.3:
                print("  ⚠️  Model performs better on high-frequency emotions")
            else:
                print("  ✓ Model generalizes well across emotion frequencies")
        
        # Recall disparity
        recall_range = recalls.max() - recalls.min()
        print(f"Recall range: {recalls.min():.3f} to {recalls.max():.3f} (spread={recall_range:.3f})")
        
        if recall_range > 0.2:
            print("  ⚠️  Large performance variation across emotions")
        else:
            print("  ✓ Consistent performance across emotions")


def main():
    """Run analysis."""
    
    results_file = Path("results/evaluation_results.json")
    
    if not results_file.exists():
        print(f"ERROR: Results file not found at {results_file}")
        print("Run: python scripts/collect_results.py")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    POST-TRAINING ANALYSIS REPORT                          ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    analyze_per_class_performance(results)
    compare_model_strengths(results)
    analyze_confusion_patterns(results)
    class_imbalance_impact(results)
    model_ranking(results)
    
    print("\n" + "="*80)
    print("Generate more detailed visualizations with:")
    print("  python scripts/visualize_results.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
