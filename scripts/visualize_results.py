#!/usr/bin/env python3
"""Generate visualizations (graphs, confusion matrices, F1 comparison) from evaluation results."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def load_results(results_file: Path = Path("results/evaluation_results.json")) -> dict:
    """Load evaluation results from JSON file."""
    if not results_file.exists():
        print(f"ERROR: Results file not found at {results_file}")
        print("Run: python scripts/collect_results.py")
        return None
    
    with open(results_file) as f:
        return json.load(f)


def plot_accuracy_comparison(results: dict, output_dir: Path):
    """Create bar chart comparing accuracy across models."""
    models = []
    accuracies = []
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        models.append(model_name.replace("_", " ").title())
        accuracies.append(result["accuracy"] * 100)
    
    if not models:
        print("No valid results to plot accuracy")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison on FER-2013 Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "01_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_per_class_f1(results: dict, output_dir: Path):
    """Create grouped bar chart of per-class F1 scores."""
    emotion_names = None
    
    # Collect F1 scores for each model and emotion
    model_f1_scores = {}
    for model_name, result in results.items():
        if "error" in result:
            continue
        if emotion_names is None:
            emotion_names = list(result["per_class"].keys())
        
        f1_scores = [result["per_class"][emotion]["f1"] for emotion in emotion_names]
        model_f1_scores[model_name] = f1_scores
    
    if not model_f1_scores:
        print("No valid results to plot F1 scores")
        return
    
    x = np.arange(len(emotion_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    for i, (model_name, f1_scores) in enumerate(model_f1_scores.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, f1_scores, width, label=model_name.replace("_", " ").title(), color=colors[i])
    
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = output_dir / "02_per_class_f1_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(results: dict, output_dir: Path):
    """Create confusion matrix heatmaps for each model."""
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        cm = np.array(result["confusion_matrix"])
        emotion_names = list(result["per_class"].keys())
        
        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=emotion_names, yticklabels=emotion_names,
                    cbar_kws={'label': 'Normalized Count'}, ax=ax)
        
        ax.set_title(f'Confusion Matrix - {model_name.replace("_", " ").title()}\n(Normalized)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        output_path = output_dir / f"03_confusion_matrix_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_metrics_heatmap(results: dict, output_dir: Path):
    """Create heatmap showing accuracy, precision, recall, F1 for each model."""
    
    models = []
    model_names_display = []
    metrics_data = []
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        models.append(model_name)
        model_names_display.append(model_name.replace("_", " ").title())
        
        # Calculate macro averages
        per_class = result["per_class"]
        precision_avg = np.mean([m["precision"] for m in per_class.values()])
        recall_avg = np.mean([m["recall"] for m in per_class.values()])
        f1_avg = np.mean([m["f1"] for m in per_class.values()])
        accuracy = result["accuracy"]
        
        metrics_data.append([accuracy, precision_avg, recall_avg, f1_avg])
    
    if not metrics_data:
        print("No valid results to plot metrics heatmap")
        return
    
    metrics_array = np.array(metrics_data)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(metrics_array, annot=True, fmt='.4f', cmap='RdYlGn', 
                xticklabels=metric_names, yticklabels=model_names_display,
                vmin=0, vmax=1, cbar_kws={'label': 'Score'}, ax=ax)
    
    ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "04_metrics_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(results: dict, output_dir: Path):
    """Create and save a summary table of all metrics."""
    
    summary = []
    
    for model_name, result in results.items():
        if "error" in result:
            summary.append({
                "Model": model_name,
                "Accuracy": "ERROR",
                "Macro Precision": "ERROR",
                "Macro Recall": "ERROR",
                "Macro F1": "ERROR",
            })
            continue
        
        per_class = result["per_class"]
        precision_avg = np.mean([m["precision"] for m in per_class.values()])
        recall_avg = np.mean([m["recall"] for m in per_class.values()])
        f1_avg = np.mean([m["f1"] for m in per_class.values()])
        
        summary.append({
            "Model": model_name.replace("_", " ").title(),
            "Accuracy": f"{result['accuracy']*100:.2f}%",
            "Macro Precision": f"{precision_avg:.4f}",
            "Macro Recall": f"{recall_avg:.4f}",
            "Macro F1": f"{f1_avg:.4f}",
        })
    
    # Save as text table
    output_file = output_dir / "00_summary_metrics.txt"
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Header
        f.write(f"{'Model':<20} {'Accuracy':<15} {'Macro Precision':<18} {'Macro Recall':<15} {'Macro F1':<12}\n")
        f.write("-"*70 + "\n")
        
        for row in summary:
            f.write(f"{row['Model']:<20} {row['Accuracy']:<15} {row['Macro Precision']:<18} {row['Macro Recall']:<15} {row['Macro F1']:<12}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"Saved: {output_file}")
    
    # Also print to console
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70 + "\n")
    print(f"{'Model':<20} {'Accuracy':<15} {'Macro Precision':<18} {'Macro Recall':<15} {'Macro F1':<12}")
    print("-"*70)
    for row in summary:
        print(f"{row['Model']:<20} {row['Accuracy']:<15} {row['Macro Precision']:<18} {row['Macro Recall']:<15} {row['Macro F1']:<12}")
    print()


def main():
    """Generate all visualizations."""
    
    # Load results
    results = load_results()
    if results is None:
        return
    
    # Create output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}...\n")
    
    # Generate all plots
    create_summary_table(results, output_dir.parent)
    plot_accuracy_comparison(results, output_dir)
    plot_per_class_f1(results, output_dir)
    plot_confusion_matrices(results, output_dir)
    plot_metrics_heatmap(results, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")
    print("View the summary in: results/00_summary_metrics.txt")


if __name__ == "__main__":
    main()
