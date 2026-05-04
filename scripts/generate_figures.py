#!/usr/bin/env python3
"""Generate all report figures from training logs and metrics files.

Reads:
    results/{model}/training_log.csv
    results/{model}/metrics.json

Writes:
    results/figures/training_curves.pdf
    results/figures/confusion_matrices.pdf
    results/figures/per_class_f1.pdf
    results/figures/class_distribution.pdf

Usage:
    python scripts/generate_figures.py [--data-root data/fer2013]
"""

import argparse
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import FER2013Dataset

plt.rcParams["font.family"] = "sans-serif"

MODELS      = ["simple_cnn", "resnet", "vit"]
MODEL_LABELS = {"simple_cnn": "SimpleCNN", "resnet": "ResNet-50", "vit": "ViT-B/16"}
COLORS       = {"simple_cnn": "#1f77b4", "resnet": "#ff7f0e", "vit": "#2ca02c"}


def _load_log(model: str) -> dict | None:
    path = Path("results") / model / "training_log.csv"
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _load_metrics(model: str) -> dict | None:
    path = Path("results") / model / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_curves(out_dir: Path):
    logs = {m: _load_log(m) for m in MODELS}
    available = [m for m, d in logs.items() if d]
    if not available:
        print("  No training logs found — skipping training_curves.pdf")
        return

    fig, axes = plt.subplots(len(available), 2, figsize=(10, 3.5 * len(available)), squeeze=False)

    for row, model in enumerate(available):
        data    = logs[model]
        epochs  = [r["epoch"]     for r in data]
        t_loss  = [r["train_loss"] for r in data]
        v_loss  = [r["val_loss"]   for r in data]
        t_acc   = [r["train_acc"]  for r in data]
        v_acc   = [r["val_acc"]    for r in data]
        color   = COLORS[model]
        label   = MODEL_LABELS[model]

        ax_loss, ax_acc = axes[row]

        ax_loss.plot(epochs, t_loss, color=color, label="Train")
        ax_loss.plot(epochs, v_loss, color=color, linestyle="--", label="Val")
        ax_loss.set_title(f"{label} — Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, [a * 100 for a in t_acc], color=color, label="Train")
        ax_acc.plot(epochs, [a * 100 for a in v_acc], color=color, linestyle="--", label="Val")
        ax_acc.set_title(f"{label} — Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend()

    fig.tight_layout()
    path = out_dir / "training_curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confusion_matrices(out_dir: Path):
    metrics = {m: _load_metrics(m) for m in MODELS}
    available = [m for m, d in metrics.items() if d]
    if not available:
        print("  No metrics files found — skipping confusion_matrices.pdf")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), squeeze=False)

    for col, model in enumerate(available):
        m   = metrics[model]
        cm  = np.array(m["confusion_matrix"], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0.0)
        classes = m["class_names"]
        acc = m["accuracy"] * 100

        ax = axes[0][col]
        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        ax.set_title(f"{MODEL_LABELS[model]}\n(acc={acc:.1f}%)", fontsize=11)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        thresh = 50.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm_pct[i, j]:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_pct[i, j] > thresh else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

    fig.tight_layout()
    path = out_dir / "confusion_matrices.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_class_f1(out_dir: Path):
    metrics = {m: _load_metrics(m) for m in MODELS}
    available = [m for m, d in metrics.items() if d]
    if not available:
        print("  No metrics files found — skipping per_class_f1.pdf")
        return

    # Use class names from first available model
    class_names = list(metrics[available[0]]["per_class_f1"].keys())
    n_classes   = len(class_names)
    n_models    = len(available)
    x = np.arange(n_classes)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.2), 5))

    for i, model in enumerate(available):
        f1_vals = [metrics[model]["per_class_f1"].get(c, 0.0) for c in class_names]
        ax.bar(x + offsets[i], f1_vals, width, label=MODEL_LABELS[model], color=COLORS[model], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score by Model")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    path = out_dir / "per_class_f1.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_class_distribution(out_dir: Path, data_root: str):
    try:
        ds = FER2013Dataset(data_root, split="train")
    except Exception as e:
        print(f"  Could not load dataset for class distribution: {e}")
        return

    counts = ds.get_class_counts()
    total  = sum(counts)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(ds.classes, counts, color="#1f77b4", alpha=0.85)
    ax.bar_label(bars, labels=[f"{c}\n({c/total*100:.1f}%)" for c in counts], fontsize=8)
    ax.set_title("FER-2013 Training Set Class Distribution")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()

    path = out_dir / "class_distribution.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/fer2013")
    args = parser.parse_args()

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    plot_training_curves(out_dir)
    plot_confusion_matrices(out_dir)
    plot_per_class_f1(out_dir)
    plot_class_distribution(out_dir, args.data_root)
    print("Done.")


if __name__ == "__main__":
    main()
