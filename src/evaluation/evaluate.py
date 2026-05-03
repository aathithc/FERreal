import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """Run inference and compute classification metrics.

    Returns a dict with keys:
        accuracy        float
        per_class       dict[class_name -> {precision, recall, f1, support}]
        confusion_matrix  np.ndarray of shape (num_classes, num_classes)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    return {
        "accuracy": float(accuracy),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\nOverall Accuracy: {results['accuracy'] * 100:.2f}%\n")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for cls, metrics in results["per_class"].items():
        print(
            f"{cls:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['support']:>10d}"
        )
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(results["confusion_matrix"])
