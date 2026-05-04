"""Generic training loop.

Usage (CLI):
    python -m src.training.train --config configs/default.yaml --model resnet

The --model flag must match a key in MODEL_REGISTRY (defined in src/models/__init__.py).
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import FER2013Dataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.training.utils import set_seed, save_checkpoint, compute_class_weights
from src.models import MODEL_REGISTRY


def _build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    image_size = cfg["image_size"]
    channels = cfg["channels"]
    data_root = cfg["data_root"]

    train_tf = get_train_transforms(image_size, channels)
    eval_tf = get_eval_transforms(image_size, channels)

    train_ds = FER2013Dataset(data_root, split="train", transform=train_tf)
    # CSV mode: use PublicTest as val, PrivateTest stays held out for final eval.
    # Folder mode: falls back to "test" (no proper held-out set available).
    try:
        val_ds = FER2013Dataset(data_root, split="val", transform=eval_tf)
    except ValueError:
        val_ds = FER2013Dataset(data_root, split="test", transform=eval_tf)

    if cfg.get("use_weighted_sampler", False):
        counts = train_ds.get_class_counts()
        sample_weights = [1.0 / counts[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.classes


def _build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_type = cfg.get("scheduler", "cosine")
    epochs = cfg["epochs"]
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("lr_step_size", 10), gamma=cfg.get("lr_gamma", 0.1)
        )
    if sched_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"], epochs=epochs, steps_per_epoch=steps_per_epoch
        )
    raise ValueError(f"Unknown scheduler: {sched_type}")


def train(model_name: str, cfg: dict) -> None:
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_names = _build_dataloaders(cfg)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = MODEL_REGISTRY[model_name](num_classes=len(class_names)).to(device)

    if cfg.get("use_class_weights", True):
        weights = compute_class_weights(train_loader.dataset, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = _build_scheduler(optimizer, cfg, len(train_loader))

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / model_name
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = cfg.get("early_stopping_patience", 10)

    for epoch in range(1, cfg["epochs"] + 1):
        # --- train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # OneCycleLR steps per batch, others per epoch
            if cfg.get("scheduler") == "onecycle":
                scheduler.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        if cfg.get("scheduler", "cosine") != "onecycle":
            scheduler.step()

        # --- validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{cfg['epochs']} | "
            f"Train loss {train_loss/train_total:.4f} acc {train_acc*100:.2f}% | "
            f"Val loss {val_loss/val_total:.4f} acc {val_acc*100:.2f}% | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,
                    "cfg": cfg,
                },
                checkpoint_dir / "best.pt",
            )
            print(f"  -> Saved best checkpoint (val_acc={best_val_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs).")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train a FER-2013 model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()), help="Model to train")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(args.model, cfg)


if __name__ == "__main__":
    main()
