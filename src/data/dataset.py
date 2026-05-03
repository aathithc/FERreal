import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    """FER-2013 dataset loader.

    Expects data laid out as:
        root/train/<class_name>/*.jpg
        root/test/<class_name>/*.jpg

    Class list is inferred from subdirectory names, sorted alphabetically so
    label indices are deterministic across machines.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Args:
            root:      Path to the fer2013 data directory (contains train/ and test/).
            split:     "train" or "test".
            transform: A torchvision transform applied to each PIL image before
                       returning it. If None, images are returned as raw PIL Images.
        """
        assert split in ("train", "test"), f"split must be 'train' or 'test', got '{split}'"
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Run scripts/download_data.sh first."
            )

        # Infer class list from directory names — sorted for determinism.
        self.classes = sorted(
            d.name for d in split_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = split_dir / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # always load as grayscale

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class_counts(self) -> list[int]:
        """Returns per-class sample counts; index == class index."""
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts
