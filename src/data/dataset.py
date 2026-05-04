from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# Canonical emotion order used in the FER-2013 CSV (emotion column 0-6)
CANONICAL_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

_CSV_SPLIT_MAP = {
    "train": "Training",
    "val":   "PublicTest",   # used for checkpoint selection / early stopping
    "test":  "PrivateTest",  # held out — only touch for final numbers
}


class FER2013Dataset(Dataset):
    """FER-2013 dataset loader.

    Auto-detects two layouts:

    CSV mode (preferred):
        root/fer2013.csv   — has columns: emotion, pixels, Usage
        Splits: train=Training (28,709), val=PublicTest (3,589), test=PrivateTest (3,589)
        Download: kaggle datasets download -d deadskull7/fer2013

    Folder mode (fallback):
        root/train/<class>/*.jpg  and  root/test/<class>/*.jpg
        Splits: train / test only (no proper held-out set)
        Download: kaggle datasets download -d msambare/fer2013

    CSV mode is strongly preferred — the folder version merges PublicTest and
    PrivateTest into a single test/ directory, making it impossible to report
    results on a truly held-out set.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        csv_path = self._find_csv()
        if csv_path is not None:
            self._load_csv(csv_path, split)
        else:
            self._load_folders(split)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _find_csv(self):
        for candidate in [self.root / "fer2013.csv", self.root / "fer2013" / "fer2013.csv"]:
            if candidate.exists():
                return candidate
        return None

    def _load_csv(self, csv_path: Path, split: str):
        if split not in _CSV_SPLIT_MAP:
            raise ValueError(f"split must be one of {list(_CSV_SPLIT_MAP)}, got '{split}'")

        usage_label = _CSV_SPLIT_MAP[split]
        self.classes = CANONICAL_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []   # list of (pixels_array, label)
        self._csv_mode = True

        with open(csv_path, newline="") as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                if row["Usage"] != usage_label:
                    continue
                label = int(row["emotion"])
                pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
                self.samples.append((pixels, label))

    def _load_folders(self, split: str):
        if split not in ("train", "test"):
            raise ValueError(
                f"Folder-mode only supports 'train' and 'test' splits, got '{split}'.\n"
                "Switch to the CSV dataset (deadskull7/fer2013) for a proper val/test split."
            )

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Run scripts/download_data.sh first."
            )

        self._csv_mode = False
        self.classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = split_dir / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]

        if self._csv_mode:
            img = Image.fromarray(item)          # already 48x48 uint8 grayscale
        else:
            img = Image.open(item).convert("L")  # path -> PIL grayscale

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class_counts(self):
        """Returns per-class sample counts; index == class index."""
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts
