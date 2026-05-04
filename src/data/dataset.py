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
        import os
        import csv
        import warnings
        from pathlib import Path
        from typing import List, Tuple, Union

        from PIL import Image
        import numpy as np
        import torch
        from torch.utils.data import Dataset


        class FER2013Dataset(Dataset):
            """FER-2013 dataset loader.

            Two supported modes:
              - CSV mode: if a `fer2013.csv` file is present under `root` (recursively),
                the dataset will be loaded from the CSV. The CSV contains columns
                `emotion`, `pixels`, `Usage` (values: Training, PublicTest, PrivateTest).
              - Folder mode (backwards compatible): expects
                `root/train/<class>/*.jpg` and `root/test/<class>/*.jpg`.

            In CSV mode the canonical class order is used:
                ['angry','disgust','fear','happy','sad','surprise','neutral']
            """

            CANONICAL_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

            def __init__(self, root: str, split: str = "train", transform=None):
                """
                Args:
                    root:      Path to the fer2013 data directory (contains train/ and test/ or fer2013.csv).
                    split:     One of 'train', 'val', 'test'. In CSV mode these map to
                         