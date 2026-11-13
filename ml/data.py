"""Datasets built from the split manifests written by ml.prepare_data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ml.config import ROOT


class ManifestDataset(Dataset):
    """Reads (path, class_id) rows from a manifest CSV; images live under `image_root`.

    If `background` (a RandomBackground) and `mask_root` are given, leaves are sometimes moved
    onto generated backgrounds before `transform` runs. Masks mirror the image layout:
    <mask_root>/<class_id>/<name>.png
    """

    def __init__(
        self,
        csv_path,
        image_root,
        class_index,
        transform=None,
        limit=None,
        background=None,
        mask_root=None,
    ):
        df = pd.read_csv(csv_path)
        if limit is not None and limit < len(df):
            df = df.sample(n=limit, random_state=0)  # random subset, for smoke tests only
        self.rel = list(df["path"])
        self.paths = [Path(image_root) / p for p in self.rel]
        self.labels = [class_index[c] for c in df["class_id"]]
        self.transform = transform
        self.background = background
        self.mask_root = Path(mask_root) if mask_root is not None else None

    def __len__(self) -> int:
        return len(self.paths)

    def mask_path(self, i: int) -> Path:
        return self.mask_root / Path(self.rel[i]).with_suffix(".png")

    def mask_coverage(self) -> float:
        """Share of images that have a mask file."""
        if self.mask_root is None or not self.paths:
            return 0.0
        return sum(self.mask_path(i).exists() for i in range(len(self))) / len(self)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        if self.background is not None and self.mask_root is not None:
            image = self.background(image, self.mask_path(i))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[i]


class ListDataset(Dataset):
    """Images from an explicit list of (absolute) paths and integer labels.

    Used for the real-world photos: as training data (with a training transform) and for scoring
    (with the eval transform). No background replacement: there are no leaf masks for them.
    """

    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = [int(x) for x in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[i]


def load_class_index(cfg) -> dict[str, int]:
    path = ROOT / cfg.paths["splits_dir"] / "class_index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    if sorted(index) != sorted(cfg.class_ids):
        raise SystemExit("class_index.json does not match config/config.yaml. Re-run prepare_data.")
    return index


def class_weights(labels, num_classes: int, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency weights ** power (1.0 = full inverse frequency, 0 = no weighting)."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("a class has no training images")
    return ((counts.sum() / (num_classes * counts)) ** power).astype(np.float32)


def verify_manifests(cfg) -> None:
    """Fail early if train/val share a leaf group or a listed image file is missing."""
    splits = ROOT / cfg.paths["splits_dir"]
    root = ROOT / cfg.dataset["export_dir"]
    frames = {name: pd.read_csv(splits / f"{name}.csv") for name in ("train", "val", "test")}
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(frames[a]["group_id"]) & set(frames[b]["group_id"])
            if shared:
                raise SystemExit(f"{len(shared)} leaf groups are shared by {a} and {b}")
    for name in ("train", "val"):
        missing = [p for p in frames[name]["path"] if not (root / p).exists()]
        if missing:
            raise SystemExit(f"{len(missing)} {name} images are missing, e.g. {missing[0]}")
