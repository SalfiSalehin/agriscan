"""Save a picture of the augmentations so you can check them by eye.

    python -m ml.preview_augmentations

Writes reports/figures/augmentation_preview.png (needs the masks from ml.make_masks).
Rows: original | leaf mask | background replaced | full training augmentation.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.augment import RandomBackground
from ml.config import ROOT, load_config
from ml.transforms import IMAGENET_MEAN, IMAGENET_STD, build_train_transform

COLUMNS = 8


def to_picture(tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    return np.clip(tensor.numpy() * std + mean, 0, 1).transpose(1, 2, 0)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    root = ROOT / cfg.dataset["export_dir"]
    mask_root = ROOT / cfg.dataset["mask_dir"]
    train = pd.read_csv(ROOT / cfg.paths["splits_dir"] / "train.csv")
    random.seed(0)

    def mask_path(rel: str) -> Path:
        return mask_root / Path(rel).with_suffix(".png")

    sample = train.sample(400, random_state=0)
    have = sum(mask_path(p).exists() for p in sample["path"])
    print(f"Masks found for {have / len(sample):.0%} of {len(sample)} random training images")

    shown = train.sample(COLUMNS, random_state=1)
    train_tf = build_train_transform(cfg.data["image_size"], cfg.train["erase_prob"])
    replace = RandomBackground(p=1.0)
    fig, axes = plt.subplots(4, COLUMNS, figsize=(2 * COLUMNS, 8.4))
    for col, (_, row) in enumerate(shown.iterrows()):
        image = Image.open(root / row["path"]).convert("RGB")
        path = mask_path(row["path"])
        axes[0, col].imshow(image)
        axes[0, col].set_title(row["class_id"].replace("_", " "), fontsize=7)
        if path.exists():
            axes[1, col].imshow(Image.open(path).convert("L"), cmap="gray", vmin=0, vmax=255)
        else:
            axes[1, col].text(0.5, 0.5, "no mask", ha="center", va="center", fontsize=8)
        swapped = replace(image, path)
        axes[2, col].imshow(swapped)
        axes[3, col].imshow(to_picture(train_tf(swapped)))
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    out = ROOT / cfg.paths["reports_dir"] / "figures" / "augmentation_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
