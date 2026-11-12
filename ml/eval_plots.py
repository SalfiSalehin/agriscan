"""Figures for the evaluation report (matplotlib only, no torch)."""

import math
from pathlib import Path

import numpy as np
from PIL import Image


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion(cm: np.ndarray, class_ids: list[str], title: str, path: Path) -> None:
    """Confusion matrix: colour = share of each true class, number = image count."""
    plt = _pyplot()
    n = len(class_ids)
    support = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, support, out=np.zeros(cm.shape, dtype=float), where=support > 0)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(share, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_ids, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{c} ({int(s)})" for c, s in zip(class_ids, support[:, 0], strict=True)])
    ax.tick_params(axis="y", labelsize=7)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if share[i, j] > 0.5 else "black",
                )
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class (number of images)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.04, label="share of the true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_errors(errors: list[dict], title: str, path: Path, limit: int = 16, columns: int = 4):
    """Grid of the most confident mistakes. `errors` rows need path (openable), true, pred, conf.

    Returns False (and writes nothing) when there are no mistakes to show.
    """
    shown = errors[:limit]
    if not shown:
        return False
    plt = _pyplot()
    rows = math.ceil(len(shown) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, error in zip(axes.ravel(), shown, strict=False):
        with Image.open(error["path"]) as handle:
            ax.imshow(np.asarray(handle.convert("RGB")))
        ax.set_title(
            f"true: {error['true']}\nsaid: {error['pred']} ({error['conf']:.0%})", fontsize=7
        )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True
