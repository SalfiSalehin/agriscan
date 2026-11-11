"""Background-replacement augmentation (numpy + Pillow only).

PlantVillage leaves sit on plain backgrounds, so a model can pick up shortcuts from the
background. Putting the leaf on generated clutter pushes it to look at the leaf instead.
The leaf masks come from PlantVillage's own segmented images (see ml/make_masks.py).
This is a crude stand-in for real fields, not a substitute for real field photos.
"""

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Rough colour families behind real leaves: soil, foliage, sky, wall/paper, shade, dry ground.
PALETTES = (
    (0.35, 0.25, 0.15),
    (0.20, 0.35, 0.15),
    (0.55, 0.65, 0.85),
    (0.85, 0.85, 0.80),
    (0.15, 0.15, 0.15),
    (0.55, 0.45, 0.30),
)


def _value_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    small = (rng.random((cells, cells, 3)) * 255).astype(np.uint8)
    big = Image.fromarray(small).resize((size, size), Image.BICUBIC)
    return np.asarray(big, dtype=np.float32) / 255.0


def make_background(size: int, rng: np.random.Generator) -> Image.Image:
    """Random textured background: palette colour + multi-scale noise + a few blobs, blurred."""
    base = np.array(PALETTES[int(rng.integers(len(PALETTES)))], dtype=np.float32)
    base = np.clip(base + rng.normal(0.0, 0.08, 3), 0.0, 1.0)
    texture = np.zeros((size, size, 3), dtype=np.float32)
    weight = 1.0
    for cells in (3, 6, 12, 24, 48):
        texture += weight * _value_noise(size, cells, rng)
        weight *= 0.6
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
    strength = float(rng.uniform(0.2, 0.6))
    background = np.clip((1.0 - strength) * base + strength * texture, 0.0, 1.0)
    image = Image.fromarray((background * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(0, 6))):
        x0, y0 = (int(v) for v in rng.integers(0, size, 2))
        w, h = (int(v) for v in rng.integers(max(size // 10, 1), max(size // 3, 2), 2))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        draw.ellipse([x0, y0, x0 + w, y0 + h], fill=colour)
    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 3.0))))


class RandomBackground:
    """With probability p, paste the leaf onto a generated background using a stored mask.

    The mask is a greyscale PNG (white = leaf). Without a usable mask the image is returned
    unchanged, so a missing or odd mask can never damage a training image.
    """

    def __init__(self, p: float = 0.5, min_fraction: float = 0.10, max_fraction: float = 0.97):
        self.p = p
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

    def __call__(self, image: Image.Image, mask_path=None) -> Image.Image:
        if mask_path is None or random.random() >= self.p:
            return image
        try:
            with Image.open(mask_path) as handle:
                mask_img = handle.convert("L")
        except FileNotFoundError:
            return image
        image = image.convert("RGB")
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size)
        fraction = float(np.asarray(mask_img, dtype=np.float32).mean()) / 255.0
        if not self.min_fraction <= fraction <= self.max_fraction:
            return image
        rng = np.random.default_rng(random.getrandbits(32))
        background = make_background(max(image.size), rng).resize(image.size)
        # shrink the mask by a pixel so no grey halo of the old background survives
        soft = mask_img.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
        alpha = (np.asarray(soft, dtype=np.float32) / 255.0)[..., None]
        leaf = np.asarray(image, dtype=np.float32)
        out = alpha * leaf + (1.0 - alpha) * np.asarray(background, dtype=np.float32)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def __repr__(self) -> str:
        return f"RandomBackground(p={self.p})"
