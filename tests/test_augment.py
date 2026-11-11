import random

import numpy as np
from PIL import Image

from ml.augment import RandomBackground, make_background


def _leaf_and_mask(tmp_path, size: int = 256):
    arr = np.full((size, size, 3), 120, dtype=np.uint8)
    arr[64:192, 64:192] = (30, 140, 40)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[64:192, 64:192] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)
    return Image.fromarray(arr), mask_path


def test_replacement_keeps_leaf_and_changes_background(tmp_path):
    random.seed(0)
    leaf, mask_path = _leaf_and_mask(tmp_path)
    out = RandomBackground(p=1.0)(leaf, mask_path)
    before, after = np.asarray(leaf).astype(int), np.asarray(out).astype(int)
    assert out.size == leaf.size
    assert np.abs(before[128, 128] - after[128, 128]).max() < 10
    assert np.abs(before[:32, :32] - after[:32, :32]).mean() > 1


def test_zero_probability_leaves_image_alone(tmp_path):
    leaf, mask_path = _leaf_and_mask(tmp_path)
    assert RandomBackground(p=0.0)(leaf, mask_path) is leaf


def test_missing_mask_leaves_image_alone(tmp_path):
    leaf, _ = _leaf_and_mask(tmp_path)
    assert RandomBackground(p=1.0)(leaf, tmp_path / "nope.png") is leaf
    assert RandomBackground(p=1.0)(leaf, None) is leaf


def test_implausible_mask_is_ignored(tmp_path):
    leaf, _ = _leaf_and_mask(tmp_path)
    empty = tmp_path / "empty.png"
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(empty)
    out = RandomBackground(p=1.0)(leaf, empty)
    assert np.array_equal(np.asarray(out), np.asarray(leaf))


def test_make_background_size_and_mode():
    background = make_background(64, np.random.default_rng(0))
    assert background.size == (64, 64)
    assert background.mode == "RGB"
