import io
import zipfile

import numpy as np
from PIL import Image

from ml.make_masks import index_segmented, segmented_to_mask


def test_segmented_to_mask_finds_leaf_and_drops_speckle():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 8, (128, 128, 3)).astype(np.uint8)  # near-black background with noise
    arr[30:100, 30:100] = (40, 120, 50)
    arr[5, 5] = (255, 255, 255)  # isolated bright speck
    mask = np.asarray(segmented_to_mask(Image.fromarray(arr)))
    assert mask[60, 60] == 255
    assert mask[110, 110] == 0
    assert mask[5, 5] == 0


def test_index_segmented_matches_color_file_names():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("segmented/Tomato___Target_Spot/abc___FREC 12_final_masked.JPG", b"x")
        uuid = "007f6e89-3e7c-4e6a-8ef6-8058c7133799"
        zf.writestr(f"segmented/Corn___Rust/{uuid}___RS_Rust 1933_final_masked.jpg", b"x")
        zf.writestr("color/Tomato___Target_Spot/abc___FREC 12.JPG", b"x")
        zf.writestr("grayscale/Tomato___Target_Spot/abc___FREC 12.JPG", b"x")
    with zipfile.ZipFile(buffer) as zf:
        index = index_segmented(zf)
    assert set(index) == {
        ("Tomato___Target_Spot", "abc___frec 12"),
        ("Corn___Rust", "rs_rust 1933"),
    }
    assert index[("Tomato___Target_Spot", "abc___frec 12")].startswith("segmented/")
