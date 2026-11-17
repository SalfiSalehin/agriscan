import numpy as np
from PIL import Image

from ml.check_images import broken_images


def test_finds_truncated_and_garbage_files_but_not_good_ones(tmp_path):
    rng = np.random.default_rng(0)
    good = tmp_path / "a" / "good.jpg"
    good.parent.mkdir()
    Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(good)
    cut = tmp_path / "a" / "cut.jpg"
    cut.write_bytes(good.read_bytes()[:400])
    junk = tmp_path / "a" / "junk.png"
    junk.write_bytes(b"not an image")
    (tmp_path / "a" / "note.txt").write_text("x")
    bad = {p.name for p, _ in broken_images([tmp_path, tmp_path / "missing"])}
    assert bad == {"cut.jpg", "junk.png"}
