import numpy as np
from PIL import Image

from ml.eval_plots import plot_confusion, plot_errors


def test_plot_confusion_writes_png(tmp_path):
    cm = np.array([[5, 1, 0], [0, 4, 0], [0, 0, 0]])
    out = tmp_path / "cm.png"
    plot_confusion(cm, ["a", "b", "c"], "title", out)
    assert out.stat().st_size > 1000


def test_plot_errors_writes_png_or_nothing(tmp_path):
    images = []
    for i in range(5):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (30, 20), (i * 40, 10, 10)).save(path)
        images.append({"path": path, "true": "a", "pred": "b", "conf": 0.9 - i / 10})
    out = tmp_path / "errors.png"
    assert plot_errors(images, "title", out, limit=4) is True
    assert out.stat().st_size > 1000
    assert plot_errors([], "title", tmp_path / "none.png") is False
    assert not (tmp_path / "none.png").exists()
