"""Checks that need the real exported model and the PlantVillage images.

Skipped when onnxruntime, torch, the model or the images are missing. They answer: does the web
preprocessing equal the training/evaluation preprocessing, and does the served model still get
the lab images right?
"""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("torch")

from app.inference import Classifier, preprocess  # noqa: E402
from ml.config import ROOT, load_config  # noqa: E402
from ml.transforms import build_eval_transform  # noqa: E402

META = ROOT / "models" / "agriscan_mobilenet_v3_small.meta.json"
CFG = load_config()
VAL_CSV = ROOT / CFG.paths["splits_dir"] / "val.csv"
IMAGE_ROOT = ROOT / CFG.dataset["export_dir"]

pytestmark = pytest.mark.skipif(
    not (META.exists() and VAL_CSV.exists() and IMAGE_ROOT.exists()),
    reason="needs models/*.meta.json and the PlantVillage export",
)


def val_rows(count):
    frame = pd.read_csv(VAL_CSV)
    picks = np.linspace(0, len(frame) - 1, count).astype(int)
    return frame.iloc[picks]


def test_web_preprocessing_equals_the_evaluation_transform():
    meta = json.loads(META.read_text(encoding="utf-8"))["input"]
    reference = build_eval_transform(meta["size"])
    for rel in val_rows(12)["path"]:
        with Image.open(IMAGE_ROOT / rel) as handle:
            image = handle.convert("RGB")
        ours = preprocess(image, meta["size"], meta["mean"], meta["std"])[0]
        assert np.abs(ours - reference(image).numpy()).max() < 1e-5, rel


def test_served_model_is_accurate_on_validation_images():
    classifier = Classifier.from_files(META)
    rows = val_rows(300)
    correct = 0
    for rel, true_class in zip(rows["path"], rows["class_id"], strict=True):
        with Image.open(IMAGE_ROOT / rel) as handle:
            result = classifier.predict(handle.convert("RGB"))
        correct += result["predictions"][0]["class_id"] == true_class
    assert correct / len(rows) > 0.93  # the model scored about 0.96-0.97 on this split


def test_picking_the_right_crop_never_hurts_on_validation_images():
    classifier = Classifier.from_files(META)
    rows = val_rows(150)
    for rel, true_class in zip(rows["path"], rows["class_id"], strict=True):
        with Image.open(IMAGE_ROOT / rel) as handle:
            image = handle.convert("RGB")
        picked = classifier.predict(image, crop=true_class.split("_")[0])
        assert picked["predictions"][0]["crop"] == true_class.split("_")[0]
