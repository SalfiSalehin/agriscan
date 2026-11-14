"""Shared fixtures for the serving tests."""

import io

import numpy as np
import pytest
from PIL import Image

CLASS_IDS = ["potato_early", "potato_late", "tomato_early", "tomato_healthy"]


class FakeSession:
    """Stands in for an ONNX Runtime session: returns fixed logits and remembers its input."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.last_feed = None

    def run(self, output_names, feed):
        self.last_feed = feed
        return [self.logits[None, :]]


def make_meta(class_ids=CLASS_IDS, temperature=1.0, threshold=0.89):
    return {
        "format_version": 1,
        "model_file": "fake.onnx",
        "sha256": "ab" * 32,
        "source_run": "fake_run",
        "arch": "fake_arch",
        "input": {
            "name": "input",
            "size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "output": {"name": "logits", "class_ids": list(class_ids)},
        "calibration": {
            "temperature": temperature,
            "confidence_threshold": threshold,
            "threshold_source": "calibrated by ml.evaluate",
        },
    }


@pytest.fixture
def meta():
    return make_meta()


@pytest.fixture
def classifier_factory():
    from app.inference import Classifier

    def build(logits, **meta_kwargs):
        return Classifier(FakeSession(logits), make_meta(**meta_kwargs))

    return build


def image_bytes(fmt="JPEG", size=(120, 90), colour=(40, 160, 60), **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, fmt, **save_kwargs)
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes():
    return image_bytes()
