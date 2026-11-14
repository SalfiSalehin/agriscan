import json

import numpy as np
import pytest
from PIL import Image

from app import inference as inf
from tests.conftest import CLASS_IDS, FakeSession, make_meta

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def test_resize_shorter_side_follows_torchvision_rule():
    assert inf.resize_shorter_side(Image.new("RGB", (300, 400)), 256).size == (256, 341)
    assert inf.resize_shorter_side(Image.new("RGB", (400, 300)), 256).size == (341, 256)
    assert inf.resize_shorter_side(Image.new("RGB", (512, 512)), 256).size == (256, 256)
    same = Image.new("RGB", (256, 500))
    assert inf.resize_shorter_side(same, 256) is same  # already the right size: untouched


def test_center_crop_takes_the_middle():
    image = Image.new("RGB", (300, 256), (0, 255, 0))
    for x in range(300):
        if x < 38 or x >= 262:  # the strips a 224-wide centre crop must not contain
            for y in range(256):
                image.putpixel((x, y), (255, 0, 0))
    crop = inf.center_crop(image, 224)
    assert crop.size == (224, 224)
    assert (np.asarray(crop) == [0, 255, 0]).all()


def test_preprocess_constant_colour_gives_expected_normalised_values():
    colour = (100, 150, 200)
    batch = inf.preprocess(Image.new("RGB", (300, 400), colour), 224, MEAN, STD)
    assert batch.shape == (1, 3, 224, 224) and batch.dtype == np.float32
    expected = (np.array(colour, dtype=np.float32) / 255 - MEAN) / STD
    assert np.allclose(batch[0, :, 0, 0], expected, atol=1e-6)
    assert np.allclose(batch[0, :, 100, 50], expected, atol=1e-6)


def test_softmax_and_temperature():
    logits = np.array([2.0, 0.0, -1.0])
    assert inf.softmax(logits).sum() == pytest.approx(1.0)
    assert inf.softmax(logits, 3.0).max() < inf.softmax(logits).max()
    assert np.isfinite(inf.softmax(np.array([1e4, 0.0]))).all()  # no overflow


def test_crop_helpers():
    assert inf.crop_of("tomato_early_blight") == "tomato"
    assert inf.crop_table(CLASS_IDS) == {"potato": [0, 1], "tomato": [2, 3]}


def test_decide_auto_mode_answers_when_confident():
    out = inf.decide([8, 0, 0, 0], CLASS_IDS, 1.0, 0.89)
    assert out["status"] == "ok" and out["diagnosis"] == "potato_early"
    assert [p["class_id"] for p in out["predictions"]][0] == "potato_early"
    assert len(out["predictions"]) == 3
    probabilities = [p["probability"] for p in out["predictions"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert out["suggested_crop"] is None


def test_decide_is_unsure_below_threshold_and_names_no_disease():
    out = inf.decide([1, 0.9, 0.8, 0.7], CLASS_IDS, 1.0, 0.89)
    assert out["status"] == "unsure"
    assert out["diagnosis"] is None
    assert out["predictions"]  # candidates are still listed for the user to look at


def test_decide_with_picked_crop_uses_all_class_probability_not_renormalised():
    logits = [0.0, 0.0, 6.0, 0.0]
    ok = inf.decide(logits, CLASS_IDS, 1.0, 0.89, crop="tomato")
    assert ok["status"] == "ok" and ok["diagnosis"] == "tomato_early"
    # potato leaf, user picked tomato: tomato classes are weak in absolute terms
    wrong = inf.decide([9.0, 0.0, 1.0, 0.0], CLASS_IDS, 1.0, 0.89, crop="tomato")
    assert wrong["status"] == "unsure" and wrong["diagnosis"] is None
    assert wrong["confidence"] < 0.01  # NOT 1.0 as a renormalised in-crop softmax would give
    assert wrong["suggested_crop"] == "potato"
    assert {p["crop"] for p in wrong["predictions"]} == {"tomato"}


def test_decide_unknown_crop_and_top_k():
    with pytest.raises(ValueError):
        inf.decide([1, 2, 3, 4], CLASS_IDS, 1.0, 0.5, crop="apple")
    assert len(inf.decide([1, 2, 3, 4], CLASS_IDS, 1.0, 0.5, top_k=2)["predictions"]) == 2
    assert len(inf.decide([1, 2, 3, 4], CLASS_IDS, 1.0, 0.5, crop="potato")["predictions"]) == 2


def test_picked_crop_confidence_never_exceeds_auto_confidence():
    rng = np.random.default_rng(0)
    for _ in range(300):
        logits = rng.normal(0, 3, 4)
        auto = inf.decide(logits, CLASS_IDS, 1.3, 0.5)["confidence"]
        for crop in ("potato", "tomato"):
            assert inf.decide(logits, CLASS_IDS, 1.3, 0.5, crop=crop)["confidence"] <= auto + 1e-9


def test_classifier_predict_feeds_a_float32_batch_and_reports_timing():
    session = FakeSession([0, 0, 9, 0])
    classifier = inf.Classifier(session, make_meta())
    result = classifier.predict(Image.new("RGB", (320, 240), (10, 120, 30)), crop="tomato")
    batch = session.last_feed["input"]
    assert batch.shape == (1, 3, 224, 224) and batch.dtype == np.float32
    assert result["status"] == "ok" and result["diagnosis"] == "tomato_early"
    assert set(result["timing_ms"]) == {"preprocess", "inference"}


def test_classifier_info_and_calibration_flag():
    classifier = inf.Classifier(FakeSession([0, 0, 0, 0]), make_meta(temperature=0.75))
    info = classifier.info()
    assert info["crops"] == ["potato", "tomato"] and info["temperature"] == 0.75
    assert info["calibrated"] is True
    meta = make_meta()
    meta["calibration"]["threshold_source"] = "config default (not calibrated)"
    assert inf.Classifier(FakeSession([0, 0, 0, 0]), meta).calibrated is False


def test_from_files_rejects_missing_or_inconsistent_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        inf.Classifier.from_files(tmp_path / "nope.json")
    meta = make_meta()
    path = tmp_path / "m.json"
    path.write_text(json.dumps({**meta, "format_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        inf.Classifier.from_files(path)
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Model file"):
        inf.Classifier.from_files(path)
    (tmp_path / "fake.onnx").write_bytes(b"not really a model")
    with pytest.raises(ValueError, match="hash"):
        inf.Classifier.from_files(path)


def _write_tiny_model(folder, class_count=4, wrong_hash=False):
    import hashlib

    from tests.tiny_onnx import tiny_model

    pytest.importorskip("onnxruntime")
    # channel 0 -> class 0, channel 1 -> class 2, channel 2 -> class 3 (class 1 stays silent)
    weights = [[10, 0, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]
    data = tiny_model(weights)
    (folder / "fake.onnx").write_bytes(data)
    meta = make_meta(class_ids=[f"crop{i // 2}_class{i}" for i in range(class_count)])
    meta["sha256"] = "0" * 64 if wrong_hash else hashlib.sha256(data).hexdigest()
    path = folder / "fake.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_from_files_runs_a_real_onnx_session(tmp_path):
    path = _write_tiny_model(tmp_path)
    classifier = inf.Classifier.from_files(path, threads=1)
    green_leaf = Image.new("RGB", (200, 150), (40, 160, 60))
    result = classifier.predict(green_leaf)
    assert result["status"] == "ok" and result["diagnosis"] == "crop1_class2"
    # dynamic batch axis: three images at once
    batch = np.concatenate([inf.preprocess(green_leaf, 224, MEAN, STD)] * 3)
    assert classifier.session.run(None, {"input": batch})[0].shape == (3, 4)


def test_from_files_rejects_a_model_that_does_not_match_its_metadata(tmp_path):
    with pytest.raises(ValueError, match="hash"):
        inf.Classifier.from_files(_write_tiny_model(tmp_path, wrong_hash=True))
    with pytest.raises(ValueError, match="output size"):
        inf.Classifier.from_files(_write_tiny_model(tmp_path, class_count=5))
