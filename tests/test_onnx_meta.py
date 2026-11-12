import numpy as np
import pytest

from ml.onnx_meta import build_meta, latency_summary, parity_report


def test_parity_passes_for_tiny_differences():
    logits = np.random.default_rng(0).normal(size=(8, 5))
    report = parity_report(logits, logits + 1e-5)
    assert report["passed"] is True
    assert report["same_prediction_share"] == 1.0


def test_parity_fails_when_predictions_change():
    a = np.array([[2.0, 0.0], [0.0, 2.0]])
    b = np.array([[0.0, 2.0], [0.0, 2.0]])
    assert parity_report(a, b)["passed"] is False


def test_parity_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        parity_report(np.zeros((2, 3)), np.zeros((2, 4)))


def test_latency_summary():
    row = latency_summary([10, 20, 30, 40], threads=2)
    assert row["median_ms"] == 25.0
    assert row["threads"] == 2
    assert row["runs"] == 4


def test_build_meta_contains_what_the_backend_needs(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    meta = build_meta(
        run_name="r",
        arch="mobilenet_v3_small",
        class_ids=["a", "b"],
        image_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
        temperature=1.4,
        threshold=0.7,
        threshold_source="test",
        onnx_path=model,
        parity={"passed": True},
        latency=[],
        versions={},
    )
    assert meta["output"]["class_ids"] == ["a", "b"]
    assert meta["input"]["size"] == 224
    assert "resize the shorter side to 256" in meta["input"]["preprocessing"]
    assert meta["calibration"]["temperature"] == 1.4
    assert len(meta["sha256"]) == 64
