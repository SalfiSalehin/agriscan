import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml import analysis as an
from ml.metrics import confusion_matrix

CLASSES = ["a", "b", "c", "d"]


def test_softmax_rows_sum_to_one_and_temperature_softens():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    p = an.softmax(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert an.softmax(logits, 3.0).max(axis=1)[0] < p.max(axis=1)[0]


def test_fit_temperature_recovers_known_overconfidence():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1.5, (20000, 5))
    p = an.softmax(scores)
    labels = np.array([rng.choice(5, p=row) for row in p])
    overconfident = scores * 2.0  # the model "shouts": true temperature is 2
    assert abs(an.fit_temperature(overconfident, labels) - 2.0) < 0.2


def test_expected_calibration_error():
    assert an.expected_calibration_error([1.0] * 100, [1] * 50 + [0] * 50) == pytest.approx(0.5)
    assert an.expected_calibration_error([0.9] * 100, [1] * 90 + [0] * 10) < 0.01


def test_auroc():
    assert an.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert an.auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert an.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert np.isnan(an.auroc([], [0.5]))


def test_choose_threshold():
    unknown = [0.55, 0.60, 0.65, 0.95, 0.97]
    assert an.choose_threshold(unknown, 0.2) == 0.96
    assert an.choose_threshold([0.995] * 5, 0.2) is None
    assert an.choose_threshold([], 0.2) is None


def test_selective_stats():
    stats = an.selective_stats([0.9, 0.8, 0.4, 0.3], [True, False, True, False], 0.5)
    assert stats["coverage"] == 0.5
    assert stats["accuracy_when_answered"] == 0.5
    assert an.selective_stats([0.1], [True], 0.5)["accuracy_when_answered"] is None


def test_half_of_is_stable_and_splits():
    assert an.half_of("00000000ff") == 0
    assert an.half_of("00000001ff") == 1
    assert {an.half_of(f"{i:08x}") for i in range(10)} == {0, 1}


def test_macro_f1_min_support_ignores_small_classes():
    cm = confusion_matrix([0, 0, 0, 0, 1], [0, 0, 0, 0, 0], 3)
    assert an.macro_f1_min_support(cm, 1) < an.macro_f1_min_support(cm, 2)
    assert an.macro_f1_min_support(cm, 2) == pytest.approx(2 * 0.8 * 1.0 / 1.8)


def test_bootstrap_ci():
    labels = np.repeat(np.arange(4), 25)
    perfect = an.bootstrap_ci(labels, labels, 4, n_boot=50)
    assert perfect["accuracy"] == [1.0, 1.0]
    pred = labels.copy()
    pred[::5] = (pred[::5] + 1) % 4
    ci = an.bootstrap_ci(labels, pred, 4, n_boot=200)
    assert ci["accuracy"][0] < 0.8 < ci["accuracy"][1]
    assert an.bootstrap_ci(labels, pred, 4, n_boot=0) is None


def test_summarise_set_small_example():
    logits = np.array([[5, 0, 0, 0], [5, 0, 0, 0], [0, 5, 0, 0], [5, 0, 0, 0]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    out = an.summarise_set(logits, labels, 1.0, CLASSES, min_support=2)
    assert out["n_images"] == 4
    assert out["accuracy"] == 0.75
    assert out["classes_reported"] == 2
    assert set(out["classes_below_min_support"]) == {"c", "d"}
    assert out["per_class"]["a"]["support"] == 2
    assert "c" not in out["per_class"]
    assert out["lowest_recall"][0]["class"] == "b"


def _calibrated(rng, n, scale):
    """Logits whose softmax is honest: the label is drawn from the model's own probabilities."""
    scores = rng.normal(0, scale, (n, 4))
    probs = an.softmax(scores)
    return scores, np.array([rng.choice(4, p=row) for row in probs])


def test_analyse_run_end_to_end_on_synthetic_logits():
    rng = np.random.default_rng(1)
    val_x, val_y = _calibrated(rng, 600, 2.5)
    test_x, test_y = _calibrated(rng, 600, 2.5)
    field_x, field_y = _calibrated(rng, 200, 2.5)
    unknown_x = rng.normal(0, 0.4, (300, 4))  # nothing stands out: the model should be unsure
    logits = {"val": val_x, "test": test_x, "field": field_x, "unknown": unknown_x}
    labels = {"val": val_y, "test": test_y, "field": field_y}
    field_info = pd.DataFrame(
        {"mapping": ["exact"] * 100 + ["approximate"] * 100, "half": [0, 1] * 100}
    )
    unknown_info = pd.DataFrame({"half": [0, 1] * 150})
    verified = np.arange(600) % 2 == 0
    result = an.analyse_run(
        logits, labels, field_info, unknown_info, CLASSES, verified, 5, 0.10, 50
    )
    assert set(result["sets"]) >= {
        "val",
        "test",
        "field",
        "field_exact_mapping_only",
        "test_leaf_verified_only",
        "test_fallback_grouping_only",
    }
    assert 0.7 < result["temperature"] < 1.4  # honest logits need almost no rescaling
    assert result["sets"]["test"]["accuracy"] > 0.6
    threshold = result["threshold"]
    assert threshold["chosen_threshold"] is not None
    chosen = [row for row in threshold["table"] if row["chosen"]]
    assert len(chosen) == 1
    assert chosen[0]["unknown_held_out_false_accept"] <= 0.15
    assert chosen[0]["test"]["coverage"] > 0.3
    assert threshold["auroc_confidence_known_vs_unknown"]["test_vs_unknown_held_out"] > 0.8
    assert threshold["images"]["unknown_calibration"] == 150


def test_analyse_run_rejects_tiny_sets():
    rng = np.random.default_rng(0)
    logits = {k: rng.normal(0, 1, (10, 4)) for k in ("val", "test", "field", "unknown")}
    labels = {k: np.zeros(10, int) for k in ("val", "test", "field")}
    info = pd.DataFrame({"mapping": ["exact"] * 10, "half": [0] * 10})
    with pytest.raises(ValueError):
        an.analyse_run(logits, labels, info, info[["half"]], CLASSES, None, 1, 0.1, 0)


def _write_image(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)


def test_collect_external_rows_merges_manifest_and_local_images(tmp_path):
    _write_image(tmp_path / "data/field/a/imported.jpg", 1)
    _write_image(tmp_path / "data/nonleaf/other_crop/apple.jpg", 2)
    _write_image(tmp_path / "data/field/b/mine.JPG", 3)  # local photo
    _write_image(tmp_path / "data/nonleaf/hand.png", 4)  # local non-leaf photo
    _write_image(tmp_path / "data/field/not_a_class/x.jpg", 5)  # ignored
    manifest = pd.DataFrame(
        {
            "path": [
                "data/field/a/imported.jpg",
                "data/nonleaf/other_crop/apple.jpg",
                "data/field/a/gone.jpg",
            ],
            "role": ["field", "unknown", "field"],
            "class_id": ["a", "", "a"],
            "source": ["PlantDoc"] * 3,
            "mapping": ["approximate", "out_of_scope", "exact"],
            "sha256": ["0" * 8, "1" * 8, "2" * 8],
        }
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    frame, warnings = an.collect_external_rows(
        tmp_path,
        tmp_path / "manifest.csv",
        tmp_path / "data/field",
        tmp_path / "data/nonleaf",
        CLASSES,
    )
    assert len(frame) == 4
    assert sorted(frame["source"]) == ["PlantDoc", "PlantDoc", "local", "local"]
    local_field = frame[(frame["source"] == "local") & (frame["role"] == "field")].iloc[0]
    assert local_field["class_id"] == "b"
    assert local_field["mapping"] == "exact"
    assert frame[frame["role"] == "unknown"]["class_id"].eq("").all()
    assert set(frame["half"]) <= {0, 1}
    assert any("not a class id" in w for w in warnings)
    assert any("missing on disk" in w for w in warnings)


def test_collect_external_rows_without_manifest_or_folders(tmp_path):
    frame, warnings = an.collect_external_rows(
        tmp_path, tmp_path / "none.csv", tmp_path / "f", tmp_path / "n", CLASSES
    )
    assert frame.empty
    assert warnings == []
