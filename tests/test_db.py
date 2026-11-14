import pytest

from app.db import Store

RESULT = {
    "status": "ok",
    "diagnosis": "tomato_early",
    "confidence": 0.93,
    "crop_mode": "auto",
    "crop": None,
    "model": {"sha256": "abcdef123456"},
}


def test_record_and_feedback_roundtrip():
    store = Store(":memory:")
    pid = store.record(RESULT, "bn")
    assert pid == 1
    assert store.add_feedback(pid, "wrong", "tomato_late") is True
    assert store.add_feedback(pid, "correct", None) is False  # one rating per prediction
    assert store.add_feedback(999, "correct", None) is False  # unknown prediction
    assert store.counts() == {"predictions": 1, "feedback": 1}


def test_unsure_prediction_can_be_stored():
    store = Store(":memory:")
    unsure = dict(RESULT, status="unsure", diagnosis=None)
    assert store.record(unsure, "en") == 1


def test_bad_verdict_is_rejected():
    store = Store(":memory:")
    pid = store.record(RESULT, "en")
    with pytest.raises(ValueError):
        store.add_feedback(pid, "maybe", None)


def test_nothing_identifying_is_stored(tmp_path):
    store = Store(tmp_path / "sub" / "log.db")
    store.record(RESULT, "en")
    columns = [row[1] for row in store._db.execute("PRAGMA table_info(predictions)")]
    assert not {"ip", "filename", "image", "user_agent"} & set(columns)
