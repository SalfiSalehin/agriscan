import pytest

from app.service import ServiceError, normalise_crop, predict_upload
from app.settings import Settings
from tests.conftest import image_bytes

SETTINGS = Settings()
CROPS = ["potato", "tomato"]


def test_normalise_crop():
    assert normalise_crop(None, CROPS) is None
    assert normalise_crop("", CROPS) is None
    assert normalise_crop(" Auto ", CROPS) is None
    assert normalise_crop(" Tomato ", CROPS) == "tomato"
    with pytest.raises(ServiceError) as caught:
        normalise_crop("apple", CROPS)
    assert caught.value.status == 422 and caught.value.code == "unknown_crop"


def test_predict_upload_auto_and_picked(classifier_factory):
    classifier = classifier_factory([0, 0, 9, 0])
    auto = predict_upload(classifier, image_bytes(), None, SETTINGS)
    assert auto["crop_mode"] == "auto" and auto["crop"] is None
    assert auto["diagnosis"] == "tomato_early"
    assert auto["model"] == {"run": "fake_run", "sha256": "ab" * 6}
    picked = predict_upload(classifier, image_bytes(), "tomato", SETTINGS)
    assert picked["crop_mode"] == "picked" and picked["crop"] == "tomato"


def test_predict_upload_turns_image_problems_into_service_errors(classifier_factory):
    classifier = classifier_factory([0, 0, 9, 0])
    with pytest.raises(ServiceError) as caught:
        predict_upload(classifier, b"nonsense", None, SETTINGS)
    assert caught.value.status == 422 and caught.value.code == "not_an_image"
    with pytest.raises(ServiceError) as caught:
        predict_upload(classifier, b"", None, SETTINGS)
    assert caught.value.status == 400


def test_bad_crop_is_rejected_before_the_image_is_decoded(classifier_factory):
    classifier = classifier_factory([0, 0, 9, 0])
    with pytest.raises(ServiceError) as caught:
        predict_upload(classifier, b"nonsense", "apple", SETTINGS)
    assert caught.value.code == "unknown_crop"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("AGRISCAN_MAX_UPLOAD_MB", "2")
    monkeypatch.setenv("AGRISCAN_THREADS", "1")
    settings = Settings.from_env()
    assert settings.max_upload_bytes == 2 * 1024 * 1024 and settings.threads == 1
    monkeypatch.setenv("AGRISCAN_THREADS", "abc")
    with pytest.raises(ValueError, match="AGRISCAN_THREADS"):
        Settings.from_env()
    monkeypatch.setenv("AGRISCAN_THREADS", "999")
    with pytest.raises(ValueError, match="between"):
        Settings.from_env()


def test_predict_upload_adds_names_info_and_prediction_id(classifier_factory):
    from app.content import Content
    from app.db import Store

    doc = {
        "diseases": {
            cid: {"kind": "disease", "name": {"en": cid + "-en", "bn": cid + "-bn"}}
            for cid in ["potato_early", "potato_late", "tomato_early", "tomato_healthy"]
        },
        "disclaimer": {"en": "d-en", "bn": "d-bn"},
        "chemical_note": {"en": "c-en", "bn": "c-bn"},
        "helpline": {"verified": False, "text": {"en": "h-en", "bn": "h-bn"}},
        "unsure": {"message": {"en": "m-en", "bn": "m-bn"}, "tips": []},
    }
    content, store = Content(doc), Store(":memory:")
    classifier = classifier_factory([0, 0, 9, 0])
    ok = predict_upload(classifier, image_bytes(), None, SETTINGS, content, store, "bn")
    assert ok["lang"] == "bn" and ok["prediction_id"] == 1
    assert ok["predictions"][0]["name"] == "tomato_early-bn"
    assert ok["info"]["kind"] == "disease" and ok["info"]["disclaimer"] == "d-bn"
    unsure = predict_upload(classifier, image_bytes(), "potato", SETTINGS, content, store)
    assert unsure["status"] == "unsure" and unsure["info"]["kind"] == "unsure"
    assert unsure["lang"] == "en" and unsure["prediction_id"] == 2
    plain = predict_upload(classifier, image_bytes(), None, SETTINGS)
    assert "info" not in plain and plain["prediction_id"] is None


def test_unknown_language_is_refused(classifier_factory):
    classifier = classifier_factory([0, 0, 9, 0])
    with pytest.raises(ServiceError) as error:
        predict_upload(classifier, image_bytes(), None, SETTINGS, lang="fr")
    assert error.value.status == 422 and error.value.code == "unknown_language"
