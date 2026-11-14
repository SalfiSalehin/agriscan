"""API tests. They need FastAPI and httpx (pip install fastapi httpx) and are skipped without."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.settings import Settings  # noqa: E402
from tests.conftest import image_bytes  # noqa: E402


@pytest.fixture
def client(classifier_factory):
    app = create_app(classifier_factory([0, 0, 9, 0]), Settings(max_upload_bytes=200_000))
    with TestClient(app) as test_client:
        yield test_client


def post(client, data, crop=None, name="leaf.jpg", content_type="image/jpeg"):
    files = {"file": (name, data, content_type)}
    return client.post("/api/predict", files=files, data={"crop": crop} if crop else None)


def test_health_and_model(client):
    assert client.get("/api/health").json() == {"status": "ok", "model": "fake_run"}
    info = client.get("/api/model").json()
    assert info["crops"] == ["potato", "tomato"] and info["threshold"] == 0.89
    assert client.get("/api/health").headers["cache-control"] == "no-store"


def test_predict_auto_and_with_crop(client):
    body = post(client, image_bytes()).json()
    assert body["status"] == "ok" and body["diagnosis"] == "tomato_early"
    assert body["crop_mode"] == "auto" and len(body["predictions"]) == 3
    picked = post(client, image_bytes(), crop="tomato").json()
    assert picked["crop_mode"] == "picked" and picked["crop"] == "tomato"


def test_predict_unsure_when_wrong_crop_is_picked(client):
    body = post(client, image_bytes(), crop="potato").json()
    assert body["status"] == "unsure" and body["diagnosis"] is None
    assert body["suggested_crop"] == "tomato"


def test_errors_have_a_stable_shape(client):
    r = post(client, b"not an image")
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_an_image"
    r = post(client, image_bytes(), crop="apple")
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_crop"
    r = post(client, b"")
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_file"
    assert client.post("/api/predict").status_code == 422  # no file at all


def test_too_large_upload_is_refused(client):
    r = post(client, b"\xff" * 300_000)
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_busy_server_answers_503_with_retry_after(classifier_factory):
    app = create_app(classifier_factory([0, 0, 9, 0]), Settings(max_concurrent=1))
    with TestClient(app) as test_client:
        app.state.slots.acquire()  # pretend another prediction is running
        r = post(test_client, image_bytes())
        assert r.status_code == 503 and r.headers["retry-after"] == "2"
        app.state.slots.release()
        assert post(test_client, image_bytes()).status_code == 200


def test_slot_is_released_after_an_error(classifier_factory):
    app = create_app(classifier_factory([0, 0, 9, 0]), Settings(max_concurrent=1))
    with TestClient(app) as test_client:
        assert post(test_client, b"nonsense").status_code == 422
        assert post(test_client, image_bytes()).status_code == 200


def test_predict_returns_advice_in_the_requested_language(classifier_factory):
    from tests.conftest import FakeSession, make_meta

    ids = ["potato_early_blight", "potato_late_blight", "tomato_early_blight", "tomato_healthy"]
    from app.content import Content
    from app.db import Store
    from app.inference import Classifier
    from app.settings import DEFAULT_CONTENT

    classifier = Classifier(FakeSession([0, 0, 9, 0]), make_meta(class_ids=ids))
    app = create_app(classifier, Settings(), Content.load(DEFAULT_CONTENT), Store(":memory:"))
    with TestClient(app) as c:
        files = {"file": ("leaf.jpg", image_bytes(), "image/jpeg")}
        body = c.post("/api/predict", files=files, data={"lang": "bn"}).json()
        assert body["status"] == "ok" and body["lang"] == "bn" and body["prediction_id"] == 1
        assert body["info"]["kind"] == "disease" and body["info"]["actions"]
        assert any("\u0980" <= ch <= "\u09ff" for ch in body["info"]["name"])
        files = {"file": ("leaf.jpg", image_bytes(), "image/jpeg")}
        assert c.post("/api/predict", files=files, data={"lang": "fr"}).status_code == 422
        page = c.get("/api/diseases/tomato_late_blight?lang=en").json()
        assert page["class_id"] == "tomato_late_blight" and page["sources"]
        assert c.get("/api/diseases/nope").json()["error"]["code"] == "unknown_class"
        assert c.get("/api/diseases/tomato_late_blight?lang=xx").status_code == 422
        r = c.post(
            "/api/feedback",
            json={"prediction_id": 1, "verdict": "wrong", "true_class": "tomato_healthy"},
        )
        assert r.status_code == 200
        again = c.post("/api/feedback", json={"prediction_id": 1, "verdict": "correct"})
        assert again.status_code == 404
        bad = c.post(
            "/api/feedback", json={"prediction_id": 1, "verdict": "wrong", "true_class": "mango"}
        )
        assert bad.status_code == 422
        assert c.post("/api/feedback", json={"prediction_id": 1, "verdict": "x"}).status_code == 422


def test_content_gap_stops_startup(classifier_factory):
    from app.content import Content, ContentError

    app = create_app(classifier_factory([0, 0, 9, 0]), Settings(), Content({"diseases": {}}))
    with pytest.raises(ContentError):
        with TestClient(app):
            pass


def test_feedback_off_without_a_store(client):
    r = client.post("/api/feedback", json={"prediction_id": 1, "verdict": "correct"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "feedback_off"
