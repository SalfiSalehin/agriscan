"""Tests for the web page and the small endpoints it uses. Need FastAPI and httpx."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from app.content import Content  # noqa: E402
from app.inference import Classifier  # noqa: E402
from app.main import create_app  # noqa: E402
from app.settings import DEFAULT_CONTENT, STATIC_DIR, Settings  # noqa: E402
from tests.conftest import FakeSession, make_meta  # noqa: E402

IDS = ["potato_early_blight", "potato_late_blight", "tomato_early_blight", "tomato_healthy"]


@pytest.fixture
def client():
    classifier = Classifier(FakeSession([0, 0, 9, 0]), make_meta(class_ids=IDS))
    app = create_app(classifier, Settings(), Content.load(DEFAULT_CONTENT))
    with TestClient(app) as test_client:
        yield test_client


def test_index_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "AgriScan" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
    assert client.get("/nope.txt").status_code == 404


def test_page_has_a_strict_csp_and_api_does_not_need_one(client):
    csp = client.get("/").headers["content-security-policy"]
    assert "script-src 'self'" in csp and "'unsafe-inline'" not in csp and "frame-ancestors" in csp
    assert "content-security-policy" not in client.get("/api/health").headers  # Swagger needs CDN


def test_page_uses_no_inline_script_or_style():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "<style" not in html and " style=" not in html
    assert "onclick=" not in html


def test_api_routes_are_not_shadowed_by_the_static_mount(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/nothing").status_code == 404


def test_classes_endpoint_lists_names_in_both_languages(client):
    en = client.get("/api/classes?lang=en").json()["classes"]
    bn = client.get("/api/classes?lang=bn").json()["classes"]
    assert [c["class_id"] for c in en] == IDS and en[0]["crop"] == "potato"
    assert en[0]["name"] != bn[0]["name"]
    assert client.get("/api/classes?lang=xx").status_code == 422


def test_unsure_advice_is_available_for_language_switching(client):
    block = client.get("/api/diseases/unsure?lang=bn").json()
    assert block["kind"] == "unsure" and block["tips"]


def test_every_ui_string_exists_in_both_languages():
    source = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    start, end = source.index("const STR = {"), source.index("const CROPS")
    block = source[start:end]
    en = block[block.index("  en: {") : block.index("  bn: {")]
    bn = block[block.index("  bn: {") :]

    def keys(text):
        return sorted(line.strip().split(":")[0] for line in text.splitlines()[1:] if ": " in line)

    assert keys(en) == keys(bn)
    assert json.dumps(keys(en))  # non-empty


def limited_client(**settings):
    classifier = Classifier(FakeSession([0, 0, 9, 0]), make_meta(class_ids=IDS))
    app = create_app(classifier, Settings(**settings), Content.load(DEFAULT_CONTENT))
    return TestClient(app)


def png():
    from tests.conftest import image_bytes

    return {"file": ("leaf.png", image_bytes("PNG"), "image/png")}


def test_rate_limit_answers_429_with_retry_after_and_security_headers():
    with limited_client(rate_per_min=2) as c:
        assert c.post("/api/predict", files=png()).status_code == 200
        assert c.post("/api/predict", files=png()).status_code == 200
        blocked = c.post("/api/predict", files=png())
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "rate_limited"
        assert int(blocked.headers["retry-after"]) >= 1
        assert (
            blocked.headers["x-content-type-options"] == "nosniff"
        )  # headers also on early replies
        assert c.get("/api/health").status_code == 200  # reading is never limited
        assert c.get("/").status_code == 200


def test_rate_limit_is_per_visitor_when_a_proxy_is_trusted():
    with limited_client(rate_per_min=1, trust_proxy=True) as c:
        first = c.post("/api/predict", files=png(), headers={"x-forwarded-for": "1.1.1.1"})
        again = c.post("/api/predict", files=png(), headers={"x-forwarded-for": "1.1.1.1"})
        other = c.post("/api/predict", files=png(), headers={"x-forwarded-for": "2.2.2.2"})
        assert (first.status_code, again.status_code, other.status_code) == (200, 429, 200)


def test_forged_forwarded_header_does_not_dodge_the_limit_without_a_proxy():
    with limited_client(rate_per_min=1) as c:
        assert (
            c.post("/api/predict", files=png(), headers={"x-forwarded-for": "1.1.1.1"}).status_code
            == 200
        )
        assert (
            c.post("/api/predict", files=png(), headers={"x-forwarded-for": "2.2.2.2"}).status_code
            == 429
        )


def test_docs_can_be_switched_off_and_headers_are_set(client):
    assert client.get("/api/docs").status_code == 200
    assert "permissions-policy" in client.get("/").headers
    assert client.get("/").headers["referrer-policy"] == "no-referrer"
    with limited_client(docs=False) as c:
        assert c.get("/api/docs").status_code == 404
        assert c.get("/api/openapi.json").status_code == 404
