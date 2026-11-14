import copy

import pytest

from app.content import Content, ContentError, localize
from app.settings import DEFAULT_CONTENT

REAL_CLASS_IDS = [
    "corn_common_rust", "corn_gray_leaf_spot", "corn_healthy", "corn_northern_leaf_blight",
    "pepper_bacterial_spot", "pepper_healthy", "potato_early_blight", "potato_healthy",
    "potato_late_blight", "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy",
    "tomato_late_blight", "tomato_leaf_mold", "tomato_mosaic_virus", "tomato_septoria",
    "tomato_spider_mites", "tomato_target_spot", "tomato_yellow_leaf_curl",
]  # fmt: skip


@pytest.fixture(scope="module")
def content():
    return Content.load(DEFAULT_CONTENT)


def test_shipped_content_covers_every_class_in_both_languages(content):
    content.validate(REAL_CLASS_IDS)  # raises on any gap
    for class_id in REAL_CLASS_IDS:
        for lang in ("en", "bn"):
            assert content.disease(class_id, lang)["name"]


def test_bangla_pages_contain_bangla_script(content):
    def has_bangla(text):
        return any("ঀ" <= ch <= "৿" for ch in text)

    for class_id in REAL_CLASS_IDS:
        page = content.disease(class_id, "bn")
        assert has_bangla(page["name"]) and has_bangla(page["disclaimer"])
        assert not has_bangla(content.disease(class_id, "en")["name"])


def test_disease_pages_carry_notes_and_healthy_pages_do_not_recommend_sprays(content):
    page = content.disease("tomato_late_blight", "en")
    assert page["kind"] == "disease" and page["actions"] and page["sources"]
    assert "does not recommend any product" in page["chemical_note"]
    assert page["helpline"]["verified"] is False
    healthy = content.disease("tomato_healthy", "en")
    assert healthy["kind"] == "healthy" and "chemical_note" not in healthy


def test_no_product_names_or_doses_in_the_text(content):
    banned = ["mancozeb", "chlorothalonil", "copper oxychloride", "metalaxyl", "ml/l", "g/l"]
    text = str(content.doc).lower()
    assert [word for word in banned if word in text] == []


def test_validate_reports_gaps(content):
    with pytest.raises(ContentError, match="no entry for class mango_rust"):
        content.validate(["mango_rust"])
    broken = copy.deepcopy(content.doc)
    broken["diseases"]["tomato_late_blight"]["name"]["bn"] = " "
    with pytest.raises(ContentError, match="empty text"):
        Content(broken).validate(REAL_CLASS_IDS)
    broken = copy.deepcopy(content.doc)
    broken["diseases"]["tomato_late_blight"]["sources"][0]["url"] = "http://x.example"
    with pytest.raises(ContentError, match="not https"):
        Content(broken).validate(REAL_CLASS_IDS)
    broken = copy.deepcopy(content.doc)
    broken["diseases"]["corn_common_rust"]["sources"] = []
    with pytest.raises(ContentError, match="no sources"):
        Content(broken).validate(REAL_CLASS_IDS)


def test_load_errors(tmp_path):
    with pytest.raises(ContentError, match="not found"):
        Content.load(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ContentError, match="not valid JSON"):
        Content.load(bad)


def test_unknown_class_and_unsure_block(content):
    assert content.disease("mango_rust", "en") is None and content.name("mango_rust", "en") is None
    block = content.unsure("bn")
    assert block["kind"] == "unsure" and block["tips"] and block["disclaimer"]


def test_localize_leaves_other_values_alone():
    value = {"a": {"en": "x", "bn": "y"}, "n": 3, "l": [{"en": "p", "bn": "q"}]}
    assert localize(value, "bn") == {"a": "y", "n": 3, "l": ["q"]}
