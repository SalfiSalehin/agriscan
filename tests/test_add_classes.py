import yaml

from ml import add_classes as ac
from ml.config import Config

BASE = """classes:
  - { id: a, crop: A, disease: X, raw_name: "r" }
  - { id: b, crop: B, disease: Y, raw_name: "s" }


dataset:
  x: 1
"""


def test_inserts_after_last_class_and_is_idempotent():
    text, added = ac.add_to_config_text(BASE)
    assert len(added) == 12
    lines = text.splitlines()
    assert lines[3].lstrip().startswith("- { id: rice_brown_spot")
    assert "dataset:" in text
    again, added_again = ac.add_to_config_text(text)
    assert added_again == [] and again == text
    ids = [c["id"] for c in yaml.safe_load(text)["classes"]]
    assert ids[:2] == ["a", "b"] and len(ids) == len(set(ids)) == 14


def test_new_entries_load_with_the_real_schema():
    text, _ = ac.add_to_config_text(BASE)
    data = yaml.safe_load(text)
    assert all(c["raw_name"] == "" for c in data["classes"][2:])
    assert {c["crop"] for c in data["classes"][2:]} == {"Rice", "Eggplant"}
    assert Config.model_fields["classes"]


def test_extend_index_keeps_old_numbers():
    out = ac.extend_index({"a": 0, "b": 1}, ["a", "b", "c", "d"])
    assert out == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert ac.extend_index(out, ["a", "b", "c", "d"]) == out
