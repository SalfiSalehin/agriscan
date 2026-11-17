"""Rice / eggplant import: folder mapping, de-duplication and the grouped train/test split."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml import import_crops as ic
from ml.analysis import assign_use, collect_external_rows


def _photo(path, seed, size=(120, 90)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (9, 12, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base).resize(size, Image.NEAREST).save(path)


def test_normalise_and_parse_index():
    assert ic.normalise("Rice Turngo") == "riceturngo"
    assert ic.normalise("Browon Spot") == "browonspot"
    assert ic.parse_index("Aphids (12)") == ("aphids", 12)
    assert ic.parse_index("bs_wb_7") == ("bs_wb", 7)
    assert ic.parse_index("plain") == ("plain", None)


def test_group_key_puts_neighbours_together():
    a = ic.group_key("c", "raw", "Leaf (3)", 8)
    b = ic.group_key("c", "raw", "Leaf (7)", 8)
    c = ic.group_key("c", "raw", "Leaf (9)", 8)
    assert a == b != c


def test_link_groups_merges_similar_photos_only():
    hashes = np.zeros((3, 32), dtype=np.uint8)
    hashes[2] = 255  # completely different
    out = ic.link_groups(["a", "b", "c"], hashes)
    assert out[0] == out[1] != out[2]


def test_assign_split_keeps_groups_whole_and_leaves_train():
    classes = ["x"] * 10 + ["y"] * 3
    groups = [f"g{i // 2}" for i in range(10)] + ["solo"] * 3
    split = ic.assign_split(classes, groups, 0.2)
    for g in set(groups):
        assert len({s for s, gg in zip(split, groups, strict=True) if gg == g}) == 1
    assert "train" in split[:10] and "test" in split[:10]
    assert set(split[10:]) == {"train"}  # a single group is never held out


def test_assign_split_is_stable():
    classes = ["x"] * 20
    groups = [f"g{i}" for i in range(20)]
    assert ic.assign_split(classes, groups, 0.2) == ic.assign_split(classes, groups, 0.2)


def test_unknown_folder_stops_the_import(tmp_path):
    _photo(tmp_path / "Dhan-Shomadhan" / "Field" / "Mystery" / "a.jpg", 1)
    with pytest.raises(SystemExit, match="no mapping"):
        ic.locate_images(tmp_path, "rice")


def test_eggplant_reads_only_raw_and_skips_unmodelled_folders(tmp_path):
    _photo(tmp_path / "Brinjal" / "RawDataset" / "Healthy Leaf" / "h (1).jpg", 1)
    _photo(tmp_path / "Brinjal" / "RawDataset" / "Aphids" / "a (1).jpg", 2)
    _photo(tmp_path / "Brinjal" / "AugmentedDataset" / "Healthy Leaf" / "h (1).jpg", 3)
    items, report = ic.locate_images(tmp_path, "eggplant")
    assert [(p.name, c) for p, c, _ in items] == [("h (1).jpg", "eggplant_healthy")]
    assert report["skipped_folders"] == {"Aphids": 1}


def test_rice_spelling_mistakes_map_to_the_right_class(tmp_path):
    for i, folder in enumerate(["Browon Spot", "Rice Turngo", "Shath Blight", "Leaf Scaled"]):
        _photo(tmp_path / "Dhan-Shomadhan" / "Field Background" / folder / f"{i}.jpg", i)
    items, _ = ic.locate_images(tmp_path, "rice")
    assert {c for _, c, _ in items} == {
        "rice_brown_spot",
        "rice_tungro",
        "rice_sheath_blight",
        "rice_leaf_scald",
    }


def test_import_source_dedupes_resizes_and_splits_by_group(tmp_path):
    folder = tmp_path / "Brinjal" / "RawDataset" / "Powdery Mildew"
    for i in range(1, 41):
        _photo(folder / f"pm ({i}).jpg", i)
    (folder / "copy.jpg").write_bytes((folder / "pm (1).jpg").read_bytes())  # exact duplicate
    tmp = tmp_path / "out"
    rows, report = ic.import_source(tmp_path, "eggplant", tmp, 0.25, block=4)
    assert report["skipped"]["duplicate"] == 1
    assert len(rows) == 40
    assert len(list(tmp.iterdir())) == 40
    frame = pd.DataFrame(rows)
    by_group = frame.groupby("group")["source_split"].nunique()
    assert (by_group == 1).all()
    assert set(frame["source_split"]) == {"train", "test"}
    assert frame["class_id"].eq("eggplant_powdery_mildew").all()
    assert frame["license"].eq("CC BY 4.0").all()


def test_group_column_reaches_assign_use_so_a_group_stays_on_one_side(tmp_path):
    rows = []
    for g in range(30):
        for k in range(4):
            rows.append(
                {
                    "path": f"p{g}_{k}.jpg",
                    "role": "field",
                    "class_id": "c",
                    "source": "s",
                    "source_split": "train",
                    "mapping": "exact",
                    "sha256": f"{(g * 4 + k) * 7919:032x}",
                    "group": ic.group_id(f"grp{g}"),
                }
            )
    frame = pd.DataFrame(rows)
    use = assign_use(frame, 0.3)
    frame["use"] = use
    assert (frame.groupby("group")["use"].nunique() == 1).all()
    assert set(use) == {"train", "val"}


def test_manifest_group_is_read_back(tmp_path):
    (tmp_path / "data" / "field" / "c").mkdir(parents=True)
    _photo(tmp_path / "data" / "field" / "c" / "a.jpg", 1)
    manifest = pd.DataFrame(
        [
            {
                "path": "data/field/c/a.jpg",
                "role": "field",
                "class_id": "c",
                "source": "s",
                "source_class": "x",
                "source_split": "train",
                "mapping": "exact",
                "license": "l",
                "sha256": "ab" * 16,
                "width": 1,
                "height": 1,
                "group": "g1",
            }
        ]
    )
    manifest.to_csv(tmp_path / "m.csv", index=False)
    frame, _ = collect_external_rows(
        tmp_path, tmp_path / "m.csv", tmp_path / "data" / "field", tmp_path / "nl", {"c"}
    )
    assert frame["group"].tolist() == ["g1"]


def test_lookalike_letters_match_by_similarity_but_unrelated_names_still_stop(tmp_path):
    _photo(tmp_path / "Dhan-Shomadhan" / "Field" / "Rice Tur\u03b7go" / "a.jpg", 1)  # Greek eta
    items, report = ic.locate_images(tmp_path, "rice")
    assert [c for _, c, _ in items] == ["rice_tungro"]
    assert report["matched_by_similarity"]
    _photo(tmp_path / "Dhan-Shomadhan" / "Field" / "Weeds" / "b.jpg", 2)
    with pytest.raises(SystemExit):
        ic.locate_images(tmp_path, "rice")
