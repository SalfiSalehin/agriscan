import pandas as pd
import pytest

from ml.config import load_config
from ml.prepare_data import build_label_map, check_leakage, leaf_id_for, normalise, split_groups

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _fake_df() -> pd.DataFrame:
    rows = []
    for cid in ("a", "b"):
        for leaf in range(100):
            for img in range(1 + leaf % 4):
                rows.append(
                    {"class_id": cid, "group_id": f"{cid}|{leaf}", "sha1": f"{cid}{leaf}{img}"}
                )
    return pd.DataFrame(rows)


def test_normalise_matches_mirror_spellings():
    assert normalise("Corn_(maize)___Common_rust_") == normalise("Corn___Common_rust")
    assert normalise("Pepper,_bell___healthy") == normalise("Pepper__bell___healthy")


def test_label_map_covers_all_classes():
    cfg = load_config()
    names = [c.raw_name for c in cfg.classes] + ["Apple___healthy"]
    mapping = build_label_map(names, cfg.classes)
    assert sorted(mapping.values()) == sorted(cfg.class_ids)


def test_label_map_fails_loudly_when_class_missing():
    with pytest.raises(SystemExit):
        build_label_map(["Apple___healthy"], load_config().classes)


def test_split_has_no_leaf_overlap_and_is_stratified():
    df = _fake_df()
    df["split"] = split_groups(df, RATIOS, seed=42)
    assert df.groupby("group_id")["split"].nunique().max() == 1
    assert check_leakage(df)["passed"]
    for _, part in df.groupby("class_id"):
        share = part["split"].value_counts(normalize=True)
        for split, ratio in RATIOS.items():
            assert abs(share[split] - ratio) < 0.03


def test_split_is_deterministic():
    df = _fake_df()
    first = split_groups(df, RATIOS, seed=42)
    second = split_groups(df.sample(frac=1, random_state=1), RATIOS, seed=42)
    assert first.equals(second.reindex(first.index))


def test_leak_check_detects_overlap():
    df = pd.DataFrame({"group_id": ["g1", "g1"], "sha1": ["a", "b"], "split": ["train", "test"]})
    assert not check_leakage(df)["passed"]


def test_leaf_id_uses_map_then_falls_back():
    leaf_map = {
        "frec_scab 3335": ["Apple___Apple_scab_L1"],
        "dup 1": ["Other_x", "Tomato___Early_blight_y"],
    }
    apple = "Apple___Apple_scab"
    tomato = "Tomato___Early_blight"
    assert leaf_id_for("00075aa8___FREC_Scab 3335.JPG", apple, leaf_map) == (
        "Apple___Apple_scab_L1",
        "leaf_map",
    )
    assert leaf_id_for("x___Dup 1.JPG", tomato, leaf_map) == ("Tomato___Early_blight_y", "leaf_map")
    assert leaf_id_for("x___Nope 9.JPG", tomato, leaf_map) == ("fallback_Nope 9", "fallback")
