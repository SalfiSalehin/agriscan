from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from ml import import_external as ie

CLASSES = {
    "corn_common_rust",
    "corn_gray_leaf_spot",
    "corn_healthy",
    "corn_northern_leaf_blight",
    "pepper_bacterial_spot",
    "pepper_healthy",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria",
    "tomato_spider_mites",
    "tomato_target_spot",
    "tomato_yellow_leaf_curl",
}


def _noise(path, seed: int, size: int = 120, quality: int = 95):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=quality)
    return arr


def _dest(tmp_path):
    return {"field": tmp_path / "field", "unknown": tmp_path / "nonleaf", "rel": tmp_path}


def _tree(tmp_path):
    root = tmp_path / "plantdoc"
    for folder in ("Tomato_leaf", "Apple_leaf", "Mystery_leaf"):
        (root / "train" / folder).mkdir(parents=True)
    _noise(root / "train" / "Tomato_leaf" / "a.jpg", 1)
    _noise(root / "train" / "Tomato_leaf" / "b.jpg", 2)
    _noise(root / "train" / "Apple_leaf" / "c.jpg", 3)
    _noise(root / "train" / "Mystery_leaf" / "d.jpg", 4)
    return root


def test_mapping_roles_and_unmapped_folders(tmp_path):
    rows, report = ie.import_plantdoc(_tree(tmp_path), CLASSES, None, _dest(tmp_path))
    by_role = pd.DataFrame(rows).groupby("role").size().to_dict()
    assert by_role == {"field": 2, "unknown": 1}
    assert report["folders_without_mapping"] == ["Mystery_leaf"]
    field = [r for r in rows if r["role"] == "field"]
    assert {r["class_id"] for r in field} == {"tomato_healthy"}
    assert all(r["path"].startswith("field/tomato_healthy/") for r in field)


def test_exact_and_near_duplicates_inside_source_are_skipped(tmp_path):
    root = _tree(tmp_path)
    arr = _noise(root / "train" / "Tomato_leaf" / "x.jpg", 1)  # same picture as a.jpg
    shrunk = Image.fromarray(arr).resize((100, 100))
    shrunk.save(root / "train" / "Tomato_leaf" / "y.jpg", quality=80)  # re-encoded copy
    rows, report = ie.import_plantdoc(root, CLASSES, None, _dest(tmp_path))
    assert len(rows) == 3
    assert report["skipped"]["duplicate_in_source"] == 2


def test_duplicates_of_plantvillage_are_skipped(tmp_path):
    root = _tree(tmp_path)
    with Image.open(root / "train" / "Tomato_leaf" / "a.jpg") as im:
        pv = np.stack([ie.dhash(im)])
    rows, report = ie.import_plantdoc(root, CLASSES, pv, _dest(tmp_path))
    assert len(rows) == 2
    assert report["skipped"]["duplicate_of_plantvillage"] == 1


def test_write_outputs_copies_files_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    cfg = SimpleNamespace(paths={"reports_dir": "reports"})
    rows, report = ie.import_plantdoc(_tree(tmp_path), CLASSES, None, _dest(tmp_path))
    ie.write_outputs(rows, report, cfg, dry_run=False)
    manifest = pd.read_csv(tmp_path / "data" / "external_manifest.csv")
    assert list(manifest.columns) == ie.MANIFEST_COLUMNS
    assert len(manifest) == 3
    assert all((tmp_path / p).exists() for p in manifest["path"])
    assert set(manifest["license"]) == {"CC BY 4.0"}


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ie, "ROOT", tmp_path)
    cfg = SimpleNamespace(paths={"reports_dir": "reports"})
    rows, report = ie.import_plantdoc(_tree(tmp_path), CLASSES, None, _dest(tmp_path))
    ie.write_outputs(rows, report, cfg, dry_run=True)
    assert not (tmp_path / "field").exists()
    assert not (tmp_path / "data").exists()


def test_every_mapped_class_is_one_of_the_19_classes():
    mapped = {c for c, _ in ie.PLANTDOC_MAP.values() if c is not None}
    assert mapped <= CLASSES
