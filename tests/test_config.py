import pytest

from ml.config import Config, load_config


def test_config_loads_19_classes():
    cfg = load_config()
    assert len(cfg.classes) == 19
    assert len(set(cfg.class_ids)) == 19


def test_bad_split_rejected():
    cfg = load_config().model_dump()
    cfg["data"]["split"] = {"train": 0.8, "val": 0.15, "test": 0.15}
    with pytest.raises(ValueError):
        Config(**cfg)
