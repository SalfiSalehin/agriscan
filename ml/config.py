"""Typed loader for config/config.yaml. Every other module imports from here."""

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


class ClassSpec(BaseModel):
    id: str
    crop: str
    disease: str
    raw_name: str


class Split(BaseModel):
    train: float
    val: float
    test: float

    @model_validator(mode="after")
    def _sums_to_one(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


class Config(BaseModel):
    project: dict
    paths: dict
    data: dict
    inference: dict
    classes: list[ClassSpec]

    @field_validator("classes")
    @classmethod
    def _unique_ids(cls, v):
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("class ids must be unique")
        return v

    @model_validator(mode="after")
    def _check_split(self):
        Split(**self.data["split"])
        return self

    @property
    def seed(self) -> int:
        return int(self.project["seed"])

    @property
    def class_ids(self) -> list[str]:
        return [c.id for c in self.classes]


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config(**yaml.safe_load(f))
