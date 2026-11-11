"""Per-class mask coverage, plus the file names where masks are missing.

    python -m ml.mask_coverage

Needs `$env:HF_HOME = "E:\\hf_cache"` in the terminal (it peeks into the cached zip).
"""

import zipfile
from pathlib import Path

import pandas as pd

from ml.config import ROOT, load_config
from ml.make_masks import cached_zip


def main() -> None:
    cfg = load_config()
    train = pd.read_csv(ROOT / cfg.paths["splits_dir"] / "train.csv")
    mask_root = ROOT / cfg.dataset["mask_dir"]
    train["has_mask"] = [(mask_root / Path(p).with_suffix(".png")).exists() for p in train["path"]]
    table = train.groupby("class_id")["has_mask"].agg(["sum", "count"])
    table["coverage"] = (table["sum"] / table["count"]).round(3)
    print(table.to_string())

    gaps = train[~train["has_mask"]]
    if gaps.empty:
        print("No gaps.")
        return
    worst = gaps["class_id"].value_counts().index[0]
    folder = gaps[gaps["class_id"] == worst]["source_file"].iloc[0].split("/")[0]
    print(f"\nMost gaps: {worst} ({(gaps['class_id'] == worst).sum()} images)")
    print("Missing, e.g.:", *gaps[gaps["class_id"] == worst]["source_file"].head(3), sep="\n  ")
    with zipfile.ZipFile(cached_zip(cfg)) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
    for variant in ("segmented", "color"):
        found = sorted(n for n in names if n.startswith(f"{variant}/{folder}/"))
        print(f"\n{variant}/{folder}: {len(found)} files, e.g.:", *found[:4], sep="\n  ")


if __name__ == "__main__":
    main()
