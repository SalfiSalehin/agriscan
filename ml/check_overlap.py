"""Find PlantDoc test photos that also appear (nearly identically) among the training photos.

    python -m ml.check_overlap

Web-scraped datasets often contain the same photo twice, sometimes in both the train and the test
folder. If the model trains on one copy, scoring it on the other copy would flatter the result.
This writes data/field_test_overlap.csv; ml.evaluate and ml.train then leave those test images
out ("excluded"). Run it once after ml.import_external, before training.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from PIL import Image

from ml.analysis import collect_external_rows
from ml.config import ROOT, load_config
from ml.import_external import DUPLICATE_BITS, dhash, hamming

OUTPUT = ROOT / "data" / "field_test_overlap.csv"


def find_overlaps(root, test_paths, train_paths, max_bits: int = DUPLICATE_BITS) -> pd.DataFrame:
    """Test images whose 256-bit dHash is within `max_bits` of some training image."""

    def hashes(paths):
        rows = []
        for p in paths:
            with Image.open(root / p) as im:
                rows.append(dhash(im))
        return np.stack(rows) if rows else np.zeros((0, 32), dtype=np.uint8)

    train_hashes = hashes(train_paths)
    found = []
    if len(train_hashes) == 0:
        return pd.DataFrame(columns=["path", "bits", "matches"])
    for path, one in zip(test_paths, hashes(test_paths), strict=True):
        distance = hamming(train_hashes, one)
        best = int(distance.argmin())
        if distance[best] <= max_bits:
            found.append({"path": path, "bits": int(distance[best]), "matches": train_paths[best]})
    return pd.DataFrame(found, columns=["path", "bits", "matches"])


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    external, warnings = collect_external_rows(
        ROOT,
        ROOT / "data" / "external_manifest.csv",
        ROOT / cfg.paths["field_dir"],
        ROOT / cfg.paths["nonleaf_dir"],
        cfg.class_ids,
    )
    for warning in warnings:
        print("WARNING:", warning)
    field = external[external["role"] == "field"]
    train = field[field["source_split"] == "train"]
    test = field[field["source_split"] != "train"]
    print(f"Comparing {len(test)} test photos against {len(train)} PlantDoc training photos ...")
    overlap = find_overlaps(ROOT, list(test["path"]), list(train["path"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(OUTPUT, index=False)
    print(f"{len(overlap)} test photo(s) are near-duplicates of a training photo.")
    if len(overlap):
        print(overlap.head(10).to_string(index=False))
    print(f"Wrote {OUTPUT}. These images are excluded from the field test set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
