"""Import external real-world images (PlantDoc) as TEST-ONLY data.

    python -m ml.import_external plantdoc E:\\external\\plantdoc
    python -m ml.import_external plantdoc E:\\external\\plantdoc --dry-run

What it does
- Maps PlantDoc folders onto the 19 AgriScan classes (images of plants the model does not
  support go to the unknown set instead). Each row says whether the mapping is "exact" or
  "approximate" so that evaluation can report them separately.
- Skips exact duplicates and near-duplicates, both inside PlantDoc and against every
  PlantVillage image of the 19 classes (PlantDoc was scraped from the web, and PlantVillage
  photos are on the web too). The near-duplicate test cannot catch heavily cropped copies.
- Copies the original files (no re-encoding) into data/field/<class>/ and
  data/nonleaf/other_crop/, and records everything in data/external_manifest.csv.

These images are never used for training.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.config import ROOT, load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HASH_SIZE = 16  # 16x16 difference hash = 256 bits
DUPLICATE_BITS = 16  # at most this many differing bits counts as the same picture
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
MANIFEST_COLUMNS = [
    "path",
    "role",
    "class_id",
    "source",
    "source_class",
    "source_split",
    "mapping",
    "license",
    "sha256",
    "width",
    "height",
]

# PlantDoc folder -> (AgriScan class id or None, mapping quality)
# "exact": same crop and disease. "approximate": PlantDoc's name is broader than ours.
# None: a plant or crop the model does not support (goes to the unknown set).
PLANTDOC_MAP: dict[str, tuple[str | None, str]] = {
    "Bell_pepper_leaf": ("pepper_healthy", "exact"),
    "Bell_pepper_leaf_spot": ("pepper_bacterial_spot", "approximate"),
    "Corn_Gray_leaf_spot": ("corn_gray_leaf_spot", "exact"),
    "Corn_leaf_blight": ("corn_northern_leaf_blight", "approximate"),
    "Corn_rust_leaf": ("corn_common_rust", "approximate"),
    "Potato_leaf_early_blight": ("potato_early_blight", "exact"),
    "Potato_leaf_late_blight": ("potato_late_blight", "exact"),
    "Tomato_Early_blight_leaf": ("tomato_early_blight", "exact"),
    "Tomato_leaf": ("tomato_healthy", "exact"),
    "Tomato_leaf_bacterial_spot": ("tomato_bacterial_spot", "exact"),
    "Tomato_leaf_late_blight": ("tomato_late_blight", "exact"),
    "Tomato_leaf_mosaic_virus": ("tomato_mosaic_virus", "exact"),
    "Tomato_leaf_yellow_virus": ("tomato_yellow_leaf_curl", "approximate"),
    "Tomato_mold_leaf": ("tomato_leaf_mold", "exact"),
    "Tomato_Septoria_leaf_spot": ("tomato_septoria", "exact"),
    "Tomato_two_spotted_spider_mites_leaf": ("tomato_spider_mites", "exact"),
    "Apple_leaf": (None, "out_of_scope"),
    "Apple_rust_leaf": (None, "out_of_scope"),
    "Apple_Scab_Leaf": (None, "out_of_scope"),
    "Blueberry_leaf": (None, "out_of_scope"),
    "Cherry_leaf": (None, "out_of_scope"),
    "grape_leaf": (None, "out_of_scope"),
    "grape_leaf_black_rot": (None, "out_of_scope"),
    "Peach_leaf": (None, "out_of_scope"),
    "Raspberry_leaf": (None, "out_of_scope"),
    "Soyabean_leaf": (None, "out_of_scope"),
    "Squash_Powdery_mildew_leaf": (None, "out_of_scope"),
    "Strawberry_leaf": (None, "out_of_scope"),
}


def dhash(image: Image.Image) -> np.ndarray:
    """256-bit difference hash, packed into 32 bytes."""
    small = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    return np.packbits((arr[:, 1:] > arr[:, :-1]).ravel())


def hamming(hashes: np.ndarray, one: np.ndarray) -> np.ndarray:
    """Bit distance between each row of `hashes` (n, 32) and `one` (32,)."""
    return POPCOUNT[hashes ^ one].sum(axis=1, dtype=np.int32)


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plantvillage_hashes(cfg) -> np.ndarray:
    """Hashes of every exported PlantVillage image (train, val and test)."""
    from tqdm import tqdm

    splits = ROOT / cfg.paths["splits_dir"]
    root = ROOT / cfg.dataset["export_dir"]
    paths = []
    for name in ("train", "val", "test"):
        paths += [root / p for p in pd.read_csv(splits / f"{name}.csv")["path"]]
    if not paths or not paths[0].exists():
        raise SystemExit(
            "PlantVillage images not found. Run ml.prepare_data first (the duplicate check "
            "needs them), or pass --skip-plantvillage-check to ignore this (not recommended)."
        )
    rows = []
    for path in tqdm(paths, desc="hashing PlantVillage", unit="img"):
        with Image.open(path) as im:
            rows.append(dhash(im))
    return np.stack(rows)


def import_plantdoc(source: Path, class_ids: set[str], pv: np.ndarray | None, dest: dict):
    """Returns (manifest rows, report dict). `dest` holds target roots: field, unknown, rel."""
    unmapped = sorted(
        p.name
        for split in ("train", "test")
        if (source / split).is_dir()
        for p in (source / split).iterdir()
        if p.is_dir() and p.name not in PLANTDOC_MAP
    )
    bad_targets = [c for c, _ in PLANTDOC_MAP.values() if c is not None and c not in class_ids]
    if bad_targets:
        raise SystemExit(f"mapping points to unknown classes: {sorted(set(bad_targets))}")

    rows, seen_sha, accepted = [], set(), []
    skipped = {"duplicate_in_source": 0, "duplicate_of_plantvillage": 0, "unreadable": 0}
    examples: dict[str, list[str]] = {k: [] for k in skipped}

    def skip(reason: str, path: Path) -> None:
        skipped[reason] += 1
        if len(examples[reason]) < 5:
            examples[reason].append(str(path))

    for split in ("train", "test"):
        if not (source / split).is_dir():
            continue
        for folder in sorted(p for p in (source / split).iterdir() if p.is_dir()):
            if folder.name not in PLANTDOC_MAP:
                continue
            class_id, mapping = PLANTDOC_MAP[folder.name]
            for path in find_images(folder):
                try:
                    with Image.open(path) as im:
                        width, height = im.size
                        bits = dhash(im)
                    digest = sha256_of(path)
                except Exception:
                    skip("unreadable", path)
                    continue
                if digest in seen_sha or (
                    accepted and hamming(np.stack(accepted), bits).min() <= DUPLICATE_BITS
                ):
                    skip("duplicate_in_source", path)
                    continue
                if pv is not None and hamming(pv, bits).min() <= DUPLICATE_BITS:
                    skip("duplicate_of_plantvillage", path)
                    continue
                seen_sha.add(digest)
                accepted.append(bits)
                if class_id is None:
                    role, target = "unknown", dest["unknown"] / "other_crop"
                else:
                    role, target = "field", dest["field"] / class_id
                name = f"plantdoc_{digest[:12]}{path.suffix.lower()}"
                rows.append(
                    {
                        "path": (target / name).relative_to(dest["rel"]).as_posix(),
                        "role": role,
                        "class_id": class_id or "",
                        "source": "PlantDoc",
                        "source_class": folder.name,
                        "source_split": split,
                        "mapping": mapping,
                        "license": "CC BY 4.0",
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "_from": path,
                        "_to": target / name,
                    }
                )
    report = {
        "source": "PlantDoc",
        "imported": len(rows),
        "skipped": skipped,
        "skipped_examples": examples,
        "folders_without_mapping": unmapped,
    }
    return rows, report


def write_outputs(rows: list[dict], report: dict, cfg, dry_run: bool) -> None:
    if not dry_run:
        for row in rows:
            row["_to"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["_from"], row["_to"])
        manifest_path = ROOT / "data" / "external_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows).reindex(columns=MANIFEST_COLUMNS)
        if manifest_path.exists():
            old = pd.read_csv(manifest_path)
            new = pd.concat([old[old["source"] != "PlantDoc"], new], ignore_index=True)
        new.to_csv(manifest_path, index=False)
        out = ROOT / cfg.paths["reports_dir"] / "external_import_plantdoc.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        table = frame.groupby(["role", "class_id", "mapping"]).size().rename("images")
        print(table.to_string())
    print(json.dumps(report, indent=2))
    if dry_run:
        print("Dry run: nothing was copied or written.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["plantdoc"])
    parser.add_argument("folder", type=Path, help="folder with PlantDoc's train/ and test/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plantvillage-check", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a folder")
    pv = None if args.skip_plantvillage_check else plantvillage_hashes(cfg)
    dest = {
        "field": ROOT / cfg.paths["field_dir"],
        "unknown": ROOT / cfg.paths["nonleaf_dir"],
        "rel": ROOT,
    }
    rows, report = import_plantdoc(args.folder, set(cfg.class_ids), pv, dest)
    write_outputs(rows, report, cfg, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
