"""Build leaf masks for the background augmentation from PlantVillage's own segmented images.

The segmented variant has the background removed (black). It is used ONLY to cut out leaf
shapes for the background-replacement augmentation. Training, validation and test images
stay the colour originals.

    python -m ml.make_masks               # masks for the training split (all that is needed)
    python -m ml.make_masks --force       # rebuild existing masks

Needs the cached data.zip: run `$env:HF_HOME = "E:\\hf_cache"` first in a new terminal.
"""

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from ml.config import ROOT, load_config

DARK_THRESHOLD = 20  # background pixels in the segmented images are (almost) pure black
SUFFIX = "_final_masked"
UUID_PREFIX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}___")


def segmented_to_mask(image: Image.Image, dark_threshold: int = DARK_THRESHOLD) -> Image.Image:
    """Greyscale mask (255 = leaf) from a segmented image, cleaned of JPEG speckle."""
    arr = np.asarray(image.convert("RGB"))
    mask = Image.fromarray(((arr.max(axis=2) > dark_threshold) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MedianFilter(5))
    return mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))  # closing


def index_segmented(zf: zipfile.ZipFile) -> dict[tuple[str, str], str]:
    """(class folder, file stem, lower-case) -> zip member name.

    Strips '_final_masked' and, for classes where segmented names carry one (corn rust),
    a leading '<uuid>___' that the colour file names don't have.
    """
    index = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        if parts and parts[0] == "raw":
            parts = parts[1:]
        if len(parts) != 3 or parts[0] != "segmented":
            continue
        stem = UUID_PREFIX.sub("", PurePosixPath(parts[2]).stem.lower()).removesuffix(SUFFIX)
        index[(parts[1], stem)] = info.filename
    return index


def cached_zip(cfg) -> Path:
    from huggingface_hub import hf_hub_download

    d = cfg.dataset
    try:
        found = hf_hub_download(
            d["hf_repo"], d["data_file"], repo_type="dataset", local_files_only=True
        )
    except Exception as exc:
        raise SystemExit(
            "data.zip is not in the Hugging Face cache. In this terminal run\n"
            '    $env:HF_HOME = "E:\\hf_cache"\nand try again.'
        ) from exc
    return Path(found)


def main() -> int:
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()

    splits = ROOT / cfg.paths["splits_dir"]
    manifest = pd.concat([pd.read_csv(splits / f"{s}.csv") for s in args.splits])
    mask_root = ROOT / cfg.dataset["mask_dir"]

    written, skipped, fractions = 0, 0, []
    missing: list[str] = []
    sizes: Counter = Counter()
    with zipfile.ZipFile(cached_zip(cfg)) as zf:
        segmented = index_segmented(zf)
        print(f"{len(segmented)} segmented images found in the zip")
        for row in tqdm(manifest.itertuples(index=False), total=len(manifest), unit="mask"):
            dest = mask_root / Path(row.path).with_suffix(".png")
            if dest.exists() and not args.force:
                skipped += 1
                continue
            folder, file_name = row.source_file.split("/", 1)
            member = segmented.get((folder, PurePosixPath(file_name).stem.lower()))
            if member is None:
                missing.append(row.source_file)
                continue
            with zf.open(member) as handle:
                image = Image.open(handle)
                image.load()
            mask = segmented_to_mask(image)
            dest.parent.mkdir(parents=True, exist_ok=True)
            mask.save(dest, optimize=True)
            written += 1
            sizes[str(image.size)] += 1
            fractions.append(float(np.asarray(mask).mean()) / 255.0)

    report = {
        "splits": args.splits,
        "rows": len(manifest),
        "written": written,
        "skipped_existing": skipped,
        "missing_segmented": len(missing),
        "missing_examples": missing[:5],
        "image_sizes": dict(sizes),
    }
    if fractions:
        q = np.percentile(fractions, [0, 1, 5, 50, 95, 100]).round(3).tolist()
        report["leaf_fraction_percentiles_0_1_5_50_95_100"] = q
        report["masks_outside_5_to_97_percent"] = int(sum(f < 0.05 or f > 0.97 for f in fractions))
    out = ROOT / cfg.paths["reports_dir"] / "mask_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if missing:
        print("Some images have no mask and will simply not get a background replacement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
