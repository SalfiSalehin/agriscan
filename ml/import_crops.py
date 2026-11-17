"""Import rice and eggplant leaf photos as extra real-world training and test data.

    python -m ml.import_crops rice E:\\AgriScan\\data\\external\\rice_dhan_shomadhan --dry-run
    python -m ml.import_crops rice E:\\AgriScan\\data\\external\\rice_dhan_shomadhan
    python -m ml.import_crops eggplant E:\\AgriScan\\data\\external\\eggplant_brinjal

What it does
- Maps the dataset's class folders onto AgriScan class ids (folders for things we do not model,
  such as eggplant fruit, aphids or magnesium deficiency, are skipped on purpose). A folder name
  that is neither mapped nor on the skip list stops the import, so nothing is silently dropped.
- Eggplant: only the RawDataset folder is read. The AugmentedDataset folder holds edited copies of
  the same photos and would leak between training and test.
- Removes exact and near-duplicate photos inside the dataset.
- Stores upright copies (EXIF rotation applied), at most 1024 pixels on the long side, in
  data/field/<class id>/ and records them in data/external_manifest.csv.
- Splits into "train" and "test" BY GROUP, never by single photo. Photos with neighbouring file
  numbers and photos that look alike are put in one group, because the authors sometimes took
  several shots of the same leaf and such shots must not land on both sides of the split.
  The groups are then dealt out per class so every class keeps a test share of about 20%.
"""

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ml.config import ROOT, load_config
from ml.import_external import (
    DUPLICATE_BITS,
    IMAGE_SUFFIXES,
    MANIFEST_COLUMNS,
    dhash,
    hamming,
    sha256_of,
)

LINK_BITS = 40  # photos at most this far apart (of 256 bits) are treated as one leaf
MAX_SIDE = 1024
JPEG_QUALITY = 92

# Folder names are compared after dropping case, spaces and punctuation, so the datasets'
# own spelling mistakes ("Turngo", "Shath", "Browon", "Scaled") map onto the right class.
SOURCES: dict[str, dict] = {
    "rice": {
        "source": "Dhan-Shomadhan",
        "license": "CC BY 4.0",
        "classes": {
            "brownspot": "rice_brown_spot",
            "browonspot": "rice_brown_spot",
            "leafscaled": "rice_leaf_scald",
            "leafscald": "rice_leaf_scald",
            "riceblast": "rice_blast",
            "riceturngo": "rice_tungro",
            "ricetungro": "rice_tungro",
            "sheathblight": "rice_sheath_blight",
            "shathblight": "rice_sheath_blight",
        },
        "skipped": set(),
        "only_under": None,
    },
    "eggplant": {
        "source": "Brinjal-BD",
        "license": "CC BY 4.0",
        "classes": {
            "cercosporaleafspot": "eggplant_cercospora_leaf_spot",
            "earlyblight": "eggplant_early_blight",
            "healthyleaf": "eggplant_healthy",
            "mosaicvirus": "eggplant_mosaic_virus",
            "phomopsisblight": "eggplant_phomopsis_blight",
            "phytophthorablight": "eggplant_phytophthora_blight",
            "powderymildew": "eggplant_powdery_mildew",
        },
        "skipped": {"aphids", "defectfruit", "healthyfruit", "magnesiumdeficiency"},
        "only_under": "RawDataset",
    },
}


def normalise(name: str) -> str:
    """'Rice Turngo' -> 'riceturngo'."""
    return re.sub(r"[^a-z]", "", name.lower())


def resolve(key: str, spec: dict, fuzzy: dict) -> tuple[str | None, str | None]:
    """(class id, None), (None, 'skip') or (None, None). A name that is almost, but not exactly,
    a known one (typos, look-alike letters) is matched to the closest known name and noted in
    `fuzzy` so the report shows it."""
    if key in spec["classes"]:
        return spec["classes"][key], None
    if key in spec["skipped"]:
        return None, "skip"
    known = list(spec["classes"]) + list(spec["skipped"])
    close = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
    if not close:
        return None, None
    fuzzy[key] = close[0]
    return resolve(close[0], spec, {})


def parse_index(stem: str) -> tuple[str, int | None]:
    """('Aphids (12)') -> ('aphids', 12); ('bs_wb_7') -> ('bs_wb', 7); no number -> (stem, None)."""
    match = re.search(r"(\d+)\)?\s*$", stem)
    if not match:
        return stem.lower(), None
    return stem[: match.start()].strip(" _(").lower(), int(match.group(1))


def group_key(class_id: str, subset: str, stem: str, block: int) -> str:
    """Photos with neighbouring numbers in the same folder share a key."""
    prefix, number = parse_index(stem)
    if number is None:
        return f"{class_id}|{subset}|{stem.lower()}"
    return f"{class_id}|{subset}|{prefix}|{number // block}"


class _Union:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)  # deterministic root


def link_groups(keys: list[str], hashes: np.ndarray, bits: int = LINK_BITS) -> list[str]:
    """Merge keys whose photos look alike. Returns the merged group name for every photo."""
    union = _Union(set(keys))
    for i in range(len(keys) - 1):
        distance = hamming(hashes[i + 1 :], hashes[i])
        for offset in np.nonzero(distance <= bits)[0]:
            union.union(keys[i], keys[i + 1 + int(offset)])
    return [union.find(k) for k in keys]


def group_id(name: str) -> str:
    """Stable hex id of a group (the training code derives its validation split from it)."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]


def assign_split(classes: list[str], groups: list[str], test_fraction: float) -> list[str]:
    """'train' or 'test' for every photo. Whole groups go to the same side. Per class the groups
    are taken in a stable pseudo-random order into test until about `test_fraction` of its photos
    are there; at least one group always stays in train, so a class with a single group has
    nothing to hold out."""
    split = ["train"] * len(classes)
    by_class: dict[str, dict[str, list[int]]] = {}
    for i, (class_id, group) in enumerate(zip(classes, groups, strict=True)):
        by_class.setdefault(class_id, {}).setdefault(group, []).append(i)
    for members in by_class.values():
        total = sum(len(v) for v in members.values())
        wanted = math.ceil(test_fraction * total)
        ordered = sorted(members, key=group_id)
        taken = 0
        for group in ordered[:-1]:  # the last group in the order stays in train
            if taken >= wanted:
                break
            for i in members[group]:
                split[i] = "test"
            taken += len(members[group])
    return split


def locate_images(folder: Path, kind: str) -> tuple[list[tuple[Path, str, str]], dict]:
    """[(path, class id, subset)] plus a report of skipped folders. Stops on unknown folders."""
    spec = SOURCES[kind]
    base = folder
    if spec["only_under"]:
        found = sorted(p for p in folder.rglob("*") if p.is_dir() and p.name == spec["only_under"])
        if len(found) != 1:
            raise SystemExit(
                f"expected exactly one folder named {spec['only_under']} under {folder}, "
                f"found {len(found)}"
            )
        base = found[0]
    items, skipped, unmapped, fuzzy = [], {}, {}, {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = normalise(path.parent.name)
        subset = normalise(path.parent.parent.name) if kind == "rice" else "raw"
        class_id, action = resolve(key, spec, fuzzy)
        if class_id:
            items.append((path, class_id, subset))
        elif action == "skip":
            skipped[path.parent.name] = skipped.get(path.parent.name, 0) + 1
        else:
            unmapped[path.parent.name] = unmapped.get(path.parent.name, 0) + 1
    if unmapped:
        raise SystemExit(
            f"folders with no mapping (add them to SOURCES or the skip list): {unmapped}"
        )
    return items, {"skipped_folders": skipped, "matched_by_similarity": fuzzy}


def import_source(folder: Path, kind: str, tmp: Path | None, test_fraction=0.2, block=8):
    """Returns (manifest rows, report). Images are written to `tmp` unless it is None (dry run)."""
    spec = SOURCES[kind]
    items, report = locate_images(folder, kind)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
    accepted, seen_sha, rows = [], set(), []
    skipped = {"duplicate": 0, "unreadable": 0}
    for path, class_id, subset in items:
        try:
            digest = sha256_of(path)
            with Image.open(path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
        except Exception:
            skipped["unreadable"] += 1
            continue
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        bits = dhash(image)
        if digest in seen_sha or (
            accepted and hamming(np.stack(accepted), bits).min() <= DUPLICATE_BITS
        ):
            skipped["duplicate"] += 1
            continue
        seen_sha.add(digest)
        accepted.append(bits)
        name = f"{kind}_{digest[:12]}.jpg"
        if tmp is not None:
            image.save(tmp / name, "JPEG", quality=JPEG_QUALITY, optimize=True)
        rows.append(
            {
                "class_id": class_id,
                "source": spec["source"],
                "source_class": f"{subset}/{path.parent.name}",
                "mapping": "exact",
                "license": spec["license"],
                "sha256": digest,
                "width": image.width,
                "height": image.height,
                "_name": name,
                "_key": group_key(class_id, subset, path.stem, block),
                "_hash": bits,
            }
        )
    classes = [r["class_id"] for r in rows]
    merged = [""] * len(rows)
    for class_id in sorted(set(classes)):
        idx = [i for i, c in enumerate(classes) if c == class_id]
        names = link_groups(
            [rows[i]["_key"] for i in idx], np.stack([rows[i]["_hash"] for i in idx])
        )
        for i, name in zip(idx, names, strict=True):
            merged[i] = name
    for row, name, side in zip(
        rows, merged, assign_split(classes, merged, test_fraction), strict=True
    ):
        row["group"] = group_id(name)
        row["source_split"] = side
    report.update({"source": spec["source"], "imported": len(rows), "skipped": skipped})
    return rows, report


def summarise(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    table = frame.groupby(["class_id", "source_split"]).size().unstack(fill_value=0)
    table["groups"] = frame.groupby("class_id")["group"].nunique()
    return table


def write_outputs(rows, report, tmp: Path | None, kind: str, cfg, dry_run: bool) -> None:
    spec = SOURCES[kind]
    if rows:
        print(summarise(rows).to_string())
    print(json.dumps(report, indent=2))
    if dry_run:
        print("Dry run: nothing was copied or written.")
        return
    manifest_path = ROOT / "data" / "external_manifest.csv"
    old = (
        pd.read_csv(manifest_path, keep_default_na=False)
        if manifest_path.exists()
        else pd.DataFrame(columns=MANIFEST_COLUMNS)
    )
    for stale in old[old["source"] == spec["source"]]["path"]:  # earlier import of this source
        (ROOT / stale).unlink(missing_ok=True)
    field_dir = ROOT / cfg.paths["field_dir"]
    new_rows = []
    for row in rows:
        target = field_dir / row["class_id"] / row["_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / row["_name"]), target)
        new_rows.append(
            {
                "path": target.relative_to(ROOT).as_posix(),
                "role": "field",
                **{k: row[k] for k in MANIFEST_COLUMNS if k in row},
            }
        )
    shutil.rmtree(tmp, ignore_errors=True)
    frame = pd.DataFrame(new_rows).reindex(columns=MANIFEST_COLUMNS)
    kept = old[old["source"] != spec["source"]].reindex(columns=MANIFEST_COLUMNS)
    pd.concat([kept, frame], ignore_index=True).fillna("").to_csv(manifest_path, index=False)
    out = ROOT / cfg.paths["reports_dir"] / f"external_import_{kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(new_rows)} photos to {field_dir} and updated {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(SOURCES))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--block", type=int, default=8, help="neighbouring file numbers per group")
    args = parser.parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} is not a folder")
    cfg = load_config()
    wanted = {c for c in SOURCES[args.kind]["classes"].values()}
    missing = sorted(wanted - set(cfg.class_ids))
    if missing:
        raise SystemExit(f"add these classes to config/config.yaml first: {missing}")
    tmp = None if args.dry_run else ROOT / "data" / "_import_tmp" / args.kind
    rows, report = import_source(args.folder, args.kind, tmp, args.test_fraction, args.block)
    write_outputs(rows, report, tmp, args.kind, cfg, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
