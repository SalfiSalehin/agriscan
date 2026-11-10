"""Phase 1 data pipeline for AgriScan.

    python -m ml.prepare_data --inspect   # look at the dataset, change nothing
    python -m ml.prepare_data             # export images, split by leaf, write manifests
    python -m ml.prepare_data --verify    # re-check the saved manifests for leakage

Reads the PlantVillage zip straight from the Hugging Face cache (no `datasets` loader) and
rebuilds leaf_id with the same lookup rule as the dataset's own loader script.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd
from tqdm import tqdm

from ml.config import ROOT, load_config

SPLITS = ("train", "val", "test")
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
CROP_PREFIXES = ("potato", "tomato", "corn", "pepper")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def normalise(name: str) -> str:
    """Make label spellings from different PlantVillage mirrors comparable."""
    name = re.sub(r"\([^)]*\)", "", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_labels(label_names, classes) -> dict[str, str]:
    """Class folder name in the dataset -> our class id (only for the classes we want)."""
    wanted = {normalise(c.raw_name): c.id for c in classes}
    return {n: wanted[normalise(n)] for n in label_names if normalise(n) in wanted}


def build_label_map(label_names, classes) -> dict[str, str]:
    mapping = match_labels(label_names, classes)
    missing = sorted({c.id for c in classes} - set(mapping.values()))
    if missing:
        hints = [n for n in label_names if normalise(n).startswith(CROP_PREFIXES)]
        raise SystemExit(
            f"No dataset folder matched these class ids: {missing}\n"
            f"Dataset folders for our crops: {hints}\n"
            "Fix raw_name in config/config.yaml and run again."
        )
    return mapping


def fetch_files(cfg) -> tuple[Path, dict]:
    """Download (or reuse from the cache) the zip and the leaf map."""
    from huggingface_hub import hf_hub_download

    d = cfg.dataset
    zip_path = Path(hf_hub_download(d["hf_repo"], d["data_file"], repo_type="dataset"))
    map_path = Path(hf_hub_download(d["hf_repo"], d["leaf_map_file"], repo_type="dataset"))
    leaf_map = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(leaf_map, dict):
        raise SystemExit("leaf-map.json is not a JSON object. Paste me its first 300 characters.")
    return zip_path, leaf_map


def list_images(zf: zipfile.ZipFile, variant: str) -> list[tuple[str, str, str]]:
    """(zip member, class folder, file name) for every image of the chosen variant."""
    found = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        if parts and parts[0] == "raw":
            parts = parts[1:]
        if len(parts) != 3 or parts[0] != variant:
            continue
        if PurePosixPath(parts[2]).suffix.lower() in IMAGE_SUFFIXES:
            found.append((info.filename, parts[1], parts[2]))
    return found


def leaf_id_for(file_name: str, class_folder: str, leaf_map: dict) -> tuple[str, str]:
    """Same rule as the dataset's own loader: look the image up in leaf-map.json.

    Returns (leaf_id, source) where source is "leaf_map" or "fallback". A fallback id is
    one group per image, so it cannot prove that photos of one leaf were kept together.
    """
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for suffix in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(suffix, "")
    ident = ident.strip()

    suggestions = leaf_map.get(ident.lower())
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if suggestions:
        if len(suggestions) == 1:
            return suggestions[0], "leaf_map"
        for suggestion in suggestions:
            if class_folder in suggestion:
                return suggestion, "leaf_map"
    return f"fallback_{ident}", "fallback"


def build_catalogue(entries, label_map: dict[str, str], leaf_map: dict) -> pd.DataFrame:
    rows = []
    for member, folder, file_name in entries:
        if folder not in label_map:
            continue
        leaf_id, source = leaf_id_for(file_name, folder, leaf_map)
        rows.append(
            {
                "zip_name": member,
                "class_id": label_map[folder],
                "leaf_id": leaf_id,
                "leaf_id_source": source,
                "source_file": f"{folder}/{file_name}",
            }
        )
    df = pd.DataFrame(rows).sort_values("zip_name").reset_index(drop=True)
    # prefix with the class so two classes can never share a group
    df["group_id"] = df["class_id"] + "|" + df["leaf_id"]
    return df


def inspect_dataset(cfg) -> None:
    variant = cfg.dataset["variant"]
    zip_path, leaf_map = fetch_files(cfg)
    print(f"zip: {zip_path}")
    print(f"leaf-map.json: {len(leaf_map)} keys; first entries: {list(leaf_map.items())[:3]}")
    with zipfile.ZipFile(zip_path) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        top = Counter(PurePosixPath(n).parts[0] for n in names)
        print(f"zip top-level folders (files in each): {dict(top)}")
        entries = list_images(zf, variant)

    folders = sorted({folder for _, folder, _ in entries})
    mapping = match_labels(folders, cfg.classes)
    print(f"\n{len(folders)} class folders in '{variant}' (OK = one of our classes):")
    for name in folders:
        print(f"  {'OK' if name in mapping else '--'} {name}")
    missing = sorted({c.id for c in cfg.classes} - set(mapping.values()))
    print(f"\nMatched {len(set(mapping.values()))}/{len(cfg.classes)} classes. Missing: {missing}")
    if not mapping:
        return

    df = build_catalogue(entries, mapping, leaf_map)
    df["fallback"] = df["leaf_id_source"] == "fallback"
    stats = df.groupby("class_id").agg(
        images=("leaf_id", "size"),
        leaves=("leaf_id", "nunique"),
        fallback_share=("fallback", "mean"),
    )
    stats["images_per_leaf"] = (stats["images"] / stats["leaves"]).round(2)
    stats["fallback_share"] = stats["fallback_share"].round(3)
    print(f"\nSelected {len(df)} images, {df['leaf_id'].nunique()} distinct leaf_id values")
    print(stats.to_string())
    print(f"\nleaf_id from the fallback rule: {df['fallback'].mean():.1%} of selected images")
    print("\nExamples:")
    print(df[["source_file", "leaf_id", "leaf_id_source"]].head(4).to_string(index=False))


def export_images(zf: zipfile.ZipFile, catalogue: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rel_paths, hashes = [], []
    for i, member in enumerate(tqdm(catalogue["zip_name"], desc="exporting", unit="img")):
        data = zf.read(member)
        class_id = catalogue.at[i, "class_id"]
        ext = ".jpg" if data[:2] == b"\xff\xd8" else ".png"
        rel = Path(class_id) / f"{i:06d}{ext}"
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        rel_paths.append(rel.as_posix())
        hashes.append(hashlib.sha1(data).hexdigest())
    return catalogue.assign(path=rel_paths, sha1=hashes)


def split_groups(df: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    """Leaf-grouped, per-class stratified split.

    Per class: shuffle the leaf groups (seeded), then give each group to whichever split is
    furthest below its image-count target. A leaf never lands in two splits.
    """
    assignment = {}
    for class_id, part in df.groupby("class_id"):
        sizes = part.groupby("group_id").size().sort_index()  # sorted => deterministic
        groups = list(sizes.index)
        random.Random(f"{seed}-{class_id}").shuffle(groups)
        target = {s: ratios[s] * int(sizes.sum()) for s in SPLITS}
        have = dict.fromkeys(SPLITS, 0)
        for group in groups:
            split = max(SPLITS, key=lambda s: target[s] - have[s])
            assignment[group] = split
            have[split] += int(sizes[group])
    return df["group_id"].map(assignment)


def check_leakage(df: pd.DataFrame) -> dict:
    groups = {s: set(df.loc[df["split"] == s, "group_id"]) for s in SPLITS}
    files = {s: set(df.loc[df["split"] == s, "sha1"]) for s in SPLITS}
    group_overlap = {f"{a}&{b}": len(groups[a] & groups[b]) for a, b in PAIRS}
    file_overlap = {f"{a}&{b}": len(files[a] & files[b]) for a, b in PAIRS}
    return {
        "leaf_group_overlap": group_overlap,
        "identical_file_overlap": file_overlap,
        "passed": all(v == 0 for v in group_overlap.values()),
    }


def class_counts(df: pd.DataFrame, class_ids: list[str]) -> pd.DataFrame:
    counts = pd.crosstab(df["class_id"], df["split"]).reindex(columns=list(SPLITS), fill_value=0)
    counts = counts.reindex(class_ids, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    return counts


def plot_distribution(counts: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = counts[list(SPLITS)].plot(kind="barh", stacked=True, figsize=(9, 7))
    ax.set_xlabel("images")
    ax.set_title("PlantVillage (color): selected classes by split")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def write_outputs(cfg, df: pd.DataFrame, leak: dict) -> pd.DataFrame:
    splits_dir = ROOT / cfg.paths["splits_dir"]
    reports_dir = ROOT / cfg.paths["reports_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    cols = ["path", "class_id", "group_id", "leaf_id_source", "source_file", "sha1"]
    for split in SPLITS:
        part = df[df["split"] == split].sort_values("path")
        part[cols].to_csv(splits_dir / f"{split}.csv", index=False)
    index = {cid: i for i, cid in enumerate(cfg.class_ids)}
    (splits_dir / "class_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    counts = class_counts(df, cfg.class_ids)
    counts.to_csv(reports_dir / "class_counts.csv")
    plot_distribution(counts, reports_dir / "figures" / "class_distribution.png")

    by_split = df.groupby("split")
    is_fallback = df["leaf_id_source"] == "fallback"
    report = {
        "source": {
            "hf_repo": cfg.dataset["hf_repo"],
            "variant": cfg.dataset["variant"],
            "licence": cfg.dataset["licence"],
        },
        "seed": cfg.seed,
        "ratios": cfg.data["split"],
        "images_total": len(df),
        "leaf_groups_total": df["group_id"].nunique(),
        "leaf_id_fallback_share": round(float(is_fallback.mean()), 4),
        "leaf_id_fallback_share_per_class": is_fallback.groupby(df["class_id"])
        .mean()
        .round(3)
        .to_dict(),
        "images_per_leaf_mean": round(float(df.groupby("group_id").size().mean()), 2),
        "images_per_split": by_split.size().to_dict(),
        "leaf_groups_per_split": by_split["group_id"].nunique().to_dict(),
        "per_class": counts.to_dict(orient="index"),
        "leakage_check": leak,
    }
    text = json.dumps(report, indent=2, default=int)
    (reports_dir / "data_report.json").write_text(text, encoding="utf-8")
    return counts


def load_manifests(cfg) -> pd.DataFrame:
    splits_dir = ROOT / cfg.paths["splits_dir"]
    frames = [pd.read_csv(splits_dir / f"{s}.csv").assign(split=s) for s in SPLITS]
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cfg = load_config()

    if args.inspect:
        inspect_dataset(cfg)
        return 0
    if args.verify:
        leak = check_leakage(load_manifests(cfg))
        print(json.dumps(leak, indent=2))
        print("LEAKAGE CHECK:", "PASSED" if leak["passed"] else "FAILED")
        return 0 if leak["passed"] else 1

    zip_path, leaf_map = fetch_files(cfg)
    with zipfile.ZipFile(zip_path) as zf:
        entries = list_images(zf, cfg.dataset["variant"])
        folders = sorted({folder for _, folder, _ in entries})
        label_map = build_label_map(folders, cfg.classes)
        catalogue = build_catalogue(entries, label_map, leaf_map)
        df = export_images(zf, catalogue, ROOT / cfg.dataset["export_dir"])
    df["split"] = split_groups(df, cfg.data["split"], cfg.seed)

    counts = class_counts(df, cfg.class_ids)
    if (counts[list(SPLITS)] == 0).any().any():
        print(counts.to_string())
        print("A class has an empty split. Not writing manifests; paste this to me.")
        return 1
    leak = check_leakage(df)
    if not leak["passed"]:
        print(json.dumps(leak, indent=2))
        print("LEAKAGE CHECK: FAILED. Not writing manifests.")
        return 1

    counts = write_outputs(cfg, df, leak)
    print(counts.to_string())
    print(json.dumps(leak, indent=2))
    print("LEAKAGE CHECK: PASSED (zero leaf overlap between train/val/test)")
    share = (df["leaf_id_source"] == "fallback").mean()
    print(f"Leaf grouping: {share:.1%} of images use fallback ids (one group per image).")
    if any(leak["identical_file_overlap"].values()):
        print("NOTE: identical image files appear in more than one split. Tell me the numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
