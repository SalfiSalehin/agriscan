"""Add the rice and eggplant classes to config/config.yaml and data/splits/class_index.json.

    python -m ml.add_classes --dry-run
    python -m ml.add_classes

The 19 existing classes keep their numbers; the new ones are appended after them, so nothing that
refers to the old order changes. Running it twice does nothing the second time.
"""

import argparse
import json
import re
import sys

from ml.config import ROOT, load_config

NEW_CLASSES = [
    ("rice_brown_spot", "Rice", "Brown spot"),
    ("rice_blast", "Rice", "Blast"),
    ("rice_leaf_scald", "Rice", "Leaf scald"),
    ("rice_sheath_blight", "Rice", "Sheath blight"),
    ("rice_tungro", "Rice", "Tungro"),
    ("eggplant_cercospora_leaf_spot", "Eggplant", "Cercospora leaf spot"),
    ("eggplant_early_blight", "Eggplant", "Early blight"),
    ("eggplant_healthy", "Eggplant", "Healthy"),
    ("eggplant_mosaic_virus", "Eggplant", "Mosaic virus"),
    ("eggplant_phomopsis_blight", "Eggplant", "Phomopsis blight"),
    ("eggplant_phytophthora_blight", "Eggplant", "Phytophthora blight"),
    ("eggplant_powdery_mildew", "Eggplant", "Powdery mildew"),
]
CLASS_LINE = re.compile(r"^\s*-\s*\{\s*id:\s*([A-Za-z0-9_]+)\s*,")


def add_to_config_text(text: str, classes=NEW_CLASSES) -> tuple[str, list[str]]:
    """Insert the missing class lines after the last existing one. Returns (text, added ids)."""
    lines = text.splitlines()
    existing = {m.group(1) for line in lines if (m := CLASS_LINE.match(line))}
    last = max((i for i, line in enumerate(lines) if CLASS_LINE.match(line)), default=None)
    if last is None:
        raise SystemExit("could not find the class list in config.yaml")
    missing = [c for c in classes if c[0] not in existing]
    new_lines = [
        f'  - {{ id: {cid}, crop: {crop}, disease: {disease}, raw_name: "" }}'
        for cid, crop, disease in missing
    ]
    lines[last + 1 : last + 1] = new_lines
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), [c[0] for c in missing]


def extend_index(index: dict[str, int], class_ids: list[str]) -> dict[str, int]:
    """Old ids keep their number; unseen ids from `class_ids` are appended in that order."""
    out = dict(index)
    for cid in class_ids:
        if cid not in out:
            out[cid] = max(out.values(), default=-1) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "config" / "config.yaml"
    text = config_path.read_text(encoding="utf-8")
    new_text, added = add_to_config_text(text)
    print(f"config.yaml: {len(added)} class(es) to add {added}")
    if not args.dry_run and added:
        config_path.write_text(new_text, encoding="utf-8")
    cfg = load_config(config_path) if not args.dry_run else None
    index_path = ROOT / "data" / "splits" / "class_index.json"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found: run ml.prepare_data first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    wanted = cfg.class_ids if cfg else list(index) + added
    extended = extend_index(index, wanted)
    print(f"class_index.json: {len(index)} -> {len(extended)} classes")
    if not args.dry_run:
        index_path.write_text(json.dumps(extended, indent=2), encoding="utf-8")
        print("Written. Old class numbers are unchanged.")
    else:
        print("Dry run: nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
