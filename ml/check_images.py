"""Find photos that cannot be read completely (cut-off downloads, damaged files).

    python -m ml.check_images            # only lists them
    python -m ml.check_images --move     # moves them to data/_broken/ so training skips them

Every image in data/field and data/nonleaf is fully decoded once, which is what training does.
Moved files are simply missing afterwards: the training code already ignores manifest rows whose
file is gone.
"""

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from ml.config import ROOT, load_config

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def broken_images(folders) -> list[tuple[Path, str]]:
    """[(path, reason)] for every image under `folders` that fails a full decode."""
    bad = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            try:
                with Image.open(path) as handle:
                    handle.convert("RGB")
            except Exception as error:  # noqa: BLE001 - any decode problem counts
                bad.append((path, f"{type(error).__name__}: {error}"))
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--move", action="store_true", help="move broken files to data/_broken/")
    args = parser.parse_args()
    cfg = load_config()
    folders = [ROOT / cfg.paths["field_dir"], ROOT / cfg.paths["nonleaf_dir"]]
    bad = broken_images(folders)
    print(f"{len(bad)} unreadable image(s)")
    for path, reason in bad:
        print(f"  {path.relative_to(ROOT)}  ({reason})")
        if args.move:
            target = ROOT / "data" / "_broken" / path.relative_to(ROOT / "data")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), target)
    if bad and args.move:
        print("Moved to data/_broken/. Start training again.")
    elif bad:
        print("Run again with --move to take them out of training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
