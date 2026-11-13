"""Phase 3A: evaluate trained models on data they never saw.

    python -m ml.evaluate --run runs/<mobilenet run> --run runs/<resnet18 run>

Sets
- test:    the held-out PlantVillage split (lab photos)
- field:   real-world photos the model never trained on: PlantDoc's official test folder, plus
           anything you add to data/field/<class id>/ (minus near-duplicates of training photos,
           see ml.check_overlap)
- unknown: leaves of other crops (plus anything you add to data/nonleaf/)

The validation split(s) are used only to fit the temperature (probability calibration): lab
validation, plus the field validation photos for runs that trained on field photos. The "unsure"
threshold is chosen on one half of the unknown images and reported on the other half.
Results go to reports/eval/<run>/ (metrics.json, confusion matrices, error lists and pictures)
and reports/eval/comparison.csv / thresholds.csv.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.analysis import analyse_run, assign_use, collect_external_rows, load_overlap, softmax
from ml.config import ROOT, load_config
from ml.data import ListDataset, ManifestDataset, load_class_index
from ml.eval_plots import plot_confusion, plot_errors
from ml.metrics import confusion_matrix
from ml.models import build_model
from ml.transforms import build_eval_transform

NOTES = [
    "test = held-out PlantVillage split: lab photos on plain backgrounds. Only part of it has "
    "verified leaf grouping; compare test_leaf_verified_only with test_fallback_grouping_only.",
    "field = PlantDoc's official TEST photos mapped onto our classes (plus any local photos), "
    "minus near-duplicates of training photos. No run trained on them. Labels are the dataset "
    "authors' own. Mappings marked approximate are in 'field' but not in "
    "'field_exact_mapping_only'. Classes with fewer images than the minimum are not scored.",
    "*_crop_known = the same images, but with the classes of every other crop switched off, as if "
    "the user had picked the crop first. It measures the app design, not the model alone.",
    "unknown = leaves of other crops (plus any local non-leaf photos). Without photos of hands, "
    "soil or objects in data/nonleaf, the threshold is only tested against other plants' leaves.",
    "Confidence intervals resample images as if independent. Several photos of one leaf are "
    "correlated, so the real uncertainty is larger.",
    "Temperature is fitted on lab validation (plus field validation for runs trained on field "
    "photos; see temperature_fitted_on). The threshold is chosen on one half of the unknown "
    "images (split by file hash) and reported on the other half.",
    "The field test set is small (check n_images and classes_below_min_support). Differences of a "
    "few points between runs are inside the confidence intervals.",
]


def load_run(run_dir: Path, device):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    model = build_model(meta["arch"], len(meta["class_ids"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), meta, ckpt["state"]


@torch.inference_mode()
def predict_logits(model, dataset, device, batch_size: int, workers: int) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    for images, _ in loader:
        chunks.append(model(images.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(chunks)


def leaf_verified_mask(test_csv: Path, fallback_label: str):
    """True for test images whose leaf grouping was verified, False for filename-fallback groups."""
    frame = pd.read_csv(test_csv)
    if "leaf_id_source" not in frame.columns:
        print("NOTE: test.csv has no leaf_id_source column, so no verified-leaf breakdown.")
        return None
    values = frame["leaf_id_source"].astype(str)
    print("leaf_id_source values in test.csv:", values.value_counts().to_dict())
    fallback = values.str.contains(fallback_label, case=False, regex=False)
    if not fallback.any() or fallback.all():
        print(
            f"NOTE: no split into verified/fallback using '{fallback_label}'. "
            "Pass --fallback-label with the value used for the fallback groups."
        )
        return None
    return (~fallback).to_numpy()


def error_rows(rel_paths, labels, logits, temperature, class_ids) -> list[dict]:
    """Misclassified images, most confident mistakes first."""
    probs = softmax(logits, temperature)
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    labels = np.asarray(labels)
    rows = [
        {
            "path": str(rel_paths[i]),
            "true": class_ids[labels[i]],
            "pred": class_ids[pred[i]],
            "conf": round(float(conf[i]), 4),
        }
        for i in np.flatnonzero(pred != labels)
    ]
    return sorted(rows, key=lambda r: -r["conf"])


def write_run_outputs(out_dir, name, logits, labels, rel_paths, roots, temperature, class_ids):
    """Confusion matrices and error lists/pictures for the test and field sets."""
    for key, title in (
        ("test", "held-out PlantVillage test split"),
        ("field", "real-world field set"),
    ):
        probs = softmax(logits[key], temperature)
        cm = confusion_matrix(labels[key], probs.argmax(axis=1), len(class_ids))
        plot_confusion(cm, class_ids, f"{name}: {title}", out_dir / f"confusion_{key}.png")
        errors = error_rows(rel_paths[key], labels[key], logits[key], temperature, class_ids)
        pd.DataFrame(errors, columns=["path", "true", "pred", "conf"]).to_csv(
            out_dir / f"errors_{key}.csv", index=False
        )
        shown = [{**e, "path": roots[key] / e["path"]} for e in errors]
        plot_errors(
            shown,
            f"{name}: most confident mistakes, {title}",
            out_dir / f"errors_{key}.png",
        )


def comparison_tables(all_metrics: list[dict]):
    rows, thresholds = [], []
    wanted = (
        "val",
        "test",
        "test_leaf_verified_only",
        "test_fallback_grouping_only",
        "field",
        "field_exact_mapping_only",
        "test_crop_known",
        "field_crop_known",
    )
    for m in all_metrics:
        for key in wanted:
            s = m["results"]["sets"].get(key)
            if s is None:
                continue
            ci = s.get("ci95", {})
            rows.append(
                {
                    "run": m["run"],
                    "arch": m["arch"],
                    "field_share": m["data"]["trained_with_field_share"],
                    "set": key,
                    "images": s["n_images"],
                    "classes": s["classes_reported"],
                    "accuracy": s["accuracy"],
                    "acc_ci95": ci.get("accuracy"),
                    "macro_f1": s["macro_f1"],
                    "f1_ci95": ci.get("macro_f1"),
                    "ece": s["ece"],
                }
            )
        th = m["results"]["threshold"]
        picked = next((r for r in th["table"] if r["chosen"]), None)
        thresholds.append(
            {
                "run": m["run"],
                "arch": m["arch"],
                "field_share": m["data"]["trained_with_field_share"],
                "temperature": m["results"]["temperature"],
                "threshold": th["chosen_threshold"],
                "field_coverage": picked["field_held_out"]["coverage"] if picked else None,
                "field_acc_answered": (
                    picked["field_held_out"]["accuracy_when_answered"] if picked else None
                ),
                "test_coverage": picked["test"]["coverage"] if picked else None,
                "unknown_false_accept": (
                    picked["unknown_held_out_false_accept"] if picked else None
                ),
                **{f"auroc_{k}": v for k, v in th["auroc_confidence_known_vs_unknown"].items()},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path, help="run folder")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-false-accept", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--fallback-label", default="fallback")
    args = parser.parse_args()

    cfg = load_config()
    class_index = load_class_index(cfg)
    class_ids = sorted(class_index, key=class_index.get)
    splits = ROOT / cfg.paths["splits_dir"]
    export_root = ROOT / cfg.dataset["export_dir"]
    workers = cfg.train["num_workers"] if args.workers is None else args.workers
    min_support = int(cfg.test_sets["field_min_per_class"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    external, warnings = collect_external_rows(
        ROOT,
        ROOT / "data" / "external_manifest.csv",
        ROOT / cfg.paths["field_dir"],
        ROOT / cfg.paths["nonleaf_dir"],
        class_ids,
    )
    for warning in warnings:
        print("WARNING:", warning)
    overlap_path = ROOT / "data" / "field_test_overlap.csv"
    overlap = load_overlap(overlap_path)
    if not overlap_path.exists():
        print("NOTE: no data/field_test_overlap.csv. Run `python -m ml.check_overlap` first.")
    default_fraction = float(cfg.train.get("field_val_fraction", 0.15))
    unknown = external[external["role"] == "unknown"].reset_index(drop=True)
    preview = assign_use(external, default_fraction, overlap)
    n_test = int((preview == "test").sum())
    if n_test == 0 or unknown.empty:
        raise SystemExit(
            "Need both field test and unknown images. Run ml.import_external first "
            f"(field test: {n_test}, unknown: {len(unknown)})."
        )
    dropped = int((preview == "excluded").sum())
    print(
        f"Field test images: {n_test} (left out as duplicates: {dropped})  "
        f"unknown images: {len(unknown)}  device: {device}"
    )
    leaf_verified = leaf_verified_mask(splits / "test.csv", args.fallback_label)

    out_root = ROOT / cfg.paths["reports_dir"] / "eval"
    all_metrics = []
    for run_arg in args.run:
        run_dir = run_arg if run_arg.is_absolute() else ROOT / run_arg
        model, meta, state = load_run(run_dir, device)
        if meta["class_ids"] != class_ids:
            raise SystemExit(f"{run_dir.name}: class order differs from class_index.json")
        hparams = {}
        if (run_dir / "run_config.json").exists():
            saved = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            hparams = saved.get("hparams", {})
        field_share = float(hparams.get("field_share", 0.0))  # old runs: lab only
        fraction = float(hparams.get("field_val_fraction", default_fraction))
        external["use"] = assign_use(external, fraction, overlap)
        field = external[external["use"] == "test"].reset_index(drop=True)
        field_val = external[external["use"] == "val"].reset_index(drop=True)
        use_field_val = field_share > 0 and not field_val.empty
        transform = build_eval_transform(meta["image_size"])
        val_ds = ManifestDataset(splits / "val.csv", export_root, class_index, transform)
        test_ds = ManifestDataset(splits / "test.csv", export_root, class_index, transform)
        field_labels = field["class_id"].map(class_index).to_numpy()
        field_ds = ListDataset([ROOT / p for p in field["path"]], field_labels, transform)
        field_val_labels = field_val["class_id"].map(class_index).to_numpy()
        field_val_ds = ListDataset(
            [ROOT / p for p in field_val["path"]], field_val_labels, transform
        )
        unknown_ds = ListDataset(
            [ROOT / p for p in unknown["path"]], [-1] * len(unknown), transform
        )

        began = time.time()
        logits = {
            name: predict_logits(model, ds, device, args.batch_size, workers)
            for name, ds in (
                ("val", val_ds),
                ("test", test_ds),
                ("field", field_ds),
                ("unknown", unknown_ds),
            )
        }
        if use_field_val:
            fv_logits = predict_logits(model, field_val_ds, device, args.batch_size, workers)
        print(
            f"{run_dir.name}: scored {sum(len(v) for v in logits.values())} images "
            f"in {time.time() - began:.0f}s"
        )
        labels = {
            "val": np.asarray(val_ds.labels),
            "test": np.asarray(test_ds.labels),
            "field": field_labels,
        }
        results = analyse_run(
            logits,
            labels,
            field[["mapping", "half"]],
            unknown[["half"]],
            class_ids,
            leaf_verified,
            min_support,
            args.max_false_accept,
            args.bootstrap,
            seed=cfg.seed,
            field_val=(fv_logits, field_val_labels) if use_field_val else None,
        )
        out_dir = out_root / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        temperature = results["temperature"]
        write_run_outputs(
            out_dir,
            run_dir.name,
            logits,
            labels,
            {"test": test_ds.rel, "field": list(field["path"])},
            {"test": export_root, "field": ROOT},
            temperature,
            class_ids,
        )
        metrics = {
            "run": run_dir.name,
            "arch": meta["arch"],
            "best_epoch": state["best_epoch"],
            "created": datetime.now().isoformat(timespec="seconds"),
            "data": {
                "test_images": len(test_ds),
                "field_images": len(field),
                "field_val_images_used_for_calibration": len(field_val) if use_field_val else 0,
                "field_test_excluded_as_duplicates": int((external["use"] == "excluded").sum()),
                "trained_with_field_share": field_share,
                "unknown_images": len(unknown),
                "field_min_images_per_class": min_support,
                "external_sources": external["source"].value_counts().to_dict(),
            },
            "temperature": temperature,
            "results": results,
            "notes": NOTES,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        all_metrics.append(metrics)
        print(f"Wrote {out_dir}")
        del model

    comparison, thresholds = comparison_tables(all_metrics)
    out_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_root / "comparison.csv", index=False)
    thresholds.to_csv(out_root / "thresholds.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + comparison.to_string(index=False))
    print("\n" + thresholds.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
