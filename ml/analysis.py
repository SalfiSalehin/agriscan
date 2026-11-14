"""Torch-free helpers for Phase 3: calibration, the "unsure" threshold, confidence intervals and
the bookkeeping for the external test sets.

Everything here works on plain numpy arrays and pandas frames, so it can be tested without a GPU.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml.metrics import accuracy, confusion_matrix, per_class_prf

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)  # always shown in the threshold table
THRESHOLD_GRID = tuple(round(0.5 + 0.01 * i, 2) for i in range(50))  # 0.50 ... 0.99
TEMPERATURE_GRID = np.linspace(0.3, 5.0, 471)


def _num(x) -> float | None:
    """JSON-friendly number: rounded, and None instead of NaN."""
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


# ---------------------------------------------------------------- probabilities and calibration
def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, labels, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the true class."""
    p = softmax(logits, temperature)
    picked = p[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def fit_temperature(logits, labels) -> float:
    """Temperature T that minimises the NLL of softmax(logits / T) (grid search, no scipy)."""
    losses = [nll(logits, labels, t) for t in TEMPERATURE_GRID]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def fit_temperature_balanced(pairs) -> float:
    """Temperature that minimises the *average of the per-set* NLLs. `pairs` is a list of
    (logits, labels), e.g. lab validation and field validation, so a big easy set cannot drown
    out a small hard one."""
    pairs = [(x, y) for x, y in pairs if len(y) > 0]
    if not pairs:
        raise ValueError("no data to fit a temperature on")
    losses = [np.mean([nll(x, y, t) for x, y in pairs]) for t in TEMPERATURE_GRID]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def crop_of(class_id: str) -> str:
    """'tomato_early_blight' -> 'tomato'."""
    return class_id.split("_")[0]


def crop_restricted_logits(logits, labels, class_ids) -> np.ndarray:
    """Logits as if the user had picked the right crop first: classes of every other crop are
    switched off. This measures the "pick your crop" design, not the model alone."""
    logits = np.array(logits, dtype=np.float64, copy=True)
    crops = np.array([crop_of(c) for c in class_ids])
    for crop in set(crops):
        rows = crops[np.asarray(labels)] == crop
        logits[np.ix_(rows, crops != crop)] = -1e9
    return logits


def expected_calibration_error(conf, correct, bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by how many images fall in each bin."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def auroc(positive, negative) -> float:
    """Chance that a random positive scores higher than a random negative (ties count half)."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


# ---------------------------------------------------------------- metrics with small classes
def macro_f1_min_support(cm: np.ndarray, min_support: int = 1) -> float:
    """Mean F1 over the classes that have at least `min_support` true images."""
    _, _, f1, support = per_class_prf(cm)
    keep = support >= max(min_support, 1)
    return float(f1[keep].mean()) if keep.any() else 0.0


def bootstrap_ci(labels, pred, num_classes: int, n_boot: int = 1000, seed: int = 0, min_support=1):
    """95% bootstrap intervals for accuracy and macro-F1, resampling images.

    Images are treated as independent. Photos of one leaf are correlated, so the true
    uncertainty is somewhat larger than this interval suggests.
    """
    if n_boot <= 0 or len(labels) == 0:
        return None
    labels, pred = np.asarray(labels), np.asarray(pred)
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        cm = confusion_matrix(labels[idx], pred[idx], num_classes)
        accs.append(accuracy(cm))
        f1s.append(macro_f1_min_support(cm, min_support))
    return {
        "accuracy": [_num(v) for v in np.percentile(accs, [2.5, 97.5])],
        "macro_f1": [_num(v) for v in np.percentile(f1s, [2.5, 97.5])],
    }


def summarise_set(
    logits, labels, temperature, class_ids, min_support: int = 1, n_boot: int = 0, seed: int = 0
) -> dict:
    """All the numbers for one evaluation set, after temperature scaling."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    num = len(class_ids)
    probs = softmax(logits, temperature)
    raw = softmax(logits)
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    cm = confusion_matrix(labels, pred, num)
    precision, recall, f1, support = per_class_prf(cm)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    reportable = support >= max(min_support, 1)
    lowest = sorted(
        (i for i in range(num) if reportable[i]), key=lambda i: (recall[i], class_ids[i])
    )[:3]
    out = {
        "n_images": int(len(labels)),
        "accuracy": _num(accuracy(cm)),
        "macro_f1": _num(macro_f1_min_support(cm, min_support)),
        "top3_accuracy": _num((top3 == labels[:, None]).any(axis=1).mean()),
        "nll": _num(nll(logits, labels, temperature)),
        "ece": _num(expected_calibration_error(conf, pred == labels)),
        "nll_before_temperature": _num(nll(logits, labels)),
        "ece_before_temperature": _num(
            expected_calibration_error(raw.max(axis=1), raw.argmax(axis=1) == labels)
        ),
        "classes_reported": int(reportable.sum()),
        "classes_below_min_support": {
            class_ids[i]: int(support[i]) for i in range(num) if not reportable[i]
        },
        "lowest_recall": [
            {"class": class_ids[i], "recall": _num(recall[i]), "support": int(support[i])}
            for i in lowest
        ],
        "per_class": {
            class_ids[i]: {
                "precision": _num(precision[i]),
                "recall": _num(recall[i]),
                "f1": _num(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num)
            if support[i] > 0
        },
    }
    ci = bootstrap_ci(labels, pred, num, n_boot, seed, min_support)
    if ci is not None:
        out["ci95"] = ci
    return out


# ---------------------------------------------------------------- the "unsure" threshold
def selective_stats(conf, correct, threshold: float) -> dict:
    """If the app only answers when confidence >= threshold: how many images, how accurate."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(conf) == 0:
        return {"n": 0, "coverage": None, "accuracy_when_answered": None}
    answered = conf >= threshold
    count = int(answered.sum())
    return {
        "n": int(len(conf)),
        "coverage": _num(count / len(conf)),
        "accuracy_when_answered": _num(correct[answered].mean()) if count else None,
    }


def choose_threshold(unknown_conf, max_false_accept: float) -> float | None:
    """Smallest threshold (0.50 to 0.99) at which at most `max_false_accept` of the images the
    model should not recognise still get an answer. None if no threshold gets there."""
    unknown_conf = np.asarray(unknown_conf, dtype=float)
    if len(unknown_conf) == 0:
        return None
    for t in THRESHOLD_GRID:
        if (unknown_conf >= t).mean() <= max_false_accept:
            return float(t)
    return None


def half_of(sha256_hex: str) -> int:
    """Stable 50/50 split of a file by its content hash: 0 = calibration, 1 = held out."""
    return int(str(sha256_hex)[:8], 16) % 2


def analyse_run(
    logits: dict,
    labels: dict,
    field_info: pd.DataFrame,
    unknown_info: pd.DataFrame,
    class_ids: list[str],
    test_leaf_verified,
    min_support: int,
    max_false_accept: float,
    n_boot: int,
    seed: int = 0,
    field_val=None,
) -> dict:
    """Everything evaluate.py reports for one model.

    `logits` has keys val, test, field, unknown; `labels` has val, test, field.
    `field_info` needs columns mapping and half (aligned with the field logits);
    `unknown_info` needs the column half. `field_val` is an optional (logits, labels) pair of
    real-world validation photos that are NOT in the field test set.
    The temperature is fitted on the validation split (lab only, or the balanced average of lab
    and field validation when `field_val` is given).
    The threshold is chosen on the calibration half of the unknown images and every threshold
    number is reported on the other half, so the threshold never grades its own homework.
    """
    has_field_val = field_val is not None and len(field_val[1]) > 0
    if has_field_val:
        temperature = fit_temperature_balanced(
            [(logits["val"], labels["val"]), (field_val[0], field_val[1])]
        )
    else:
        temperature = fit_temperature(logits["val"], labels["val"])
    sets = {
        "val": summarise_set(logits["val"], labels["val"], temperature, class_ids),
        "test": summarise_set(
            logits["test"], labels["test"], temperature, class_ids, n_boot=n_boot, seed=seed
        ),
        "field": summarise_set(
            logits["field"],
            labels["field"],
            temperature,
            class_ids,
            min_support=min_support,
            n_boot=n_boot,
            seed=seed,
        ),
    }
    for name in ("test", "field"):  # the "pick your crop first" design: right crop is given
        sets[f"{name}_crop_known"] = summarise_set(
            crop_restricted_logits(logits[name], labels[name], class_ids),
            labels[name],
            temperature,
            class_ids,
            min_support=min_support if name == "field" else 1,
            n_boot=n_boot,
            seed=seed,
        )
    exact = field_info["mapping"].to_numpy() == "exact"
    if exact.any() and not exact.all():
        sets["field_exact_mapping_only"] = summarise_set(
            logits["field"][exact], labels["field"][exact], temperature, class_ids, min_support
        )
    if test_leaf_verified is not None:
        verified = np.asarray(test_leaf_verified, dtype=bool)
        for name, keep in (
            ("test_leaf_verified_only", verified),
            ("test_fallback_grouping_only", ~verified),
        ):
            if keep.any():
                sets[name] = summarise_set(
                    logits["test"][keep], np.asarray(labels["test"])[keep], temperature, class_ids
                )

    def conf_and_correct(name, mask=None):
        probs = softmax(logits[name], temperature)
        conf, pred = probs.max(axis=1), probs.argmax(axis=1)
        correct = pred == np.asarray(labels[name]) if name in labels else None
        if mask is not None:
            conf = conf[mask]
            correct = correct[mask] if correct is not None else None
        return conf, correct

    field_half = field_info["half"].to_numpy()
    unknown_half = unknown_info["half"].to_numpy()
    if not ((field_half == 1).any() and (unknown_half == 0).any() and (unknown_half == 1).any()):
        raise ValueError("the field and unknown sets are too small to split into two halves")
    test_conf, test_correct = conf_and_correct("test")
    field_conf, field_correct = conf_and_correct("field", field_half == 1)
    unknown_cal, _ = conf_and_correct("unknown", unknown_half == 0)
    unknown_held, _ = conf_and_correct("unknown", unknown_half == 1)

    def crop_known_conf_and_correct(name, mask=None):
        """Serving rule when the user picked the crop: the answer is the best class of that crop,
        the confidence is its probability among ALL classes (not renormalised), so the threshold
        chosen on the unknown images still applies and a wrong crop pick ends up as "unsure"."""
        restricted = crop_restricted_logits(logits[name], labels[name], class_ids)
        pred = restricted.argmax(axis=1)
        conf = softmax(logits[name], temperature)[np.arange(len(pred)), pred]
        correct = pred == np.asarray(labels[name])
        if mask is not None:
            conf, correct = conf[mask], correct[mask]
        return conf, correct

    test_ck_conf, test_ck_correct = crop_known_conf_and_correct("test")
    field_ck_conf, field_ck_correct = crop_known_conf_and_correct("field", field_half == 1)

    chosen = choose_threshold(unknown_cal, max_false_accept)
    grid = sorted(set(THRESHOLDS) | ({chosen} if chosen is not None else set()))
    table = [
        {
            "threshold": t,
            "chosen": chosen is not None and abs(t - chosen) < 1e-9,
            "test": selective_stats(test_conf, test_correct, t),
            "field_held_out": selective_stats(field_conf, field_correct, t),
            "test_crop_known": selective_stats(test_ck_conf, test_ck_correct, t),
            "field_held_out_crop_known": selective_stats(field_ck_conf, field_ck_correct, t),
            "unknown_held_out_false_accept": _num((unknown_held >= t).mean()),
        }
        for t in grid
    ]
    return {
        "temperature": _num(temperature),
        "temperature_fitted_on": (
            f"lab validation + field validation ({len(field_val[1])} field images), "
            "averaged per set"
            if has_field_val
            else "lab validation only"
        ),
        "sets": sets,
        "threshold": {
            "selection_rule": (
                f"smallest threshold from 0.50 to 0.99 at which at most {max_false_accept:.0%} of "
                "the calibration-half unknown images still get an answer"
            ),
            "chosen_threshold": chosen,
            "images": {
                "unknown_calibration": int((unknown_half == 0).sum()),
                "unknown_held_out": int((unknown_half == 1).sum()),
                "field_calibration": int((field_half == 0).sum()),
                "field_held_out": int((field_half == 1).sum()),
            },
            "auroc_confidence_known_vs_unknown": {
                "test_vs_unknown_held_out": _num(auroc(test_conf, unknown_held)),
                "field_held_out_vs_unknown_held_out": _num(auroc(field_conf, unknown_held)),
            },
            "table": table,
        },
    }


# ---------------------------------------------------------------- external test sets
def collect_external_rows(root, manifest_path, field_dir, nonleaf_dir, class_ids):
    """All field and unknown images: the imported ones (data/external_manifest.csv) plus any
    image you drop into data/field/<class id>/ or data/nonleaf/ yourself ("local").

    Returns (DataFrame with path, role, class_id, source, source_split, mapping, sha256, half;
    warnings). Paths are relative to `root`. `source_split` is PlantDoc's own train/test folder
    ("" for local photos).
    """
    root = Path(root)
    warnings: list[str] = []
    rows, known = [], set()
    if Path(manifest_path).exists():
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        for r in manifest.itertuples(index=False):
            known.add(r.path)
            rows.append(
                {
                    "path": r.path,
                    "role": r.role,
                    "class_id": r.class_id,
                    "source": r.source,
                    "source_split": getattr(r, "source_split", ""),
                    "mapping": r.mapping,
                    "sha256": r.sha256,
                }
            )
    ignored = []
    for base, role in ((Path(field_dir), "field"), (Path(nonleaf_dir), "unknown")):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            if rel in known:
                continue
            class_id = ""
            if role == "field":
                parts = path.relative_to(base).parts
                class_id = parts[0] if len(parts) > 1 else ""
                if class_id not in class_ids:
                    ignored.append(rel)
                    continue
            rows.append(
                {
                    "path": rel,
                    "role": role,
                    "class_id": class_id,
                    "source": "local",
                    "source_split": "",
                    "mapping": "exact",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    if ignored:
        warnings.append(
            f"{len(ignored)} image(s) in the field folder were ignored because their folder is "
            f"not a class id, e.g. {ignored[0]}"
        )
    columns = ["path", "role", "class_id", "source", "source_split", "mapping", "sha256"]
    frame = pd.DataFrame(rows, columns=columns)
    present = np.array([(root / p).exists() for p in frame["path"]], dtype=bool)
    if not present.all():
        warnings.append(
            f"{int((~present).sum())} listed image(s) are missing on disk and were skipped"
        )
        frame = frame[present]
    frame = frame.reset_index(drop=True)
    frame["half"] = [half_of(s) for s in frame["sha256"]]
    return frame, warnings


def val_share(sha256_hex: str) -> float:
    """Stable number in [0, 1) from a file hash (different hex digits than `half_of`)."""
    return int(str(sha256_hex)[8:16], 16) / 16**8


def load_overlap(path) -> set[str]:
    """Paths listed in the overlap file written by ml.check_overlap (empty set if it is missing)."""
    path = Path(path)
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["path"])


def assign_use(frame: pd.DataFrame, val_fraction: float, exclude=()) -> pd.Series:
    """What each external image is used for.

    - field, PlantDoc train folder -> "train" (mixed into training) or, for a stable
      `val_fraction` of them chosen by file hash, "val" (early stopping and calibration only)
    - field, PlantDoc test folder or your own local photos -> "test" (never trained on)
    - field images listed in `exclude` (near-duplicates of training photos) -> "excluded"
    - unknown images -> "unknown" (never trained on; they only choose the "unsure" threshold)
    """
    exclude = set(exclude)
    use = []
    for r in frame.itertuples(index=False):
        if r.role != "field":
            use.append("unknown")
        elif r.path in exclude:
            use.append("excluded")
        elif r.source_split == "train":
            use.append("val" if val_share(r.sha256) < val_fraction else "train")
        else:
            use.append("test")
    return pd.Series(use, index=frame.index, dtype=object)
