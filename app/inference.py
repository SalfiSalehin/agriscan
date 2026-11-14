"""Serve the exported ONNX model: preprocessing, calibration, crop choice, the "not sure" rule.

No torch and no web framework in here, so it is easy to test. The preprocessing reproduces
ml.transforms.build_eval_transform exactly (resize the shorter side to 256 with PIL bilinear,
centre-crop 224, scale to 0-1, ImageNet mean/std).

The "not sure" rule. The threshold in the model's meta file was chosen on images the model should
not recognise, using the probability over ALL classes. So the confidence here is always the
probability among all classes of the answer given, also when the user picked the crop first:
- picked crop: the answer is the best class of that crop. Its all-class probability can only be
  lower than or equal to the overall top probability, so wrong answers on unfamiliar images are
  not more likely than without the pick, and a wrong crop pick ends up as "unsure".
- no crop given: the answer is the overall top class.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_META_FORMAT = 1
EVAL_RESIZE_RATIO = 256 / 224


# ------------------------------------------------------------------------------ preprocessing
def resize_shorter_side(image: Image.Image, target: int) -> Image.Image:
    """torchvision.transforms.Resize(target) for a PIL image."""
    width, height = image.size
    short, long = (width, height) if width <= height else (height, width)
    if short == target:
        return image
    new_short, new_long = target, int(target * long / short)
    new_size = (new_short, new_long) if width <= height else (new_long, new_short)
    return image.resize(new_size, Image.Resampling.BILINEAR)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """torchvision.transforms.CenterCrop(size) for a PIL image at least `size` wide and tall."""
    width, height = image.size
    left = int(round((width - size) / 2.0))
    top = int(round((height - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def preprocess(image: Image.Image, size: int, mean, std) -> np.ndarray:
    """RGB PIL image -> float32 array of shape (1, 3, size, size) for the model."""
    resized = resize_shorter_side(image.convert("RGB"), round(size * EVAL_RESIZE_RATIO))
    pixels = np.asarray(center_crop(resized, size), dtype=np.uint8).astype(np.float32)
    pixels = pixels / np.float32(255.0)
    pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


# ------------------------------------------------------------------------------ decision rule
def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def crop_of(class_id: str) -> str:
    return class_id.split("_")[0]


def crop_table(class_ids) -> dict[str, list[int]]:
    """crop id -> indices of its classes, e.g. 'tomato' -> [12, 13, ...]."""
    table: dict[str, list[int]] = {}
    for i, class_id in enumerate(class_ids):
        table.setdefault(crop_of(class_id), []).append(i)
    return table


def decide(logits, class_ids, temperature, threshold, crop=None, top_k: int = 3) -> dict:
    """Turn one image's logits into the answer. Raises ValueError for an unknown crop."""
    probs = softmax(logits, temperature)
    crops = crop_table(class_ids)
    if crop is None:
        candidates = np.arange(len(class_ids))
    elif crop in crops:
        candidates = np.asarray(crops[crop])
    else:
        raise ValueError(f"unknown crop {crop!r}")
    order = candidates[np.argsort(-probs[candidates], kind="stable")][:top_k]
    confidence = float(probs[order[0]])
    status = "ok" if confidence >= threshold else "unsure"
    overall = int(np.argmax(probs))
    suggested = None
    if crop is not None and crop_of(class_ids[overall]) != crop and probs[overall] >= threshold:
        suggested = crop_of(class_ids[overall])  # the photo looks like another crop
    return {
        "status": status,
        "diagnosis": class_ids[order[0]] if status == "ok" else None,
        "confidence": round(confidence, 4),
        "threshold": float(threshold),
        "predictions": [
            {
                "class_id": class_ids[i],
                "crop": crop_of(class_ids[i]),
                "probability": round(float(probs[i]), 4),
            }
            for i in order
        ],
        "suggested_crop": suggested,
    }


# ------------------------------------------------------------------------------ the classifier
class Classifier:
    """An ONNX Runtime session plus the metadata that says how to use it."""

    def __init__(self, session, meta: dict):
        self.session = session
        self.meta = meta
        self.class_ids = list(meta["output"]["class_ids"])
        self.size = int(meta["input"]["size"])
        self.mean, self.std = meta["input"]["mean"], meta["input"]["std"]
        self.input_name = meta["input"]["name"]
        calibration = meta["calibration"]
        self.temperature = float(calibration["temperature"])
        self.threshold = float(calibration["confidence_threshold"])
        self.calibrated = "not calibrated" not in str(calibration.get("threshold_source", ""))
        self.crops = sorted(crop_table(self.class_ids))
        self.run_name = meta.get("source_run", "unknown")
        self.sha256 = meta.get("sha256", "")

    @classmethod
    def from_files(cls, meta_path, threads: int = 2, verify_hash: bool = True) -> "Classifier":
        meta_path = Path(meta_path)
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Model metadata not found: {meta_path}. Run `python -m ml.export_onnx --run ...`."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format_version") != SUPPORTED_META_FORMAT:
            raise ValueError(f"unsupported model metadata format: {meta.get('format_version')!r}")
        model_path = meta_path.parent / meta["model_file"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found next to its metadata: {model_path}")
        if verify_hash:
            actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual != meta["sha256"]:
                raise ValueError(
                    f"{model_path.name} does not match the hash in {meta_path.name}. "
                    "Re-run the export so both files come from the same run."
                )
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].name != meta["input"]["name"]:
            raise ValueError("the model's input name differs from its metadata")
        if session.get_outputs()[0].shape[-1] != len(meta["output"]["class_ids"]):
            raise ValueError("the model's output size differs from the class list in its metadata")
        return cls(session, meta)

    def predict(self, image: Image.Image, crop: str | None = None, top_k: int = 3) -> dict:
        began = time.perf_counter()
        batch = preprocess(image, self.size, self.mean, self.std)
        prepared = time.perf_counter()
        logits = self.session.run(None, {self.input_name: batch})[0][0]
        finished = time.perf_counter()
        result = decide(logits, self.class_ids, self.temperature, self.threshold, crop, top_k)
        result["timing_ms"] = {
            "preprocess": round((prepared - began) * 1000, 1),
            "inference": round((finished - prepared) * 1000, 1),
        }
        return result

    def info(self) -> dict:
        return {
            "run": self.run_name,
            "arch": self.meta.get("arch"),
            "sha256": self.sha256[:12],
            "classes": self.class_ids,
            "crops": self.crops,
            "image_size": self.size,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
        }
