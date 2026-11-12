"""Torch-free helpers for the ONNX export: parity check, latency summary, metadata file."""

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np

PARITY_MAX_ABS_DIFF = 1e-3


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parity_report(torch_logits, onnx_logits) -> dict:
    """Compare PyTorch and ONNX Runtime outputs on the same inputs. Raises if they disagree."""
    a = np.asarray(torch_logits, dtype=np.float64)
    b = np.asarray(onnx_logits, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"output shapes differ: {a.shape} vs {b.shape}")
    report = {
        "images": int(len(a)),
        "max_abs_logit_diff": float(np.abs(a - b).max()),
        "same_prediction_share": float((a.argmax(axis=1) == b.argmax(axis=1)).mean()),
    }
    report["passed"] = (
        report["max_abs_logit_diff"] < PARITY_MAX_ABS_DIFF
        and report["same_prediction_share"] == 1.0
    )
    return report


def latency_summary(times_ms, threads: int) -> dict:
    t = np.asarray(times_ms, dtype=float)
    return {
        "threads": int(threads),
        "runs": int(len(t)),
        "median_ms": round(float(np.median(t)), 2),
        "p95_ms": round(float(np.percentile(t, 95)), 2),
        "mean_ms": round(float(t.mean()), 2),
    }


def build_meta(
    *,
    run_name: str,
    arch: str,
    class_ids: list[str],
    image_size: int,
    mean,
    std,
    temperature: float,
    threshold: float,
    threshold_source: str,
    onnx_path,
    parity: dict,
    latency: list[dict],
    versions: dict,
) -> dict:
    """Everything the web backend needs to use the model exactly the way it was evaluated."""
    onnx_path = Path(onnx_path)
    return {
        "format_version": 1,
        "model_file": onnx_path.name,
        "sha256": file_sha256(onnx_path),
        "size_mb": round(onnx_path.stat().st_size / 2**20, 2),
        "source_run": run_name,
        "arch": arch,
        "input": {
            "name": "input",
            "layout": "NCHW float32, dynamic batch",
            "size": int(image_size),
            "mean": list(mean),
            "std": list(std),
            "preprocessing": (
                f"RGB; resize the shorter side to {round(image_size * 256 / 224)} (bilinear); "
                f"centre-crop {image_size}x{image_size}; scale to 0-1; subtract mean; divide by std"
            ),
        },
        "output": {"name": "logits", "class_ids": list(class_ids)},
        "calibration": {
            "temperature": float(temperature),
            "how_to_use": "probabilities = softmax(logits / temperature)",
            "confidence_threshold": float(threshold),
            "threshold_source": threshold_source,
            "below_threshold": "answer 'not sure', do not name a disease",
        },
        "parity_vs_pytorch": parity,
        "cpu_latency_batch1_model_only": latency,
        "environment": versions,
        "exported": datetime.now().isoformat(timespec="seconds"),
    }
