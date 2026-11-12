"""Phase 3A: export a trained checkpoint to ONNX, check it against PyTorch, measure size and speed.

    pip install onnx onnxruntime
    python -m ml.export_onnx --run runs/<run name>

Writes models/agriscan_<arch>.onnx and models/agriscan_<arch>.meta.json (class order,
preprocessing, temperature, confidence threshold) and a copy of the meta file in reports/onnx/.
The temperature and threshold are read from reports/eval/<run>/metrics.json, written by
ml.evaluate. Without it the config's default threshold is used and the model is marked
"not calibrated".
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ml.config import ROOT, load_config
from ml.data import ManifestDataset, load_class_index
from ml.models import build_model
from ml.onnx_meta import build_meta, latency_summary, parity_report
from ml.transforms import build_eval_transform


def export(model, size: int, path: Path, opset: int) -> None:
    """torch.onnx.export across torch versions (the `dynamo` argument only exists in newer ones)."""
    dummy = torch.zeros(1, 3, size, size)
    kwargs = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset,
    }
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)


def parity_batch(cfg, class_index, size: int, count: int) -> torch.Tensor:
    """`count` validation images spread evenly over the split, preprocessed for the model."""
    dataset = ManifestDataset(
        ROOT / cfg.paths["splits_dir"] / "val.csv",
        ROOT / cfg.dataset["export_dir"],
        class_index,
        build_eval_transform(size),
    )
    picks = np.linspace(0, len(dataset) - 1, count).astype(int)
    return torch.stack([dataset[int(i)][0] for i in picks])


def benchmark(path: Path, size: int, threads: int, runs: int, warmup: int = 20) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((1, 3, size, size)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"input": x})
    times = []
    for _ in range(runs):
        began = time.perf_counter()
        session.run(None, {"input": x})
        times.append((time.perf_counter() - began) * 1000)
    return latency_summary(times, threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--parity-images", type=int, default=64)
    parser.add_argument("--bench-runs", type=int, default=200)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("Install the ONNX packages first: pip install onnx onnxruntime") from None

    cfg = load_config()
    class_index = load_class_index(cfg)
    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    if meta["class_ids"] != sorted(class_index, key=class_index.get):
        raise SystemExit("class order in the checkpoint differs from class_index.json")
    size = int(meta["image_size"])
    model = build_model(meta["arch"], len(meta["class_ids"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = ROOT / cfg.paths["models_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"agriscan_{meta['arch']}"
    onnx_path = out_dir / f"{stem}.onnx"
    print(f"torch {torch.__version__}: exporting {run_dir.name} (opset {args.opset}) ...")
    export(model, size, onnx_path, args.opset)
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 2**20:.2f} MB)")

    # does ONNX Runtime give the same answers as PyTorch? (also exercises the dynamic batch axis)
    batch = parity_batch(cfg, class_index, size, args.parity_images)
    with torch.inference_mode():
        expected = model(batch).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"input": batch.numpy()})[0]
    parity = parity_report(expected, got)
    single = session.run(None, {"input": batch[:1].numpy()})[0]
    parity["batch_of_one_matches_batch"] = bool(np.allclose(single[0], got[0], atol=1e-4))
    print("Parity vs PyTorch:", json.dumps(parity))
    if not parity["passed"] or not parity["batch_of_one_matches_batch"]:
        raise SystemExit("ONNX output does not match PyTorch. Do not use this file.")

    latency = [benchmark(onnx_path, size, t, args.bench_runs) for t in (1, 2)]
    for row in latency:
        print(
            f"CPU, {row['threads']} thread(s), batch 1: median {row['median_ms']} ms, "
            f"p95 {row['p95_ms']} ms (model only: no image decoding or resizing)"
        )

    eval_path = ROOT / cfg.paths["reports_dir"] / "eval" / run_dir.name / "metrics.json"
    if eval_path.exists():
        results = json.loads(eval_path.read_text(encoding="utf-8"))["results"]
        temperature = results["temperature"]
        chosen = results["threshold"]["chosen_threshold"]
        if chosen is None:
            threshold, source = (
                cfg.inference["confidence_threshold"],
                "config default (calibration found no threshold)",
            )
        else:
            threshold, source = chosen, f"calibrated by ml.evaluate ({eval_path.name})"
    else:
        print("WARNING: no evaluation found for this run, so the model is NOT calibrated.")
        temperature = 1.0
        threshold, source = cfg.inference["confidence_threshold"], "config default (not calibrated)"

    info = build_meta(
        run_name=run_dir.name,
        arch=meta["arch"],
        class_ids=meta["class_ids"],
        image_size=size,
        mean=meta["mean"],
        std=meta["std"],
        temperature=temperature,
        threshold=threshold,
        threshold_source=source,
        onnx_path=onnx_path,
        parity=parity,
        latency=latency,
        versions={
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "os": platform.platform(),
        },
    )
    text = json.dumps(info, indent=2)
    (out_dir / f"{stem}.meta.json").write_text(text, encoding="utf-8")
    report_dir = ROOT / cfg.paths["reports_dir"] / "onnx"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{stem}.meta.json").write_text(text, encoding="utf-8")
    print(f"Wrote models/{stem}.meta.json (temperature {temperature}, threshold {threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
