"""Phase 2: train a leaf-disease classifier (two-stage transfer learning, AMP).

    python -m ml.train --arch mobilenet_v3_small --smoke    # pipeline check, a few minutes
    python -m ml.train --arch mobilenet_v3_small            # the real run
    python -m ml.train --arch resnet18                      # baseline
    python -m ml.train --resume runs/<run name>/last.pt     # continue an interrupted run

Stage 1 trains only the classifier head (backbone and its BatchNorm statistics frozen).
Stage 2 fine-tunes everything with a lower learning rate, cosine schedule and early stopping on
validation loss. Only the validation split is used for model selection; the test split is never
touched here.
"""

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ml.augment import RandomBackground
from ml.config import ROOT, load_config
from ml.data import ManifestDataset, class_weights, load_class_index, verify_manifests
from ml.metrics import accuracy, confusion_matrix, macro_f1, per_class_prf
from ml.models import (
    ARCHS,
    build_model,
    count_params,
    set_backbone_trainable,
    set_train_mode,
    split_params,
)
from ml.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform, build_train_transform

MIN_DELTA = 1e-4  # a validation-loss drop smaller than this does not count as improvement


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--arch", choices=ARCHS, default="mobilenet_v3_small")
    p.add_argument("--name", default=None, help="run name (default: <arch>_s<seed>_<time>)")
    p.add_argument("--seed", type=int, default=None, help="default: project.seed from the config")
    p.add_argument("--bg-prob", type=float, default=None, help="chance of replacing the background")
    p.add_argument("--erase-prob", type=float, default=None, help="chance of random erasing")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--stage1-epochs", type=int, default=None)
    p.add_argument(
        "--stage2-epochs", type=int, default=None, help="maximum, early stopping applies"
    )
    p.add_argument(
        "--smoke", action="store_true", help="tiny subset, 1+1 epochs, nothing in reports/"
    )
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision (debugging)")
    p.add_argument("--resume", type=Path, default=None, help="path to a run's last.pt")
    return p.parse_args()


def resolve_hparams(args: argparse.Namespace, cfg) -> dict:
    hp = dict(cfg.train)
    overrides = {
        "bg_prob": args.bg_prob,
        "erase_prob": args.erase_prob,
        "num_workers": args.workers,
        "batch_size": args.batch_size,
        "stage1_epochs": args.stage1_epochs,
        "stage2_max_epochs": args.stage2_epochs,
    }
    hp.update({k: v for k, v in overrides.items() if v is not None})
    if args.smoke:
        hp["stage1_epochs"] = 1
        hp["stage2_max_epochs"] = 1
    return hp


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_info() -> dict:
    def run(*cmd: str) -> str:
        done = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def make_loaders(cfg, hp: dict, smoke: bool, seed: int, class_index: dict):
    root = ROOT / cfg.dataset["export_dir"]
    splits = ROOT / cfg.paths["splits_dir"]
    size = cfg.data["image_size"]
    train_limit, val_limit = (512, 256) if smoke else (None, None)
    background = RandomBackground(hp["bg_prob"]) if hp["bg_prob"] > 0 else None
    train_ds = ManifestDataset(
        splits / "train.csv",
        root,
        class_index,
        build_train_transform(size, hp["erase_prob"]),
        train_limit,
        background=background,
        mask_root=ROOT / cfg.dataset["mask_dir"],
    )
    if background is not None:
        coverage = train_ds.mask_coverage()
        print(f"Background masks found for {coverage:.1%} of the training images")
        if coverage < 0.9:
            raise SystemExit(
                "Too few leaf masks. Run: python -m ml.make_masks (or use --bg-prob 0)"
            )
    val_ds = ManifestDataset(
        splits / "val.csv", root, class_index, build_eval_transform(size), val_limit
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = hp["num_workers"]
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=hp["batch_size"] * 2, shuffle=False, **common)
    return train_loader, val_loader


def configure_stage(model: nn.Module, arch: str, stage: int, hp: dict, n_epochs: int):
    set_backbone_trainable(model, arch, stage == 2)
    backbone, head = split_params(model, arch)
    if stage == 1:
        return AdamW(head, lr=hp["stage1_lr"], weight_decay=hp["weight_decay"]), None
    optimizer = AdamW(
        [
            {"params": backbone, "lr": hp["stage2_lr_backbone"]},
            {"params": head, "lr": hp["stage2_lr_head"]},
        ],
        weight_decay=hp["weight_decay"],
    )
    return optimizer, CosineAnnealingLR(optimizer, T_max=n_epochs)


def to_device(images: torch.Tensor, labels: torch.Tensor, device: torch.device):
    images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return images, labels.to(device, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, hp, use_amp):
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = to_device(images, labels, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        loss = criterion(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.numel()
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, use_amp: bool):
    """Plain (unweighted, unsmoothed) cross-entropy, accuracy, macro-F1 and the confusion matrix."""
    model.eval()
    total_loss, seen = 0.0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in loader:
        images, labels = to_device(images, labels, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        total_loss += nn.functional.cross_entropy(logits.float(), labels, reduction="sum").item()
        seen += labels.numel()
        cm += confusion_matrix(labels.cpu().numpy(), logits.argmax(1).cpu().numpy(), num_classes)
    metrics = {"loss": total_loss / seen, "acc": accuracy(cm), "macro_f1": macro_f1(cm)}
    return metrics, cm


def plot_curves(history: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(11, 4))
    x = history["epoch"]
    ax_loss.plot(x, history["train_loss"], label="train (weighted + label smoothing)")
    ax_loss.plot(x, history["val_loss"], label="val (plain cross-entropy)")
    ax_loss.set_ylabel("loss")
    ax_score.plot(x, history["val_acc"], label="val accuracy")
    ax_score.plot(x, history["val_macro_f1"], label="val macro-F1")
    ax_score.set_ylabel("score")
    for ax in (ax_loss, ax_score):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
        if (history["stage"] == 1).any():
            ax.axvline(history.loc[history["stage"] == 1, "epoch"].max() + 0.5, ls="--", c="grey")
    fig.suptitle(f"{title} (dashed line: stage 1 -> stage 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config()

    resume_ckpt = None
    if args.resume:
        run_dir = args.resume.resolve().parent
        saved = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        arch, hp, seed = saved["arch"], saved["hparams"], saved["seed"]
        name, smoke = saved["name"], saved["smoke"]
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
    else:
        arch, smoke = args.arch, args.smoke
        seed = args.seed if args.seed is not None else cfg.seed
        hp = resolve_hparams(args, cfg)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        name = args.name or f"{'smoke_' if smoke else ''}{arch}_s{seed}_{stamp}"
        run_dir = ROOT / "runs" / name
    report_dir = ROOT / cfg.paths["reports_dir"] / "training" / name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cpu":
        print("WARNING: CUDA is not available; training on the CPU will be very slow.")
    seed_everything(seed)

    verify_manifests(cfg)
    class_index = load_class_index(cfg)
    class_ids = sorted(class_index, key=class_index.get)
    num_classes = len(class_ids)
    train_loader, val_loader = make_loaders(cfg, hp, smoke, seed, class_index)

    splits = ROOT / cfg.paths["splits_dir"]
    train_labels = pd.read_csv(splits / "train.csv")["class_id"].map(class_index).to_numpy()
    weights = class_weights(train_labels, num_classes, hp["class_weight_power"])

    model = build_model(arch, num_classes, pretrained=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device), label_smoothing=hp["label_smoothing"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    meta = {
        "arch": arch,
        "class_ids": class_ids,
        "image_size": cfg.data["image_size"],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "seed": seed,
    }

    state = {"stage": 1, "epoch_in_stage": 0, "epoch": 0, "best_loss": float("inf"),
             "best_epoch": 0, "bad_epochs": 0, "stopped_early": False, "history": []}  # fmt: skip
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        state = resume_ckpt["state"]
        print(f"Resuming {name} after epoch {state['epoch']} (stage {state['stage']})")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "name": name,
            "arch": arch,
            "seed": seed,
            "smoke": smoke,
            "amp": use_amp,
            "hparams": hp,
            "started": datetime.now().isoformat(timespec="seconds"),
            "environment": {
                "python": platform.python_version(),
                "os": platform.platform(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            },
            "git": git_info(),
            "data": {
                "manifest_digests": {
                    f"{s}.csv": file_digest(splits / f"{s}.csv") for s in ("train", "val", "test")
                },
                "train_images_used": len(train_loader.dataset),
                "val_images_used": len(val_loader.dataset),
            },
            "class_weights": {
                c: round(float(w), 4) for c, w in zip(class_ids, weights, strict=True)
            },
            "params_million": round(count_params(model) / 1e6, 3),
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print(f"Run {name}: {arch}, {run_config['params_million']} M parameters, device {device}")

    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    stage_epochs = {1: hp["stage1_epochs"], 2: hp["stage2_max_epochs"]}
    total_epochs = sum(stage_epochs.values())

    for stage in (1, 2):
        if stage < state["stage"]:
            continue
        optimizer, scheduler = configure_stage(model, arch, stage, hp, stage_epochs[stage])
        if resume_ckpt is not None and stage == state["stage"]:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            if scheduler is not None and resume_ckpt["scheduler"] is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
            scaler.load_state_dict(resume_ckpt["scaler"])
            start = state["epoch_in_stage"]
        else:
            start = 0
            state["bad_epochs"] = 0
        state["stage"] = stage

        for epoch_in_stage in range(start, stage_epochs[stage]):
            began = time.time()
            set_train_mode(model, arch, backbone_frozen=(stage == 1))
            lr = optimizer.param_groups[-1]["lr"]
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, hp, use_amp
            )
            val, _ = evaluate(model, val_loader, device, num_classes, use_amp)
            if scheduler is not None:
                scheduler.step()

            state["epoch"] += 1
            state["epoch_in_stage"] = epoch_in_stage + 1
            row = {
                "epoch": state["epoch"],
                "stage": stage,
                "lr": float(lr),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val["loss"]),
                "val_acc": float(val["acc"]),
                "val_macro_f1": float(val["macro_f1"]),
                "seconds": round(time.time() - began, 1),
            }
            state["history"].append(row)

            improved = val["loss"] < state["best_loss"] - MIN_DELTA
            if improved:
                state["best_loss"], state["best_epoch"], state["bad_epochs"] = (
                    float(val["loss"]),
                    state["epoch"],
                    0,
                )
                torch.save({"model": model.state_dict(), "meta": meta, "state": state}, best_path)
            else:
                state["bad_epochs"] += 1
            if stage == 2 and state["bad_epochs"] >= hp["early_stop_patience"]:
                state["stopped_early"] = True

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(),
                    "state": state,
                    "meta": meta,
                },
                last_path,
            )
            pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
            print(
                f"[stage {stage}] epoch {state['epoch']:>2}/{total_epochs} "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val['loss']:.4f} acc {val['acc']:.4f} f1 {val['macro_f1']:.4f} | "
                f"lr {lr:.1e} | {row['seconds']:.0f}s{' *best*' if improved else ''}"
            )
            if state["stopped_early"]:
                print(f"Early stopping: no val-loss improvement for {state['bad_epochs']} epochs.")
                break

    # ---- final report on the best checkpoint (validation split only) ----
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    val, cm = evaluate(model, val_loader, device, num_classes, use_amp)
    precision, recall, f1, support = per_class_prf(cm)
    history = pd.DataFrame(state["history"])
    summary = {
        "run": name,
        "arch": arch,
        "params_million": round(count_params(model) / 1e6, 3),
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
        "stopped_early": state["stopped_early"],
        "val_at_best": {k: round(float(v), 4) for k, v in val.items()},
        "train_seconds_total": round(float(history["seconds"].sum()), 1),
        "seconds_per_epoch_mean": round(float(history["seconds"].mean()), 1),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None
        ),
        "best_checkpoint_mb_pytorch": round(best_path.stat().st_size / 2**20, 2),
        "val_per_class": {
            c: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, c in enumerate(class_ids)
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_curves(history, run_dir / "curves.png", name)
    if not smoke:  # small files that belong in git (checkpoints stay in runs/, which is ignored)
        report_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("run_config.json", "history.csv", "summary.json", "curves.png"):
            shutil.copy(run_dir / filename, report_dir / filename)

    print(
        f"\nBest epoch {summary['best_epoch']} of {summary['epochs_run']}: {summary['val_at_best']}"
    )
    worst = sorted(summary["val_per_class"].items(), key=lambda kv: kv[1]["recall"])[:3]
    print("Lowest validation recall:", ", ".join(f"{c} {m['recall']:.3f}" for c, m in worst))
    print(f"Checkpoints: {run_dir}")
    if not smoke:
        print(f"Report files: {report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
