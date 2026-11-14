"""Runtime settings, read from environment variables (all optional)."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "models" / "agriscan_mobilenet_v3_small.meta.json"
DEFAULT_CONTENT = ROOT / "app" / "content" / "diseases.json"
STATIC_DIR = ROOT / "app" / "static"
DEFAULT_DB = ROOT / "data" / "agriscan.db"


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _flag_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be 1/0, true/false, yes/no or on/off, got {raw!r}")


def _db_path() -> Path | None:
    """AGRISCAN_DB: a file path, or "off" to store nothing. Default: data/agriscan.db."""
    raw = os.environ.get("AGRISCAN_DB", "")
    if raw.lower() == "off":
        return None
    return Path(raw) if raw else DEFAULT_DB


@dataclass(frozen=True)
class Settings:
    model_meta: Path = DEFAULT_META
    max_upload_bytes: int = 8 * 1024 * 1024  # largest accepted upload
    max_pixels: int = 40_000_000  # width x height; guards against decompression bombs
    min_side: int = 64  # shorter side in pixels; smaller images carry no usable detail
    threads: int = 2  # ONNX Runtime threads per request
    max_concurrent: int = 4  # predictions running at once; more get a 503 "busy"
    rate_per_min: int = 0  # requests per visitor per minute on POST endpoints; 0 = unlimited
    trust_proxy: bool = False  # read the visitor address from X-Forwarded-For (behind a proxy)
    docs: bool = True  # serve the Swagger page at /api/docs
    content_path: Path | None = None  # disease texts; None = answers carry no advice
    db_path: Path | None = None  # prediction/feedback log; None = nothing is stored

    @classmethod
    def from_env(cls) -> "Settings":
        meta = os.environ.get("AGRISCAN_MODEL_META")
        content = os.environ.get("AGRISCAN_CONTENT")
        return cls(
            model_meta=Path(meta) if meta else DEFAULT_META,
            max_upload_bytes=_int_env("AGRISCAN_MAX_UPLOAD_MB", 8, 1, 50) * 1024 * 1024,
            max_pixels=_int_env("AGRISCAN_MAX_PIXELS", 40_000_000, 1_000_000, 200_000_000),
            min_side=_int_env("AGRISCAN_MIN_SIDE", 64, 16, 1024),
            threads=_int_env("AGRISCAN_THREADS", 2, 1, 16),
            max_concurrent=_int_env("AGRISCAN_MAX_CONCURRENT", 4, 1, 64),
            rate_per_min=_int_env("AGRISCAN_RATE_PER_MIN", 30, 0, 10_000),
            trust_proxy=_flag_env("AGRISCAN_TRUST_PROXY", False),
            docs=_flag_env("AGRISCAN_DOCS", True),
            content_path=Path(content) if content else DEFAULT_CONTENT,
            db_path=_db_path(),
        )
