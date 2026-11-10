"""Environment check. Copy the output into your README's 'Training environment' section."""

import platform
import sys

sys.path.insert(0, ".")
from ml.config import load_config  # noqa: E402

print(f"Python      : {sys.version.split()[0]} on {platform.platform()}")

try:
    import torch

    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU         : {p.name}, {p.total_memory / 1024**3:.1f} GB")
        print(f"CUDA runtime: {torch.version.cuda}")
        x = torch.randn(256, 256, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            y = x @ x
        print(f"AMP matmul  : OK ({y.dtype})")
except ImportError:
    print("PyTorch     : NOT INSTALLED")

try:
    import onnxruntime

    print(f"ONNX Runtime: {onnxruntime.__version__}")
except ImportError:
    print("ONNX Runtime: not installed yet (fine, needed in Phase 3)")

cfg = load_config()
print(f"Config      : OK, {len(cfg.classes)} classes, seed={cfg.seed}")
