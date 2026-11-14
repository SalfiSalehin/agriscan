"""Entry point for hosts that look for a top-level `index.py` (Vercel). Locally use:
uvicorn app.main:app
"""

from app.main import app  # noqa: F401
