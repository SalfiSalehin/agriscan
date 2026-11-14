# AgriScan web app for Hugging Face Spaces (Docker SDK) or any container host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Spaces run containers as a non-root user with id 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user models/agriscan_mobilenet_v3_small.onnx models/agriscan_mobilenet_v3_small.meta.json ./models/

USER user

# Behind the Spaces proxy: count visitors by the address the proxy saw, hide Swagger, and keep
# the (optional) feedback log in /tmp, the only place guaranteed writable. It is lost on restart.
ENV AGRISCAN_TRUST_PROXY=1 \
    AGRISCAN_DOCS=off \
    AGRISCAN_DB=/tmp/agriscan.db \
    AGRISCAN_THREADS=2

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-server-header"]
