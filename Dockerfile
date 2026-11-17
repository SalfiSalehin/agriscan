# AgriScan web app, for any container host (Render, Cloud Run, Fly.io, ...).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as a non-root user.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user models/agriscan_resnet18.onnx models/agriscan_resnet18.meta.json ./models/

USER user

# Behind a host's reverse proxy: count visitors by the address the proxy saw, hide Swagger, and
# keep the (optional) feedback log in /tmp, which is writable but lost on restart. The limit is
# generous because a proxy chain can make several visitors look like one address.
ENV AGRISCAN_TRUST_PROXY=1 \
    AGRISCAN_DOCS=off \
    AGRISCAN_DB=/tmp/agriscan.db \
    AGRISCAN_THREADS=2 \
    AGRISCAN_RATE_PER_MIN=60

# The host tells us the port in $PORT (Render uses 10000); 7860 is the local default.
ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ['PORT'], timeout=3)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --no-server-header"]
