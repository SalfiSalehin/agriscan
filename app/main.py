"""The AgriScan web API.

    uvicorn app.main:app --reload            # development, http://127.0.0.1:8000/api/docs

Endpoints
    GET  /api/health    is the service up, which model is loaded
    GET  /api/model     classes, crops, temperature, threshold of the loaded model
    POST /api/predict   multipart form: `file` (image), optional `crop` ("tomato"), `lang` (en|bn)
    GET  /api/diseases/{class_id}?lang=en|bn   the advice page for one class
    GET  /api/classes?lang=en|bn   the model's classes with display names
    GET  /            the web page (app/static)
    POST /api/feedback  JSON {prediction_id, verdict: correct|wrong, true_class?}

Uploaded photos are decoded in memory and never written to disk or kept.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.content import Content
from app.db import Store
from app.inference import Classifier
from app.ratelimit import RateLimiter, client_ip
from app.service import ServiceError, normalise_lang, predict_upload
from app.settings import STATIC_DIR, Settings

log = logging.getLogger("agriscan")
# The page loads only its own scripts, styles and images (photos preview from blob: URLs).
PAGE_CSP = (
    "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
    "form-action 'self'"
)
BODY_OVERHEAD = 64 * 1024  # multipart framing around the file, allowed on top of the file limit


def error_response(status: int, code: str, message: str, headers: dict | None = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


class FeedbackIn(BaseModel):
    prediction_id: int
    verdict: Literal["correct", "wrong"]
    true_class: str | None = None


def create_app(
    classifier: Classifier | None = None,
    settings: Settings | None = None,
    content: Content | None = None,
    store: Store | None = None,
) -> FastAPI:
    """Build the app. Tests pass a ready `classifier`; otherwise the model is loaded at startup
    and a missing or inconsistent model stops the server instead of failing every request."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or Classifier.from_files(
            settings.model_meta, threads=settings.threads
        )
        loaded = app.state.classifier
        log.info("model %s loaded (threshold %.2f)", loaded.run_name, loaded.threshold)
        loaded_content = content
        if loaded_content is None and settings.content_path is not None:
            loaded_content = Content.load(settings.content_path)
        if loaded_content is not None:
            loaded_content.validate(loaded.class_ids)  # a gap stops the server at startup
        app.state.content = loaded_content
        opened = store
        if opened is None and settings.db_path is not None:
            opened = Store(settings.db_path)
        app.state.store = opened
        yield
        if opened is not None and store is None:
            opened.close()

    app = FastAPI(
        title="AgriScan",
        version="0.4.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.docs else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.docs else None,
    )
    app.state.slots = threading.BoundedSemaphore(settings.max_concurrent)
    limiter = RateLimiter(settings.rate_per_min) if settings.rate_per_min else None
    app.state.limiter = limiter
    limited_paths = ("/api/predict", "/api/feedback")

    @app.middleware("http")
    async def guard_and_headers(request: Request, call_next):
        path = request.url.path
        early = None
        if path == "/api/predict":
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > settings.max_upload_bytes + BODY_OVERHEAD:
                early = error_response(413, "too_large", "The upload is too large.")
        if early is None and limiter and request.method == "POST" and path in limited_paths:
            who = client_ip(
                request.client.host if request.client else None,
                request.headers.get("x-forwarded-for"),
                settings.trust_proxy,
            )
            allowed, wait = limiter.allow(f"{who}|{path}")
            if not allowed:
                early = error_response(
                    429,
                    "rate_limited",
                    "Too many requests. Please wait a moment and try again.",
                    {"Retry-After": str(wait)},
                )
        response = early or await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        else:
            response.headers["Content-Security-Policy"] = PAGE_CSP
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return error_response(error.status, error.code, error.message, error.headers)

    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", "model": request.app.state.classifier.run_name}

    @app.get("/api/model")
    def model_info(request: Request):
        return request.app.state.classifier.info()

    @app.post("/api/predict")
    def predict(
        request: Request,
        file: Annotated[UploadFile, File()],
        crop: Annotated[str | None, Form()] = None,
        lang: Annotated[str | None, Form()] = None,
    ):
        slots = request.app.state.slots
        if not slots.acquire(blocking=False):
            raise ServiceError(
                503,
                "busy",
                "The server is busy. Please try again in a moment.",
                {"Retry-After": "2"},
            )
        began = time.perf_counter()
        try:
            data = file.file.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise ServiceError(413, "too_large", "The upload is too large.")
            result = predict_upload(
                request.app.state.classifier,
                data,
                crop,
                settings,
                content=request.app.state.content,
                store=request.app.state.store,
                lang=lang,
            )
        finally:
            slots.release()
            file.file.close()
        log.info(
            "predict status=%s crop=%s total_ms=%.0f",
            result["status"],
            result["crop"],
            (time.perf_counter() - began) * 1000,
        )
        return result

    @app.get("/api/diseases/{class_id}")
    def disease(request: Request, class_id: str, lang: str = "en"):
        lang_id = normalise_lang(lang)
        loaded = request.app.state.content
        if loaded is not None and class_id == "unsure":  # the "not sure" advice block
            return loaded.unsure(lang_id)
        page = loaded.disease(class_id, lang_id) if loaded is not None else None
        if page is None:
            raise ServiceError(404, "unknown_class", "No information for this class.")
        return page

    @app.get("/api/classes")
    def classes(request: Request, lang: str = "en"):
        lang_id = normalise_lang(lang)
        loaded = request.app.state.content
        if loaded is None:
            raise ServiceError(404, "no_content", "No class names available.")
        return {"classes": loaded.classes(request.app.state.classifier.class_ids, lang_id)}

    @app.post("/api/feedback")
    def feedback(request: Request, body: FeedbackIn):
        db = request.app.state.store
        if db is None:
            raise ServiceError(404, "feedback_off", "Feedback is not being collected.")
        classifier_ids = request.app.state.classifier.class_ids
        if body.true_class is not None and body.true_class not in classifier_ids:
            raise ServiceError(422, "unknown_class", "Unknown class in true_class.")
        if body.verdict == "correct" and body.true_class is not None:
            raise ServiceError(422, "bad_feedback", "true_class is only for verdict 'wrong'.")
        if not db.add_feedback(body.prediction_id, body.verdict, body.true_class):
            raise ServiceError(404, "unknown_prediction", "No such prediction, or already rated.")
        return {"status": "thanks"}

    if STATIC_DIR.is_dir():  # last, so it never shadows an /api route
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")

    return app


app = create_app()
