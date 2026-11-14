"""The prediction use case, independent of the web framework: bytes in, answer or error out."""

from app.content import LANGS, Content
from app.db import Store
from app.imaging import ImageRejected, load_image
from app.inference import Classifier
from app.settings import Settings


class ServiceError(Exception):
    """An error with an HTTP status and a stable, translatable code."""

    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.headers = headers or {}


def normalise_crop(value: str | None, valid: list[str]) -> str | None:
    """None, '' and 'auto' mean "no crop picked". Anything else must be a known crop id."""
    if value is None:
        return None
    crop = value.strip().lower()
    if crop in ("", "auto"):
        return None
    if crop not in valid:
        raise ServiceError(422, "unknown_crop", f"Unknown crop. Choose one of: {', '.join(valid)}.")
    return crop


def normalise_lang(value: str | None) -> str:
    """None and '' mean English; anything else must be 'en' or 'bn'."""
    lang = (value or "en").strip().lower()
    if lang not in LANGS:
        raise ServiceError(422, "unknown_language", "Language must be 'en' or 'bn'.")
    return lang


def predict_upload(
    classifier: Classifier,
    data: bytes,
    crop: str | None,
    settings: Settings,
    content: Content | None = None,
    store: Store | None = None,
    lang: str | None = None,
):
    lang_id = normalise_lang(lang)
    crop_id = normalise_crop(crop, classifier.crops)
    try:
        image = load_image(data, settings.max_pixels, settings.min_side)
    except ImageRejected as error:
        raise ServiceError(error.status, error.code, error.message) from error
    result = classifier.predict(image, crop_id)
    result["crop_mode"] = "picked" if crop_id else "auto"
    result["crop"] = crop_id
    result["model"] = {"run": classifier.run_name, "sha256": classifier.sha256[:12]}
    result["lang"] = lang_id
    if content is not None:
        for item in result["predictions"]:
            item["name"] = content.name(item["class_id"], lang_id)
        if result["status"] == "ok":
            result["info"] = content.disease(result["diagnosis"], lang_id)
        else:
            result["info"] = content.unsure(lang_id)
    result["prediction_id"] = store.record(result, lang_id) if store is not None else None
    return result
