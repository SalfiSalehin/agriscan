"""Safe decoding of user uploads. Nothing is written to disk and nothing is kept."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}  # decided from the file's bytes, not its name or header


class ImageRejected(Exception):
    """The upload cannot be used. `code` is stable (the front end translates it)."""

    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _to_rgb(image: Image.Image) -> Image.Image:
    """RGB copy. Transparent areas become white, not black."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return image.convert("RGB")


def load_image(data: bytes, max_pixels: int, min_side: int) -> Image.Image:
    """Decode an uploaded image, or raise ImageRejected.

    The size is checked from the header before any pixels are decoded. EXIF rotation is applied
    (phone photos are often stored sideways); the training pipeline did not do this, but leaf
    orientation does not matter to the model and a sideways photo is still a leaf.
    """
    if not data:
        raise ImageRejected("empty_file", "The uploaded file is empty.", 400)
    try:
        probe = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise ImageRejected("too_many_pixels", "The image has too many pixels.") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageRejected("not_an_image", "The file is not an image we can read.") from None
    with probe:
        if probe.format not in ALLOWED_FORMATS:
            raise ImageRejected("unsupported_format", "Please upload a JPEG, PNG or WebP image.")
        width, height = probe.size
        if width * height > max_pixels:
            raise ImageRejected("too_many_pixels", "The image has too many pixels.")
        if min(width, height) < min_side:
            raise ImageRejected("too_small", f"The image is too small (minimum {min_side} px).")
        try:
            probe.seek(0)  # animated files: first frame only
            probe.load()
        except (OSError, SyntaxError, ValueError, EOFError, Image.DecompressionBombError):
            raise ImageRejected("corrupt_image", "The image file is damaged.") from None
        try:
            oriented = ImageOps.exif_transpose(probe)
        except (OSError, SyntaxError, ValueError, TypeError, KeyError):
            oriented = probe  # unreadable EXIF block: use the pixels as they are
        return _to_rgb(oriented)
