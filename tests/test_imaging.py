import io

import numpy as np
import pytest
from PIL import Image

from app.imaging import ImageRejected, load_image
from tests.conftest import image_bytes

MAX_PIXELS, MIN_SIDE = 4_000_000, 64


def load(data):
    return load_image(data, MAX_PIXELS, MIN_SIDE)


def rejected(data, **kwargs):
    with pytest.raises(ImageRejected) as caught:
        load_image(data, kwargs.get("max_pixels", MAX_PIXELS), kwargs.get("min_side", MIN_SIDE))
    return caught.value


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_accepts_the_allowed_formats(fmt):
    image = load(image_bytes(fmt))
    assert image.mode == "RGB" and image.size == (120, 90)


def test_rejects_empty_garbage_and_other_formats():
    assert rejected(b"").code == "empty_file" and rejected(b"").status == 400
    assert rejected(b"this is not an image at all").code == "not_an_image"
    assert rejected(image_bytes("GIF")).code == "unsupported_format"
    assert rejected(image_bytes("BMP")).code == "unsupported_format"


def test_rejects_too_small_and_too_many_pixels():
    assert rejected(image_bytes(size=(40, 200))).code == "too_small"
    assert rejected(image_bytes(size=(300, 300)), max_pixels=50_000).code == "too_many_pixels"


def test_rejects_truncated_files():
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    noisy.save(buffer, "JPEG", quality=95)
    data = buffer.getvalue()
    assert rejected(data[: len(data) // 2]).code == "corrupt_image"


def test_exif_rotation_is_applied():
    exif = Image.Exif()
    exif[274] = 6  # "rotate 90 degrees clockwise to display"
    image = load(image_bytes("JPEG", size=(100, 70), exif=exif))
    assert image.size == (70, 100)


def test_transparent_png_is_composited_on_white_not_black():
    rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    buffer = io.BytesIO()
    rgba.save(buffer, "PNG")
    image = load(buffer.getvalue())
    assert image.getpixel((50, 50)) == (255, 255, 255)


def test_animated_image_uses_first_frame():
    frames = [Image.new("RGB", (100, 100), c) for c in ((255, 0, 0), (0, 0, 255))]
    buffer = io.BytesIO()
    frames[0].save(buffer, "WEBP", save_all=True, append_images=frames[1:], duration=50)
    image = load(buffer.getvalue())
    r, _, b = image.getpixel((50, 50))
    assert r > 200 and b < 60
