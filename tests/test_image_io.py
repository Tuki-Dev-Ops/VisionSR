"""Decode/encode round-trips.

These need no checkpoints: they are about what the file says, not about what the
network produced. Everything here was visible in the before/after canvas — a result
that came back rotated, or opaque, next to the source it was made from.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageCms


def _jpeg_with_orientation(size: tuple[int, int], orientation: int) -> bytes:
    """A JPEG whose stored pixels are `size` and whose EXIF asks for a rotation."""
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x010F] = "TestCam"  # a tag with no opinion about geometry
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(buffer, "JPEG", exif=exif.tobytes())
    return buffer.getvalue()


def test_encoded_result_is_not_rotated_twice():
    """The exported EXIF must not re-apply an orientation we already baked in.

    The bug: ``load_image`` read ``info["exif"]`` *before* ``exif_transpose`` and
    handed that blob to the encoder. The pixels were upright, but the block still
    said "rotate 90", so every viewer rotated them again. In the compare canvas the
    original (rotated once, correctly) sat next to a result lying on its side.
    """
    from visionsr.preprocessing.io import encode_image, load_image

    image, metadata = load_image(_jpeg_with_orientation((400, 200), 6))
    assert image.shape[:2] == (400, 200), "exif_transpose did not upright the source"

    encoded = encode_image(image, format="jpeg", metadata=metadata)
    result = Image.open(io.BytesIO(encoded))

    assert result.getexif().get(0x0112) in (None, 1), (
        "the result still carries an Orientation tag: a viewer will rotate the "
        "already-upright pixels a second time"
    )
    # What a browser actually paints, for both files.
    from PIL import ImageOps

    assert ImageOps.exif_transpose(result).size == (200, 400)


def test_export_keeps_the_camera_tags_it_can_still_vouch_for():
    from visionsr.preprocessing.io import encode_image, load_image

    image, metadata = load_image(_jpeg_with_orientation((400, 200), 6))
    result = Image.open(io.BytesIO(encode_image(image, format="jpeg", metadata=metadata)))

    assert result.getexif().get(0x010F) == "TestCam"


def test_exported_exif_describes_the_exported_frame():
    """PixelXDimension/PixelYDimension must follow the pixels through an upscale."""
    from visionsr.preprocessing.io import encode_image, load_image

    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA002] = 400
    exif.get_ifd(0x8769)[0xA003] = 200
    buffer = io.BytesIO()
    Image.new("RGB", (400, 200), (10, 90, 200)).save(buffer, "JPEG", exif=exif.tobytes())

    image, metadata = load_image(buffer.getvalue())
    upscaled = np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)

    result = Image.open(io.BytesIO(encode_image(upscaled, format="jpeg", metadata=metadata)))
    reported = result.getexif().get_ifd(0x8769)

    assert result.size == (1600, 800)
    assert (reported.get(0xA002), reported.get(0xA003)) == (1600, 800)


def test_wide_gamut_source_keeps_its_alpha(monkeypatch: pytest.MonkeyPatch):
    """The sRGB conversion must not flatten a transparent source.

    ``ImageCms.profileToProfile`` only emits RGB. Handing it an RGBA image returned a
    fully opaque one, so a cutout of a Display-P3 PNG came back as a solid rectangle —
    the background removal looked like it had done nothing at all.

    Pillow only ships built-in profiles that ``_to_srgb`` recognises as sRGB and skips,
    so the description is faked: the transform itself is real, and it is the transform
    that was eating the alpha.
    """
    from visionsr.preprocessing import io as image_io

    monkeypatch.setattr(
        image_io.ImageCms, "getProfileDescription", lambda _profile: "Display P3"
    )

    source = Image.new("RGBA", (16, 16), (220, 40, 40, 255))
    source.putalpha(Image.new("L", (16, 16), 0))  # fully transparent

    buffer = io.BytesIO()
    source.save(
        buffer, "PNG", icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    )

    image, metadata = image_io.load_image(buffer.getvalue())

    assert image.shape[2] == 4, "the alpha channel was dropped by the colour conversion"
    assert metadata["has_alpha"] is True
    assert int(image[:, :, 3].max()) == 0, "the transparent source came back opaque"
    # Converted pixels are sRGB now — the old profile must not ride along and be
    # applied a second time on export.
    assert metadata["icc_profile"] is None


@pytest.mark.parametrize("fmt", ["png", "webp", "jpeg"])
def test_formats_round_trip_with_metadata(fmt: str):
    """Every output format must survive being handed a source's EXIF."""
    from visionsr.preprocessing.io import encode_image, load_image

    image, metadata = load_image(_jpeg_with_orientation((64, 32), 8))
    decoded = Image.open(io.BytesIO(encode_image(image, format=fmt, metadata=metadata)))

    assert decoded.size == (32, 64)
    assert decoded.getexif().get(0x0112) in (None, 1)
