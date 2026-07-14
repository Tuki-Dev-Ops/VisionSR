"""Image decode/encode.

OpenCV is used for pixel work (fast, and already a dependency for the analyser)
but Pillow owns file I/O, because cv2.imread silently returns None on a bad path,
mangles non-ASCII paths on Windows, and drops EXIF and ICC profiles.

Everything inside the engine is **RGB**. The BGR/RGB flip happens exactly here and
in :func:`save_image`, nowhere else.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms, ImageOps

from ..core.config import get_settings
from ..core.errors import ImageTooLargeError, InvalidImageError

log = logging.getLogger(__name__)

# Pillow refuses very large files by default as a decompression-bomb guard. Our own
# megapixel ceiling is the real limit, so lift Pillow's and let ours speak.
PILImage.MAX_IMAGE_PIXELS = None

SUPPORTED_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jfif", ".avif"}
)


def load_image(source: Path | bytes | io.BytesIO) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode to HWC uint8 RGB(A) plus the metadata needed to write it back out.

    Returns:
        (image, metadata). Metadata carries ``exif``, ``icc_profile`` and ``format``
        so an export can round-trip them.
    """
    try:
        handle = PILImage.open(source if isinstance(source, Path) else io.BytesIO(_as_bytes(source)))
    except Exception as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc

    metadata: dict[str, Any] = {
        "format": handle.format,
        "mode": handle.mode,
        "exif": handle.info.get("exif"),
        "icc_profile": handle.info.get("icc_profile"),
    }

    # Phones store orientation in EXIF rather than rotating pixels. Apply it now, or
    # the output comes out sideways.
    handle = ImageOps.exif_transpose(handle)

    handle = _to_srgb(handle, metadata)

    if handle.mode not in ("RGB", "RGBA"):
        handle = handle.convert("RGBA" if "A" in handle.mode or handle.mode == "P" else "RGB")

    image = np.asarray(handle, dtype=np.uint8)
    _check_size(image)

    metadata["has_alpha"] = image.shape[2] == 4
    return image, metadata


def _as_bytes(source: bytes | io.BytesIO) -> bytes:
    return source.getvalue() if isinstance(source, io.BytesIO) else source


def _to_srgb(handle: PILImage.Image, metadata: dict[str, Any]) -> PILImage.Image:
    """Convert a wide-gamut source (Display P3, Adobe RGB) into sRGB.

    The models were trained on sRGB. Feeding them P3 data without converting makes
    saturated colours drift — subtly, but visibly on skin tones.
    """
    profile = metadata.get("icc_profile")
    if not profile:
        return handle

    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        if ImageCms.getProfileDescription(src).strip().lower().startswith("srgb"):
            return handle

        converted = ImageCms.profileToProfile(
            handle, src, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
        log.debug("converted %s -> sRGB", ImageCms.getProfileDescription(src).strip())
        # The pixels are sRGB now; keeping the old profile would double-convert on save.
        metadata["icc_profile"] = None
        return converted or handle
    except Exception as exc:
        log.warning("ICC conversion failed (%s); treating input as sRGB", exc)
        return handle


def _check_size(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise InvalidImageError(f"Expected an RGB or RGBA image, got shape {image.shape}.")

    limit = get_settings().max_input_megapixels
    megapixels = (image.shape[0] * image.shape[1]) / 1_000_000
    if megapixels > limit:
        raise ImageTooLargeError(megapixels, limit)


def save_image(
    image: np.ndarray,
    path: Path,
    *,
    format: str | None = None,
    quality: int = 95,
    metadata: dict[str, Any] | None = None,
    preserve_exif: bool = True,
) -> Path:
    """Encode HWC uint8 RGB(A) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = (format or path.suffix.lstrip(".") or "png").lower()
    if fmt == "jpg":
        fmt = "jpeg"

    handle = PILImage.fromarray(_as_uint8(image))

    if fmt == "jpeg" and handle.mode == "RGBA":
        # JPEG has no alpha. Compositing on white is the least surprising choice;
        # a caller who cares should have asked for PNG or WebP.
        background = PILImage.new("RGB", handle.size, (255, 255, 255))
        background.paste(handle, mask=handle.split()[3])
        handle = background

    params: dict[str, Any] = {}
    if fmt in ("jpeg", "webp"):
        params["quality"] = int(quality)
    if fmt == "jpeg":
        params["subsampling"] = 0  # 4:4:4 — do not throw away the chroma we just rebuilt
        params["optimize"] = True
    if fmt == "png":
        params["compress_level"] = 6
    if fmt == "webp":
        params["method"] = 6

    if preserve_exif and metadata:
        if metadata.get("exif"):
            params["exif"] = metadata["exif"]
        if metadata.get("icc_profile"):
            params["icc_profile"] = metadata["icc_profile"]

    handle.save(path, format=fmt.upper(), **params)
    return path


def encode_image(
    image: np.ndarray,
    format: str = "png",
    quality: int = 95,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Same as :func:`save_image` but to memory — used by the HTTP layer."""
    buffer = io.BytesIO()
    fmt = "jpeg" if format.lower() in ("jpg", "jpeg") else format.lower()

    handle = PILImage.fromarray(_as_uint8(image))
    if fmt == "jpeg" and handle.mode == "RGBA":
        background = PILImage.new("RGB", handle.size, (255, 255, 255))
        background.paste(handle, mask=handle.split()[3])
        handle = background

    params: dict[str, Any] = {}
    if fmt in ("jpeg", "webp"):
        params["quality"] = int(quality)
    if fmt == "jpeg":
        params["subsampling"] = 0
    if metadata and metadata.get("exif"):
        params["exif"] = metadata["exif"]

    handle.save(buffer, format=fmt.upper(), **params)
    return buffer.getvalue()


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image * 255.0 if image.dtype.kind == "f" else image, 0, 255).astype(np.uint8)


def to_float(image: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]."""
    if image.dtype == np.float32:
        return image
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    """float32 [0,1] -> uint8 [0,255], with rounding rather than truncation."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Separate RGB from alpha. Alpha never goes through the SR network."""
    if image.shape[2] == 4:
        return np.ascontiguousarray(image[:, :, :3]), np.ascontiguousarray(image[:, :, 3])
    return image, None


def merge_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return rgb
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    return np.concatenate([rgb, alpha], axis=2)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES
