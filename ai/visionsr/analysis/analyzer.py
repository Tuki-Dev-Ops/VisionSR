"""The analysis stage: everything the pipeline learns before it picks a model."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..core.types import ImageAnalysis
from . import quality
from .classifier import classify
from .faces import get_detector

log = logging.getLogger(__name__)

# Analysis statistics are scale-invariant enough that a downscaled copy gives the
# same answer for a fraction of the cost. Noise and blockiness are the exceptions —
# they live at the pixel level — so those run on the full-resolution image.
_ANALYSIS_MAX_EDGE = 1024


def analyze(image: np.ndarray, metadata: dict[str, Any] | None = None) -> ImageAnalysis:
    """Profile an image: content type, degradation levels, faces.

    Args:
        image: HWC uint8 RGB or RGBA.
        metadata: from ``load_image`` — EXIF is carried through to the result.
    """
    metadata = metadata or {}
    height, width = image.shape[:2]
    channels = image.shape[2]

    rgb = image[:, :, :3]
    small = _downscale(rgb, _ANALYSIS_MAX_EDGE)

    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Pixel-level degradation: measure on the original, resampling would erase it.
    noise = quality.estimate_noise(gray_full)
    blockiness = quality.estimate_blockiness(gray_full)
    blur = quality.estimate_blur(gray_full)

    faces = _detect_faces(rgb)
    face_ratio = _largest_face_ratio(faces, width, height)

    result = classify(small, has_faces=bool(faces), face_ratio=face_ratio)

    analysis = ImageAnalysis(
        width=width,
        height=height,
        channels=channels,
        has_alpha=channels == 4,
        content_type=result.content_type,
        content_confidence=result.confidence,
        noise_level=noise,
        blur_level=blur,
        compression_level=blockiness,
        quality_score=quality.quality_score(noise, blur, blockiness),
        faces=[f.bbox for f in faces],
        exif=_readable_exif(metadata),
        scores=result.scores,
    )

    log.info(
        "analysed %dx%d: %s (%.0f%% conf), quality %.2f, %d face(s) "
        "[noise %.2f blur %.2f jpeg %.2f]",
        width,
        height,
        analysis.content_type.value,
        analysis.content_confidence * 100,
        analysis.quality_score,
        len(faces),
        noise,
        blur,
        blockiness,
    )
    return analysis


def _downscale(image: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_edge:
        return image

    scale = max_edge / max(h, w)
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def _detect_faces(rgb: np.ndarray) -> list:
    """Faces, or an empty list if the detector was never downloaded.

    A missing detector must not fail an ordinary upscale — it only means face
    restoration is unavailable, which the pipeline reports separately.
    """
    detector = get_detector()
    if not detector.is_available():
        log.debug("face detector not installed; skipping face analysis")
        return []

    try:
        return detector.detect(rgb)
    except Exception as exc:
        log.warning("face detection failed (%s); continuing without faces", exc)
        return []


def _largest_face_ratio(faces: list, width: int, height: int) -> float:
    if not faces:
        return 0.0
    return float(faces[0].area / (width * height))


def _readable_exif(metadata: dict[str, Any]) -> dict[str, Any]:
    """Pull the handful of EXIF tags worth surfacing; drop the raw blob."""
    raw = metadata.get("exif")
    if not raw:
        return {}

    try:
        from PIL import ExifTags, Image

        exif = Image.Exif()
        exif.load(raw)

        wanted = {"Make", "Model", "DateTime", "ISOSpeedRatings", "FNumber", "ExposureTime"}
        return {
            ExifTags.TAGS[tag]: str(value)
            for tag, value in exif.items()
            if tag in ExifTags.TAGS and ExifTags.TAGS[tag] in wanted
        }
    except Exception:
        return {}
