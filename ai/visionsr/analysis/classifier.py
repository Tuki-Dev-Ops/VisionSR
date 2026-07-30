"""Content classification.

Picking the wrong model is the single most visible failure this system can have:
run a photo model over line art and it stipples texture into flat cel-shaded
regions; run an anime model over a portrait and skin turns plastic. So the
classifier's job is not academic accuracy, it is *not making that mistake*.

This is a deliberate, documented heuristic ensemble rather than a learned model:

* It is inspectable. When it misroutes an image, ``ImageAnalysis.scores`` says
  which signal was responsible, and a user can override with ``--model``.
* It needs no weights, no download, no GPU, and runs in ~10ms.
* The decision boundaries that matter (illustration vs photograph, text vs
  pictorial) are genuinely separable by low-level statistics.

A learned classifier is the right long-term answer and slots in behind the same
:func:`classify` signature — see ``docs/roadmap.md``. The scores below are what a
future model would be trained to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.types import ContentType
from . import quality


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    content_type: ContentType
    confidence: float
    scores: dict[str, float]


def classify(image: np.ndarray, has_faces: bool = False, face_ratio: float = 0.0) -> ClassificationResult:
    """Decide what kind of image this is.

    Args:
        image: HWC uint8 RGB.
        has_faces: from the face detector — a strong prior the pixels cannot give.
        face_ratio: fraction of the frame the largest face occupies.
    """
    rgb = image[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    signals = {
        "flatness": _flatness(gray),
        "edge_density": _edge_density(gray),
        "palette_concentration": _palette_concentration(rgb),
        "saturation": quality.estimate_saturation(rgb),
        "colorfulness": quality.estimate_colorfulness(rgb),
        "bimodality": _bimodality(gray),
        "axis_alignment": _axis_alignment(gray),
        "noise": quality.estimate_noise(gray),
        "grayness": _grayness(rgb),
        "sepia": _sepia(rgb),
        "face_ratio": face_ratio,
    }

    votes = _score_types(signals, has_faces, image.shape[:2])

    best = max(votes, key=lambda t: votes[t])
    ordered = sorted(votes.values(), reverse=True)

    # Confidence is the *margin* over the runner-up, not the raw score: a 0.9 that
    # beat another 0.88 is a coin flip and should be reported as one.
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    confidence = float(np.clip(0.45 + margin, 0.0, 1.0))

    return ClassificationResult(
        content_type=best,
        confidence=confidence,
        scores={**signals, **{f"vote_{k.value}": v for k, v in votes.items()}},
    )


def _score_types(
    s: dict[str, float], has_faces: bool, shape: tuple[int, int]
) -> dict[ContentType, float]:
    """Turn low-level signals into a score per content type.

    Each expression below is a claim about what that content type *looks like* in
    these statistics. They are additive so no single signal can dominate, and every
    term is commented with the claim it encodes.
    """
    height, width = shape
    votes: dict[ContentType, float] = {}

    # Illustration: large flat colour regions, few distinct colours, hard edges,
    # and (crucially) little sensor noise — a drawing has no grain.
    votes[ContentType.ANIME] = (
        0.35 * s["flatness"]
        + 0.25 * s["palette_concentration"]
        + 0.20 * s["saturation"]
        + 0.20 * (1.0 - s["noise"])
    )

    # Manga: illustration that is also nearly monochrome.
    votes[ContentType.MANGA] = (
        0.30 * s["flatness"]
        + 0.30 * s["grayness"]
        + 0.20 * s["edge_density"]
        + 0.20 * s["bimodality"]
    )

    # Document/scan: luminance piles up at black and white, low saturation, and the
    # strokes are strongly axis-aligned (text lines and page edges).
    votes[ContentType.DOCUMENT] = (
        0.40 * s["bimodality"]
        + 0.30 * (1.0 - s["saturation"])
        + 0.30 * s["axis_alignment"]
    )

    # Screenshot/UI: flat fills like illustration, but overwhelmingly rectilinear,
    # and with a tiny palette.
    votes[ContentType.SCREENSHOT] = (
        0.35 * s["axis_alignment"]
        + 0.35 * s["flatness"]
        + 0.30 * s["palette_concentration"]
    )

    # Old photo: desaturated, sepia-shifted, noisy, and soft.
    votes[ContentType.OLD_PHOTO] = (
        0.35 * s["sepia"]
        + 0.30 * s["noise"]
        + 0.35 * (1.0 - s["colorfulness"])
    )

    # Portrait: the face detector said so, and the face is a meaningful part of the
    # frame. A 20px face in a landscape is not a portrait.
    votes[ContentType.PORTRAIT] = (
        (0.60 + 0.40 * min(s["face_ratio"] * 6.0, 1.0)) if has_faces else 0.0
    )

    # Photograph: the null hypothesis. Continuous tone, varied colour, some noise.
    votes[ContentType.PHOTO] = (
        0.35 * (1.0 - s["flatness"])
        + 0.30 * s["colorfulness"]
        + 0.20 * (1.0 - s["palette_concentration"])
        + 0.15 * min(s["noise"] * 3.0, 1.0)
    )

    # Pixel art: only credible on a small canvas whose colours snap to a tiny
    # palette and whose "edges" sit on a coarse grid.
    is_small = width <= 512 and height <= 512
    votes[ContentType.PIXEL_ART] = (
        (0.5 * s["palette_concentration"] + 0.5 * s["flatness"]) if is_small else 0.0
    )

    return votes


def _flatness(gray: np.ndarray) -> float:
    """Fraction of pixels sitting in a locally-uniform region.

    Illustration is mostly flat fill; photography almost never is, because sensor
    noise and real-world shading keep the local gradient non-zero everywhere.
    """
    gradient = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float((np.abs(gradient) < 2.0).mean())


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels Canny calls an edge. Line art is edge-rich."""
    edges = cv2.Canny(gray, 80, 200)
    # ~12% edge pixels already reads as dense line art, so saturate there.
    return float(np.clip((edges > 0).mean() * 8.0, 0.0, 1.0))


def _palette_concentration(rgb: np.ndarray) -> float:
    """How much of the image is covered by its most common colours.

    Quantised to 5 bits/channel. A cel-shaded frame puts most of its pixels into a
    handful of bins; a photograph spreads them across thousands.
    """
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    # Widen before shifting, not after: identical for uint8 input, and it keeps the
    # operand types legible to a type checker.
    quantised = small.astype(np.uint32) >> 3
    keys = (quantised[:, :, 0] << 10) | (quantised[:, :, 1] << 5) | quantised[:, :, 2]

    counts = np.bincount(keys.ravel())
    counts = np.sort(counts)[::-1]

    # Share of pixels held by the top 32 colours.
    return float(counts[:32].sum() / keys.size)


def _bimodality(gray: np.ndarray) -> float:
    """How much of the histogram sits at the two extremes.

    Scanned text is ink-on-paper: near-black and near-white, very little between.
    """
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-9)
    return float(hist[:3].sum() + hist[-3:].sum())


def _axis_alignment(gray: np.ndarray) -> float:
    """Share of gradient energy pointing exactly horizontally or vertically.

    Text baselines, table rules, window chrome and UI borders are axis-aligned.
    Natural scenes are not — foliage and faces have gradients at every angle.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)
    strong = magnitude > np.percentile(magnitude, 90)
    if not strong.any():
        return 0.0

    angle = np.arctan2(np.abs(gy[strong]), np.abs(gx[strong]))
    # Within ~11 degrees of an axis.
    aligned = (angle < 0.2) | (angle > np.pi / 2 - 0.2)
    return float(aligned.mean())


def _grayness(rgb: np.ndarray) -> float:
    """1.0 when R==G==B everywhere."""
    channels = rgb.astype(np.float32)
    spread = channels.max(axis=2) - channels.min(axis=2)
    return float(np.clip(1.0 - spread.mean() / 32.0, 0.0, 1.0))


def _sepia(rgb: np.ndarray) -> float:
    """Warm cast typical of aged prints: red high, blue low, and low saturation."""
    r, g, b = (rgb[:, :, i].astype(np.float32).mean() for i in range(3))
    total = r + g + b + 1e-6

    warmth = (r - b) / total  # positive when the image leans amber
    if warmth <= 0:
        return 0.0

    low_sat = 1.0 - quality.estimate_saturation(rgb)
    return float(np.clip(warmth * 6.0, 0.0, 1.0) * low_sat)
