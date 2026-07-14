"""Post-processing.

Deliberately conservative. The network has already done the hard reconstruction;
anything applied here can only trade one artefact for another, so each filter is
off by default and each has a documented reason to exist.
"""

from __future__ import annotations

import cv2
import numpy as np


def unsharp_mask(
    image: np.ndarray, amount: float = 0.5, radius: float = 1.0, threshold: int = 3
) -> np.ndarray:
    """Local-contrast sharpening.

    The ``threshold`` is what separates this from a naive unsharp: pixels whose
    local contrast is below it are left alone, so flat areas (sky, skin, paper) do
    not get their noise amplified while edges still get crisper.

    Args:
        amount: 0 = off, 1.0 = strong. Above ~1.5 it starts ringing.
        radius: gaussian sigma of the blur that defines "local".
        threshold: minimum 0-255 difference before sharpening applies.
    """
    if amount <= 0:
        return image

    blurred = cv2.GaussianBlur(image, (0, 0), radius)

    source = image.astype(np.float32)
    detail = source - blurred.astype(np.float32)

    if threshold > 0:
        mask = np.abs(detail) >= threshold
        detail = detail * mask

    return np.clip(source + amount * detail, 0, 255).astype(np.uint8)


def enhance_document(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Lift text legibility on a scan.

    CLAHE on the luminance channel only, so the paper's colour cast is preserved
    and only local contrast moves. Global histogram equalisation would blow out the
    page white and lose faint pencil or low-toner text — exactly the content that
    needed help.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)

    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2RGB)


def denoise(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Edge-preserving denoise.

    Rarely needed: the Real-ESRGAN degradation model already includes noise, so the
    network removes most of it. This exists for inputs so noisy that the network
    treats the grain as signal and sharpens it.
    """
    if strength <= 0:
        return image

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=strength * 10,
        hColor=strength * 10,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def match_color(enhanced: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Force the output's colour statistics back onto the input's.

    GAN-based SR models drift: over a long run they can warm or cool the whole
    frame by a few percent. Matching per-channel mean and standard deviation in LAB
    pulls that back without touching the detail the network added.
    """
    src = cv2.resize(source, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_AREA)

    enhanced_lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)

    for channel in range(3):
        e = enhanced_lab[:, :, channel]
        s = source_lab[:, :, channel]

        e_std = e.std()
        if e_std < 1e-5:
            continue

        enhanced_lab[:, :, channel] = (e - e.mean()) * (s.std() / e_std) + s.mean()

    return cv2.cvtColor(np.clip(enhanced_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
