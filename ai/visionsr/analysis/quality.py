"""No-reference quality estimation.

Every measure here works on the input alone — there is no ground truth to compare
against at inference time, which rules out PSNR/SSIM and forces blind estimators.
Each returns a 0..1 score with a documented mapping, because the selector compares
them against thresholds and a hidden scale change there is a silent behaviour change.
"""

from __future__ import annotations

import cv2
import numpy as np

# Immerkær's noise-estimation kernel: a Laplacian-of-Laplacian that is blind to
# smooth gradients and to first-order edges, so what survives is mostly noise.
_NOISE_KERNEL = np.array(
    [
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1],
    ],
    dtype=np.float32,
)


def estimate_noise(gray: np.ndarray) -> float:
    """Gaussian noise sigma, normalised to 0..1.

    J. Immerkær, "Fast Noise Variance Estimation" (1996). The kernel above has zero
    response to planes and to ideal step edges, so ``E|I * K|`` over the image is
    dominated by noise; the sqrt(pi/2) factor converts mean-absolute to sigma.

    Mapping: sigma of 0 -> 0.0, sigma of 25/255 (visibly grainy) -> 1.0.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0

    response = cv2.filter2D(gray.astype(np.float32), -1, _NOISE_KERNEL)
    sigma = float(np.abs(response).mean()) * np.sqrt(np.pi / 2) / 6.0

    return float(np.clip(sigma / 25.0, 0.0, 1.0))


def estimate_blur(gray: np.ndarray) -> float:
    """Blurriness, 0 (razor sharp) .. 1 (mush).

    Variance of the Laplacian is the standard cheap sharpness proxy, but its raw
    value is scene-dependent (a picture of a blank wall scores as "blurry"). It is
    normalised here by the image's own contrast, so a low-contrast but sharp image
    is not misread as blurred.
    """
    laplacian_var = float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())
    contrast = float(gray.std()) + 1e-6

    sharpness = laplacian_var / (contrast**2)

    # sharpness ~0.02 is soft, ~0.5+ is crisp. Log-map because the quantity spans
    # orders of magnitude.
    normalised = np.log10(sharpness + 1e-6)
    blur = 1.0 - np.clip((normalised + 2.0) / 2.0, 0.0, 1.0)
    return float(blur)


def estimate_blockiness(gray: np.ndarray) -> float:
    """JPEG artefact level, 0 (clean) .. 1 (heavy 8x8 blocking).

    JPEG quantises in 8x8 blocks independently, so discontinuities land
    *specifically* on the multiple-of-8 grid. Comparing the mean gradient across
    block boundaries with the mean gradient inside blocks isolates that: real image
    content has no reason to prefer the 8-pixel grid, compression artefacts do.
    """
    h, w = gray.shape
    if h < 24 or w < 24:
        return 0.0

    gray = gray.astype(np.float32)

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))

    # Columns 7, 15, 23 ... are the last column of each 8-wide block.
    boundary_cols = np.arange(7, dx.shape[1], 8)
    boundary_rows = np.arange(7, dy.shape[0], 8)
    if boundary_cols.size == 0 or boundary_rows.size == 0:
        return 0.0

    interior_cols = np.setdiff1d(np.arange(dx.shape[1]), boundary_cols)
    interior_rows = np.setdiff1d(np.arange(dy.shape[0]), boundary_rows)

    boundary = (dx[:, boundary_cols].mean() + dy[boundary_rows, :].mean()) / 2
    interior = (dx[:, interior_cols].mean() + dy[interior_rows, :].mean()) / 2

    if interior < 1e-6:
        return 0.0

    # 1.0 => boundaries are no different from interior => no blocking.
    ratio = float(boundary / interior)
    return float(np.clip((ratio - 1.0) / 0.6, 0.0, 1.0))


def estimate_saturation(image: np.ndarray) -> float:
    """Mean HSV saturation, 0..1. Near-zero on scans, greyscale and old photos."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean() / 255.0)


def estimate_colorfulness(image: np.ndarray) -> float:
    """Hasler & Süsstrunk colourfulness, normalised to 0..1.

    Distinct from saturation: it rewards *variety* of colour, not intensity. A
    monochrome-but-vivid red image is saturated yet not colourful. Used to separate
    illustration (few, strong colours) from photography (many, subtle ones).
    """
    r, g, b = (image[:, :, i].astype(np.float32) for i in range(3))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)

    # Hasler & Süsstrunk report ~109 as "extremely colourful".
    return float(np.clip((std + 0.3 * mean) / 109.0, 0.0, 1.0))


def quality_score(noise: float, blur: float, blockiness: float) -> float:
    """Collapse the degradation measures into one 0 (ruined) .. 1 (pristine) figure.

    Weights are not equal: blur destroys information irrecoverably, whereas noise
    and blocking are largely removable, so blur is penalised hardest.
    """
    damage = 0.5 * blur + 0.3 * noise + 0.2 * blockiness
    return float(np.clip(1.0 - damage, 0.0, 1.0))
