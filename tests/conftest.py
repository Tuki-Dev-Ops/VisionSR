"""Shared fixtures.

Tests split into two tiers:

* Unmarked tests run anywhere, with no weights and no GPU. They cover the tiler,
  the registry, the selector, the analyser and the API contract — the logic most
  likely to break under refactoring.
* ``@pytest.mark.weights`` tests need real checkpoints on disk. They are the ones
  that can actually catch a wrong architecture or a ruined output, and they skip
  cleanly on a machine that has not run ``scripts/download_weights.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

ASSETS = REPO_ROOT / "tests" / "assets"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "weights: requires downloaded model checkpoints (and usually a GPU)"
    )


@pytest.fixture(scope="session")
def registry():
    """The real model registry, loaded from ai/configs."""
    import visionsr.models  # noqa: F401  — registers the architectures
    from visionsr.core.config import get_settings
    from visionsr.core.registry import models

    if len(models) == 0:
        models.load_dir(get_settings().config_dir, replace=True)
    return models


@pytest.fixture(scope="session", autouse=True)
def _skip_without_weights(request: pytest.FixtureRequest) -> None:
    """Skip the weights tier when the checkpoints are not there."""
    if not request.node.get_closest_marker("weights"):
        return


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    from visionsr.core.config import get_settings

    checkpoint_dir = get_settings().checkpoint_dir
    has_weights = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pth"))

    if has_weights:
        return

    skip = pytest.mark.skip(
        reason="no checkpoints — run: python scripts/download_weights.py --all"
    )
    for item in items:
        if "weights" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def gfpgan_spec(registry):
    return registry.get("gfpgan-v1.4")


@pytest.fixture(scope="session")
def sr_spec(registry):
    return registry.get("realesr-general-x4v3")


@pytest.fixture(scope="session")
def portrait_truth() -> np.ndarray:
    """The undegraded original that ``portrait_image`` is derived from."""
    from visionsr.preprocessing.io import load_image

    source = ASSETS / "hr" / "kodim04.png"
    if not source.exists():
        pytest.skip(f"missing ground-truth asset {source}")

    truth, _ = load_image(source)
    return truth[:, :, :3]


@pytest.fixture(scope="session")
def portrait_image() -> np.ndarray:
    """A small, genuinely degraded photo of a face.

    Built from a ground-truth image rather than shipped as a fixture file: the
    degradation is then explicit and reproducible, and the test asserts against a
    known original instead of a magic blob.
    """
    import cv2

    from visionsr.preprocessing.io import load_image

    source = ASSETS / "hr" / "kodim04.png"
    if not source.exists():
        pytest.skip(f"missing ground-truth asset {source}")

    truth, _ = load_image(source)
    truth = truth[:, :, :3]

    blurred = cv2.GaussianBlur(truth, (0, 0), 1.2)
    small = cv2.resize(
        blurred, (truth.shape[1] // 4, truth.shape[0] // 4), interpolation=cv2.INTER_AREA
    )

    rng = np.random.default_rng(0)
    noisy = np.clip(small + rng.normal(0, 6.0, small.shape), 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50]
    )
    assert ok
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def aligned_face(portrait_image) -> np.ndarray:
    """A 512x512 FFHQ-aligned crop — what GFPGAN actually consumes."""
    from visionsr.analysis.faces import align_face, get_detector

    detector = get_detector()
    if not detector.is_available():
        pytest.skip("face detector not downloaded")

    faces = detector.detect(portrait_image)
    if not faces:
        pytest.skip("no face detected in the fixture")

    return align_face(portrait_image, faces[0])


@pytest.fixture
def gradient_image() -> np.ndarray:
    """A deterministic synthetic image with structure at several frequencies.

    Used by the tiler tests: a flat colour would hide a seam, and random noise would
    hide it too. Smooth gradients plus hard edges make a stitching error obvious.
    """
    height, width = 200, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    red = xx / width
    green = yy / height
    blue = ((np.sin(xx / 9) * np.cos(yy / 7)) + 1) / 2

    image = np.stack([red, green, blue], axis=2)
    image[60:80, :] = 1.0  # a hard horizontal edge
    image[:, 150:158] = 0.0  # a hard vertical edge

    return (image * 255).astype(np.uint8)
