"""VisionSR — enterprise AI image super resolution and restoration.

    from visionsr import enhance_file, EnhanceOptions

    result = enhance_file("photo.jpg", EnhanceOptions(scale=4))
    result.image  # HWC uint8 RGB numpy array

The engine picks the model, the backend, the precision and the tile size for you.
Everything is overridable through :class:`EnhanceOptions`.
"""

from __future__ import annotations

from pathlib import Path

from .core.errors import (
    BackendUnavailableError,
    ImageTooLargeError,
    InvalidImageError,
    ModelNotFoundError,
    OutOfMemoryError,
    VisionSRError,
    WeightsNotFoundError,
)
from .core.types import (
    Backend,
    ContentType,
    EnhanceOptions,
    EnhanceResult,
    ImageAnalysis,
    ModelSpec,
    Precision,
    Task,
)

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BackendUnavailableError",
    "ContentType",
    "EnhanceOptions",
    "EnhanceResult",
    "ImageAnalysis",
    "ImageTooLargeError",
    "InvalidImageError",
    "ModelNotFoundError",
    "ModelSpec",
    "OutOfMemoryError",
    "Precision",
    "Task",
    "VisionSRError",
    "WeightsNotFoundError",
    "__version__",
    "analyze_file",
    "enhance",
    "enhance_file",
    "list_models",
]


def enhance_file(path: str | Path, options: EnhanceOptions | None = None) -> EnhanceResult:
    """Enhance an image file. The one-liner entry point."""
    from .inference.engine import get_engine

    return get_engine().enhance_file(Path(path), options)


def enhance(image, options: EnhanceOptions | None = None) -> EnhanceResult:
    """Enhance an in-memory HWC uint8 RGB(A) numpy array."""
    from .inference.engine import get_engine

    return get_engine().enhance(image, options)


def analyze_file(path: str | Path) -> ImageAnalysis:
    """Profile an image without enhancing it."""
    from .analysis.analyzer import analyze as _analyze
    from .preprocessing.io import load_image

    image, metadata = load_image(Path(path))
    return _analyze(image, metadata)


def list_models() -> list[ModelSpec]:
    """Every model in the registry."""
    from .core.registry import models
    from .inference.engine import _ensure_registry

    _ensure_registry()
    return models.all()
