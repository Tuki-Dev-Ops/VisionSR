"""Typed error hierarchy.

Every failure the engine can produce is one of these, so the API layer can map
them to status codes in one place instead of guessing from strings.
"""

from __future__ import annotations


class VisionSRError(Exception):
    """Base class for every engine error."""


class ConfigError(VisionSRError):
    """Malformed or contradictory configuration."""


class ModelNotFoundError(VisionSRError):
    def __init__(self, model_id: str, available: list[str] | None = None) -> None:
        self.model_id = model_id
        self.available = available or []
        hint = f" Available: {', '.join(sorted(self.available))}" if self.available else ""
        super().__init__(f"No model registered with id {model_id!r}.{hint}")


class ArchitectureNotFoundError(VisionSRError):
    def __init__(self, architecture: str, available: list[str] | None = None) -> None:
        self.architecture = architecture
        self.available = available or []
        hint = f" Registered: {', '.join(sorted(self.available))}" if self.available else ""
        super().__init__(f"No architecture registered as {architecture!r}.{hint}")


class WeightsNotFoundError(VisionSRError):
    def __init__(self, model_id: str, path: str) -> None:
        self.model_id = model_id
        self.path = path
        super().__init__(
            f"Weights for {model_id!r} are missing at {path}. "
            f"Run: python scripts/download_weights.py --model {model_id}"
        )


class WeightsCorruptError(VisionSRError):
    def __init__(self, path: str, expected: str, actual: str) -> None:
        super().__init__(
            f"Checksum mismatch for {path}: expected sha256 {expected[:12]}…, got {actual[:12]}…. "
            "Delete the file and re-download."
        )


class BackendUnavailableError(VisionSRError):
    def __init__(self, backend: str, reason: str = "") -> None:
        self.backend = backend
        suffix = f" ({reason})" if reason else ""
        super().__init__(f"Backend {backend!r} is not available on this machine{suffix}.")


class UnsupportedBackendError(VisionSRError):
    def __init__(self, model_id: str, backend: str, supported: list[str]) -> None:
        super().__init__(
            f"Model {model_id!r} does not support backend {backend!r}. "
            f"Supported: {', '.join(supported)}"
        )


class OutOfMemoryError(VisionSRError):
    """VRAM exhausted even at the minimum tile size."""

    def __init__(self, tile_size: int, detail: str = "") -> None:
        self.tile_size = tile_size
        suffix = f" {detail}" if detail else ""
        super().__init__(
            f"Out of GPU memory at tile size {tile_size}px, which is the floor.{suffix} "
            "Try --backend cpu, or a smaller scale."
        )


class InvalidImageError(VisionSRError):
    """Input could not be decoded, or has an unusable shape."""


class ImageTooLargeError(VisionSRError):
    def __init__(self, megapixels: float, limit: float) -> None:
        super().__init__(
            f"Input is {megapixels:.1f} MP, over the {limit:.0f} MP limit for this deployment."
        )


class OutputTooLargeError(VisionSRError):
    """The requested scale would produce an image too big to hold in memory.

    Raised *before* any work starts, because the alternative is not a graceful
    failure: an allocation that large does not raise something catchable, it gets the
    process killed, taking every other queued job with it.
    """

    def __init__(self, megapixels: float, limit: float, scale: int, required_mb: float) -> None:
        self.megapixels = megapixels
        self.limit = limit
        super().__init__(
            f"x{scale} would produce a {megapixels:.0f} MP image, needing roughly "
            f"{required_mb / 1024:.1f} GB of RAM to hold — over the {limit:.0f} MP "
            f"limit. Use a smaller scale, or raise VISIONSR_MAX_OUTPUT_MEGAPIXELS if "
            f"this machine has the memory."
        )
