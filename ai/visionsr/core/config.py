"""Runtime settings and filesystem layout.

Every path the engine touches is resolved here, so a deployment can relocate
checkpoints/cache with env vars (``VISIONSR_CHECKPOINT_DIR=...``) without code
changes. Defaults are repo-relative so a fresh clone just works.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _frozen() -> bool:
    """True inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _bundle_root() -> Path:
    """Where read-only bundled data lives.

    A frozen app has no source tree, so ``Path(__file__).parents[3]`` — which is how
    this used to be computed — resolves to a path inside PyInstaller's archive and then
    to nothing at all. The symptom is not an obvious crash: the registry loads zero
    models and the first job fails with "no model for task super_resolution", which
    reads like a configuration mistake rather than a packaging one.
    """
    if _frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]

    # .../VisionSR/ai/visionsr/core/config.py -> .../VisionSR
    return Path(__file__).resolve().parents[3]


def _data_root() -> Path:
    """Where *writable* data lives: checkpoints, ONNX graphs, job storage.

    Separate from the bundle, and not optional. An installed app's program directory is
    read-only for a standard user, so anything the app downloads — the 650MB of weights
    and graphs it deliberately does not ship — has to go somewhere else. On Windows that
    is LOCALAPPDATA; a checkout uses the repo, where the developer expects to find it.
    """
    if not _frozen():
        return _bundle_root() / "ai"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")

    return Path(base) / "VisionSR"


REPO_ROOT = _bundle_root()
AI_ROOT = REPO_ROOT / "ai"
DATA_ROOT = _data_root()


class Settings(BaseSettings):
    """Engine-wide configuration. Overridable via env or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONSR_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # Read-only, and shipped: the model registry is part of the build.
    config_dir: Path = Field(default=AI_ROOT / "configs")

    # Writable, and not shipped: 650MB of weights and ONNX graphs, fetched on first run.
    # In a checkout these land in the repo; in an installed app, in the user's data
    # directory, because Program Files is not writable.
    checkpoint_dir: Path = Field(default=DATA_ROOT / "checkpoints")
    dataset_dir: Path = Field(default=DATA_ROOT / "datasets")
    storage_dir: Path = Field(default=DATA_ROOT / "storage")

    # Hard ceiling on input size — a 200MP TIFF should be rejected, not OOM the box.
    max_input_megapixels: float = 100.0

    # Ceiling on *output* size. This used to be the binding constraint — the tiler
    # assembled into a full-size float32 accumulator at 16 bytes per output pixel, so
    # 80MP was already 1.3GB just to stitch. The tiler now streams row-bands into a
    # uint8 canvas (3 bytes/px, and a float band that does not grow with the image),
    # so the limit is an order of magnitude looser and exists only to stop something
    # absurd from exhausting RAM anyway.
    #
    # 500MP is a 22000x22000 image: ~1.5GB for the canvas, plus as much again to
    # encode it. Beyond that a machine really should be told, not surprised.
    max_output_megapixels: float = 500.0
    # Keep at most this many models resident. On a 4GB card, 2 is the practical max
    # (an SR model plus a face model); the cache evicts LRU beyond it.
    model_cache_size: int = 2

    # Tiles below this are dominated by overlap and stop being worth it.
    min_tile_size: int = 64
    max_tile_size: int = 1024
    # Leave this much VRAM unclaimed so the driver and other apps do not get starved.
    vram_headroom_mb: int = 512

    download_timeout_s: float = 300.0
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        for path in (self.checkpoint_dir, self.storage_dir):
            path.mkdir(parents=True, exist_ok=True)

    def weight_path(self, filename: str) -> Path:
        return self.checkpoint_dir / filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
