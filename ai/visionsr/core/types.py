"""Domain types shared across the engine.

Nothing in here imports torch — the core layer stays framework-agnostic so that
backends (torch, onnx, openvino, tensorrt) can be swapped without touching it.
Image data crosses layer boundaries as HWC uint8/float32 numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# HWC, uint8 [0,255] or float32 [0,1]. RGB or RGBA.
Image = np.ndarray


class Task(StrEnum):
    """What a model is for. Drives automatic model selection."""

    SUPER_RESOLUTION = "super_resolution"
    FACE_RESTORATION = "face_restoration"
    ANIME_RESTORATION = "anime_restoration"
    DENOISE = "denoise"
    DEBLUR = "deblur"
    JPEG_ARTIFACT_REMOVAL = "jpeg_artifact_removal"
    DOCUMENT_ENHANCEMENT = "document_enhancement"
    COLORIZE = "colorize"
    BACKGROUND_REMOVAL = "background_removal"


class Backend(StrEnum):
    """Execution backend. Every model must declare which of these it supports."""

    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    DIRECTML = "directml"
    OPENVINO = "openvino"
    MPS = "mps"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class ContentType(StrEnum):
    """Output of the image classifier. Maps to a model-selection policy."""

    PHOTO = "photo"
    PORTRAIT = "portrait"
    ANIME = "anime"
    MANGA = "manga"
    DOCUMENT = "document"
    SCREENSHOT = "screenshot"
    PIXEL_ART = "pixel_art"
    GAME = "game"
    ARTWORK = "artwork"
    MEDICAL = "medical"
    SATELLITE = "satellite"
    OLD_PHOTO = "old_photo"
    LOGO = "logo"
    TEXT = "text"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------------------
# Model description
# --------------------------------------------------------------------------------------


class WeightSource(BaseModel):
    """Where a checkpoint comes from and how to verify it.

    Weights are never committed. `download_weights.py` resolves these.
    """

    model_config = ConfigDict(protected_namespaces=())

    url: str
    filename: str
    sha256: str | None = None
    size_bytes: int | None = None


class ModelSpec(BaseModel):
    """Declarative description of a model.

    This is the *only* place model knowledge lives. Adding a model means adding a
    spec (YAML or code), never editing the engine. Architectures are looked up by
    `architecture` in the architecture registry, so a new arch is one decorator.
    """

    model_config = ConfigDict(protected_namespaces=(), frozen=True)

    id: str
    name: str
    task: Task
    architecture: str
    scale: int = 4

    # Constructor kwargs handed verbatim to the architecture factory.
    params: dict[str, Any] = Field(default_factory=dict)

    weights: WeightSource | None = None
    # Key inside the checkpoint dict holding the state_dict (varies per release).
    state_dict_key: str | None = None

    # How the network wants its input, and how to read its output back. Networks
    # disagree about both, and every disagreement is a silent failure rather than a
    # crash — so it is declared per model and handled once, in the backend, leaving
    # every caller with a single [0,1] contract. See backends/base.py.
    #
    # Applied to [0,1] RGB as (x - mean) / std:
    #   SR models        mean 0,   std 1     (identity — they want [0,1])
    #   GFPGAN           mean 0.5, std 0.5   (i.e. [-1,1])
    #   segmentation     ImageNet channel statistics
    input_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    input_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ...and how to bring the output back to [0,1]:
    #   identity  already [0,1]                        (SR models)
    #   affine    undo the input normalisation         (GFPGAN's [-1,1])
    #   minmax    rescale by the output's own extremes (segmentation logits, which
    #             are unbounded — clipping them to [0,1] would flatten the mask)
    output_norm: Literal["identity", "affine", "minmax"] = "identity"

    backends: list[Backend] = Field(default_factory=lambda: [Backend.CPU, Backend.CUDA])
    precisions: list[Precision] = Field(default_factory=lambda: [Precision.FP32, Precision.FP16])

    # Selection policy inputs.
    content_types: list[ContentType] = Field(default_factory=list)
    # Higher wins when several models match the same content type.
    priority: int = 0

    # Whether the shipped ONNX graph may be fp16.
    #
    # It halves the file, which matters: the graphs are the bulk of the installer. And
    # it is usually faster, because DirectML prefers half precision.
    #
    # But it is not free, and it is not a guess — every export is measured against its
    # torch model and the build fails if the deviation exceeds 2/255. GFPGAN is the
    # cautionary case: in *torch* fp16 its StyleGAN2 decoder collapses outright (output
    # std 0.343 -> 0.123, a near-constant dark smear with no error raised). Whether ONNX
    # Runtime's fp16 behaves the same way is an empirical question, and this flag exists
    # so the answer can differ per model.
    onnx_fp16: bool = True

    # Whether the ONNX export may leave H and W dynamic.
    #
    # True for anything fully convolutional — which is every SR model, and is required
    # of them, because the tiler's edge tiles are smaller than its interior ones and a
    # graph frozen at one size would reject every last row and column.
    #
    # False for a network with a shape-dependent layer. GFPGAN flattens its encoder
    # output into a Linear, so its input must be exactly the 512x512 the weights were
    # trained on. Exporting it with dynamic axes produces a graph that builds without
    # complaint and then fails on the first inference.
    dynamic_shape: bool = True

    # Inference hints.
    tile_size: int = 256
    tile_overlap: int = 16
    # Pixels of padding the arch needs on each side (window-based models e.g. SwinIR).
    window_multiple: int = 1
    # Rough VRAM cost per megapixel of *input*, in MB. Used to auto-size tiles.
    vram_mb_per_megapixel: float = 900.0

    description: str = ""
    license: str = ""

    @property
    def requires_weights(self) -> bool:
        return self.weights is not None


# --------------------------------------------------------------------------------------
# Requests / results
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DeviceInfo:
    backend: Backend
    name: str
    total_vram_mb: int | None = None
    free_vram_mb: int | None = None
    compute_capability: tuple[int, int] | None = None

    @property
    def is_gpu(self) -> bool:
        return self.backend in (Backend.CUDA, Backend.TENSORRT, Backend.DIRECTML, Backend.MPS)


@dataclass(slots=True)
class ImageAnalysis:
    """Everything the analyzer learned about an input. Feeds model selection."""

    width: int
    height: int
    channels: int
    has_alpha: bool
    content_type: ContentType
    content_confidence: float

    noise_level: float  # 0..1
    blur_level: float  # 0..1, higher = blurrier
    compression_level: float  # 0..1, higher = more JPEG artifacts
    quality_score: float  # 0..1, higher = better source

    faces: list[tuple[int, int, int, int]] = field(default_factory=list)
    exif: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0


class EnhanceOptions(BaseModel):
    """User-facing knobs for one enhancement run."""

    model_config = ConfigDict(protected_namespaces=())

    scale: Literal[1, 2, 4, 8, 16] = 4

    # None => the selector decides from the image analysis.
    model_id: str | None = None

    face_restore: bool | None = None  # None => auto (on when faces detected)
    face_restore_weight: float = 0.5  # blend of restored face vs original

    # Cut the subject out. Produces RGBA, so it forces a format that has an alpha
    # channel — asking for JPEG here would silently composite the transparency away.
    remove_background: bool = False
    background_model_id: str | None = None  # None => the registry's best
    background_feather: int = 0  # extra edge blur, px. 0 keeps the model's own softness.

    denoise_strength: float = 0.5
    sharpen: float = 0.0

    backend: Backend | None = None  # None => best available
    precision: Precision | None = None  # None => fp16 on GPU, fp32 on CPU

    tile_size: int | None = None  # None => auto from free VRAM
    tile_overlap: int | None = None

    output_format: Literal["png", "jpeg", "webp"] = "png"
    output_quality: int = 95
    preserve_exif: bool = True

    seed: int | None = None

    @model_validator(mode="after")
    def _alpha_needs_a_format_that_has_alpha(self) -> EnhanceOptions:
        """JPEG has no alpha channel.

        Left alone, the encoder composites the cutout onto white and writes a normal
        photo — the request succeeds, the file opens, and the transparency the user
        asked for is simply gone. Refusing is the only honest option; silently doing
        something else is how a user discovers the problem in someone else's design
        tool a week later.
        """
        if self.remove_background and self.output_format == "jpeg":
            raise ValueError(
                "remove_background produces transparency, and JPEG cannot store an "
                "alpha channel — the cutout would be silently flattened onto white. "
                "Use png or webp."
            )
        return self


@dataclass(slots=True)
class ModelRun:
    """One model's contribution to a result. A pipeline may chain several."""

    model_id: str
    backend: Backend
    precision: Precision
    duration_ms: float
    tiles: int
    peak_vram_mb: float | None = None


@dataclass(slots=True)
class EnhanceResult:
    image: Image
    analysis: ImageAnalysis
    runs: list[ModelRun]
    total_duration_ms: float
    source_path: Path | None = None

    @property
    def output_size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h
