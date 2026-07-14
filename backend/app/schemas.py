"""HTTP request/response models.

Deliberately separate from the engine's domain types. The engine's dataclasses
carry numpy arrays and internal fields that must never reach the wire, and an API
contract has to stay stable while the engine's internals move. Translation happens
in one place: :func:`analysis_to_schema` and :func:`result_to_schema`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from visionsr.core.types import EnhanceResult, ImageAnalysis, ModelSpec


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    backends: list[str]
    device: str
    models_loaded: list[str]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    name: str
    task: str
    architecture: str
    scale: int
    content_types: list[str]
    description: str
    installed: bool
    backends: list[str]
    precisions: list[str]


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class AnalysisSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    width: int
    height: int
    megapixels: float
    has_alpha: bool

    content_type: str
    content_confidence: float

    noise_level: float
    blur_level: float
    compression_level: float
    quality_score: float

    faces: list[tuple[int, int, int, int]]

    recommended_model: str
    recommended_face_model: str | None


class ResultSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    width: int
    height: int
    duration_ms: float
    models_used: list[str]
    backend: str
    tiles: int
    analysis: AnalysisSchema


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    stage: str = ""
    progress: float = Field(0.0, ge=0.0, le=1.0)
    error: str | None = None
    result: ResultSchema | None = None


class JobEvent(BaseModel):
    """One frame on the SSE stream. Kept minimal — it is sent many times a second."""

    status: JobStatus
    stage: str
    progress: float
    error: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    code: str


# ---------------------------------------------------------------------------
# domain -> wire
# ---------------------------------------------------------------------------


def analysis_to_schema(
    analysis: ImageAnalysis, model_id: str, face_model_id: str | None
) -> AnalysisSchema:
    return AnalysisSchema(
        width=analysis.width,
        height=analysis.height,
        megapixels=round(analysis.megapixels, 3),
        has_alpha=analysis.has_alpha,
        content_type=analysis.content_type.value,
        content_confidence=round(analysis.content_confidence, 4),
        noise_level=round(analysis.noise_level, 4),
        blur_level=round(analysis.blur_level, 4),
        compression_level=round(analysis.compression_level, 4),
        quality_score=round(analysis.quality_score, 4),
        faces=analysis.faces,
        recommended_model=model_id,
        recommended_face_model=face_model_id,
    )


def result_to_schema(result: EnhanceResult) -> ResultSchema:
    width, height = result.output_size
    models_used = [run.model_id for run in result.runs]

    sr_model = models_used[0] if models_used else "none"
    face_model = next(
        (run.model_id for run in result.runs if run.model_id != sr_model),
        None,
    )

    return ResultSchema(
        width=width,
        height=height,
        duration_ms=round(result.total_duration_ms, 1),
        models_used=models_used,
        backend=result.runs[0].backend.value if result.runs else "none",
        tiles=sum(run.tiles for run in result.runs),
        analysis=analysis_to_schema(result.analysis, sr_model, face_model),
    )


def model_to_schema(spec: ModelSpec, installed: bool) -> ModelInfo:
    return ModelInfo(
        id=spec.id,
        name=spec.name,
        task=spec.task.value,
        architecture=spec.architecture,
        scale=spec.scale,
        content_types=[c.value for c in spec.content_types],
        description=" ".join(spec.description.split()),
        installed=installed,
        backends=[b.value for b in spec.backends],
        precisions=[p.value for p in spec.precisions],
    )
