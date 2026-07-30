"""HTTP routes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from visionsr import __version__
from visionsr.core.errors import InvalidImageError, VisionSRError
from visionsr.core.types import EnhanceOptions

from .jobs import get_manager
from .schemas import (
    AnalysisSchema,
    HealthResponse,
    JobCreated,
    JobState,
    ModelsResponse,
    analysis_to_schema,
    model_to_schema,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

#: Refuse an upload above this before reading it into memory.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from visionsr.backends.factory import cache
    from visionsr.core.device import available_backends, best_backend, device_info

    backends = available_backends()
    info = device_info(best_backend())

    return HealthResponse(
        status="ok",
        version=__version__,
        backends=[b.value for b in backends],
        device=info.name,
        models_loaded=[key.model_id for key in cache.resident],
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    from visionsr.backends.weights import is_runnable
    from visionsr.core.registry import models
    from visionsr.inference.engine import _ensure_registry

    _ensure_registry()

    # `installed` means "this model can run here", not "its .pth is on disk". The
    # packaged app ships ONNX graphs and no PyTorch, so the two answers differ.
    return ModelsResponse(
        models=[model_to_schema(spec, is_runnable(spec)) for spec in models.all()]
    )


@router.post("/analyze", response_model=AnalysisSchema)
async def analyze(file: Annotated[UploadFile, File()]) -> AnalysisSchema:
    """Profile an image and report which model would be chosen. Does not enhance."""
    payload = await _read_upload(file)

    # Decoding a 24MP TIFF, running the quality estimators over it and doing face
    # detection is 100-500ms of straight CPU work. Doing that inline in an async
    # handler blocks the event loop — every other request, including the progress
    # polls of a job that is already running and the health check the frontend uses
    # to decide whether the backend is up, stalls behind it. `async def` is not a
    # thread; it is a promise not to block, and this function cannot keep it.
    try:
        return await asyncio.to_thread(_analyze_blocking, payload)
    except VisionSRError as exc:
        raise _http_error(exc) from exc


def _analyze_blocking(payload: bytes) -> AnalysisSchema:
    from visionsr.analysis.analyzer import analyze as run_analysis
    from visionsr.analysis.selector import ModelSelector
    from visionsr.inference.engine import _ensure_registry
    from visionsr.preprocessing.io import load_image

    _ensure_registry()

    image, metadata = load_image(payload)
    analysis = run_analysis(image, metadata)
    plan = ModelSelector().select(analysis, EnhanceOptions())

    return analysis_to_schema(
        analysis,
        plan.sr_model.id,
        plan.face_model.id if plan.face_model else None,
    )


#: Scales the API accepts. Not a Literal on the parameter: multipart form fields
#: arrive as strings, and pydantic will not coerce "4" into Literal[4] — it 422s with
#: a message that blames the caller for something the transport did.
_SCALES = (2, 4, 8, 16)


@router.post("/jobs", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    file: Annotated[UploadFile, File()],
    scale: Annotated[int, Form()] = 4,
    model_id: Annotated[str | None, Form()] = None,
    face_restore: Annotated[str, Form()] = "",
    face_restore_weight: Annotated[float, Form(ge=0.0, le=1.0)] = 0.5,
    remove_background: Annotated[str, Form()] = "false",
    background_model_id: Annotated[str | None, Form()] = None,
    background_feather: Annotated[int, Form(ge=0, le=32)] = 0,
    sharpen: Annotated[float, Form(ge=0.0, le=1.5)] = 0.0,
    output_format: Annotated[Literal["png", "jpeg", "webp"], Form()] = "png",
) -> JobCreated:
    """Queue an enhancement. Returns immediately; poll or stream for progress."""
    if scale not in _SCALES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"scale must be one of {', '.join(map(str, _SCALES))}; got {scale}.",
        )

    payload = await _read_upload(file)

    try:
        options = EnhanceOptions(
            scale=scale,  # type: ignore[arg-type]
            model_id=model_id or None,
            # An empty string means "auto" — the frontend cannot send a null in a
            # multipart form, so tri-state has to be encoded as "" | "true" | "false".
            face_restore=_tri_state(face_restore),
            face_restore_weight=face_restore_weight,
            remove_background=_tri_state(remove_background) or False,
            background_model_id=background_model_id or None,
            background_feather=background_feather,
            sharpen=sharpen,
            output_format=output_format,
        )
    except ValidationError as exc:
        # EnhanceOptions refuses combinations that would silently lose data — most
        # notably a transparent cutout written to JPEG. Surface the reason, not a
        # pydantic dump.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors()),
        ) from exc

    job = await get_manager().submit(payload, file.filename or "upload", options)
    return JobCreated(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str) -> JobState:
    job = await get_manager().get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")
    return job.to_state()


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-sent progress stream. Closes when the job reaches a terminal state."""
    manager = get_manager()
    job = await manager.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")

    async def event_stream() -> AsyncIterator[str]:
        async for event in manager.stream(job):
            yield f"data: {json.dumps(event.model_dump())}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # nginx buffers text/event-stream by default, which turns a live progress
            # bar into a single update at the end.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/jobs/{job_id}/result")
async def job_result(job_id: str) -> Response:
    job = await get_manager().get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")

    if job.status.value == "failed":
        raise HTTPException(status.HTTP_409_CONFLICT, job.error or "Job failed.")
    if job.output is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Job has not finished yet.")

    suffix = job.media_type.split("/")[-1]
    return Response(
        content=job.output,
        media_type=job.media_type,
        headers={
            "Content-Disposition": f'inline; filename="{job.id}.{suffix}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/jobs/{job_id}/source")
async def job_source(job_id: str) -> Response:
    """The original upload — the frontend needs it for the before/after slider."""
    job = await get_manager().get(job_id)
    if job is None or not job.source:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")

    return Response(
        content=job.source,
        media_type="application/octet-stream",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str) -> Response:
    if not await get_manager().delete(job_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------


async def _read_upload(file: UploadFile) -> bytes:
    payload = await file.read()

    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Upload is {len(payload) // 1024**2}MB, over the "
            f"{MAX_UPLOAD_BYTES // 1024**2}MB limit.",
        )

    return payload


def _tri_state(value: str) -> bool | None:
    """"" -> auto, "true" -> on, "false" -> off."""
    normalised = value.strip().lower()
    if normalised in ("", "auto", "null", "none"):
        return None
    return normalised in ("true", "1", "yes", "on")


def _http_error(exc: VisionSRError) -> HTTPException:
    code = (
        status.HTTP_400_BAD_REQUEST
        if isinstance(exc, InvalidImageError)
        else status.HTTP_422_UNPROCESSABLE_CONTENT
    )
    return HTTPException(code, str(exc))
