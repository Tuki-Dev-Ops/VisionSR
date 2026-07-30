"""FastAPI application.

    uvicorn backend.app.main:app --reload

Serves the engine over HTTP for the web frontend and the Electron desktop shell.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from visionsr import __version__
from visionsr.core.errors import (
    BackendUnavailableError,
    ImageTooLargeError,
    InvalidImageError,
    ModelNotFoundError,
    OutOfMemoryError,
    VisionSRError,
    WeightsNotFoundError,
)

from .jobs import get_manager
from .routes import router
from .schemas import ErrorResponse

log = logging.getLogger(__name__)

#: One place mapping every engine error to a status code, so no route has to guess.
_ERROR_STATUS: dict[type[VisionSRError], int] = {
    InvalidImageError: status.HTTP_400_BAD_REQUEST,
    ImageTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    ModelNotFoundError: status.HTTP_404_NOT_FOUND,
    WeightsNotFoundError: status.HTTP_503_SERVICE_UNAVAILABLE,
    BackendUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    OutOfMemoryError: status.HTTP_507_INSUFFICIENT_STORAGE,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from visionsr.core.device import summary
    from visionsr.inference.engine import _ensure_registry

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Load the registry at startup, not on the first request: a misconfigured
    # models.yaml should fail the deploy, not the first user.
    _ensure_registry()

    log.info("VisionSR %s ready — %s", __version__, summary())

    yield

    get_manager().shutdown()
    log.info("VisionSR shut down")


app = FastAPI(
    title="VisionSR",
    description="Enterprise AI image super resolution and restoration.",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    # The desktop shell loads from file:// (origin "null") and the dev frontend from
    # localhost on a port that moves. This is a local-first app whose API binds to
    # loopback; tightening this is a cloud-deployment concern, and is done there.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(VisionSRError)
async def engine_error_handler(request: Request, exc: VisionSRError) -> JSONResponse:
    code = _ERROR_STATUS.get(type(exc), status.HTTP_422_UNPROCESSABLE_CONTENT)
    log.warning("%s %s -> %d: %s", request.method, request.url.path, code, exc)

    return JSONResponse(
        status_code=code,
        content=ErrorResponse(detail=str(exc), code=type(exc).__name__).model_dump(),
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"name": "VisionSR", "version": __version__, "docs": "/docs"}
