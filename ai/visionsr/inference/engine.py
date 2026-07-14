"""The enhancement engine.

Owns the pipeline the spec describes:

    analyse -> classify -> select model -> tiled inference -> face restore
            -> post-process -> quality check

Everything above this layer (CLI, HTTP API, desktop app, batch worker) calls
:meth:`Engine.enhance` and nothing else. Everything below is swappable through the
registries. This is the only class that knows the *order* of the stages.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from ..analysis.analyzer import analyze
from ..analysis.selector import ModelSelector, Plan
from ..backends.factory import get_model
from ..core.config import get_settings
from ..core.device import resolve_backend
from ..core.errors import OutputTooLargeError
from ..core.registry import models as model_registry
from ..core.types import (
    EnhanceOptions,
    EnhanceResult,
    ImageAnalysis,
    ModelRun,
)
from ..pipelines.face import FacePipeline
from ..pipelines.matting import MattingPipeline
from ..postprocessing import filters
from ..preprocessing.io import (
    load_image,
    merge_alpha,
    split_alpha,
)
from .tiler import TiledRunner

log = logging.getLogger(__name__)

#: (stage name, fraction done 0..1)
ProgressFn = Callable[[str, float], None]


class Engine:
    """Stateless per-request; the expensive state (loaded models) lives in the cache."""

    def __init__(self, selector: ModelSelector | None = None) -> None:
        self._selector = selector or ModelSelector()
        self._faces = FacePipeline()
        self._matting = MattingPipeline()

    # -- public API ------------------------------------------------------------

    def enhance_file(
        self,
        path: Path,
        options: EnhanceOptions | None = None,
        progress: ProgressFn | None = None,
    ) -> EnhanceResult:
        image, metadata = load_image(path)
        result = self.enhance(image, options, progress, metadata=metadata)
        result.source_path = path
        return result

    def enhance(
        self,
        image: np.ndarray,
        options: EnhanceOptions | None = None,
        progress: ProgressFn | None = None,
        metadata: dict | None = None,
    ) -> EnhanceResult:
        """Run the full pipeline on an HWC uint8 RGB(A) image."""
        options = options or EnhanceOptions()
        started = time.perf_counter()
        runs: list[ModelRun] = []

        reporter = _Progress(progress)

        reporter.stage("analyzing", 0.0)
        analysis = analyze(image, metadata)

        plan = self._selector.select(analysis, options, options.backend)
        _check_output_size(analysis, plan, options)

        reporter.plan(
            passes=plan.passes,
            faces=plan.face_model is not None,
            background=plan.background_model is not None,
        )
        reporter.stage("analyzing", 1.0)

        # Alpha is not RGB and must not go through an RGB network — it is upscaled
        # separately with Lanczos, which is correct for a coverage mask.
        rgb, alpha = split_alpha(image)

        canvas = self._run_super_resolution(rgb, plan, options, runs, reporter)

        if plan.face_model is not None:
            reporter.stage("restoring faces", 0.0)
            upscale = round(canvas.shape[0] / rgb.shape[0])
            canvas, face_run = self._faces.restore(
                canvas,
                rgb,
                plan.face_model,
                upscale=upscale,
                weight=options.face_restore_weight,
                backend=options.backend,
                precision=options.precision,
            )
            if face_run is not None:
                runs.append(face_run)
            reporter.stage("restoring faces", 1.0)

        # Land exactly on the requested scale. The planner may have overshot (two
        # x4 passes for a x8 request), so this is usually a downscale — which is
        # information-preserving, unlike the upscale it replaced.
        canvas = self._resample_to_target(canvas, rgb.shape[:2], options.scale)

        reporter.stage("post-processing", 0.0)
        canvas = self._post_process(canvas, analysis, options)

        if plan.background_model is not None:
            # Last, deliberately. The segmenter downsamples whatever it is handed to a
            # fixed square, so it wants the *upscaled* image — a 512px canvas resized
            # to 1024 is a far better-defined input than a 128px original, and the edge
            # quality of a cutout is entirely a question of detail. Running it first
            # would throw away the detail super resolution just reconstructed.
            reporter.stage("removing background", 0.0)
            canvas, matting_run = self._matting.run(
                canvas,
                plan.background_model,
                backend=options.backend,
                precision=options.precision,
                feather=options.background_feather,
            )
            runs.append(matting_run)
            reporter.stage("removing background", 1.0)

            # The cutout *is* the alpha. Any alpha the source arrived with described a
            # different image and must not be merged back over it.
            alpha = None

        if alpha is not None:
            canvas = merge_alpha(canvas, _resize_alpha(alpha, canvas.shape[:2]))
        reporter.stage("post-processing", 1.0)

        total_ms = (time.perf_counter() - started) * 1000
        log.info(
            "done in %.2fs: %dx%d -> %dx%d",
            total_ms / 1000,
            analysis.width,
            analysis.height,
            canvas.shape[1],
            canvas.shape[0],
        )

        return EnhanceResult(
            image=canvas,
            analysis=analysis,
            runs=runs,
            total_duration_ms=total_ms,
        )

    def plan(self, image: np.ndarray, options: EnhanceOptions | None = None) -> Plan:
        """Explain what ``enhance`` would do, without doing it. Used by `visionsr plan`."""
        options = options or EnhanceOptions()
        return self._selector.select(analyze(image), options, options.backend)

    # -- stages ----------------------------------------------------------------

    def _run_super_resolution(
        self,
        rgb: np.ndarray,
        plan: Plan,
        options: EnhanceOptions,
        runs: list[ModelRun],
        reporter: _Progress,
    ) -> np.ndarray:
        if plan.passes == 0:
            return rgb

        backend = resolve_backend(options.backend, plan.sr_model.backends)
        model = get_model(plan.sr_model, backend, options.precision)
        runner = TiledRunner(model)

        # uint8 all the way through, including between passes. The tiler converts each
        # tile to float on its way into the network; a float32 canvas would cost 12
        # bytes per pixel of an image that may be hundreds of megapixels, for no gain —
        # see TiledRunner.run.
        canvas = rgb

        for index in range(plan.passes):
            stage = (
                f"upscaling (pass {index + 1}/{plan.passes})" if plan.passes > 1 else "upscaling"
            )
            started = time.perf_counter()

            def on_tile(done: int, total: int, stage: str = stage, index: int = index) -> None:
                reporter.pass_progress(index, stage, done / total)

            canvas, tiles = runner.run(
                canvas,
                tile_size=options.tile_size,
                overlap=options.tile_overlap,
                progress=on_tile,
            )

            runs.append(
                ModelRun(
                    model_id=plan.sr_model.id,
                    backend=model.backend,
                    precision=model.precision,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    tiles=tiles,
                )
            )

        return canvas

    def _resample_to_target(
        self, canvas: np.ndarray, source_shape: tuple[int, int], target_scale: int
    ) -> np.ndarray:
        target_h = source_shape[0] * target_scale
        target_w = source_shape[1] * target_scale

        if (canvas.shape[0], canvas.shape[1]) == (target_h, target_w):
            return canvas

        # INTER_AREA for shrinking (it integrates, so it does not alias), Lanczos for
        # the rare case where we have to grow.
        shrinking = target_h < canvas.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4

        log.debug(
            "resampling %dx%d -> %dx%d to hit x%d exactly",
            canvas.shape[1],
            canvas.shape[0],
            target_w,
            target_h,
            target_scale,
        )
        return cv2.resize(canvas, (target_w, target_h), interpolation=interpolation)

    def _post_process(
        self, canvas: np.ndarray, analysis: ImageAnalysis, options: EnhanceOptions
    ) -> np.ndarray:
        if options.sharpen > 0:
            canvas = filters.unsharp_mask(canvas, amount=options.sharpen)

        # Documents are the one content type where a hard contrast/threshold pass
        # genuinely helps; on photographs it would crush the tonal range.
        if analysis.content_type.value in ("document", "text"):
            canvas = filters.enhance_document(canvas)

        return canvas


def _resize_alpha(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Upscale the alpha channel to match the enhanced RGB.

    Deliberately *not* run through the SR model: alpha is a coverage mask, not an
    image. An SR network trained on photographs would hallucinate texture into it
    and produce a fringed, speckled matte.
    """
    return cv2.resize(alpha, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)


#: Host memory per output pixel: the uint8 canvas (3 B), plus roughly as much again
#: for the encode buffer the result eventually goes into. The tiler's float blend band
#: is bounded by tile height and does not scale with the image, so it is not counted.
_BYTES_PER_OUTPUT_PIXEL = 6


def _check_output_size(analysis: ImageAnalysis, plan: Plan, options: EnhanceOptions) -> None:
    """Refuse a request that would exhaust memory, before any work is done.

    The failure this prevents is not a slow job — it is the *process* dying. An
    allocation of several gigabytes does not fail with an exception you can catch and
    report; the OS kills the worker, and every queued job goes with it. A clear error
    beforehand is strictly better than a crash afterwards.

    The check uses the *planner's* peak, not the final size: two x4 passes to reach x8
    pass through a x16 intermediate, and it is that intermediate that has to fit.
    """
    settings = get_settings()

    peak_scale = plan.sr_model.scale**plan.passes if plan.passes else options.scale
    peak_megapixels = analysis.megapixels * (peak_scale**2)

    if peak_megapixels > settings.max_output_megapixels:
        raise OutputTooLargeError(
            megapixels=peak_megapixels,
            limit=settings.max_output_megapixels,
            scale=options.scale,
            required_mb=peak_megapixels * 1_000_000 * _BYTES_PER_OUTPUT_PIXEL / 1024**2,
        )


class _Progress:
    """Turns per-stage progress into a single monotonic 0..1 for the whole job.

    The API contract promises ``progress: 0..1``, and a progress bar that resets to
    zero four times is not that. Each stage owns a slice of the range, sized roughly
    by how long it actually takes: analysis is fast, upscaling dominates, face
    restoration is a second or two, post-processing is instant.

    The upscaling slice is divided evenly across passes, so a three-pass x8 run
    advances smoothly from 5% to 85% rather than sweeping to 100% three times.
    """

    #: (stage prefix, share of the whole job). Must sum to 1.0.
    _WEIGHTS: ClassVar[dict[str, float]] = {
        "analyzing": 0.05,
        "upscaling": 0.75,
        "restoring faces": 0.10,
        "removing background": 0.05,
        "post-processing": 0.05,
    }

    #: Stages that only happen sometimes. When one is skipped its slice is handed to
    #: upscaling rather than left as a gap the bar visibly jumps over.
    _OPTIONAL: ClassVar[tuple[str, ...]] = ("restoring faces", "removing background")

    def __init__(self, callback: ProgressFn | None) -> None:
        self._callback = callback
        self._passes = 1
        self._active: set[str] = set()
        self._last = 0.0

    def plan(self, passes: int, faces: bool, background: bool = False) -> None:
        self._passes = max(1, passes)
        self._active = set()
        if faces:
            self._active.add("restoring faces")
        if background:
            self._active.add("removing background")

    def stage(self, name: str, fraction: float) -> None:
        base, weight = self._slice(name)
        self._emit(name, base + weight * fraction)

    def pass_progress(self, index: int, name: str, fraction: float) -> None:
        base, weight = self._slice("upscaling")
        share = weight / self._passes
        self._emit(name, base + share * (index + fraction))

    def _slice(self, name: str) -> tuple[float, float]:
        """(offset, width) of this stage within the whole job."""
        skipped = sum(
            self._WEIGHTS[stage] for stage in self._OPTIONAL if stage not in self._active
        )

        offset = 0.0
        for stage, weight in self._WEIGHTS.items():
            if stage == "upscaling":
                weight += skipped  # absorb whatever is not going to run

            if name.startswith(stage):
                return offset, weight

            if stage not in self._OPTIONAL or stage in self._active:
                offset += weight

        return offset, 0.0

    def _emit(self, stage: str, value: float) -> None:
        if self._callback is None:
            return
        # Never go backwards: a bar that retreats reads as a bug even when the
        # underlying number is honest.
        self._last = min(1.0, max(self._last, value))
        self._callback(stage, self._last)


_engine: Engine | None = None


def get_engine() -> Engine:
    """Process-wide engine. Shares the model cache across requests."""
    global _engine
    if _engine is None:
        # Populate the registry on first use rather than at import: it reads YAML
        # from disk, and importing a library should not touch the filesystem.
        _ensure_registry()
        _engine = Engine()
    return _engine


def _ensure_registry() -> None:
    from ..core.config import get_settings

    if len(model_registry) > 0:
        return

    import visionsr.models  # noqa: F401  — import registers the architectures

    config_dir = get_settings().config_dir
    if config_dir.exists():
        model_registry.load_dir(config_dir, replace=True)
