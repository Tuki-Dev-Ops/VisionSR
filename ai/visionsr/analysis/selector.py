"""Model selection.

Turns an :class:`ImageAnalysis` into a concrete plan: which SR model, whether to
run face restoration, and how many times to chain the SR model to reach the
requested scale.

The rule from the spec is that no model may be named in engine code. So this file
names none: it asks the registry for models matching a *task* and *content type*,
and ranks the candidates. Deleting every model from ``models.yaml`` and adding
different ones changes the selection without touching this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.errors import ModelNotFoundError
from ..core.registry import ModelRegistry
from ..core.registry import models as default_registry
from ..core.types import Backend, ContentType, EnhanceOptions, ImageAnalysis, ModelSpec, Task

log = logging.getLogger(__name__)

#: Content types that route to the anime task rather than generic SR.
_ILLUSTRATION = frozenset({ContentType.ANIME, ContentType.MANGA})

#: VRAM-per-megapixel of the heaviest model we expect to see. Used only to put
#: ``vram_mb_per_megapixel`` on a 0..1 scale, so a new model with a higher figure
#: simply saturates at 1.0 rather than breaking the ranking.
_REFERENCE_CAPACITY_MB_PER_MP = 2400.0

#: The size at which an expensive model's runtime becomes the user's problem. The
#: size penalty is scaled against this, so the crossover from "best model" to "fast
#: model" lands at roughly 3-4MP for a clean image, and later for a damaged one.
_LARGE_IMAGE_MEGAPIXELS = 8.0

#: How much a badly degraded image is willing to pay for a bigger network...
_DAMAGE_WEIGHT = 40.0

#: ...and how much a large one is unwilling to. Larger than _DAMAGE_WEIGHT, so on a
#: genuinely huge input, size wins even over heavy degradation.
_SIZE_WEIGHT = 70.0

#: A face smaller than this fraction of the frame is not worth restoring — the
#: 512px crop would be mostly upsampled mush, and GFPGAN would invent a stranger.
_MIN_FACE_RATIO = 0.0015


def _capacity(spec: ModelSpec) -> float:
    """Model cost on a 0 (cheap) .. 1 (expensive) scale.

    VRAM per megapixel is used as the proxy rather than parameter count, because it
    is what actually constrains this system: it is what decides the tile size, and
    therefore the wall-clock time.
    """
    return min(spec.vram_mb_per_megapixel / _REFERENCE_CAPACITY_MB_PER_MP, 1.0)


@dataclass(slots=True)
class Plan:
    """What the pipeline is about to do, and why."""

    sr_model: ModelSpec
    #: Times to run the SR model. 2 passes of a x4 model gives x16.
    passes: int
    #: Final resample factor after the passes, to land exactly on the request.
    #: <1 means downscale (e.g. x4 model, x3 requested -> 0.75).
    final_resample: float
    face_model: ModelSpec | None
    background_model: ModelSpec | None
    reason: str

    @property
    def effective_scale(self) -> float:
        return (self.sr_model.scale**self.passes) * self.final_resample


class ModelSelector:
    """Ranks registry models against an analysis. Stateless."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or default_registry

    def select(
        self,
        analysis: ImageAnalysis,
        options: EnhanceOptions,
        backend: Backend | None = None,
    ) -> Plan:
        sr_model = (
            self._registry.get(options.model_id)
            if options.model_id
            else self._select_sr(analysis, options, backend)
        )

        passes, final_resample = _plan_scaling(sr_model.scale, options.scale)
        face_model = self._select_face(analysis, options, backend)
        background_model = self._select_background(options, backend)

        plan = Plan(
            sr_model=sr_model,
            passes=passes,
            final_resample=final_resample,
            face_model=face_model,
            background_model=background_model,
            reason=self._explain(
                analysis, options, sr_model, face_model, background_model, passes
            ),
        )
        log.info("plan: %s", plan.reason)
        return plan

    def _select_sr(
        self,
        analysis: ImageAnalysis,
        options: EnhanceOptions,
        backend: Backend | None,
    ) -> ModelSpec:
        task = (
            Task.ANIME_RESTORATION
            if analysis.content_type in _ILLUSTRATION
            else Task.SUPER_RESOLUTION
        )

        candidates = self._registry.find(
            task=task, content_type=analysis.content_type, backend=backend
        )

        # Nothing claims this content type — fall back to the task's generalists,
        # then to any SR model at all. An unknown content type must still upscale.
        if not candidates:
            candidates = self._registry.find(task=task, backend=backend)
        if not candidates:
            candidates = self._registry.find(task=Task.SUPER_RESOLUTION, backend=backend)
        if not candidates:
            raise ModelNotFoundError(
                f"<no model for task {task.value}>",
                [m.id for m in self._registry.all()],
            )

        return max(candidates, key=lambda spec: self._rank(spec, analysis, options))

    def _rank(self, spec: ModelSpec, analysis: ImageAnalysis, options: EnhanceOptions) -> float:
        """Higher is better.

        The policy, in order:

        **Quality first.** The best model for the task wins by default. This is an
        upscaler; people run it because they want the best result, and on a 2MP photo
        the heavy model costs a couple of seconds. Trading visible quality to save
        those seconds is a bad deal, and it is the deal the previous version of this
        function made — it matched model capacity to *degradation*, so a merely-decent
        photo got the cheap model even when the expensive one was nearly free.

        **Cost, in proportion to the work.** A heavy model on a 24MP input is hundreds
        of tiles and minutes of GPU. So the penalty for an expensive model scales with
        the size of the image, not with anything else. Below a few megapixels it is
        negligible and quality wins; above that it takes over and the light model does.

        **Damage earns capacity.** A ruined image genuinely benefits from the larger
        network, which partly offsets the size penalty.

        Callers who disagree with all of this pass ``--model`` and are obeyed.
        """
        cost = _capacity(spec)  # 0 (cheap) .. 1 (expensive)
        need = 1.0 - analysis.quality_score  # 0 (pristine) .. 1 (ruined)
        pressure = min(analysis.megapixels / _LARGE_IMAGE_MEGAPIXELS, 1.5)

        # Priority is the quality ranking of models for this task. It leads.
        score = float(spec.priority)

        # A model whose native scale divides cleanly into the request needs fewer
        # passes, and every extra pass compounds the previous one's artefacts.
        if spec.scale == options.scale:
            score += 50
        elif options.scale % spec.scale == 0:
            score += 20

        score += _DAMAGE_WEIGHT * cost * need
        score -= _SIZE_WEIGHT * cost * pressure

        return score

    def _select_face(
        self,
        analysis: ImageAnalysis,
        options: EnhanceOptions,
        backend: Backend | None,
    ) -> ModelSpec | None:
        if options.face_restore is False:
            return None

        # Auto mode: only when faces were actually found and they are big enough
        # to hold real detail.
        if options.face_restore is None:
            if not analysis.has_faces:
                return None
            largest = max(w * h for _, _, w, h in analysis.faces)
            if largest / (analysis.width * analysis.height) < _MIN_FACE_RATIO:
                log.debug("faces present but too small to restore; skipping")
                return None

        candidates = self._registry.find(task=Task.FACE_RESTORATION, backend=backend)
        if not candidates:
            if options.face_restore:
                log.warning("face restoration requested but no face model is registered")
            return None

        return candidates[0]  # find() is already priority-ordered

    def _select_background(
        self, options: EnhanceOptions, backend: Backend | None
    ) -> ModelSpec | None:
        """Never automatic. Cutting the background out is a destructive edit, and no
        image analysis can tell you whether the user wanted it."""
        if not options.remove_background:
            return None

        if options.background_model_id:
            return self._registry.get(options.background_model_id)

        candidates = self._registry.find(task=Task.BACKGROUND_REMOVAL, backend=backend)
        if not candidates:
            raise ModelNotFoundError(
                "<no background removal model>", [m.id for m in self._registry.all()]
            )
        return candidates[0]  # find() is priority-ordered

    def _explain(
        self,
        analysis: ImageAnalysis,
        options: EnhanceOptions,
        sr: ModelSpec,
        face: ModelSpec | None,
        background: ModelSpec | None,
        passes: int,
    ) -> str:
        bits = [
            f"{analysis.content_type.value} ({analysis.content_confidence:.0%})",
            f"quality {analysis.quality_score:.2f}",
            f"-> {sr.id}" + (f" x{passes} passes" if passes > 1 else ""),
        ]
        if face:
            bits.append(f"+ {face.id} on {len(analysis.faces)} face(s)")
        if background:
            bits.append(f"+ {background.id} (background removal)")
        if options.model_id:
            bits.append("(model pinned by caller)")
        return ", ".join(bits)


def _plan_scaling(model_scale: int, target_scale: int) -> tuple[int, float]:
    """How many model passes, plus a final resample, to land on ``target_scale``.

    A x4 model asked for x8 runs twice (x16) and then downsamples by 0.5. That
    beats a single pass followed by a 2x bicubic upsample: the network's second
    pass reconstructs real detail, whereas bicubic only interpolates. Downscaling
    afterwards is lossless in perceived sharpness.
    """
    if target_scale <= 1:
        return 0, float(target_scale)

    passes = 1
    while model_scale**passes < target_scale:
        passes += 1

    return passes, target_scale / (model_scale**passes)
