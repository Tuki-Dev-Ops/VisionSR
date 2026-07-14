"""Face pipeline: detect -> align -> restore -> blend back.

Runs *after* super-resolution, on the upscaled canvas. That ordering is the whole
trick. GFPGAN always emits a 512x512 face, so restoring first and then upscaling
would push a 512px face through the SR net and re-soften everything it just
rebuilt. Restoring second means the 512px restoration lands on a canvas where the
face region is already close to 512px, and nothing downstream degrades it.

Faces are detected on the *original* image (the detector is calibrated for natural
resolutions, and it is cheaper), then the alignment transform is scaled up to the
output canvas.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..analysis.faces import DetectedFace, FaceDetector, align_face, get_detector, paste_face
from ..backends.factory import get_model
from ..core.types import Backend, ModelRun, ModelSpec, Precision
from ..preprocessing.io import to_float, to_uint8

log = logging.getLogger(__name__)


class FacePipeline:
    """Restores every detected face in an already-upscaled image."""

    def __init__(self, detector: FaceDetector | None = None) -> None:
        self._detector = detector or get_detector()

    def restore(
        self,
        canvas: np.ndarray,
        original: np.ndarray,
        spec: ModelSpec,
        upscale: int,
        *,
        weight: float = 0.5,
        backend: Backend | None = None,
        precision: Precision | None = None,
        faces: list[DetectedFace] | None = None,
    ) -> tuple[np.ndarray, ModelRun | None]:
        """Restore faces in ``canvas``.

        Args:
            canvas: HWC uint8 RGB, the upscaled image. Modified copy is returned.
            original: HWC uint8 RGB, the input — faces are detected here.
            spec: the face model to run.
            upscale: canvas size / original size. Scales the alignment transform.
            weight: 0 keeps the original face, 1 takes the restoration wholesale.
                Below 1 it is blended with the SR-upscaled face, which is what keeps
                identity when GFPGAN gets creative.
            faces: pre-detected faces; re-detected if omitted.

        Returns:
            (canvas, run record). Run is None when there was nothing to do.
        """
        if faces is None:
            faces = self._detector.detect(original)

        if not faces:
            return canvas, None

        model = get_model(spec, backend, precision)
        started = time.perf_counter()
        restored_count = 0

        for face in faces:
            crop = align_face(original, face)  # 512x512 uint8 RGB

            # Plain [0,1] in, [0,1] out. GFPGAN's native [-1,1] convention is
            # declared in its ModelSpec and handled by the backend — see
            # backends/base.py.
            restored = to_uint8(model.infer(to_float(crop)[None, ...])[0])

            if weight < 1.0:
                restored = self._blend_with_source(restored, canvas, face, upscale, weight)

            canvas = paste_face(canvas, restored, face, upscale=upscale)
            restored_count += 1

        duration_ms = (time.perf_counter() - started) * 1000
        log.info("restored %d face(s) in %.0fms", restored_count, duration_ms)

        return canvas, ModelRun(
            model_id=spec.id,
            backend=model.backend,
            precision=model.precision,
            duration_ms=duration_ms,
            tiles=restored_count,
        )

    def _blend_with_source(
        self,
        restored: np.ndarray,
        canvas: np.ndarray,
        face: DetectedFace,
        upscale: int,
        weight: float,
    ) -> np.ndarray:
        """Mix the restored face with the SR model's version of the same face.

        GFPGAN's prior is strong enough that at weight 1.0 it will happily replace a
        blurry face with a sharp *different* one. Blending against the SR output —
        which is faithful but soft — trades some of that sharpness back for identity.
        Sampled from the canvas through the same alignment transform so the two are
        pixel-aligned.
        """
        import cv2

        if face.affine is None:
            return restored

        # `affine` maps original -> template. We need canvas -> template, and
        # canvas = original * upscale, so substitute o = c/upscale:
        #     t = A@o + b  =>  t = (A/upscale)@c + b
        # The linear block is divided; the translation is left alone. (Scaling the
        # translation instead — the intuitive-looking mistake — samples the canvas
        # far outside the face and blends in garbage.)
        affine = face.affine.copy()
        affine[:, :2] /= upscale

        sr_face = cv2.warpAffine(
            canvas, affine, (restored.shape[1], restored.shape[0]), flags=cv2.INTER_LINEAR
        )

        return np.clip(
            restored.astype(np.float32) * weight + sr_face.astype(np.float32) * (1 - weight),
            0,
            255,
        ).astype(np.uint8)
