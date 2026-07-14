"""Background removal.

Runs *after* super resolution, on the upscaled canvas, for the same reason the face
pipeline does: the segmenter downsamples whatever it is given to a fixed 1024px
square, and a 512px upscaled image resized to 1024 is a far better-defined input
than a 128px original resized to 1024. Segmenting first would throw away the detail
super resolution just reconstructed — and the edge quality of a cutout is entirely a
question of detail.

The model gives a soft alpha, and that softness is the point: hair, fur and motion
blur are genuinely semi-transparent, and a hard threshold turns them into a jagged
silhouette. So the mask is kept continuous all the way to the output, and the only
cleanup applied is the one that fixes a real artefact of these networks — a faint
non-zero haze over the whole background, which is invisible on screen and becomes an
ugly grey wash the moment anyone composites onto white.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ..backends.factory import get_model
from ..core.types import Backend, ModelRun, ModelSpec, Precision
from ..preprocessing.io import to_float

log = logging.getLogger(__name__)

#: Below this, alpha is treated as "definitely background" and pushed to zero; above
#: the upper bound, as "definitely foreground". In between it is stretched linearly
#: and left soft.
#:
#: These are not arbitrary. IS-Net's min-max normalised output leaves a low-level haze
#: across the background — values around 0.02-0.08 that read as black on screen but
#: composite as a visible grey film. And it rarely reaches a true 1.0 even in the
#: middle of a solid object, so a foreground left un-stretched comes out slightly
#: translucent. Clamping both ends and stretching the middle fixes both without
#: hardening the edge, which is the thing that must stay soft.
_ALPHA_FLOOR = 0.10
_ALPHA_CEILING = 0.92


class MattingPipeline:
    """Produces an alpha channel that isolates the subject."""

    def run(
        self,
        canvas: np.ndarray,
        spec: ModelSpec,
        *,
        backend: Backend | None = None,
        precision: Precision | None = None,
        feather: int = 0,
    ) -> tuple[np.ndarray, ModelRun]:
        """Cut the background out of ``canvas``.

        Args:
            canvas: HWC uint8 RGB.
            spec: the segmentation model to run.
            feather: extra blur on the alpha edge, in pixels. 0 leaves the model's own
                softness alone, which is usually right.

        Returns:
            (HWC uint8 **RGBA**, run record).
        """
        model = get_model(spec, backend, precision)
        started = time.perf_counter()

        height, width = canvas.shape[:2]
        size = spec.tile_size  # the graph's fixed input square

        # INTER_AREA going down (it integrates, so thin structures survive as partial
        # coverage rather than being point-sampled away), Lanczos coming back up.
        small = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

        raw = model.infer(to_float(small)[None, ...])[0]
        mask = raw[:, :, 0] if raw.ndim == 3 else raw

        alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LANCZOS4)
        alpha = self._clean(alpha, feather)

        rgba = np.dstack([canvas, (alpha * 255.0 + 0.5).astype(np.uint8)])

        duration_ms = (time.perf_counter() - started) * 1000
        coverage = float((alpha > 0.5).mean())
        log.info(
            "removed background with %s in %.0fms (subject covers %.0f%% of the frame)",
            spec.id,
            duration_ms,
            coverage * 100,
        )

        return rgba, ModelRun(
            model_id=spec.id,
            backend=model.backend,
            precision=model.precision,
            duration_ms=duration_ms,
            tiles=1,
        )

    def _clean(self, alpha: np.ndarray, feather: int) -> np.ndarray:
        """Kill the background haze, solidify the interior, keep the edge soft."""
        alpha = np.clip(
            (alpha - _ALPHA_FLOOR) / (_ALPHA_CEILING - _ALPHA_FLOOR), 0.0, 1.0
        ).astype(np.float32)

        if feather > 0:
            # Odd kernel, as cv2 requires.
            alpha = cv2.GaussianBlur(alpha, (feather | 1, feather | 1), 0)

        return alpha


def composite(rgba: np.ndarray, background: tuple[int, int, int] | None = None) -> np.ndarray:
    """Flatten an RGBA cutout onto a solid colour. Used to render previews.

    Straight alpha, not premultiplied — the pipeline never premultiplies, so the RGB
    under a transparent pixel still holds the original background colour. Compositing
    with anything other than this formula would let that colour bleed back in as a
    halo around the subject, which is the classic "cheap cutout" look.
    """
    if rgba.shape[2] != 4:
        return rgba

    colour = np.asarray(background or (255, 255, 255), dtype=np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)

    return np.clip(rgb * alpha + colour * (1.0 - alpha), 0, 255).astype(np.uint8)
