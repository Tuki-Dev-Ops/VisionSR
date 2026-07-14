"""Tiler tests.

The tiler is the component most likely to be quietly wrong: a bad overlap or a bad
blend produces an image that is the right size and looks fine at a glance, with a
faint grid of seams you only notice at 100% zoom. So these tests check coverage and
reconstruction numerically, using a fake model whose correct output is known
exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionsr.backends.base import RunnableModel
from visionsr.core.types import Backend, ModelSpec, Precision, Task
from visionsr.inference.tiler import TiledRunner, feather_mask, plan_tiles


class NearestUpscaler(RunnableModel):
    """A model whose output is exactly reproducible: nearest-neighbour upscaling.

    That is the point. If the tiler splits an image, runs this on each tile, and
    reassembles, the result must equal a single nearest-neighbour upscale of the
    whole image — to within blending error. Any deviation is the tiler's fault, and
    nothing else's. A real network could not give that guarantee.
    """

    def __init__(self, scale: int = 2) -> None:
        spec = ModelSpec(
            id="fake-nearest",
            name="fake",
            task=Task.SUPER_RESOLUTION,
            architecture="<none>",
            scale=scale,
            tile_overlap=8,
        )
        super().__init__(spec, Backend.CPU, Precision.FP32)
        self.calls = 0

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.repeat(np.repeat(batch, self.scale, axis=1), self.scale, axis=2)

    def unload(self) -> None:
        self._loaded = False


# -- planning ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "tile", "overlap"),
    [
        (100, 100, 64, 8),
        (320, 200, 128, 16),
        (64, 64, 64, 8),  # exactly one tile
        (65, 65, 64, 8),  # one pixel over — the off-by-one case
        (1000, 37, 256, 32),  # extreme aspect ratio
        (7, 9, 64, 8),  # smaller than a tile in both axes
    ],
)
def test_tiles_cover_every_pixel(width, height, tile, overlap):
    """Every pixel must belong to at least one tile. A gap is a hole in the output."""
    coverage = np.zeros((height, width), dtype=np.int32)

    for t in plan_tiles(width, height, tile, overlap):
        coverage[t.y : t.y + t.h, t.x : t.x + t.w] += 1

    assert coverage.min() >= 1, f"{(coverage == 0).sum()} pixel(s) covered by no tile"


def test_tiles_stay_inside_the_image():
    for t in plan_tiles(300, 180, 128, 16):
        assert t.x >= 0 and t.y >= 0
        assert t.x + t.w <= 300
        assert t.y + t.h <= 180


def test_edge_flags_mark_the_border():
    tiles = plan_tiles(300, 180, 128, 16)

    assert any(t.edge_left and t.edge_top for t in tiles)
    assert any(t.edge_right and t.edge_bottom for t in tiles)

    for t in tiles:
        assert t.edge_left == (t.x == 0)
        assert t.edge_right == (t.x + t.w >= 300)


def test_overlap_must_leave_forward_progress():
    """tile_size <= 2*overlap means stride <= 0 — an infinite loop, not a slow one."""
    with pytest.raises(ValueError, match="forward progress"):
        plan_tiles(100, 100, 64, 32)


# -- blending ---------------------------------------------------------------


def test_feather_mask_does_not_ramp_at_image_borders():
    """Ramping at the outer border would fade the picture out into nothing."""
    tiles = plan_tiles(300, 180, 128, 16)
    corner = next(t for t in tiles if t.edge_left and t.edge_top)

    mask = feather_mask(corner.h, corner.w, 16, corner)

    assert mask[0, 0, 0] == pytest.approx(1.0), "top-left corner of the image was feathered"
    assert mask[-1, -1, 0] < 0.9, "the interior edges should still ramp"


def test_feather_mask_is_never_zero():
    """A strictly-zero weight would divide by zero if a pixel had only that tile."""
    # Large enough to actually contain a tile that touches no border — a 300x180
    # grid at 128px does not, and every tile there is an edge tile.
    tiles = plan_tiles(600, 500, 128, 16)
    interior = next(
        t for t in tiles if not (t.edge_left or t.edge_top or t.edge_right or t.edge_bottom)
    )
    assert feather_mask(interior.h, interior.w, 16, interior).min() > 0


# -- reconstruction ---------------------------------------------------------


@pytest.mark.parametrize(("tile", "overlap"), [(64, 8), (96, 16), (128, 4), (48, 12)])
def test_tiled_result_matches_the_untiled_one(gradient_image, tile, overlap):
    """The whole point: tiling must be invisible.

    Nearest-neighbour upscaling is piecewise-constant, so a correct blend of
    overlapping tiles reproduces it *exactly*, not approximately. A tolerance of one
    8-bit level means "no visible seam" is not a judgement call — and it still holds
    now that the output is assembled band-by-band rather than in one buffer, which is
    exactly what this test exists to guarantee.
    """
    model = NearestUpscaler(scale=2)

    tiled, count = TiledRunner(model).run(gradient_image, tile_size=tile, overlap=overlap)
    expected = np.repeat(np.repeat(gradient_image, 2, axis=0), 2, axis=1)

    assert count >= 1
    assert tiled.dtype == np.uint8
    assert tiled.shape == expected.shape

    deviation = np.abs(tiled.astype(np.int16) - expected.astype(np.int16)).max()
    assert deviation <= 1, (
        f"tiled output deviates by {deviation}/255 from the untiled result at "
        f"tile={tile}, overlap={overlap} — the blend is leaving seams."
    )


def test_single_tile_path_is_exact(gradient_image):
    """An image smaller than one tile must not be padded, blended or otherwise touched."""
    model = NearestUpscaler(scale=2)

    tiled, count = TiledRunner(model).run(gradient_image, tile_size=512, overlap=16)

    assert count == 1
    assert model.calls == 1
    expected = np.repeat(np.repeat(gradient_image, 2, axis=0), 2, axis=1)
    assert np.array_equal(tiled, expected)


def test_oom_backs_off_and_retries(gradient_image):
    """A device OOM should halve the tile and retry, not fail the request."""

    class FlakyModel(NearestUpscaler):
        def __init__(self) -> None:
            super().__init__(scale=2)
            self.attempted: list[int] = []

        def infer(self, batch: np.ndarray) -> np.ndarray:
            edge = batch.shape[1]
            self.attempted.append(edge)
            # Anything above 64px "does not fit".
            if edge > 64:
                raise RuntimeError("CUDA error: out of memory")
            return super().infer(batch)

    model = FlakyModel()

    result, _ = TiledRunner(model).run(gradient_image, tile_size=128, overlap=8)

    assert max(model.attempted) > 64, "the first attempt should have used the large tile"
    assert result.shape == (gradient_image.shape[0] * 2, gradient_image.shape[1] * 2, 3)


def test_oom_at_minimum_tile_gives_up_clearly(gradient_image):
    """When even the smallest tile OOMs, say so — do not spin forever."""
    from visionsr.core.errors import OutOfMemoryError

    class AlwaysOOM(NearestUpscaler):
        def infer(self, batch: np.ndarray) -> np.ndarray:
            raise RuntimeError("CUDA error: out of memory")

    with pytest.raises(OutOfMemoryError, match="floor"):
        TiledRunner(AlwaysOOM()).run(gradient_image, tile_size=128, overlap=8)


# -- streaming assembly -----------------------------------------------------


def test_the_blend_buffer_does_not_grow_with_image_height():
    """Memory must be O(tile height x width), not O(output pixels).

    This is the property the whole band-streaming design exists for, and it is the one
    a refactor would silently lose: reverting to a full-size accumulator would keep
    every other test green while quietly reintroducing a 6GB allocation on a 24MP job,
    and an out-of-memory kill that takes the worker down with it.

    So: run the same width at two very different heights and assert the float buffer
    never holds more than a couple of tile rows either way.
    """
    tile, overlap, scale = 64, 8, 2

    short = _peak_band_rows(height=200, tile=tile, overlap=overlap, scale=scale)
    tall = _peak_band_rows(height=2000, tile=tile, overlap=overlap, scale=scale)

    # Ten times the height must not mean ten times the resident rows.
    assert tall <= short + tile * scale, (
        f"the blend buffer grew with image height: {short} rows at 200px, {tall} rows "
        f"at 2000px. The full-size accumulator is back."
    )
    # And the bound is genuinely a couple of tile rows, not the whole image.
    assert tall < 4 * tile * scale, f"band held {tall} rows, expected fewer than {4 * tile * scale}"


def _peak_band_rows(height: int, tile: int, overlap: int, scale: int) -> int:
    """Largest number of output rows the blend band held during one run."""
    from visionsr.inference import tiler

    peak = 0

    class Instrumented(tiler._Band):
        def add(self, *args, **kwargs) -> None:
            super().add(*args, **kwargs)
            nonlocal peak
            peak = max(peak, self.resident_rows)

    original = tiler._Band
    tiler._Band = Instrumented
    try:
        TiledRunner(NearestUpscaler(scale=scale)).run(
            np.zeros((height, 320, 3), dtype=np.uint8), tile_size=tile, overlap=overlap
        )
    finally:
        tiler._Band = original

    return peak
