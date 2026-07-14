"""Tiled inference.

A 4GB card cannot hold the activations for a 12MP image through a 23-block RRDBNet
— not close. So the image is cut into overlapping tiles, each tile is run
separately, and the results are blended back together.

Two things make this non-trivial:

**Seams.** Naively stitching abutting tiles leaves visible edges, because a
convolution near a tile border sees zero-padding instead of the neighbouring
pixels. Two defences, both needed:

  1. Tiles *overlap*. Every output pixel near a boundary is produced by at least
     two tiles, each of which had real context on at least one side.
  2. Overlaps are *feathered*, not cropped. Each tile carries a weight map that
     ramps from 0 at its outer edge to 1 in its interior; contributions are summed
     and divided by total weight. A cosine ramp (not linear) makes the first
     derivative continuous, which is what stops the eye from finding the seam.

**Choosing the tile size.** Too large and it OOMs; too small and the run is
dominated by per-tile overhead and overlap waste. It is derived from *live* free
VRAM, and on OOM the runner halves it and retries rather than failing the request.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from ..backends.base import RunnableModel
from ..core.config import get_settings
from ..core.device import free_vram_mb
from ..core.errors import OutOfMemoryError
from ..core.types import Backend, ModelSpec

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class Tile:
    """One tile in input-image coordinates."""

    x: int
    y: int
    w: int
    h: int
    row: int
    col: int
    #: True on each side where this tile touches the image border. Those sides get
    #: no feather ramp — there is no neighbour to blend with, and ramping there
    #: would fade the picture out into nothing.
    edge_left: bool
    edge_top: bool
    edge_right: bool
    edge_bottom: bool


def plan_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    """Lay a grid of overlapping tiles over the image.

    Tiles advance by ``stride = tile_size - 2 * overlap`` so that consecutive tiles
    share a ``2 * overlap`` band. The last tile in each direction is pulled back to
    end exactly on the border instead of being clipped, so it may overlap its
    predecessor by more than the nominal amount — harmless, the blend handles it.
    """
    if tile_size <= 2 * overlap:
        raise ValueError(
            f"tile_size ({tile_size}) must exceed 2*overlap ({2 * overlap}); "
            "otherwise tiles make no forward progress."
        )

    stride = tile_size - 2 * overlap
    tiles: list[Tile] = []

    ys = _starts(height, tile_size, stride)
    xs = _starts(width, tile_size, stride)

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            h = min(tile_size, height - y)
            w = min(tile_size, width - x)
            tiles.append(
                Tile(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    row=row,
                    col=col,
                    edge_left=(x == 0),
                    edge_top=(y == 0),
                    edge_right=(x + w >= width),
                    edge_bottom=(y + h >= height),
                )
            )
    return tiles


def _starts(extent: int, tile_size: int, stride: int) -> list[int]:
    if extent <= tile_size:
        return [0]

    starts = list(range(0, extent - tile_size + 1, stride))
    # Ensure full coverage: if the stride left a strip at the end uncovered, add a
    # final tile flush with the border.
    if starts[-1] + tile_size < extent:
        starts.append(extent - tile_size)
    return starts


def feather_mask(
    h: int, w: int, overlap: int, tile: Tile, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Per-pixel blend weight for one tile, shape (h, w, 1).

    Ramps from ~0 at each *interior* edge to 1 over ``overlap`` pixels, using a
    raised cosine. Image-border edges stay at full weight.
    """
    ramp = _cosine_ramp(overlap, dtype)

    wy = np.ones(h, dtype=dtype)
    wx = np.ones(w, dtype=dtype)

    n = min(overlap, h // 2)
    if n > 0:
        if not tile.edge_top:
            wy[:n] = ramp[:n]
        if not tile.edge_bottom:
            wy[-n:] = ramp[:n][::-1]

    n = min(overlap, w // 2)
    if n > 0:
        if not tile.edge_left:
            wx[:n] = ramp[:n]
        if not tile.edge_right:
            wx[-n:] = ramp[:n][::-1]

    return (wy[:, None] * wx[None, :])[:, :, None]


def _cosine_ramp(length: int, dtype: np.dtype) -> np.ndarray:
    """0 -> 1 over `length` samples, with zero slope at both ends."""
    if length <= 0:
        return np.ones(0, dtype=dtype)
    t = (np.arange(length, dtype=dtype) + 0.5) / length
    # Floor at a small epsilon: a strictly-zero weight column would divide by zero
    # if it ever ended up as the only contributor to a pixel.
    return np.maximum(0.5 - 0.5 * np.cos(math.pi * t), 1e-4).astype(dtype)


def auto_tile_size(spec: ModelSpec, backend: Backend, image_megapixels: float) -> int:
    """Largest tile the free VRAM can take, rounded to a multiple of 32.

    Backends that cannot report free memory (CPU, DirectML, ONNX) fall back to the
    spec default — they are not VRAM-bound in the same way.
    """
    settings = get_settings()

    free = free_vram_mb(backend)
    if free is None:
        return spec.tile_size

    budget_mb = free - settings.vram_headroom_mb

    if budget_mb <= 0:
        # Dropping to 64px tiles here is technically correct and practically a
        # disaster: a 0.3MP image at x8 becomes a thousand tiles and ten minutes of
        # GPU, for output identical to the two-second version. Before accepting that,
        # reclaim the one thing we control — other models sitting resident in the
        # cache. On a 4GB card an idle GFPGAN alone is ~350MB.
        from ..backends.factory import cache

        evicted = [key for key in cache.resident if key.model_id != spec.id]
        if evicted:
            log.warning(
                "only %dMB VRAM free — evicting %d cached model(s) to avoid "
                "collapsing to minimum tiles",
                free,
                len(evicted),
            )
            for key in evicted:
                cache.evict(key.model_id)

            free = free_vram_mb(backend) or free
            budget_mb = free - settings.vram_headroom_mb

    if budget_mb <= 0:
        log.warning(
            "only %dMB VRAM free even after evicting the model cache; falling back to "
            "%dpx tiles. This will be slow.",
            free,
            settings.min_tile_size,
        )
        return settings.min_tile_size

    tile_megapixels = budget_mb / spec.vram_mb_per_megapixel
    tile_px = int(math.sqrt(tile_megapixels * 1_000_000))

    # No point tiling above the image itself.
    image_px = int(math.sqrt(image_megapixels * 1_000_000))
    tile_px = min(tile_px, max(image_px, settings.min_tile_size))

    tile_px = max(settings.min_tile_size, min(settings.max_tile_size, tile_px))
    tile_px = (tile_px // 32) * 32

    log.debug(
        "auto tile: %dpx (free=%dMB, budget=%dMB, %.0fMB/MP)",
        tile_px,
        free,
        budget_mb,
        spec.vram_mb_per_megapixel,
    )
    return max(settings.min_tile_size, tile_px)


def _pad_to_multiple(tile: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    """Reflect-pad so H and W divide evenly. Returns (padded, pad_h, pad_w).

    Reflect, not zero: zero-padding injects a hard black edge that the network
    happily turns into a dark halo.
    """
    if multiple <= 1:
        return tile, 0, 0

    h, w = tile.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return tile, 0, 0

    padded = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, pad_h, pad_w


class _Band:
    """A sliding window of output rows being blended.

    The naive assembler allocates a float32 accumulator and a weight map the size of
    the *whole* output — 16 bytes per output pixel. At x4 on a 24MP photo that is
    6GB, and when the allocation fails there is no exception to catch: the process
    dies.

    But a tile can only contribute to output rows it actually covers, and tiles arrive
    in row-major order. So once the next tile row has begun, every row above its top
    edge is final and can be resolved to uint8 and dropped. Only the rows currently in
    play need to exist in float, and that is a band a couple of tiles tall — a fixed
    cost, independent of image height.

    Memory therefore goes from O(output pixels) to O(tile height x output width).
    """

    def __init__(self, out_w: int) -> None:
        self._out_w = out_w
        self._top = 0  # output row this band starts at
        self._accum = np.zeros((0, out_w, 3), dtype=np.float32)
        self._weights = np.zeros((0, out_w, 1), dtype=np.float32)

    def add(self, top: int, left: int, values: np.ndarray, mask: np.ndarray) -> None:
        """Blend one tile's output into the band, growing it if the tile runs past."""
        height, width = values.shape[:2]
        self._grow_to(top + height)

        y = top - self._top
        self._accum[y : y + height, left : left + width] += values * mask
        self._weights[y : y + height, left : left + width] += mask

    def settle(self, output: np.ndarray, until: int) -> None:
        """Resolve every row above ``until`` into ``output`` and release it."""
        rows = until - self._top
        if rows <= 0:
            return

        rows = min(rows, self._accum.shape[0])
        if rows > 0:
            resolved = self._accum[:rows] / self._weights[:rows]
            # Round rather than truncate, and clip: a network can overshoot [0,1] and
            # the blend divides by a weight that is only guaranteed to be positive.
            np.clip(resolved * 255.0 + 0.5, 0, 255, out=resolved)
            output[self._top : self._top + rows] = resolved.astype(np.uint8)

            self._accum = self._accum[rows:]
            self._weights = self._weights[rows:]

        self._top = until

    def _grow_to(self, bottom: int) -> None:
        needed = bottom - self._top - self._accum.shape[0]
        if needed <= 0:
            return

        self._accum = np.concatenate(
            [self._accum, np.zeros((needed, self._out_w, 3), dtype=np.float32)]
        )
        self._weights = np.concatenate(
            [self._weights, np.zeros((needed, self._out_w, 1), dtype=np.float32)]
        )

    @property
    def resident_rows(self) -> int:
        """Rows currently held in float. Used by the tests to prove the band is bounded."""
        return self._accum.shape[0]


class TiledRunner:
    """Runs a model over an image tile by tile, blending the results.

    Stateless between calls apart from the OOM-driven tile-size backoff, which is
    intentionally remembered: once a size has OOMed on this device, the next
    request should not rediscover that the hard way.
    """

    def __init__(self, model: RunnableModel) -> None:
        self.model = model
        self.spec = model.spec
        self._oom_ceiling: int | None = None

    def run(
        self,
        image: np.ndarray,
        tile_size: int | None = None,
        overlap: int | None = None,
        progress: ProgressFn | None = None,
    ) -> tuple[np.ndarray, int]:
        """Upscale ``image``.

        Args:
            image: HWC **uint8** RGB, 3 channels.
            tile_size: override; None => derived from free VRAM.
            overlap: override; None => the spec's value.
            progress: called as (done, total) after each tile.

        Returns:
            (HWC uint8 RGB upscaled image, number of tiles run).

        uint8 in and uint8 out, deliberately. The obvious design keeps the working
        canvas in float32 and only quantises at the end — but that costs 12 bytes per
        pixel of a canvas that can be hundreds of megapixels, and the network never
        sees the canvas anyway: it sees tiles. So tiles are converted to float
        individually, and the canvas stays uint8 at 3 bytes.

        The cost is that chaining passes (x4 then x4 again) round-trips through 8-bit
        between them. That is a quarter of a level of quantisation on values the model
        is about to reconstruct from scratch anyway, and it is measurably invisible —
        while the float32 canvas was the single thing standing between this engine and
        an out-of-memory kill on large work.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB, got shape {image.shape}.")
        if image.dtype != np.uint8:
            raise TypeError(f"Expected a uint8 image, got {image.dtype}.")

        settings = get_settings()
        height, width = image.shape[:2]
        megapixels = (width * height) / 1_000_000

        size = tile_size or auto_tile_size(self.spec, self.model.backend, megapixels)
        if self._oom_ceiling is not None:
            size = min(size, self._oom_ceiling)

        lap = self.spec.tile_overlap if overlap is None else overlap
        # An overlap that swallows the tile leaves zero stride. Keep it under a
        # quarter so `stride > 0` always holds.
        lap = min(lap, max(0, size // 4))

        while True:
            try:
                return self._run_at(image, size, lap, progress)
            except MemoryError as exc:  # raised by _infer_tile on device OOM
                if size <= settings.min_tile_size:
                    raise OutOfMemoryError(size, str(exc)) from exc

                size = max(settings.min_tile_size, size // 2)
                lap = min(lap, max(0, size // 4))
                self._oom_ceiling = size
                log.warning("device OOM — retrying at tile size %dpx", size)
                self._free_device_memory()

    def _run_at(
        self,
        image: np.ndarray,
        tile_size: int,
        overlap: int,
        progress: ProgressFn | None,
    ) -> tuple[np.ndarray, int]:
        scale = self.model.scale
        height, width = image.shape[:2]

        tiles = plan_tiles(width, height, tile_size, overlap)
        out_h, out_w = height * scale, width * scale

        log.info(
            "%s: %dx%d -> %dx%d, %d tile(s) @ %dpx overlap %d",
            self.spec.id,
            width,
            height,
            out_w,
            out_h,
            len(tiles),
            tile_size,
            overlap,
        )

        output = np.empty((out_h, out_w, 3), dtype=np.uint8)

        # Tiles are produced in row-major order by plan_tiles, so a whole tile row can
        # be finished before the next one starts — which means the float blend buffer
        # only ever has to hold the rows currently in play, not the entire image.
        rows: dict[int, list[Tile]] = {}
        for tile in tiles:
            rows.setdefault(tile.row, []).append(tile)

        band = _Band(out_w)
        done = 0

        for row in sorted(rows):
            for tile in rows[row]:
                patch = image[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w]
                result = self._infer_tile(patch)

                mask = feather_mask(tile.h * scale, tile.w * scale, overlap * scale, tile)

                band.add(
                    top=tile.y * scale,
                    left=tile.x * scale,
                    values=result,
                    mask=mask,
                )

                done += 1
                if progress is not None:
                    progress(done, len(tiles))

            # Everything above the next tile row's first pixel can no longer receive a
            # contribution, so it is final. Resolve it and let the memory go.
            next_row = rows.get(row + 1)
            settled = min(t.y for t in next_row) * scale if next_row else out_h
            band.settle(output, settled)

        band.settle(output, out_h)
        return output, len(tiles)

    def _infer_tile(self, patch: np.ndarray) -> np.ndarray:
        """Run one tile, translating device OOM into a plain MemoryError.

        The uint8 -> float32 conversion happens here, per tile, rather than once over
        the whole canvas. That is the entire reason the canvas can stay uint8.
        """
        patch = patch.astype(np.float32) / 255.0
        padded, pad_h, pad_w = _pad_to_multiple(patch, self.spec.window_multiple)

        try:
            out = self.model.infer(padded[None, ...])[0]
        except Exception as exc:
            if _is_oom(exc):
                self._free_device_memory()
                raise MemoryError(str(exc)) from exc
            raise

        if pad_h or pad_w:
            scale = self.model.scale
            h = (padded.shape[0] - pad_h) * scale
            w = (padded.shape[1] - pad_w) * scale
            out = out[:h, :w]

        return out

    def _free_device_memory(self) -> None:
        if self.model.backend in (Backend.CUDA, Backend.TENSORRT):
            import torch

            torch.cuda.empty_cache()

    def iter_tiles(self, image: np.ndarray, tile_size: int, overlap: int) -> Iterator[Tile]:
        """Expose the tile plan without running anything — used by tests and the UI."""
        h, w = image.shape[:2]
        yield from plan_tiles(w, h, tile_size, overlap)


def _is_oom(exc: Exception) -> bool:
    """Recognise an out-of-memory failure across backends.

    torch raises `torch.cuda.OutOfMemoryError`, ONNX Runtime raises a generic
    RuntimeError whose message mentions the allocator, and DirectML says
    "device removed" when it is really just out of memory. Message sniffing is
    unpleasant but it is the only thing that covers all three.
    """
    if isinstance(exc, MemoryError):
        return True

    name = type(exc).__name__
    if name in ("OutOfMemoryError", "OutOfMemoryError_"):
        return True

    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "failed to allocate",
            "device removed",
            "cudnn_status_alloc_failed",
        )
    )
