"""Backend selection and the loaded-model cache.

Loading an SR model costs 0.5-3s (disk read + weight init + host->device copy),
far more than running one tile. Serving many requests therefore requires the
model to stay resident, but a 4GB card cannot hold everything at once — so this
is an LRU cache with a hard resident-count ceiling, evicting (and freeing VRAM)
on the way out.

Hot-swap: :func:`reload_model` drops a model from the cache so the next request
rebuilds it from a changed spec, without restarting the process.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

from ..core.config import get_settings
from ..core.device import default_precision, resolve_backend
from ..core.errors import BackendUnavailableError, UnsupportedBackendError
from ..core.types import Backend, ModelSpec, Precision
from .base import BackendAdapter, RunnableModel
from .onnx_backend import OnnxAdapter
from .openvino_backend import OpenVinoAdapter
from .torch_backend import TorchAdapter

log = logging.getLogger(__name__)

_ADAPTERS: tuple[BackendAdapter, ...] = (TorchAdapter(), OnnxAdapter(), OpenVinoAdapter())


@dataclass(frozen=True, slots=True)
class CacheKey:
    model_id: str
    backend: Backend
    precision: Precision


def adapter_for(backend: Backend) -> BackendAdapter:
    for adapter in _ADAPTERS:
        if backend in adapter.handles:
            if not adapter.is_available():
                raise BackendUnavailableError(
                    backend.value, f"{type(adapter).__name__} runtime is not installed"
                )
            return adapter
    raise BackendUnavailableError(backend.value, "no adapter handles this backend")


class ModelCache:
    """Bounded LRU of device-resident models."""

    def __init__(self, max_size: int | None = None) -> None:
        self._max_size = max_size or get_settings().model_cache_size
        self._entries: OrderedDict[CacheKey, RunnableModel] = OrderedDict()
        self._lock = threading.RLock()

    def get(
        self,
        spec: ModelSpec,
        backend: Backend | None = None,
        precision: Precision | None = None,
    ) -> RunnableModel:
        """Return a ready model, loading it if it is not already resident."""
        resolved_backend = resolve_backend(backend, spec.backends)
        if resolved_backend not in spec.backends:
            raise UnsupportedBackendError(
                spec.id, resolved_backend.value, [b.value for b in spec.backends]
            )

        resolved_precision = precision or default_precision(resolved_backend)
        if resolved_precision not in spec.precisions:
            log.debug(
                "%s does not support %s; using %s",
                spec.id,
                resolved_precision.value,
                spec.precisions[0].value,
            )
            resolved_precision = spec.precisions[0]

        key = CacheKey(spec.id, resolved_backend, resolved_precision)

        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached.is_loaded:
                self._entries.move_to_end(key)
                return cached

            # Evict *before* loading: on a 4GB card, loading first would spike to
            # peak(old + new) and OOM at exactly the moment we were trying to avoid.
            while len(self._entries) >= self._max_size:
                self._evict_oldest()

            adapter = adapter_for(resolved_backend)
            model = adapter.load(spec, resolved_backend, resolved_precision)
            self._entries[key] = model
            return model

    def _evict_oldest(self) -> None:
        key, model = self._entries.popitem(last=False)
        log.debug("evicting %s from model cache", key.model_id)
        model.unload()

    def evict(self, model_id: str) -> int:
        """Drop every resident variant of one model. Returns how many were freed."""
        with self._lock:
            keys = [k for k in self._entries if k.model_id == model_id]
            for key in keys:
                self._entries.pop(key).unload()
        return len(keys)

    def clear(self) -> None:
        with self._lock:
            while self._entries:
                self._evict_oldest()

    @property
    def resident(self) -> list[CacheKey]:
        with self._lock:
            return list(self._entries)


# Process-wide cache. The API layer shares one across requests.
cache = ModelCache()


def get_model(
    spec: ModelSpec,
    backend: Backend | None = None,
    precision: Precision | None = None,
) -> RunnableModel:
    return cache.get(spec, backend, precision)


def reload_model(model_id: str) -> int:
    """Hot-swap: force the next request for this model to rebuild it."""
    freed = cache.evict(model_id)
    log.info("hot-swap: dropped %d resident instance(s) of %s", freed, model_id)
    return freed


def backend_capabilities() -> dict[str, dict]:
    """Per-adapter capability report for `visionsr doctor` and /health."""
    return {type(a).__name__.replace("Adapter", "").lower(): a.capabilities() for a in _ADAPTERS}
