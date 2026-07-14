"""Pluggable registries.

Two separate concerns, deliberately kept apart:

* ``architectures`` maps an architecture name -> a factory that builds the network.
  Adding a new network is one decorator; the engine never learns its name.
* ``models`` maps a model id -> a :class:`ModelSpec` (weights, task, hints).
  Adding a new *checkpoint* of a known architecture is a YAML entry, no code.

Nothing else in the engine may reference a concrete model by name. The rule from
the spec — "never hardcode models" — is enforced here by construction: the
engine only ever asks the registry for "a model that does X".
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, TypeVar

import yaml

from .errors import ArchitectureNotFoundError, ConfigError, ModelNotFoundError
from .types import Backend, ContentType, ModelSpec, Task

log = logging.getLogger(__name__)

#: Architecture sentinel meaning "there is no torch architecture — the checkpoint is
#: already a runnable graph". A model declaring this joins the registry as a YAML entry
#: and nothing else, which is the point of the registry existing.
ONNX_GRAPH = "onnx-graph"


class Architecture(Protocol):
    """Anything the architecture registry can build.

    Deliberately structural: an architecture is any callable returning an object
    the torch backend can ``load_state_dict`` into and call. That keeps the core
    free of a torch import while still typing the contract.
    """

    def __call__(self, **params: Any) -> Any: ...


F = TypeVar("F", bound=Callable[..., Any])


class ArchitectureRegistry:
    """Name -> network factory."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}
        self._lock = threading.RLock()

    def register(self, name: str) -> Callable[[F], F]:
        """Decorator. ``@architectures.register("rrdbnet")``."""

        def decorator(factory: F) -> F:
            with self._lock:
                if name in self._factories:
                    raise ConfigError(f"Architecture {name!r} is already registered.")
                self._factories[name] = factory
            return factory

        return decorator

    def build(self, name: str, **params: Any) -> Any:
        with self._lock:
            factory = self._factories.get(name)
        if factory is None:
            raise ArchitectureNotFoundError(name, list(self._factories))
        return factory(**params)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories


class ModelRegistry:
    """Model id -> spec, plus the queries the selector needs."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._lock = threading.RLock()

    # -- mutation ----------------------------------------------------------------

    def register(self, spec: ModelSpec, *, replace: bool = False) -> ModelSpec:
        """Add a spec. ``replace=True`` supports hot-swapping a model definition."""
        with self._lock:
            if spec.id in self._specs and not replace:
                raise ConfigError(
                    f"Model {spec.id!r} is already registered. Pass replace=True to override."
                )
            self._specs[spec.id] = spec
        log.debug("registered model %s (%s, %s)", spec.id, spec.architecture, spec.task.value)
        return spec

    def unregister(self, model_id: str) -> None:
        with self._lock:
            self._specs.pop(model_id, None)

    def load_yaml(self, path: Path, *, replace: bool = False) -> list[ModelSpec]:
        """Load specs from a YAML file: a top-level ``models:`` list of specs."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

        entries = raw.get("models")
        if not isinstance(entries, list):
            raise ConfigError(f"{path} must contain a top-level 'models:' list.")

        loaded = []
        for entry in entries:
            try:
                spec = ModelSpec.model_validate(entry)
            except Exception as exc:
                got = entry.get("id", "<no id>") if isinstance(entry, dict) else "<not a mapping>"
                raise ConfigError(f"{path}: invalid model spec {got!r}: {exc}") from exc
            loaded.append(self.register(spec, replace=replace))
        log.info("loaded %d model spec(s) from %s", len(loaded), path.name)
        return loaded

    def load_dir(self, directory: Path, *, replace: bool = False) -> list[ModelSpec]:
        loaded: list[ModelSpec] = []
        for path in sorted(directory.glob("*.y*ml")):
            loaded.extend(self.load_yaml(path, replace=replace))
        return loaded

    # -- queries -----------------------------------------------------------------

    def get(self, model_id: str) -> ModelSpec:
        with self._lock:
            spec = self._specs.get(model_id)
        if spec is None:
            raise ModelNotFoundError(model_id, list(self._specs))
        return spec

    def all(self) -> list[ModelSpec]:
        with self._lock:
            return sorted(self._specs.values(), key=lambda s: (-s.priority, s.id))

    def find(
        self,
        *,
        task: Task | None = None,
        content_type: ContentType | None = None,
        scale: int | None = None,
        backend: Backend | None = None,
    ) -> list[ModelSpec]:
        """All specs matching every supplied filter, best (highest priority) first."""
        results = []
        for spec in self.all():
            if task is not None and spec.task != task:
                continue
            # An empty content_types list means "generalist": it matches anything,
            # which is what keeps an unknown or misclassified image upscaling.
            if (
                content_type is not None
                and spec.content_types
                and content_type not in spec.content_types
            ):
                continue
            if scale is not None and spec.scale != scale:
                continue
            if backend is not None and backend not in spec.backends:
                continue
            results.append(spec)
        return results

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self.all())


# Process-wide singletons. Tests build their own instances instead of mutating these.
architectures = ArchitectureRegistry()
models = ModelRegistry()


def register_architecture(name: str) -> Callable[[F], F]:
    """Module-level shorthand for ``architectures.register``."""
    return architectures.register(name)
