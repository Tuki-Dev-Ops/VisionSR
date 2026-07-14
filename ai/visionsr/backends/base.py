"""The backend contract.

The engine talks to every execution provider through :class:`RunnableModel`, whose
tensor exchange format is deliberately numpy, not torch: it is the only format
all six required backends (CPU, CUDA, TensorRT, ONNX, DirectML, OpenVINO) can
agree on. Backends convert at their own boundary.

Exchange format
---------------
``infer(batch)`` takes and returns **NHWC float32 in [0, 1]**, RGB — always, for
every model.

That is a promise the *backend* keeps, not the network. Networks disagree about
their value range: SR models want [0,1], anything with a StyleGAN decoder (GFPGAN,
CodeFormer) wants [-1,1]. Each :class:`ModelSpec` declares its ``value_range`` and
the backend converts on the way in and on the way out, so no pipeline above this
layer has to know or care. Pushing that knowledge upward is how you end up
clamping a [-1,1] output to [0,1] and silently deleting half the tonal range.

Alpha is handled above this layer (the pipeline upscales it separately), so a
backend never sees 4 channels.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from ..core.types import Backend, ModelSpec, Precision


class RunnableModel(abc.ABC):
    """A model that has been built, had its weights loaded, and is on a device.

    Instances are stateful and device-bound. They are cached and reused across
    requests; :meth:`unload` frees the device memory.
    """

    def __init__(self, spec: ModelSpec, backend: Backend, precision: Precision) -> None:
        self.spec = spec
        self.backend = backend
        self.precision = precision
        self._loaded = True

    @property
    def scale(self) -> int:
        return self.spec.scale

    @abc.abstractmethod
    def infer(self, batch: np.ndarray) -> np.ndarray:
        """Run the network.

        Args:
            batch: NHWC float32 [0,1] RGB.
        Returns:
            NHWC float32 [0,1] RGB, spatial dims multiplied by ``self.scale``.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Release device memory. The instance is unusable afterwards."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.spec.id!r}, "
            f"backend={self.backend.value}, precision={self.precision.value})"
        )


class BackendAdapter(abc.ABC):
    """Builds :class:`RunnableModel` instances for one execution provider."""

    #: Which :class:`Backend` values this adapter serves.
    handles: tuple[Backend, ...] = ()

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when the runtime for this backend is importable and functional."""

    @abc.abstractmethod
    def load(self, spec: ModelSpec, backend: Backend, precision: Precision) -> RunnableModel:
        """Instantiate the architecture, load weights, place it on the device."""

    def capabilities(self) -> dict[str, Any]:
        """Free-form info for `visionsr doctor`."""
        return {}
