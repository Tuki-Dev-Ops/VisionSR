"""Backend/device discovery.

Imports of torch, onnxruntime and openvino are deferred: the engine must be
importable (and the registry inspectable) on a machine where none of them are
installed, which is what makes the API layer testable without a GPU.
"""

from __future__ import annotations

import functools
import logging
from importlib.util import find_spec

from .errors import BackendUnavailableError
from .types import Backend, DeviceInfo, Precision

log = logging.getLogger(__name__)

# Preference order when the caller does not pin a backend. First available wins.
_PREFERENCE: tuple[Backend, ...] = (
    Backend.TENSORRT,
    Backend.CUDA,
    Backend.DIRECTML,
    Backend.MPS,
    Backend.OPENVINO,
    Backend.ONNX,
    Backend.CPU,
)


def _installed(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@functools.cache
def available_backends() -> tuple[Backend, ...]:
    """Backends actually usable on this machine, in preference order.

    Cached: probing CUDA initialises a context, which is slow and we only want
    it once per process.
    """
    found: list[Backend] = []

    if _installed("torch"):
        import torch

        # Backend.CPU means "torch on the CPU" — it is served by the torch adapter and
        # nothing else. Listing it unconditionally would advertise a backend that
        # cannot load a model, which matters because the shipped desktop runtime has no
        # torch at all: it runs ONNX Runtime with the DirectML provider, and its CPU
        # path is the ONNX CPU execution provider, not this one.
        found.append(Backend.CPU)

        if torch.cuda.is_available():
            found.append(Backend.CUDA)
            if _installed("tensorrt") and _installed("torch_tensorrt"):
                found.append(Backend.TENSORRT)
        if torch.backends.mps.is_available():
            found.append(Backend.MPS)

    if _installed("onnxruntime"):
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        if providers & {"CUDAExecutionProvider", "CPUExecutionProvider"}:
            found.append(Backend.ONNX)
        if "DmlExecutionProvider" in providers:
            found.append(Backend.DIRECTML)

    if _installed("openvino"):
        found.append(Backend.OPENVINO)

    return tuple(b for b in _PREFERENCE if b in found)


def best_backend(supported: list[Backend] | None = None) -> Backend:
    """Highest-preference backend that is both available and supported by the model."""
    usable = available_backends()
    if supported:
        allowed = set(supported)
        usable = tuple(b for b in usable if b in allowed)
    if not usable:
        raise BackendUnavailableError(
            "any",
            f"model supports {[b.value for b in supported or []]}, "
            f"machine has {[b.value for b in available_backends()]}",
        )
    return usable[0]


def resolve_backend(requested: Backend | None, supported: list[Backend]) -> Backend:
    """Validate an explicit backend request, or pick the best one."""
    if requested is None:
        return best_backend(supported)
    if requested not in available_backends():
        raise BackendUnavailableError(requested.value)
    return requested


def default_precision(backend: Backend) -> Precision:
    """fp16 pays for itself on GPU; on CPU it is emulated and slower than fp32."""
    if backend in (Backend.CUDA, Backend.TENSORRT, Backend.DIRECTML):
        return Precision.FP16
    return Precision.FP32


def device_info(backend: Backend) -> DeviceInfo:
    """Name and live VRAM figures for the active device."""
    if backend in (Backend.CUDA, Backend.TENSORRT):
        import torch

        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        return DeviceInfo(
            backend=backend,
            name=props.name,
            total_vram_mb=int(total_b / 1024**2),
            free_vram_mb=int(free_b / 1024**2),
            compute_capability=(props.major, props.minor),
        )

    if backend == Backend.MPS:
        return DeviceInfo(backend=backend, name="Apple Silicon GPU")

    if backend == Backend.DIRECTML:
        return DeviceInfo(backend=backend, name="DirectML device")

    return DeviceInfo(backend=backend, name=_cpu_name())


def free_vram_mb(backend: Backend) -> int | None:
    """VRAM actually available to us right now, or None if unknowable.

    Not the same as what the driver reports free. PyTorch's caching allocator holds
    on to blocks it has finished with rather than returning them to the driver, so
    once a model has run, ``mem_get_info`` can report a handful of megabytes free
    while hundreds of megabytes sit reusable inside torch's own pool. Trusting the
    driver's figure made the tiler drop to its minimum tile size after the first
    inference and stay there — correct output, several times slower than necessary.

    Available = what the driver has left + what torch is holding but not using.
    """
    if backend not in (Backend.CUDA, Backend.TENSORRT):
        return None

    try:
        import torch

        device = torch.cuda.current_device()
        driver_free, _ = torch.cuda.mem_get_info(device)

        # Reserved by torch from the driver, minus what is live in tensors: this is
        # cached and immediately reusable without touching the driver.
        cached_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)

        return int((driver_free + cached_free) / 1024**2)
    except Exception as exc:  # pragma: no cover - driver-dependent
        log.debug("could not read free VRAM: %s", exc)
        return None


def torch_device(backend: Backend) -> str:
    """Map a Backend to a torch device string."""
    return {
        Backend.CUDA: "cuda",
        Backend.TENSORRT: "cuda",
        Backend.MPS: "mps",
    }.get(backend, "cpu")


def _cpu_name() -> str:
    import platform

    return platform.processor() or platform.machine() or "CPU"


def summary() -> str:
    """One-line capability report, used by `visionsr doctor` and startup logs."""
    parts = []
    for backend in available_backends():
        info = device_info(backend)
        if info.total_vram_mb:
            parts.append(f"{backend.value}:{info.name} ({info.total_vram_mb}MB)")
        else:
            parts.append(f"{backend.value}:{info.name}")
    return " | ".join(parts)
