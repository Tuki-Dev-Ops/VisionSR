"""PyTorch backend — serves CPU, CUDA, MPS and (via torch-tensorrt) TensorRT."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..core.device import torch_device
from ..core.registry import architectures
from ..core.types import Backend, ModelSpec, Precision
from .base import BackendAdapter, RunnableModel
from .normalize import from_network, to_network
from .weights import load_state_dict

log = logging.getLogger(__name__)

_TORCH_DTYPE = {
    Precision.FP32: "float32",
    Precision.FP16: "float16",
    Precision.BF16: "bfloat16",
}


class TorchModel(RunnableModel):
    """A torch nn.Module placed on a device, wrapped in the numpy contract."""

    def __init__(
        self,
        spec: ModelSpec,
        backend: Backend,
        precision: Precision,
        module: Any,
        device: str,
    ) -> None:
        super().__init__(spec, backend, precision)
        self._module = module
        self._device = device

    def infer(self, batch: np.ndarray) -> np.ndarray:
        import torch

        if batch.ndim != 4:
            raise ValueError(f"Expected NHWC batch, got shape {batch.shape}.")

        dtype = getattr(torch, _TORCH_DTYPE[self.precision])
        normalised = to_network(batch, self.spec)

        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(normalised)).permute(0, 3, 1, 2)
            tensor = tensor.to(device=self._device, dtype=dtype, non_blocking=True)

            out = self._module(tensor)
            out = out.float().permute(0, 2, 3, 1).cpu().numpy()

        return from_network(out, self.spec)

    def unload(self) -> None:
        import torch

        del self._module
        self._module = None
        self._loaded = False
        if self._device.startswith("cuda"):
            torch.cuda.empty_cache()

    @contextlib.contextmanager
    def track_peak_vram(self) -> Iterator[dict[str, float]]:
        """Measure peak allocation over a block, for the ModelRun record."""
        stats: dict[str, float] = {"peak_mb": 0.0}
        if not self._device.startswith("cuda"):
            yield stats
            return

        import torch

        torch.cuda.reset_peak_memory_stats()
        try:
            yield stats
        finally:
            stats["peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2


class TorchAdapter(BackendAdapter):
    handles = (Backend.CPU, Backend.CUDA, Backend.MPS, Backend.TENSORRT)

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, spec: ModelSpec, backend: Backend, precision: Precision) -> RunnableModel:
        import torch

        device = torch_device(backend)

        # fp16 on CPU is emulated and ~10x slower than fp32 — silently correcting
        # is friendlier than failing, but it must be visible in the logs.
        if device == "cpu" and precision in (Precision.FP16, Precision.BF16):
            log.warning("%s is not usable on CPU; falling back to fp32.", precision.value)
            precision = Precision.FP32

        module = architectures.build(spec.architecture, **spec.params)

        if spec.requires_weights:
            state = load_state_dict(spec)
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing:
                log.warning(
                    "%s: %d weight(s) missing from checkpoint (e.g. %s)",
                    spec.id,
                    len(missing),
                    missing[:3],
                )
            if unexpected:
                log.debug("%s: %d unused key(s) in checkpoint", spec.id, len(unexpected))

        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

        dtype = getattr(torch, _TORCH_DTYPE[precision])
        module = module.to(device=device, dtype=dtype)

        if backend == Backend.TENSORRT:
            module = self._compile_tensorrt(module, spec, dtype)

        log.info(
            "loaded %s on %s/%s (%.1fM params)",
            spec.id,
            backend.value,
            precision.value,
            sum(p.numel() for p in module.parameters()) / 1e6,
        )
        return TorchModel(spec, backend, precision, module, device)

    def _compile_tensorrt(self, module: Any, spec: ModelSpec, dtype: Any) -> Any:
        """AOT-compile to a TensorRT engine.

        Shapes are dynamic across the tile grid's edge tiles, so the engine is
        built with a min/opt/max range rather than a single static shape.
        """
        import torch
        import torch_tensorrt

        tile = spec.tile_size
        return torch_tensorrt.compile(
            module,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 3, 64, 64),
                    opt_shape=(1, 3, tile, tile),
                    max_shape=(1, 3, tile * 2, tile * 2),
                    dtype=dtype,
                )
            ],
            enabled_precisions={dtype},
            truncate_long_and_double=True,
            device=torch.device("cuda"),
        )

    def capabilities(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError:
            return {"installed": False}

        caps: dict[str, Any] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            caps["cuda_version"] = torch.version.cuda
            caps["device"] = torch.cuda.get_device_name(0)
        return caps
