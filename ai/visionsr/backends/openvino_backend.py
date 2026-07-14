"""OpenVINO backend — Intel CPU/iGPU/NPU.

OpenVINO reads the exported ONNX graph directly, so it reuses the same on-demand
export as the ONNX Runtime backend rather than introducing a third artefact.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.config import get_settings
from ..core.types import Backend, ModelSpec, Precision
from .base import BackendAdapter, RunnableModel
from .normalize import from_network, to_network

log = logging.getLogger(__name__)


class OpenVinoModel(RunnableModel):
    def __init__(
        self, spec: ModelSpec, backend: Backend, precision: Precision, compiled: Any
    ) -> None:
        super().__init__(spec, backend, precision)
        self._compiled = compiled
        self._request = compiled.create_infer_request()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        nchw = to_network(batch, self.spec).transpose(0, 3, 1, 2)

        result = self._request.infer({0: np.ascontiguousarray(nchw.astype(np.float32))})

        # As in the ONNX backend: the first output is the fused prediction; the rest
        # are training-time side outputs.
        out = next(iter(result.values())).transpose(0, 2, 3, 1)

        return from_network(out, self.spec)

    def unload(self) -> None:
        del self._request
        del self._compiled
        self._compiled = None
        self._loaded = False


class OpenVinoAdapter(BackendAdapter):
    handles = (Backend.OPENVINO,)

    def is_available(self) -> bool:
        try:
            import openvino  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, spec: ModelSpec, backend: Backend, precision: Precision) -> RunnableModel:
        import openvino as ov

        from ..core.registry import ONNX_GRAPH
        from ..exporters.onnx_exporter import export_onnx
        from .weights import resolve_weight_path

        if spec.architecture == ONNX_GRAPH:
            # Already a graph — see OnnxAdapter._ensure_graph.
            graph = resolve_weight_path(spec)
        else:
            onnx_dir = get_settings().checkpoint_dir / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            graph = onnx_dir / f"{spec.id}.onnx"
            if not graph.exists():
                export_onnx(spec, graph, precision=Precision.FP32)

        core = ov.Core()
        model = core.read_model(graph)

        # AUTO lets OpenVINO pick GPU/NPU/CPU; hint at throughput since we feed it
        # a tile grid, not a single latency-critical request.
        compiled = core.compile_model(
            model, "AUTO", {"PERFORMANCE_HINT": "THROUGHPUT"}
        )
        log.info("loaded %s on OpenVINO (%s)", spec.id, core.available_devices)
        return OpenVinoModel(spec, backend, precision, compiled)

    def capabilities(self) -> dict[str, Any]:
        try:
            import openvino as ov
        except ImportError:
            return {"installed": False}
        return {"installed": True, "version": ov.__version__, "devices": ov.Core().available_devices}
