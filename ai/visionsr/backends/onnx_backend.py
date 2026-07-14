"""ONNX Runtime backend — serves ONNX (CPU/CUDA EP) and DirectML.

DirectML matters on Windows: it runs on AMD and Intel GPUs, and on NVIDIA cards
whose driver stack lacks CUDA. It is the only GPU path that works without CUDA.

ONNX graphs are produced on demand from the torch checkpoint the first time a
model is requested on this backend, then cached on disk. That keeps a single
source of truth (the .pth in the registry) instead of shipping two artefacts.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import get_settings
from ..core.errors import BackendUnavailableError, WeightsNotFoundError
from ..core.registry import ONNX_GRAPH
from ..core.types import Backend, ModelSpec, Precision
from .base import BackendAdapter, RunnableModel
from .normalize import from_network, to_network

log = logging.getLogger(__name__)

_PROVIDERS: dict[Backend, list[str]] = {
    Backend.ONNX: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    Backend.DIRECTML: ["DmlExecutionProvider", "CPUExecutionProvider"],
    Backend.OPENVINO: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
}


class OnnxModel(RunnableModel):
    def __init__(
        self,
        spec: ModelSpec,
        backend: Backend,
        precision: Precision,
        session: Any,
    ) -> None:
        super().__init__(spec, backend, precision)
        self._session = session

        node = session.get_inputs()[0]
        self._input_name = node.name

        # Take the dtype from the *graph*, not from the requested precision. There is
        # one graph per model, in fp32 (see _ensure_graph), and feeding it fp16 because
        # the caller asked for fp16 is a type error, not a speed-up. ONNX Runtime and
        # DirectML do their own precision planning internally.
        self._np_dtype = np.float16 if "float16" in node.type else np.float32

    def infer(self, batch: np.ndarray) -> np.ndarray:
        # ONNX graphs here are NCHW; the public contract is NHWC, so transpose at the
        # boundary.
        nchw = to_network(batch, self.spec).transpose(0, 3, 1, 2)
        nchw = np.ascontiguousarray(nchw.astype(self._np_dtype))

        outputs = self._session.run(None, {self._input_name: nchw})

        # Segmentation graphs (U²-Net, IS-Net) emit a stack of side outputs — one per
        # decoder stage, for deep supervision during training. The first is the fused,
        # full-resolution prediction and the only one worth having. Unpacking a single
        # output here, as an SR graph would allow, would raise on every one of them.
        out = outputs[0].transpose(0, 2, 3, 1)

        return from_network(out, self.spec)

    def unload(self) -> None:
        del self._session
        self._session = None
        self._loaded = False


class OnnxAdapter(BackendAdapter):
    handles = (Backend.ONNX, Backend.DIRECTML)

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, spec: ModelSpec, backend: Backend, precision: Precision) -> RunnableModel:
        import onnxruntime as ort

        wanted = _PROVIDERS[backend]
        installed = set(ort.get_available_providers())
        providers = [p for p in wanted if p in installed]
        if not providers:
            raise BackendUnavailableError(
                backend.value,
                f"onnxruntime has {sorted(installed)}, needs one of {wanted}",
            )

        graph = self._ensure_graph(spec, precision)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(graph), options, providers=providers)

        log.info("loaded %s on %s via %s", spec.id, backend.value, session.get_providers()[0])
        return OnnxModel(spec, backend, precision, session)

    def _ensure_graph(self, spec: ModelSpec, precision: Precision) -> Path:
        """Return the .onnx for this spec.

        Two kinds of model live in the registry, and the difference is exactly here:

        * A **torch** model is a ``.pth`` plus a registered architecture. Its ONNX graph
          is derived from that checkpoint on first use and cached, so there is one
          source of truth rather than two artefacts that can drift apart.

        * A **native ONNX** model *is* a graph — there is no architecture to register
          and no torch code to write. That is what lets a published model (the U²-Net
          family, say) join the registry as a YAML entry and nothing else, which is the
          whole promise of a pluggable model registry. It also means such a model
          simply cannot run on the torch backend, which its spec must reflect.
        """
        from .weights import resolve_weight_path

        if spec.architecture == ONNX_GRAPH:
            return resolve_weight_path(spec)

        # One graph per model, in fp32.
        #
        # A second fp16 graph would double the payload the desktop app ships — and buy
        # nothing: ONNX Runtime and DirectML select their own execution precision from
        # the fp32 graph, which is what they are designed to do. Encoding precision in
        # the filename made sense when the graph was a per-request export cache; it does
        # not when the graph is a shipped artefact.
        onnx_dir = get_settings().checkpoint_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        graph = onnx_dir / f"{spec.id}.onnx"

        if graph.exists():
            return graph

        # Exporting needs torch, and the shipped runtime does not have it — it is an
        # ONNX Runtime build, which is the entire point (150MB and no CUDA, against
        # 2.5GB and a driver dependency). So on a machine with no torch, a missing graph
        # is a *packaging* failure, and it deserves to say so rather than surfacing as
        # an ImportError from three frames down.
        if find_spec("torch") is None:
            raise WeightsNotFoundError(
                spec.id,
                f"{graph} (this build has no PyTorch, so the graph cannot be exported "
                f"here — it must be produced by scripts/export_onnx.py and shipped)",
            )

        log.info("no ONNX graph for %s yet — exporting from checkpoint", spec.id)
        from ..exporters.onnx_exporter import export_onnx

        export_onnx(spec, graph, precision=precision)
        return graph

    def capabilities(self) -> dict[str, Any]:
        try:
            import onnxruntime as ort
        except ImportError:
            return {"installed": False}
        return {
            "installed": True,
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
        }
