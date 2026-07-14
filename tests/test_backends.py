"""Backend equivalence.

The spec requires every model to run on CPU, CUDA, TensorRT, ONNX, DirectML and
OpenVINO. That claim is only worth something if the backends agree: a "pluggable
backend" that quietly returns a different image on each provider is not a feature,
it is a liability, and it is exactly the kind of thing nobody notices until a user
compares a cloud result against their desktop one.

So the test is not "does each backend run" — it is "does each backend produce the
same picture". Which backends exist on the machine varies, so each is skipped
independently rather than failing the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionsr import EnhanceOptions, enhance
from visionsr.core.device import available_backends
from visionsr.core.types import Backend, Precision

pytestmark = pytest.mark.weights

#: The light model: fast enough to run three times in a test, and its ONNX export is
#: 4.6MB rather than 64MB.
MODEL = "realesr-general-x4v3"


def _enhance_on(image: np.ndarray, backend: Backend) -> np.ndarray:
    result = enhance(
        image,
        EnhanceOptions(scale=4, model_id=MODEL, backend=backend, face_restore=False),
    )
    assert result.runs[0].backend == backend
    return result.image


@pytest.fixture(scope="module")
def cpu_reference(portrait_image) -> np.ndarray:
    """CPU fp32 is the reference: no half precision, no vendor kernels, no surprises."""
    return _enhance_on(portrait_image, Backend.CPU)


def test_cpu_backend_runs(cpu_reference, portrait_image):
    assert cpu_reference.shape == (
        portrait_image.shape[0] * 4,
        portrait_image.shape[1] * 4,
        3,
    )


@pytest.mark.parametrize(
    ("backend", "tolerance_mean", "tolerance_max"),
    [
        # fp16 on CUDA: rounding at every layer, so a small mean deviation is
        # expected and healthy. A *large* one means the graph diverged, not that the
        # arithmetic rounded.
        (Backend.CUDA, 0.5, 12),
        # ONNX runs the exported graph in fp32 — it should be all but identical, and
        # a real difference here means the export is lossy or wrong.
        (Backend.ONNX, 0.1, 3),
        (Backend.DIRECTML, 0.5, 12),
        (Backend.OPENVINO, 0.5, 12),
    ],
)
def test_backends_agree_with_cpu(
    portrait_image, cpu_reference, backend, tolerance_mean, tolerance_max
):
    """Same model, same input, different execution provider — same picture."""
    if backend not in available_backends():
        pytest.skip(f"{backend.value} is not available on this machine")

    output = _enhance_on(portrait_image, backend)
    assert output.shape == cpu_reference.shape

    difference = np.abs(output.astype(np.int16) - cpu_reference.astype(np.int16))

    assert difference.mean() < tolerance_mean, (
        f"{backend.value} deviates from the CPU reference by {difference.mean():.3f}/255 "
        f"on average — that is a different image, not a rounding difference."
    )
    assert difference.max() < tolerance_max, (
        f"{backend.value} has a worst-case deviation of {difference.max()}/255"
    )


def test_onnx_graph_is_exported_on_demand(registry, tmp_path, monkeypatch):
    """A missing ONNX graph must be generated from the checkpoint, not error.

    The registry's source of truth is a .pth. Shipping a second artefact per model
    per precision would double the download and guarantee they drift apart, so the
    graph is derived on first use and cached.
    """
    if Backend.ONNX not in available_backends():
        pytest.skip("onnxruntime is not installed")

    from visionsr.exporters.onnx_exporter import export_onnx

    spec = registry.get(MODEL)
    destination = tmp_path / f"{spec.id}.onnx"

    export_onnx(spec, destination, precision=Precision.FP32)

    assert destination.exists()
    assert destination.stat().st_size > 100_000, "the exported graph is suspiciously small"

    # And it must accept the varying tile shapes the tiler produces — a graph frozen
    # at one spatial size would reject every edge tile.
    import onnxruntime as ort

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    for size in (64, 96, 128):
        (output,) = session.run(None, {name: np.random.rand(1, 3, size, size).astype(np.float32)})
        assert output.shape == (1, 3, size * 4, size * 4)


def test_an_unsupported_backend_is_refused_clearly(registry):
    """Asking for a backend the model does not declare must say so, not fall back."""
    from visionsr.backends.factory import ModelCache
    from visionsr.core.errors import BackendUnavailableError, UnsupportedBackendError

    spec = registry.get(MODEL).model_copy(update={"backends": [Backend.CPU]})

    with pytest.raises((UnsupportedBackendError, BackendUnavailableError)):
        ModelCache(max_size=1).get(spec, backend=Backend.CUDA)


def test_unsupported_precision_falls_back_rather_than_failing(registry):
    """A model that cannot do fp16 must run in fp32, not refuse to run.

    This is the mechanism that keeps GFPGAN off fp16 (where it silently collapses)
    even when the CUDA default asks for it.
    """
    from visionsr.backends.factory import ModelCache

    gfpgan = registry.get("gfpgan-v1.4")
    assert Precision.FP16 not in gfpgan.precisions

    model = ModelCache(max_size=1).get(gfpgan, precision=Precision.FP16)
    assert model.precision == Precision.FP32
