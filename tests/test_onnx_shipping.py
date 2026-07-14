"""Guards on the shipped ONNX graphs.

The desktop app runs ONNX Runtime on DirectML and contains no PyTorch. That decision
takes the installer from ~2.5GB to ~250MB and makes it work on any DX12 GPU rather than
only on NVIDIA cards with a recent driver — but it moves a whole class of failure from
"impossible" to "silent".

An ONNX graph can be wrong in ways nothing catches. It builds. The session loads. The
output is the right shape. And the pixels are degraded, or the model has collapsed
entirely. The tests here exist because two such failures were caught within an hour of
each other, and only one of them by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.weights


def _providers():
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    return [
        p
        for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if p in available
    ]


def test_the_shipped_graph_exists_for_every_torch_model(registry):
    """The frozen app cannot export a graph — it has no torch. It must find one."""
    from visionsr.backends.weights import onnx_graph_path
    from visionsr.core.registry import ONNX_GRAPH

    missing = [
        spec.id
        for spec in registry.all()
        if spec.architecture != ONNX_GRAPH and not onnx_graph_path(spec).exists()
    ]

    assert not missing, (
        f"{missing} have no exported ONNX graph. The packaged app has no PyTorch and "
        "cannot make one — run scripts/export_onnx.py."
    )


@pytest.mark.parametrize("model_id", ["realesrgan-x4plus", "gfpgan-v1.4"])
def test_the_graph_runs_on_every_provider_that_ships(registry, model_id):
    """A graph that does not run on DirectML is not shippable, however it behaves on CPU.

    This is the failure that a CPU-only check cannot see. The fp16 SR graphs load and run
    perfectly on ONNX Runtime's CPU provider, and fail outright on DirectML — the very
    provider the desktop app uses — on their final Conv, with an HRESULT the Python
    binding cannot even decode into a message.
    """
    import onnxruntime as ort

    from visionsr.backends.weights import onnx_graph_path

    spec = registry.get(model_id)
    graph = onnx_graph_path(spec)

    size = spec.tile_size if not spec.dynamic_shape else 64
    sample = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)

    for provider in _providers():
        session = ort.InferenceSession(str(graph), providers=[provider])
        (out,) = session.run(None, {session.get_inputs()[0].name: sample})

        assert np.isfinite(out).all(), f"{model_id} produced non-finite values on {provider}"


def test_gfpgan_does_not_collapse_on_the_provider_it_ships_on(registry, aligned_face):
    """The one that nearly shipped.

    GFPGAN's StyleGAN2 decoder cannot survive half precision — its weight demodulation
    plus a per-layer sqrt(2) gain exceeds fp16's dynamic range. In torch that was already
    known, and the model is pinned to fp32.

    The ONNX build reintroduced it by a different door. The fp16 graph measured 1.5/255
    against torch and passed the export gate — because the gate ran on the CPU provider,
    which quietly computes much of an fp16 graph in fp32. On DirectML, which is what the
    app actually runs, the identical graph collapses exactly as torch's fp16 did: output
    std 0.343 -> 0.123, 179/255 from the truth. No error. No NaN. Just a dark smear where
    the face was.

    The reference is torch on the *same* input, not an absolute threshold: what the output
    of a healthy decoder looks like depends entirely on what went in, and a test that
    forgets that fails on noise while passing on a face.
    """
    import onnxruntime as ort
    import torch

    from visionsr.backends.weights import load_state_dict, onnx_graph_path
    from visionsr.core.registry import architectures
    from visionsr.preprocessing.io import to_float

    spec = registry.get("gfpgan-v1.4")
    assert not spec.onnx_fp16, (
        "GFPGAN is marked as fp16-safe. It is not: on DirectML the fp16 graph collapses "
        "silently. See ai/configs/models.yaml."
    )

    # The model's real input: an aligned face in [-1, 1].
    sample = ((to_float(aligned_face) - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)

    module = architectures.build(spec.architecture, **spec.params)
    module.load_state_dict(load_state_dict(spec), strict=False)
    module.eval()

    with torch.inference_mode():
        truth = module(torch.from_numpy(sample)).numpy()

    graph = onnx_graph_path(spec)

    for provider in _providers():
        session = ort.InferenceSession(str(graph), providers=[provider])
        (out,) = session.run(None, {session.get_inputs()[0].name: sample})

        # A collapsed decoder produces a near-constant field. That is the one thing it
        # cannot fake, and it is what the fp16 failure looked like: the spread fell to a
        # third of the reference.
        ratio = float(out.std() / truth.std())
        assert ratio > 0.85, (
            f"on {provider}, GFPGAN's output spread is {ratio:.2f}x the torch reference "
            f"(std {out.std():.3f} vs {truth.std():.3f}). The fp16 collapse looked like "
            f"0.36x. The decoder has degenerated."
        )

        # And it must actually match, not merely have the right shape of histogram.
        deviation = float(np.abs(truth - out).max()) / 2  # the model works in [-1,1]
        assert deviation < 2 / 255, (
            f"on {provider}, the graph is {deviation * 255:.1f}/255 from torch"
        )


def test_the_engine_runs_end_to_end_without_torch(portrait_image, monkeypatch):
    """The shipped configuration, exercised: ONNX Runtime only.

    Not a simulation of the packaged app — the same engine code, with torch made
    invisible, running the same graphs on the same provider. If this passes and the
    packaged app fails, the difference is packaging; if this fails, the engine was never
    going to work in the shipped build.
    """
    import importlib.util

    from visionsr.core.types import Backend

    real_find_spec = importlib.util.find_spec

    def without_torch(name: str, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            return None
        return real_find_spec(name, *args, **kwargs)

    from visionsr.core import device

    monkeypatch.setattr(device, "find_spec", without_torch)
    device.available_backends.cache_clear()

    try:
        backends = device.available_backends()

        assert Backend.CUDA not in backends, "cuda is a torch backend; it must vanish with torch"
        assert Backend.CPU not in backends, "cpu is the *torch* CPU backend, not ONNX's"
        assert backends, "no backend survives without torch — the shipped app cannot run"

        from visionsr import EnhanceOptions, enhance

        result = enhance(
            portrait_image,
            EnhanceOptions(scale=2, face_restore=True, remove_background=True),
        )

        assert result.image.shape[2] == 4, "the cutout did not survive"
        assert len(result.runs) == 3, f"expected SR + face + matting, got {result.runs}"
        assert all(run.backend in backends for run in result.runs)

    finally:
        device.available_backends.cache_clear()
