"""Torch -> ONNX export.

The registry's source of truth is a .pth. Every non-torch backend (ONNX Runtime,
DirectML, OpenVINO) needs an ONNX graph, so rather than shipping a second artefact
per model, the graph is generated from the checkpoint on first use and cached.

Dynamic H/W axes are essential here: the tiler produces edge tiles that are
smaller than the interior ones, and a graph frozen at one spatial size would
reject them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..backends.weights import load_state_dict
from ..core.registry import architectures
from ..core.types import ModelSpec, Precision

log = logging.getLogger(__name__)


def export_onnx(
    spec: ModelSpec,
    output: Path,
    precision: Precision = Precision.FP32,
    opset: int = 17,
    sample_size: int = 64,
) -> Path:
    """Export a registered model to ONNX.

    Args:
        spec: what to export. Its architecture must be registered and its weights
            present on disk.
        output: destination .onnx path.
        precision: fp16 halves the file and runs faster, but export it from an fp32
            graph and let the runtime cast — exporting *in* fp16 makes the tracer
            fold constants at reduced precision and can degrade output visibly.
        opset: 17 covers everything these architectures use.
        sample_size: spatial size of the tracing input. Irrelevant to the result
            because H and W are marked dynamic, but it must be a legal input size.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    module = architectures.build(spec.architecture, **spec.params)
    if spec.requires_weights:
        module.load_state_dict(load_state_dict(spec), strict=False)
    module.eval()

    if spec.dynamic_shape:
        # The tracer must see a size the architecture actually accepts: RRDBNet at
        # scale 2 pixel-unshuffles by 2, at scale 1 by 4.
        multiple = max(spec.window_multiple, 4)
        size = max(sample_size, multiple)
        size = (size // multiple) * multiple

        # H and W dynamic, because the tiler's edge tiles are smaller than its interior
        # ones and a graph frozen at one size would reject every last row and column.
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    else:
        # Fully static — not even the batch axis.
        #
        # GFPGAN's flatten-into-Linear already pins H and W to the 512x512 the weights
        # were trained on. But its StyleGAN2 decoder goes further: ModulatedConv2d
        # applies a different kernel per sample by folding the batch into a *grouped*
        # convolution (`groups=b`). ONNX's Conv requires `group` to be a static integer,
        # so the moment the batch axis is symbolic the tracer has nothing to put there
        # and the export dies mid-graph.
        #
        # That is fine, because it is also how the model is used: the face pipeline
        # aligns and restores one face at a time. Batch 1, always.
        size = spec.tile_size
        dynamic_axes = None

    sample = torch.randn(1, 3, size, size, dtype=torch.float32)

    log.info(
        "exporting %s -> %s (opset %d, %s)",
        spec.id,
        output.name,
        opset,
        "dynamic HxW" if spec.dynamic_shape else f"static 1x3x{size}x{size}",
    )

    with torch.inference_mode():
        torch.onnx.export(
            module,
            (sample,),
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    if precision == Precision.FP16:
        convert_to_fp16(output)

    log.info("exported %s (%.1f MB)", output.name, output.stat().st_size / 1024**2)
    return output


def convert_to_fp16(path: Path) -> None:
    """Cast a graph's weights to fp16 in place.

    A post-pass on the finished graph, never a trace of an fp16 module: constant folding
    must happen at full precision or the folded constants are themselves degraded.

    ``keep_io_types=True`` leaves the graph's inputs and outputs fp32, so the runtime
    casts at the boundary and the backend does not have to know or care. Only the weights
    and the intermediate activations are halved — which is where the size and the speed
    are.
    """
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(path))
    onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), str(path))

    log.info("converted %s to fp16 (%.0f MB)", path.name, path.stat().st_size / 1024**2)


def verify_onnx(spec: ModelSpec, graph: Path, tolerance: float = 2e-2) -> bool:
    """Check the exported graph against the torch model on random input.

    Worth running in CI: a silently-wrong export produces plausible-looking but
    degraded output, which is the hardest kind of bug to notice.
    """
    import numpy as np
    import onnxruntime as ort

    module = architectures.build(spec.architecture, **spec.params)
    if spec.requires_weights:
        module.load_state_dict(load_state_dict(spec), strict=False)
    module.eval()

    sample = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = module(sample).numpy()

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {session.get_inputs()[0].name: sample.numpy()})

    deviation = float(np.abs(expected - actual).max())
    ok = deviation < tolerance

    log.log(
        logging.INFO if ok else logging.ERROR,
        "onnx verification for %s: max deviation %.4f (tolerance %.4f) - %s",
        spec.id,
        deviation,
        tolerance,
        "OK" if ok else "FAILED",
    )
    return ok
