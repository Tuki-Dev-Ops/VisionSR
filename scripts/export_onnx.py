#!/usr/bin/env python
"""Export every torch model in the registry to ONNX, and prove each one is right.

This is a **build step for shipping**, and it exists because of a decision worth
stating plainly: the desktop app does not ship PyTorch.

Freezing torch with CUDA produces a ~2.5GB binary, and nobody ships a desktop image
tool that way. ONNX Runtime with the DirectML provider is ~150MB, runs on any DX12
GPU — AMD, Intel, and NVIDIA cards whose driver is too old for the CUDA toolkit the
build was made against — and needs no CUDA installation at all. So torch is a
*development* dependency (training, research, and this export), and the shipped
runtime is ONNX.

The price of that is this script. An ONNX export can be silently wrong: the graph
builds, the session loads, inference returns an image of the right size, and the
pixels are subtly degraded — the hardest kind of bug to notice and the easiest to
ship. So every export is checked against the torch model it came from, on real
input, and a deviation above tolerance fails the build.

    python scripts/export_onnx.py            # export + verify everything
    python scripts/export_onnx.py --model gfpgan-v1.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from visionsr.core.config import get_settings  # noqa: E402
from visionsr.core.registry import ONNX_GRAPH, architectures, models  # noqa: E402
from visionsr.core.types import ModelSpec, Precision  # noqa: E402
from visionsr.ui import FAIL, OK, SKIP, console  # noqa: E402

#: Maximum permitted deviation between the torch model and its ONNX graph, on a [0,1]
#: scale. 2/255 is below the quantisation step of the 8-bit image either would be
#: written to, so anything under it is provably invisible. Anything over it means the
#: export changed the model.
TOLERANCE = 2.0 / 255.0


def verify(spec: ModelSpec, graph: Path) -> tuple[float, str]:
    """Check the graph against its torch model. Return (worst deviation, worst provider).

    **On every execution provider the app might actually use**, and that is not a detail.

    Verifying on the CPU provider alone very nearly shipped a catastrophically broken
    GFPGAN. ONNX Runtime's CPU kernels quietly compute much of an fp16 graph in fp32, so
    the fp16 GFPGAN measured 1.5/255 against torch and sailed through the gate. Run the
    same graph on DirectML — which is what the desktop app does — and it collapses
    exactly as torch's fp16 did: output std 0.343 -> 0.123, a near-constant dark smear,
    179/255 from the truth. The graph was fine. The gate was measuring the wrong machine.

    Real input, not zeros: a zero tensor propagates through a broken graph as
    convincingly as through a correct one.
    """
    import onnxruntime as ort
    import torch

    from visionsr.backends.weights import load_state_dict

    module = architectures.build(spec.architecture, **spec.params)
    module.load_state_dict(load_state_dict(spec), strict=False)
    module.eval()

    size = spec.tile_size if not spec.dynamic_shape else 64
    sample = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)

    with torch.inference_mode():
        expected = module(torch.from_numpy(sample)).numpy()

    available = set(ort.get_available_providers())
    # CPU is the floor and always present; the GPU providers are the ones that ship.
    providers = [
        p
        for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if p in available
    ]

    worst, culprit = 0.0, providers[0]

    for provider in providers:
        try:
            session = ort.InferenceSession(str(graph), providers=[provider])
            (actual,) = session.run(None, {session.get_inputs()[0].name: sample})
        except Exception as exc:
            # ONNX Runtime's native error text comes back in the OS's ANSI codepage, so
            # on a non-English Windows the failure surfaces as a UnicodeDecodeError from
            # inside the binding — which says nothing at all about what went wrong.
            # Name the provider, because that is the fact that matters.
            raise RuntimeError(
                f"the graph does not run on {provider} ({type(exc).__name__})"
            ) from exc

        if expected.shape != actual.shape:
            raise ValueError(
                f"shape mismatch on {provider}: torch {expected.shape} vs onnx {actual.shape}"
            )

        deviation = float(np.abs(expected - actual).max())
        if deviation > worst:
            worst, culprit = deviation, provider

    return worst, culprit


def verify_dynamic_shapes(graph: Path, scale: int) -> None:
    """A fully-convolutional graph must accept the tiler's varying tile sizes.

    The tiler's edge tiles are smaller than its interior ones. A graph accidentally
    frozen at one spatial size passes every other check here and then raises on the
    last column of every image.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    for size in (64, 96, 128):
        sample = np.random.default_rng(1).random((1, 3, size, size), dtype=np.float32)
        (out,) = session.run(None, {name: sample})

        expected = (1, 3, size * scale, size * scale)
        if out.shape != expected:
            raise ValueError(f"at {size}px the graph returned {out.shape}, expected {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", help="Export one model. Repeatable.")
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Keep every graph at fp32, even where fp16 is provably lossless.",
    )
    args = parser.parse_args()

    import visionsr.models  # noqa: F401  — registers the architectures
    from visionsr.exporters.onnx_exporter import convert_to_fp16, export_onnx

    settings = get_settings()
    models.load_dir(settings.config_dir, replace=True)

    onnx_dir = settings.checkpoint_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    wanted = [models.get(m) for m in args.model] if args.model else models.all()
    failed: list[str] = []

    for spec in wanted:
        if spec.architecture == ONNX_GRAPH:
            console.print(f"[dim]{SKIP} {spec.id} — already an ONNX graph[/dim]")
            continue

        graph = onnx_dir / f"{spec.id}.onnx"

        try:
            export_onnx(spec, graph, precision=Precision.FP32)

            deviation, provider = verify(spec, graph)
            if spec.dynamic_shape:
                verify_dynamic_shapes(graph, spec.scale)

            if deviation > TOLERANCE:
                failed.append(spec.id)
                console.print(
                    f"[red]{FAIL} {spec.id} — even the fp32 export deviates from torch by "
                    f"{deviation * 255:.2f}/255 on {provider} "
                    f"(limit {TOLERANCE * 255:.0f}/255).[/red]"
                )
                continue

            precision = "fp32"
            size_mb = graph.stat().st_size / 1024**2

            # fp16 halves the graph, and the graphs are the bulk of the installer. So try
            # it — and then *measure*, on every provider that ships. GFPGAN is exactly why:
            # its fp16 graph is fine on the CPU provider and collapses on DirectML, and
            # the collapse is silent. If the halved graph cannot hold the model everywhere
            # it will run, keep the full one and pay the megabytes.
            if spec.onnx_fp16 and not args.no_fp16:
                convert_to_fp16(graph)
                rejection: str | None = None

                try:
                    half, half_provider = verify(spec, graph)
                    if half > TOLERANCE:
                        rejection = (
                            f"deviates by {half * 255:.1f}/255 on "
                            f"{half_provider.replace('ExecutionProvider', '')}"
                        )
                except RuntimeError as exc:
                    # An fp16 graph that will not even *run* on a provider the app ships
                    # on is not a candidate, whatever it does elsewhere. Measured on this
                    # machine: DirectML fails outright on the final Conv of every fp16 SR
                    # graph (HRESULT 0x8007023E). Letting that propagate would fail the
                    # whole build over an optimisation that is meant to be optional.
                    rejection = str(exc)

                if rejection is None:
                    precision, deviation, provider = "fp16", half, half_provider
                    size_mb = graph.stat().st_size / 1024**2
                else:
                    console.print(f"[yellow]  {spec.id}: fp16 {rejection} — using fp32.[/yellow]")
                    export_onnx(spec, graph, precision=Precision.FP32)
                    deviation, provider = verify(spec, graph)
                    size_mb = graph.stat().st_size / 1024**2

            console.print(
                f"[green]{OK}[/green] {spec.id:<22} {size_mb:>5.0f}MB  {precision}  "
                f"worst deviation {deviation * 255:.3f}/255 on {provider.replace('ExecutionProvider', '')}"
            )

        except Exception as exc:
            failed.append(spec.id)
            console.print(f"[red]{FAIL} {spec.id} — {type(exc).__name__}: {exc}[/red]")

    if failed:
        console.print(f"\n[red]{FAIL} {len(failed)} export(s) failed: {', '.join(failed)}[/red]")
        return 1

    total = sum(f.stat().st_size for f in onnx_dir.glob("*.onnx")) / 1024**2
    console.print(
        f"\n[green]{OK} Every graph matches its torch model to within "
        f"{TOLERANCE * 255:.0f}/255.[/green]  [dim]{total:.0f}MB in {onnx_dir}[/dim]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
