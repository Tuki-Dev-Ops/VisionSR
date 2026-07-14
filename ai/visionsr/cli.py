"""Command line interface.

    visionsr enhance photo.jpg -o out.png --scale 4
    visionsr enhance ./album --scale 2 --recursive     # batch
    visionsr analyze photo.jpg                          # what would it do, and why
    visionsr models                                     # what is registered/installed
    visionsr doctor                                     # what hardware can it use
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .core.errors import VisionSRError
from .core.types import Backend, EnhanceOptions, Precision
from .ui import FAIL, OK, bar, console

app = typer.Typer(
    name="visionsr",
    help="Enterprise AI image super resolution and restoration.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Errors only.")] = False,
) -> None:
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=verbose)],
    )


@app.command()
def enhance(
    source: Annotated[Path, typer.Argument(help="Image file or a directory of them.")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file or directory.")] = None,
    scale: Annotated[int, typer.Option("--scale", "-s", help="2, 4, 8 or 16.")] = 4,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Pin a model; default is auto-select.")] = None,
    face_restore: Annotated[bool | None, typer.Option("--face/--no-face", help="Default: on when faces are found.")] = None,
    face_weight: Annotated[float, typer.Option("--face-weight", help="0 = keep original face, 1 = full restoration.")] = 0.5,
    remove_bg: Annotated[bool, typer.Option("--remove-bg", help="Cut the subject out. Writes RGBA, so PNG or WebP only.")] = False,
    bg_model: Annotated[str | None, typer.Option("--bg-model", help="Pin a segmentation model; default is auto.")] = None,
    bg_feather: Annotated[int, typer.Option("--bg-feather", help="Extra alpha edge blur, px. 0 keeps the model's own softness.")] = 0,
    backend: Annotated[Backend | None, typer.Option("--backend", "-b", help="Default: best available.")] = None,
    precision: Annotated[Precision | None, typer.Option("--precision", "-p")] = None,
    tile: Annotated[int | None, typer.Option("--tile", help="Tile size px. Default: from free VRAM.")] = None,
    sharpen: Annotated[float, typer.Option("--sharpen", help="0..1.5 unsharp amount.")] = 0.0,
    quality: Annotated[int, typer.Option("--quality", help="JPEG/WebP quality.")] = 95,
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Recurse into subdirectories.")] = False,
) -> None:
    """Upscale and restore an image, or a whole folder of them."""
    from .inference.engine import get_engine
    from .preprocessing.io import save_image

    options = EnhanceOptions(
        scale=scale,  # type: ignore[arg-type]
        model_id=model,
        face_restore=face_restore,
        face_restore_weight=face_weight,
        remove_background=remove_bg,
        background_model_id=bg_model,
        background_feather=bg_feather,
        backend=backend,
        precision=precision,
        tile_size=tile,
        sharpen=sharpen,
        output_quality=quality,
    )

    sources = _collect_sources(source, recursive)
    if not sources:
        console.print(f"[red]No supported images found at {source}[/red]")
        raise typer.Exit(1)

    batch = len(sources) > 1
    if batch and output is not None and output.suffix:
        console.print("[red]--output must be a directory when enhancing multiple files.[/red]")
        raise typer.Exit(1)

    engine = get_engine()
    failures = 0

    for index, path in enumerate(sources, start=1):
        destination = _destination(path, output, source, batch, scale)
        label = f"[{index}/{len(sources)}] {path.name}" if batch else path.name

        try:
            with _progress_bar(label) as (progress, task):
                # Bind `label` and `task` as defaults rather than closing over them:
                # they are loop variables, and a late-binding closure would report
                # every file in a batch under the name of the last one.
                def on_progress(
                    stage: str,
                    fraction: float,
                    progress=progress,
                    task=task,
                    label=label,
                ) -> None:
                    progress.update(
                        task, description=f"{label} — {stage}", completed=fraction * 100
                    )

                result = engine.enhance_file(path, options, progress=on_progress)

            save_image(
                result.image,
                destination,
                quality=quality,
                metadata={"exif": None},
                preserve_exif=False,
            )
            _print_result(result, destination)

        except VisionSRError as exc:
            failures += 1
            console.print(f"[red]{path.name}: {exc}[/red]")
        except Exception as exc:
            failures += 1
            console.print(f"[red]{path.name}: unexpected error: {exc}[/red]")
            if logging.getLogger().level <= logging.DEBUG:
                console.print_exception()

    if failures:
        console.print(f"\n[yellow]{failures} of {len(sources)} failed.[/yellow]")
        raise typer.Exit(1)


@app.command()
def analyze(
    source: Annotated[Path, typer.Argument(help="Image to profile.")],
    scale: Annotated[int, typer.Option("--scale", "-s")] = 4,
) -> None:
    """Show what the engine sees in an image, and which model it would pick."""
    from .inference.engine import get_engine
    from .preprocessing.io import load_image

    image, metadata = load_image(source)
    engine = get_engine()

    from .analysis.analyzer import analyze as run_analysis

    analysis = run_analysis(image, metadata)
    plan = engine.plan(image, EnhanceOptions(scale=scale))  # type: ignore[arg-type]

    table = Table(title=source.name, show_header=False, box=None)
    table.add_column(style="cyan", justify="right")
    table.add_column()

    table.add_row("dimensions", f"{analysis.width} x {analysis.height} ({analysis.megapixels:.1f} MP)")
    table.add_row("alpha", "yes" if analysis.has_alpha else "no")
    table.add_row("content", f"{analysis.content_type.value} ({analysis.content_confidence:.0%} confident)")
    table.add_row("faces", str(len(analysis.faces)))
    table.add_row("", "")
    table.add_row("noise", bar(analysis.noise_level))
    table.add_row("blur", bar(analysis.blur_level))
    table.add_row("jpeg artefacts", bar(analysis.compression_level))
    table.add_row("quality score", bar(analysis.quality_score, good_is_high=True))
    table.add_row("", "")
    table.add_row("model", plan.sr_model.id + (f"  x{plan.passes} passes" if plan.passes > 1 else ""))
    table.add_row("face model", plan.face_model.id if plan.face_model else "—")
    table.add_row("output", f"{analysis.width * scale} x {analysis.height * scale}")

    console.print(table)
    if analysis.exif:
        console.print("\n[dim]EXIF:[/dim] " + ", ".join(f"{k}={v}" for k, v in analysis.exif.items()))


@app.command()
def models() -> None:
    """List registered models and whether each one can actually run here."""
    from .backends.weights import is_runnable
    from .core.registry import models as registry
    from .inference.engine import _ensure_registry

    _ensure_registry()

    table = Table(title="Model registry")
    table.add_column("id", style="cyan")
    table.add_column("task")
    table.add_column("arch")
    table.add_column("scale", justify="right")
    table.add_column("content types", style="dim")
    table.add_column("weights")

    for spec in registry.all():
        status = (
            "[green]ready[/green]" if is_runnable(spec) else "[yellow]not downloaded[/yellow]"
        )

        table.add_row(
            spec.id,
            spec.task.value,
            spec.architecture,
            f"x{spec.scale}",
            ", ".join(c.value for c in spec.content_types) or "—",
            status,
        )

    console.print(table)
    console.print("\n[dim]Fetch missing weights: python scripts/download_weights.py --all[/dim]")


@app.command()
def doctor() -> None:
    """Report what hardware and runtimes are usable."""
    from .analysis.faces import get_detector
    from .backends.factory import backend_capabilities
    from .core.device import available_backends, device_info

    console.print("[bold]Backends[/bold]")
    for backend in available_backends():
        info = device_info(backend)
        vram = f" — {info.free_vram_mb}/{info.total_vram_mb} MB free" if info.total_vram_mb else ""
        console.print(f"  [green]{OK}[/green] {backend.value:<10} {info.name}{vram}")

    console.print("\n[bold]Runtimes[/bold]")
    for name, caps in backend_capabilities().items():
        if not caps.get("installed"):
            console.print(f"  [dim]{FAIL} {name:<10} not installed[/dim]")
            continue

        detail = caps.get("version", "")
        if name == "torch" and not caps.get("cuda_available"):
            detail += "  [yellow](no CUDA — check your NVIDIA driver)[/yellow]"
        console.print(f"  [green]{OK}[/green] {name:<10} {detail}")

    console.print("\n[bold]Face detector[/bold]")
    detector = get_detector()
    if detector.is_available():
        console.print(f"  [green]{OK}[/green] YuNet installed")
    else:
        console.print(f"  [yellow]{FAIL} not downloaded — face restoration unavailable[/yellow]")


def _collect_sources(source: Path, recursive: bool) -> list[Path]:
    from .preprocessing.io import is_supported

    if source.is_file():
        return [source]
    if not source.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    return sorted(p for p in source.glob(pattern) if p.is_file() and is_supported(p))


def _destination(path: Path, output: Path | None, source: Path, batch: bool, scale: int) -> Path:
    if output is None:
        return path.with_name(f"{path.stem}_x{scale}.png")

    if not batch and output.suffix:
        return output

    # Directory output: mirror the input tree so a recursive run does not flatten it.
    relative = path.relative_to(source) if source.is_dir() else Path(path.name)
    return (output / relative).with_suffix(".png")


def _progress_bar(label: str):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    class _Ctx:
        def __enter__(self):
            progress.start()
            return progress, progress.add_task(label, total=100)

        def __exit__(self, *exc):
            progress.stop()
            return False

    return _Ctx()


def _print_result(result, destination: Path) -> None:
    width, height = result.output_size
    models_used = " + ".join(run.model_id for run in result.runs) or "resample only"
    backend = result.runs[0].backend.value if result.runs else "—"
    tiles = sum(run.tiles for run in result.runs)

    console.print(
        f"[green]{OK}[/green] {destination.name}  "
        f"[dim]{width}x{height} · {models_used} · {backend} · "
        f"{tiles} tiles · {result.total_duration_ms / 1000:.1f}s[/dim]"
    )


if __name__ == "__main__":
    sys.exit(app())
