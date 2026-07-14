#!/usr/bin/env python
"""Fetch model checkpoints listed in the registry.

    python scripts/download_weights.py --all
    python scripts/download_weights.py --model realesrgan-x4plus
    python scripts/download_weights.py --list

Weights are never committed — they are large, and their licences differ from this
repo's. The registry declares where each one comes from; this script resolves them
and records the sha256 it actually got into ``ai/checkpoints/registry.lock.json``,
so a later run (or a colleague, or CI) can detect a changed or corrupted file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from visionsr.analysis.faces import YUNET_FILENAME, YUNET_URL  # noqa: E402
from visionsr.core.config import get_settings  # noqa: E402
from visionsr.core.registry import models  # noqa: E402
from visionsr.core.types import ModelSpec, WeightSource  # noqa: E402
from visionsr.ui import FAIL, OK, SKIP, console  # noqa: E402

#: The face detector is not a ModelSpec — it is not an inference model, it is a
#: dependency of the face *pipeline*. It is listed here so one command gets you a
#: working install.
_FACE_DETECTOR = ModelSpec(
    id="face-detector",
    name="YuNet face detector",
    task="face_restoration",  # type: ignore[arg-type]
    architecture="<opencv-dnn>",
    weights=WeightSource(url=YUNET_URL, filename=YUNET_FILENAME),
    description="Required for face restoration and portrait classification.",
)


def load_registry() -> list[ModelSpec]:
    import visionsr.models  # noqa: F401  — registers architectures

    models.load_dir(get_settings().config_dir, replace=True)
    return [*models.all(), _FACE_DETECTOR]


def download(spec: ModelSpec, force: bool = False) -> tuple[bool, str | None]:
    """Fetch one checkpoint. Returns (downloaded, sha256)."""
    assert spec.weights is not None

    settings = get_settings()
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.weight_path(spec.weights.filename)

    if destination.exists() and not force:
        console.print(f"[dim]{SKIP} {spec.id} — already present[/dim]")
        return False, _sha256(destination)

    # Download to a temp name and rename on success. A half-written .pth that looks
    # complete is worse than no file: it fails at load time with an opaque unpickling
    # error rather than a clear "not downloaded".
    partial = destination.with_suffix(destination.suffix + ".part")

    with httpx.stream(
        "GET", spec.weights.url, follow_redirects=True, timeout=settings.download_timeout_s
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(spec.id, total=total or None)

            with partial.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1024 * 256):
                    fh.write(chunk)
                    progress.update(task, advance=len(chunk))

    partial.replace(destination)

    digest = _sha256(destination)
    size_mb = destination.stat().st_size / 1024**2
    console.print(f"[green]{OK}[/green] {spec.id} — {size_mb:.1f} MB, sha256 {digest[:12]}")

    return True, digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_lockfile(entries: dict[str, dict[str, str]]) -> Path:
    """Record what we actually downloaded, so drift is detectable later."""
    path = get_settings().checkpoint_dir / "registry.lock.json"

    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(entries)

    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Download every registered model.")
    parser.add_argument("--model", action="append", help="Download one model by id. Repeatable.")
    parser.add_argument("--list", action="store_true", help="List what is available.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    specs = load_registry()
    by_id = {spec.id: spec for spec in specs}

    if args.list:
        settings = get_settings()
        for spec in specs:
            assert spec.weights is not None
            present = settings.weight_path(spec.weights.filename).exists()
            mark = f"[green]{OK}[/green]" if present else f"[dim]{SKIP}[/dim]"
            console.print(f"{mark} [cyan]{spec.id:<28}[/cyan] {spec.description.strip().splitlines()[0][:60]}")
        return 0

    if args.all:
        wanted = specs
    elif args.model:
        unknown = [m for m in args.model if m not in by_id]
        if unknown:
            console.print(f"[red]Unknown model id(s): {', '.join(unknown)}[/red]")
            console.print(f"[dim]Available: {', '.join(by_id)}[/dim]")
            return 1
        wanted = [by_id[m] for m in args.model]
    else:
        parser.print_help()
        return 1

    lock: dict[str, dict[str, str]] = {}
    failed = []

    for spec in wanted:
        if spec.weights is None:
            continue
        try:
            _, digest = download(spec, force=args.force)
            if digest:
                lock[spec.id] = {
                    "filename": spec.weights.filename,
                    "url": spec.weights.url,
                    "sha256": digest,
                }
        except httpx.HTTPError as exc:
            failed.append(spec.id)
            console.print(f"[red]{FAIL} {spec.id} — {exc}[/red]")

    if lock:
        path = update_lockfile(lock)
        console.print(f"\n[dim]Recorded {len(lock)} checksum(s) in {path.name}[/dim]")

    if failed:
        console.print(f"[red]{len(failed)} download(s) failed: {', '.join(failed)}[/red]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
