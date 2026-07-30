#!/usr/bin/env python
"""Fetch the images the test suite measures against.

    python scripts/download_assets.py

Not committed, for the same reason the model weights are not: they are third-party
images that carry their own terms, and a repository should not quietly redistribute
them. The tests that need them skip cleanly when they are absent, so a fresh clone still
runs — it just runs fewer tests, and says so.

Two sets, doing two different jobs:

**Ground truth** (`tests/assets/hr/`) — the Kodak True Color suite. These are *undegraded*
originals, which is the whole point: quality is measured by degrading them synthetically,
reconstructing, and comparing against what was actually there. Measuring against an image
that was already low-resolution proves nothing, and an early version of the quality gate
did exactly that and reported nonsense.

**Real degraded inputs** (`tests/assets/`) — samples from the Real-ESRGAN repository, used
where the point is to exercise the pipeline on real-world damage rather than to score it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from visionsr.ui import FAIL, OK, SKIP, console  # noqa: E402

ASSETS = REPO_ROOT / "tests" / "assets"

# Kodak True Color suite: 768x512, uncompressed, the standard reference set for image
# processing work. kodim04 is a portrait (it exercises the face pipeline); kodim05 is a
# row of motorcycles, whose wheel spokes are a brutal test of a segmentation mask.
#
# Two sources per image, tried in order. r0k.us is the suite's home and stays first, but
# it now answers 401 to everything programmatic — a browser User-Agent no longer buys a
# way past it — so a mirror has to carry the load or the ground-truth half of this script
# silently gets you nothing. The mirror stores them as ``NN.png``; they are renamed to
# their canonical names on the way in.
_MIRROR = (
    "https://raw.githubusercontent.com/MohamedBakrAli/"
    "Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/{n:02d}.png"
)

GROUND_TRUTH = {
    f"kodim{n:02d}.png": (
        f"https://r0k.us/graphics/kodak/kodak/kodim{n:02d}.png",
        _MIRROR.format(n=n),
    )
    for n in (4, 5)
}

# Every image in the suite is 768x512 or that rotated. A source serving anything else is
# serving the wrong thing, and saying so here beats a decode error three tests later.
KODAK_SIZES = {(768, 512), (512, 768)}

# Real-ESRGAN's own sample inputs: already degraded, and useful as such.
DEGRADED = {
    name: f"https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/{name}"
    for name in ("0014.jpg", "00003.png", "OST_009.png", "children-alpha.png")
}


def _decoded_size(payload: bytes) -> tuple[int, int] | None:
    """Size of `payload` as an image, or None if it is not one.

    A host that has stopped serving the file may still answer 200 with an HTML error
    page. Writing that to ``kodim04.png`` produces a file that exists, has a plausible
    name, and fails much later somewhere that looks unrelated.
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as handle:
            handle.verify()
        with Image.open(io.BytesIO(payload)) as handle:
            return handle.size
    except Exception:
        return None


def fetch(urls: str | tuple[str, ...], destination: Path, sizes: set | None = None) -> bool:
    """Fetch the first of `urls` that yields a real image. Returns True on success."""
    if destination.exists():
        console.print(f"[dim]{SKIP} {destination.name} — already here[/dim]")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates = (urls,) if isinstance(urls, str) else urls
    problems = []

    for url in candidates:
        try:
            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            problems.append(f"{url} — {exc}")
            continue

        size = _decoded_size(response.content)
        if size is None:
            problems.append(f"{url} — 200, but the body is not an image")
            continue
        if sizes is not None and size not in sizes:
            problems.append(f"{url} — wrong image ({size[0]}x{size[1]})")
            continue

        destination.write_bytes(response.content)
        console.print(
            f"[green]{OK}[/green] {destination.name} — {size[0]}x{size[1]}, "
            f"{len(response.content) / 1024:.0f} KB"
        )
        return True

    console.print(f"[red]{FAIL} {destination.name}[/red]")
    for problem in problems:
        console.print(f"  [dim]{problem}[/dim]")
    return False


def main() -> int:
    ok = True

    console.print("[bold]ground truth[/bold] [dim](undegraded originals, for scoring)[/dim]")
    for name, urls in GROUND_TRUTH.items():
        ok &= fetch(urls, ASSETS / "hr" / name, sizes=KODAK_SIZES)

    console.print("\n[bold]degraded samples[/bold] [dim](real-world damage, for exercising)[/dim]")
    for name, url in DEGRADED.items():
        ok &= fetch(url, ASSETS / name)

    if not ok:
        console.print(
            f"\n[yellow]{FAIL} Some assets could not be fetched. The tests that need them "
            "will skip.[/yellow]"
        )
        return 1

    console.print(f"\n[green]{OK} Assets ready. Run: pytest / python scripts/verify_quality.py[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
