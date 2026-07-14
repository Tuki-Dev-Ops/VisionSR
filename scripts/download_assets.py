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
GROUND_TRUTH = {
    f"kodim{n:02d}.png": f"https://r0k.us/graphics/kodak/kodak/kodim{n:02d}.png"
    for n in (4, 5)
}

# Real-ESRGAN's own sample inputs: already degraded, and useful as such.
DEGRADED = {
    name: f"https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/{name}"
    for name in ("0014.jpg", "00003.png", "OST_009.png", "children-alpha.png")
}


def fetch(url: str, destination: Path) -> bool:
    if destination.exists():
        console.print(f"[dim]{SKIP} {destination.name} — already here[/dim]")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        # r0k.us rate-limits aggressively and 401s a bare client.
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[red]{FAIL} {destination.name} — {exc}[/red]")
        return False

    destination.write_bytes(response.content)
    console.print(f"[green]{OK}[/green] {destination.name} — {len(response.content) / 1024:.0f} KB")
    return True


def main() -> int:
    ok = True

    console.print("[bold]ground truth[/bold] [dim](undegraded originals, for scoring)[/dim]")
    for name, url in GROUND_TRUTH.items():
        ok &= fetch(url, ASSETS / "hr" / name)

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
