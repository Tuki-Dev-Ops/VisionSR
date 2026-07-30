#!/usr/bin/env python
"""Build the synthetic inputs the end-to-end scripts upload.

    python scripts/make_e2e_fixtures.py

Fixtures are generated rather than committed so the degradation is explicit and
reproducible, and so a clean checkout can run the e2e suite without hunting for a
photo. Written to ``outputs/``, which is gitignored.

Two files:

``rotated_input.jpg`` — a phone-style photo, portrait to the eye and landscape on disk
with EXIF Orientation=6. Every modern phone hands you this, and it is the input that
made the before/after canvas render the result on its side while squashing the original
to match. Drawn from scratch, because what it has to carry is a metadata shape, not a
photographic one.

``portrait_input.png`` — the degraded portrait the browser and desktop e2e scripts
upload. Derived from a real photograph (run ``scripts/download_assets.py`` first),
because those scripts assert a face is found and a subject can be cut out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

# How the photo should LOOK. Small on purpose: a 4x job on it fits a 4 GB laptop GPU
# and finishes fast enough that the e2e suite stays worth running.
WIDTH, HEIGHT = 300, 500

# What the JPEG is crushed to. Low enough that super resolution has real work to do.
QUALITY = 38


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _photo() -> Image.Image:
    """A portrait frame with unmistakable up/down cues and real high-frequency detail."""
    rng = np.random.default_rng(7)
    image = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)

    # Sky gradient into ground: which way is up, without reading anything.
    for y in range(HEIGHT):
        t = y / HEIGHT
        draw.line(
            [(0, y), (WIDTH, y)],
            fill=(int(70 + 110 * (1 - t)), int(95 + 105 * (1 - t)), int(185 - 45 * t)),
        )
    draw.rectangle([0, int(HEIGHT * 0.70), WIDTH, HEIGHT], fill=(44, 88, 48))

    draw.text((WIDTH // 2, 40), "TOP", fill="white", font=_font(44), anchor="mm")
    draw.text((WIDTH // 2, HEIGHT - 34), "BOTTOM", fill="white", font=_font(34), anchor="mm")

    # A big arrow — readable even in a thumbnail, and the thing a human checks first.
    draw.polygon(
        [(WIDTH // 2, 118), (WIDTH // 2 - 52, 208), (WIDTH // 2 + 52, 208)],
        fill=(255, 208, 56),
    )
    draw.rectangle([WIDTH // 2 - 21, 208, WIDTH // 2 + 21, 318], fill=(255, 208, 56))

    # A converging picket fence: the classic resolution target.
    for i in range(40):
        x = 12 + i * 7 + (i * i) // 26
        if x > WIDTH - 12:
            break
        draw.line([(x, HEIGHT - 118), (x, HEIGHT - 74)], fill=(240, 245, 240), width=1)

    # Small text — the first thing to turn to mush at low resolution.
    draw.text(
        (14, HEIGHT - 66),
        "VisionSR 4x  ::  fine detail 0123456789",
        fill=(228, 236, 228),
        font=_font(13),
    )

    # Concentric rings over the horizon.
    cx, cy = WIDTH // 2, 380
    for r in range(6, 62, 4):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=1)

    # Sensor noise, so the analyser reads it as a photograph rather than a rendering.
    noisy = np.asarray(image).astype(np.int16) + rng.normal(0, 4.5, (HEIGHT, WIDTH, 3)).astype(
        np.int16
    )
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def write_rotated_input(path: Path) -> Path:
    """Save the photo the way a phone would: rotated on disk, uprighted by EXIF."""
    photo = _photo()

    # Orientation=6 means "rotate 90 CW to display", so store it rotated 90 CCW.
    stored = photo.transpose(Image.ROTATE_90)

    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    exif[0x010F] = "VisionSR"  # Make
    exif[0x0110] = "e2e-fixture"  # Model
    exif.get_ifd(0x8769)[0xA002] = stored.width  # PixelXDimension, as a camera writes it
    exif.get_ifd(0x8769)[0xA003] = stored.height

    path.parent.mkdir(parents=True, exist_ok=True)
    stored.save(path, "JPEG", quality=QUALITY, subsampling=2, exif=exif.tobytes())
    return path


def write_portrait_input(path: Path, truth: Path) -> bool:
    """Degrade a ground-truth portrait into the photo the e2e scripts upload.

    ``e2e-smoke``, ``e2e-cutout`` and the three desktop scripts all upload
    ``outputs/portrait_input.png`` and assert things only a real photograph can satisfy:
    that the classifier calls it a portrait, that a face is found, that a subject can be
    lifted off its background. So it is derived from a real image rather than drawn.

    The degradation mirrors the ``portrait_image`` fixture in ``tests/conftest.py`` —
    blur, downscale, sensor noise, then a hard JPEG — because the e2e should upload the
    same kind of damaged input the Python suite scores against, not a pristine one the
    upscaler has no work to do on.
    """
    if not truth.exists():
        # Plain ASCII: this prints to whatever console the developer has, and a Windows
        # cp949/cp1252 terminal raises UnicodeEncodeError on anything fancier.
        print(f"\n{path.name}: SKIPPED - no ground truth at {truth}")
        print("  Run: python scripts/download_assets.py")
        return False

    import cv2

    original = cv2.imread(str(truth), cv2.IMREAD_COLOR)
    if original is None:
        print(f"\n{path.name}: SKIPPED - could not read {truth}")
        return False

    blurred = cv2.GaussianBlur(original, (0, 0), 1.2)
    small = cv2.resize(
        blurred,
        (original.shape[1] // 4, original.shape[0] // 4),
        interpolation=cv2.INTER_AREA,
    )

    rng = np.random.default_rng(0)
    noisy = np.clip(small + rng.normal(0, 6.0, small.shape), 0, 255).astype(np.uint8)

    # Round-trip through a hard JPEG so the compression artefacts are real, then store
    # as PNG: the file the e2e uploads must not add a second generation of loss.
    ok, encoded = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ok:
        print(f"\n{path.name}: SKIPPED - JPEG round-trip failed")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.imdecode(encoded, cv2.IMREAD_COLOR))

    with Image.open(path) as handle:
        size = handle.size
    print(f"\n{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"  {path.stat().st_size / 1024:.0f} KB, PNG")
    print(f"  {size[0]}x{size[1]}, degraded from {truth.name} ({original.shape[1]}x{original.shape[0]})")
    return True


def main() -> int:
    # Not description=__doc__: argparse would print it, and the docstring is not ASCII.
    parser = argparse.ArgumentParser(description="Build the generated e2e input images.")
    parser.add_argument(
        "--outputs", type=Path, default=OUTPUTS, help="where to write (default: ./outputs)"
    )
    args = parser.parse_args()

    path = write_rotated_input(args.outputs / "rotated_input.jpg")

    check = Image.open(path)
    displayed = ImageOps.exif_transpose(check)
    size_kb = path.stat().st_size / 1024
    print(f"{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"  {size_kb:.0f} KB, JPEG q{QUALITY}")
    print(f"  stored pixels    : {check.size[0]}x{check.size[1]} (landscape on disk)")
    print(f"  EXIF orientation : {check.getexif().get(0x0112)}")
    print(f"  as displayed     : {displayed.size[0]}x{displayed.size[1]} (portrait to the eye)")

    write_portrait_input(
        args.outputs / "portrait_input.png",
        REPO_ROOT / "tests" / "assets" / "hr" / "kodim04.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
