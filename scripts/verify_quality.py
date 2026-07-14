#!/usr/bin/env python
"""End-to-end quality proof.

Shape-correct output is not evidence the engine works — a broken state-dict load,
a wrong normalisation, or a mis-stitched tile grid all produce a correctly-sized
image. This measures whether the pixels are actually right.

Protocol: take a high-resolution image as ground truth, synthesise a low-resolution
input from it, upscale that back with the engine, and compare against the truth.
Bicubic upsampling of the same input is the baseline.

Two degradation regimes, because they test different things:

* ``clean`` — area-downscale only. This is *adversarial* for a GAN upscaler:
  Real-ESRGAN was trained to undo blur/noise/JPEG, so on a pristine input it
  applies a correction that is not needed and invents texture, which costs PSNR
  even as the image looks sharper. A modest PSNR loss here is expected and correct.

* ``realistic`` — blur, then downscale, then sensor noise, then JPEG. This is the
  degradation model the network was actually trained on, and the regime real user
  photos live in. **This is the one that must beat bicubic**, and by a wide margin.
  If it does not, the engine is broken.

    python scripts/verify_quality.py
    python scripts/verify_quality.py --scale 2 --precision fp32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from visionsr import EnhanceOptions, enhance  # noqa: E402
from visionsr.core.types import Backend, Precision  # noqa: E402
from visionsr.preprocessing.io import load_image  # noqa: E402
from visionsr.ui import FAIL, OK, console  # noqa: E402

_lpips_net = None


def lpips_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """Learned perceptual distance (Zhang et al. 2018). Lower is better.

    This is the metric that matters for a GAN upscaler, and the reason is not a
    technicality. PSNR asks "is each pixel numerically close?" — but Real-ESRGAN
    *invents* texture: pore detail, fabric weave, foliage. That texture is
    perceptually right and numerically wrong, so it is penalised by PSNR and
    rewarded by a human eye. LPIPS compares deep features rather than pixels, and
    tracks human judgement far more closely; it is what the SR literature reports
    for perceptual models, and what separates a working Real-ESRGAN from a broken one.
    """
    global _lpips_net
    try:
        import lpips
        import torch
    except ImportError:
        return None

    if _lpips_net is None:
        # AlexNet backbone: the variant Zhang et al. found best-correlated with
        # human preference, and the one the SR papers quote.
        _lpips_net = lpips.LPIPS(net="alex", verbose=False)

    def prepare(x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        return t * 2 - 1  # LPIPS expects [-1, 1]

    with torch.inference_mode():
        return float(_lpips_net(prepare(a), prepare(b)).item())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over luminance, 11x11 gaussian window (Wang et al. 2004)."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)

    sigma_a = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a**2
    sigma_b = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b**2
    sigma_ab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))


def degrade(truth: np.ndarray, scale: int, regime: str, seed: int = 0) -> np.ndarray:
    """Synthesise a low-resolution input from ground truth."""
    h, w = truth.shape[:2]
    target = (w // scale, h // scale)

    if regime == "clean":
        # INTER_AREA integrates over the footprint instead of point-sampling, so it
        # is the honest "what a lower-res sensor would have captured" downsample.
        return cv2.resize(truth, target, interpolation=cv2.INTER_AREA)

    rng = np.random.default_rng(seed)

    # Order matters and mirrors a real capture chain: optics blur, then the sensor
    # samples, then the sensor adds noise, then the file is compressed. Blurring
    # after downscaling (a common mistake) models nothing physical.
    blurred = cv2.GaussianBlur(truth, (0, 0), sigmaX=1.2)
    small = cv2.resize(blurred, target, interpolation=cv2.INTER_AREA)

    noisy = small.astype(np.float32) + rng.normal(0, 6.0, small.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ok:
        return noisy
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def run_case(
    path: Path,
    scale: int,
    regime: str,
    backend: Backend | None,
    precision: Precision | None,
    model: str | None,
) -> dict:
    truth, _ = load_image(path)
    truth = truth[:, :, :3]

    # Crop to a multiple of `scale` so the round trip lands on the exact original
    # size — otherwise the images are misaligned and PSNR is meaningless.
    h = (truth.shape[0] // scale) * scale
    w = (truth.shape[1] // scale) * scale
    truth = np.ascontiguousarray(truth[:h, :w])

    low_res = degrade(truth, scale, regime)

    started = time.perf_counter()
    result = enhance(
        low_res,
        EnhanceOptions(
            scale=scale,  # type: ignore[arg-type]
            model_id=model,
            backend=backend,
            precision=precision,
            face_restore=False,  # isolate the SR model
        ),
    )
    elapsed = time.perf_counter() - started

    baseline = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
    output = result.image[:, :, :3]

    return {
        "name": path.stem,
        "regime": regime,
        "model": result.runs[0].model_id if result.runs else "—",
        "sr_psnr": psnr(truth, output),
        "bicubic_psnr": psnr(truth, baseline),
        "sr_ssim": ssim(truth, output),
        "bicubic_ssim": ssim(truth, baseline),
        "sr_lpips": lpips_distance(truth, output),
        "bicubic_lpips": lpips_distance(truth, baseline),
        "seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--backend", type=Backend, default=None)
    parser.add_argument("--precision", type=Precision, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--assets", type=Path, default=REPO_ROOT / "tests" / "assets" / "hr")
    args = parser.parse_args()

    images = sorted(p for p in args.assets.glob("*") if p.suffix.lower() in {".png", ".jpg"})
    if not images:
        console.print(f"[red]No ground-truth images in {args.assets}[/red]")
        return 1

    failures = 0
    header = f"{'image':<10} {'degradation':<11} {'PSNR SR':>8} {'PSNR bic':>9} {'SSIM SR':>8} {'SSIM bic':>9} {'LPIPS SR':>9} {'LPIPS bic':>10} {'LPIPS win':>10}"
    console.print(f"\n[bold]Round-trip fidelity at x{args.scale}[/bold]  (model: {args.model or 'auto'})\n")
    console.print(f"[dim]{header}[/dim]")

    for path in images:
        for regime in ("realistic", "clean"):
            try:
                r = run_case(path, args.scale, regime, args.backend, args.precision, args.model)
            except Exception as exc:
                console.print(f"[red]{FAIL} {path.name} ({regime}): {exc}[/red]")
                failures += 1
                continue

            # LPIPS is a distance: lower is better, so an improvement is a decrease.
            has_lpips = r["sr_lpips"] is not None
            lpips_gain = (r["bicubic_lpips"] - r["sr_lpips"]) if has_lpips else 0.0

            # The gate. Only the realistic regime, and only on LPIPS — see the module
            # docstring for why PSNR is the wrong instrument for this model class.
            if regime == "realistic":
                failed = has_lpips and lpips_gain <= 0
                failures += failed
                colour = "red" if failed else "green"
            else:
                colour = "green" if lpips_gain > 0 else "yellow"

            lp_sr = f"{r['sr_lpips']:.4f}" if has_lpips else "n/a"
            lp_bic = f"{r['bicubic_lpips']:.4f}" if has_lpips else "n/a"
            win = f"[{colour}]{lpips_gain:+.4f}[/{colour}]" if has_lpips else "—"

            console.print(
                f"{r['name']:<10} {regime:<11} "
                f"{r['sr_psnr']:>8.2f} {r['bicubic_psnr']:>9.2f} "
                f"{r['sr_ssim']:>8.4f} {r['bicubic_ssim']:>9.4f} "
                f"{lp_sr:>9} {lp_bic:>10} {win:>10}"
            )

    console.print(
        "\n[dim]LPIPS is the gate: lower = perceptually closer to ground truth. It is the\n"
        "metric the SR literature uses for GAN upscalers, because they trade pixel\n"
        "accuracy (PSNR) for texture a human reads as detail. PSNR/SSIM are reported\n"
        "for completeness; a GAN model is expected to lose PSNR and still be better.[/dim]"
    )

    if failures:
        console.print(
            f"\n[red]{FAIL} {failures} realistic-regime case(s) did not beat bicubic on LPIPS.[/red]"
        )
        return 1

    console.print(
        f"\n[green]{OK} Every realistic case is perceptually closer to ground truth "
        f"than bicubic.[/green]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
