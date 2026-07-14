"""Regression tests for bugs that were found by looking at the output.

Every one of these shipped a *correctly shaped* image. None of them raised, none
of them logged an error, and the unit tests of the day were all green. They were
caught by putting the result on screen — which is why each now has a test that
asserts something about the pixels, not about the plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionsr.core.types import Precision

pytestmark = pytest.mark.weights  # all of these need real checkpoints


def test_gfpgan_output_is_not_clipped_to_upper_half(gfpgan_spec, aligned_face):
    """The backend must map a [-1,1] model's output back to [0,1], not clamp it.

    The bug: RunnableModel promises [0,1], and the torch backend enforced it with
    ``out.clamp_(0, 1)`` — applied to GFPGAN's native [-1,1] output. Every negative
    value (i.e. everything darker than mid-grey) became 0, and the face pipeline's
    subsequent ``* 0.5 + 0.5`` mapped what survived into [0.5, 1]. Result: a flat,
    washed-out pink smear where the face used to be. No error, no NaN.

    The guard is the histogram. A real restored face uses the full range and has
    structure; the broken one was near-constant and never went dark.
    """
    from visionsr.backends.factory import get_model
    from visionsr.preprocessing.io import to_float

    model = get_model(gfpgan_spec, precision=Precision.FP32)
    out = model.infer(to_float(aligned_face)[None, ...])[0]

    assert out.min() < 0.15, (
        f"darkest pixel is {out.min():.3f}: the shadows have been clipped away. "
        "The backend is not converting the model's value_range."
    )
    assert out.max() > 0.85, f"brightest pixel is only {out.max():.3f}"
    assert out.std() > 0.15, (
        f"output std is {out.std():.3f} — that is a flat field, not a face."
    )


def test_gfpgan_is_not_offered_in_fp16(gfpgan_spec):
    """fp16 silently destroys GFPGAN, so the spec must not advertise it.

    Measured on GFPGANv1.4: fp32 output spans -1.00..+0.92 (std 0.343); the same
    graph in fp16 collapses to -0.50..-0.19 (std 0.123). The StyleGAN2 decoder's
    demodulation and its per-layer sqrt(2) gain exceed half precision's dynamic
    range. Nothing raises — it just returns a dark smear.

    Selection reads ``spec.precisions``, so keeping fp16 out of it is what prevents
    a caller (or the auto-precision default on CUDA) from ever choosing it.
    """
    assert Precision.FP16 not in gfpgan_spec.precisions, (
        "fp16 is back in the GFPGAN spec. It produces a silently ruined face; "
        "see the measurement in ai/configs/models.yaml."
    )
    assert Precision.FP32 in gfpgan_spec.precisions


def test_sr_models_still_use_fp16(registry):
    """The fp16 ban is specific to the StyleGAN decoder — SR models must keep it.

    Guards the over-correction: pinning everything to fp32 would halve throughput
    and double VRAM on the very card that needs neither.
    """
    sr = registry.get("realesrgan-x4plus")
    assert Precision.FP16 in sr.precisions


def test_free_vram_counts_torch_cached_memory():
    """Free VRAM must include torch's cached-but-unallocated pool.

    The bug: ``torch.cuda.mem_get_info`` reports what the *driver* has left. PyTorch's
    caching allocator does not return freed blocks to the driver, so after one
    inference the driver can report a few MB free while hundreds of MB sit reusable
    inside torch. The tiler read that, concluded it was out of memory, and dropped to
    64px tiles — correct output, several times slower, for the rest of the process.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    from visionsr.core.device import free_vram_mb
    from visionsr.core.types import Backend

    # Reserve a big block, then free it. It stays in torch's cache, so the driver
    # still considers it taken — exactly the state that triggered the bug.
    block = torch.empty(int(200 * 1024**2 / 4), dtype=torch.float32, device="cuda")
    del block

    driver_free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

    reported = free_vram_mb(Backend.CUDA)
    assert reported is not None

    expected = (driver_free + cached) / 1024**2
    assert reported == pytest.approx(expected, rel=0.05), (
        f"free_vram_mb reported {reported}MB but torch is holding "
        f"{cached / 1024**2:.0f}MB of reusable cache on top of the driver's "
        f"{driver_free / 1024**2:.0f}MB."
    )

    torch.cuda.empty_cache()


def test_every_architecture_matches_its_checkpoint_exactly(registry):
    """No missing and no unexpected keys, for every registered model.

    ``load_state_dict(strict=False)`` is used at runtime so a partially-updated
    checkpoint still loads — which means a genuinely wrong architecture would load
    too, quietly, with half its layers randomly initialised. This is the test that
    would catch that.
    """
    from visionsr.backends.weights import load_state_dict
    from visionsr.core.registry import ONNX_GRAPH, architectures

    for spec in registry.all():
        # A native ONNX model has no torch architecture to compare against — the
        # checkpoint *is* the graph. Its equivalent guard lives in test_packaging.py,
        # which asserts every architecture a model names actually resolves.
        if spec.architecture == ONNX_GRAPH:
            continue

        module = architectures.build(spec.architecture, **spec.params)
        missing, unexpected = module.load_state_dict(load_state_dict(spec), strict=False)

        # StyleGAN2 regenerates its noise buffers; they are not in every checkpoint.
        missing = [key for key in missing if "noises.noise" not in key]

        assert not missing, f"{spec.id}: {len(missing)} weight(s) absent, e.g. {missing[:3]}"
        assert not unexpected, (
            f"{spec.id}: checkpoint has {len(unexpected)} key(s) the architecture "
            f"does not define, e.g. {list(unexpected)[:3]}"
        )


def test_face_restoration_makes_the_face_closer_to_the_real_one(
    portrait_image, portrait_truth
):
    """Face restoration must land on the face, and make it *better*.

    This is the test for a whole class of face-pipeline bugs, all of which produce a
    plausible-looking image: the restored crop pasted at the wrong coordinates; the
    two alignment transforms (template->canvas scales every term by the upscale;
    canvas->template scales only the linear block) swapped; the [-1,1] output clipped
    to [0,1]; fp16 collapsing the decoder. Each of those was a real bug here, and none
    of them raised.

    Rather than assert on geometry — which requires arbitrary thresholds, and where
    "how much of the face box changed?" has no defensible right answer, because the
    FFHQ crop legitimately extends past the detector's box into hair and jaw — this
    asserts the thing that actually matters: with the ground-truth original in hand,
    switching face restoration on must move the face *towards* it.

    LPIPS, because the point of GFPGAN is texture a human reads as detail (eyelashes,
    hair strands, iris), which is exactly what pixel metrics fail to credit.
    """
    lpips = pytest.importorskip("lpips")
    import torch

    from visionsr import EnhanceOptions, enhance

    upscale = 4
    without = enhance(portrait_image, EnhanceOptions(scale=upscale, face_restore=False))
    with_face = enhance(
        portrait_image,
        EnhanceOptions(scale=upscale, face_restore=True, face_restore_weight=1.0),
    )

    assert with_face.analysis.has_faces, "the fixture is supposed to contain a face"
    x, y, w, h = with_face.analysis.faces[0]

    # Compare the face region only: the rest of the frame is identical by construction,
    # and averaging it in would dilute the signal to nothing.
    truth = portrait_truth[: without.image.shape[0], : without.image.shape[1]]
    box = np.s_[y * upscale : (y + h) * upscale, x * upscale : (x + w) * upscale]

    net = lpips.LPIPS(net="alex", verbose=False)

    def distance(candidate: np.ndarray) -> float:
        def prepare(a: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            return t * 2 - 1

        with torch.inference_mode():
            return float(net(prepare(candidate[box]), prepare(truth[box])).item())

    sr_only = distance(without.image)
    restored = distance(with_face.image)

    assert restored < sr_only, (
        f"face restoration moved the face AWAY from ground truth "
        f"(LPIPS {sr_only:.4f} -> {restored:.4f}). It is landing in the wrong place, "
        "or the model is producing garbage."
    )

