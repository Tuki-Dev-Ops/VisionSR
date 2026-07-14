"""Background removal.

The interesting property of a cutout is not that it produces four channels — a bug
that returned a uniform alpha would do that too, and it would pass any test that only
checked the shape. It is that the alpha *separates* something: opaque where the
subject is, transparent where it is not, and soft in between.

So these tests assert on the mask's structure, not its existence.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionsr import EnhanceOptions, enhance
from visionsr.core.types import Task

pytestmark = pytest.mark.weights


@pytest.fixture(scope="module")
def cutout(portrait_image):
    return enhance(
        portrait_image,
        EnhanceOptions(scale=2, remove_background=True, face_restore=False),
    )


def test_the_result_is_rgba(cutout):
    assert cutout.image.shape[2] == 4
    assert cutout.image.dtype == np.uint8


def test_the_segmentation_model_actually_ran(cutout, registry):
    ids = [run.model_id for run in cutout.runs]
    assert len(ids) == 2, f"expected an SR model and a segmentation model, got {ids}"

    segmentation = registry.get(ids[-1])
    assert segmentation.task == Task.BACKGROUND_REMOVAL


def test_the_alpha_separates_rather_than_covering(cutout):
    """The mask must be a decision, not a haze.

    A broken normalisation — clipping the segmenter's unbounded logits to [0,1] instead
    of rescaling by their own extremes — produces an alpha channel that is uniformly
    mid-grey. It has four channels, it has a range, and it is useless: composited onto
    white it yields a washed-out ghost of the whole frame. The signature is that almost
    nothing is *decisively* opaque or transparent.
    """
    alpha = cutout.image[:, :, 3].astype(np.float32) / 255.0

    transparent = float((alpha < 0.05).mean())
    opaque = float((alpha > 0.95).mean())

    assert transparent > 0.05, (
        f"only {transparent:.1%} of the frame is properly transparent — the mask is not "
        "removing a background, it is tinting one."
    )
    assert opaque > 0.20, f"only {opaque:.1%} of the frame is properly opaque"

    # And it must not be a hard binary mask either: hair and edges are genuinely
    # semi-transparent, and that softness is what separates a cutout from a sticker.
    soft = float(((alpha > 0.05) & (alpha < 0.95)).mean())
    assert soft > 0.005, "the alpha has no soft edge at all — it has been thresholded"


def test_the_subject_survives_and_the_corners_do_not(cutout):
    """Anchored on geometry the model cannot fake.

    The fixture is a portrait: the subject occupies the middle, and the frame corners
    are background. An inverted mask, a transposed one, or one that latched onto the
    wrong region all pass every test above and fail this one.
    """
    alpha = cutout.image[:, :, 3].astype(np.float32) / 255.0
    height, width = alpha.shape

    centre = alpha[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3].mean()

    corner = height // 8
    corners = np.mean(
        [
            alpha[:corner, :corner].mean(),
            alpha[:corner, -corner:].mean(),
            alpha[-corner:, :corner].mean(),
        ]
    )

    assert centre > 0.8, f"the subject is only {centre:.2f} opaque — the mask is inverted?"
    assert corners < 0.4, f"the frame corners are {corners:.2f} opaque — background is surviving"
    assert centre - corners > 0.5, "the mask barely distinguishes subject from background"


def test_a_source_alpha_is_replaced_not_blended(portrait_image):
    """A cutout must overwrite any alpha the input arrived with.

    Merging them would multiply two unrelated masks together and produce a subject that
    is transparent wherever the *original* file happened to be — which, for a PNG with
    a decorative border, is a hole through the middle of the person.
    """
    with_alpha = np.dstack(
        [portrait_image, np.full(portrait_image.shape[:2], 128, dtype=np.uint8)]
    )

    result = enhance(
        with_alpha,
        EnhanceOptions(scale=2, remove_background=True, face_restore=False),
    )

    alpha = result.image[:, :, 3]
    assert alpha.max() > 200, (
        "the source's flat 50% alpha is still capping the cutout — it was blended in "
        "rather than replaced."
    )


def test_jpeg_output_is_refused_rather_than_silently_flattened():
    """JPEG has no alpha channel.

    Encoding an RGBA cutout to JPEG composites it onto white and writes a normal photo.
    The request succeeds, the file opens, and the transparency the user asked for is
    simply gone. That is the worst possible outcome, and it is the default one unless
    the combination is refused outright.
    """
    with pytest.raises(ValueError, match="alpha"):
        EnhanceOptions(remove_background=True, output_format="jpeg")

    # The lossless formats are fine.
    EnhanceOptions(remove_background=True, output_format="png")
    EnhanceOptions(remove_background=True, output_format="webp")


def test_a_native_onnx_model_needs_no_torch_architecture(registry):
    """The registry's claim, tested on the case that most stresses it.

    The segmentation models are published ONNX graphs. They have no architecture module,
    no exporter and no torch code — they joined the registry as a YAML entry and nothing
    else. If that stopped being true, the pluggable-registry claim would be marketing.
    """
    from visionsr.core.registry import ONNX_GRAPH

    segmenters = registry.find(task=Task.BACKGROUND_REMOVAL)
    assert segmenters, "no background-removal model is registered"

    for spec in segmenters:
        assert spec.architecture == ONNX_GRAPH
        assert spec.weights is not None
        assert spec.weights.filename.endswith(".onnx")

        # ...and it must therefore declare that torch cannot run it.
        from visionsr.core.types import Backend

        assert Backend.CUDA not in spec.backends
        assert Backend.CPU not in spec.backends
