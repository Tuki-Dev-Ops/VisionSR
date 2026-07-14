"""Registry and model-selection tests.

The architectural rule from the spec is "never hardcode models — every model must be
pluggable". That is easy to state and easy to violate a month later with one
innocent ``if model_id == "realesrgan-x4plus"``. The tests here are the enforcement:
they build a registry containing *only invented models* and assert the engine still
routes correctly. If anyone adds a real model name to engine code, these fail.
"""

from __future__ import annotations

import pytest

from visionsr.core.errors import ConfigError, ModelNotFoundError
from visionsr.core.registry import ModelRegistry
from visionsr.core.types import (
    Backend,
    ContentType,
    EnhanceOptions,
    ImageAnalysis,
    ModelSpec,
    Task,
)


def make_spec(model_id: str, **overrides) -> ModelSpec:
    base = {
        "id": model_id,
        "name": model_id,
        "task": Task.SUPER_RESOLUTION,
        "architecture": "fictional-arch",
        "scale": 4,
    }
    return ModelSpec(**{**base, **overrides})


def make_analysis(**overrides) -> ImageAnalysis:
    base = {
        "width": 640,
        "height": 480,
        "channels": 3,
        "has_alpha": False,
        "content_type": ContentType.PHOTO,
        "content_confidence": 0.9,
        "noise_level": 0.1,
        "blur_level": 0.1,
        "compression_level": 0.1,
        "quality_score": 0.85,
    }
    return ImageAnalysis(**{**base, **overrides})


@pytest.fixture
def fake_registry() -> ModelRegistry:
    """A registry of models that do not exist. Nothing real is named here on purpose."""
    registry = ModelRegistry()

    registry.register(
        make_spec(
            "heavyweight-photo",
            content_types=[ContentType.PHOTO, ContentType.OLD_PHOTO],
            priority=100,
            vram_mb_per_megapixel=2400,
        )
    )
    registry.register(
        make_spec(
            "featherweight-photo",
            content_types=[ContentType.PHOTO, ContentType.DOCUMENT],
            priority=80,
            vram_mb_per_megapixel=700,
        )
    )
    registry.register(
        make_spec(
            "line-art-specialist",
            task=Task.ANIME_RESTORATION,
            content_types=[ContentType.ANIME, ContentType.MANGA],
            priority=100,
        )
    )
    registry.register(
        make_spec(
            "face-fixer",
            task=Task.FACE_RESTORATION,
            scale=1,
            content_types=[ContentType.PORTRAIT],
            priority=100,
        )
    )
    registry.register(make_spec("double-only", scale=2, priority=50))
    return registry


# -- registry ---------------------------------------------------------------


def test_duplicate_id_is_rejected(fake_registry):
    with pytest.raises(ConfigError, match="already registered"):
        fake_registry.register(make_spec("heavyweight-photo"))


def test_replace_enables_hot_swap(fake_registry):
    fake_registry.register(make_spec("heavyweight-photo", scale=2), replace=True)
    assert fake_registry.get("heavyweight-photo").scale == 2


def test_unknown_id_lists_what_is_available(fake_registry):
    with pytest.raises(ModelNotFoundError) as exc:
        fake_registry.get("no-such-model")
    assert "face-fixer" in str(exc.value)


def test_find_filters_and_orders_by_priority(fake_registry):
    """A model with no declared content types is a generalist: it matches everything.

    That is what makes an unknown or misclassified image still upscale instead of
    erroring, so `double-only` belongs in this result — last, on its low priority.
    """
    found = fake_registry.find(task=Task.SUPER_RESOLUTION, content_type=ContentType.PHOTO)
    assert [s.id for s in found] == [
        "heavyweight-photo",
        "featherweight-photo",
        "double-only",
    ]


def test_find_by_backend_excludes_unsupported(fake_registry):
    fake_registry.register(
        make_spec("cpu-only", backends=[Backend.CPU], content_types=[ContentType.PHOTO])
    )
    found = fake_registry.find(content_type=ContentType.PHOTO, backend=Backend.CUDA)
    assert "cpu-only" not in [s.id for s in found]


# -- selection --------------------------------------------------------------


def test_illustration_routes_to_the_anime_task(fake_registry):
    """A drawing must not go to a photo model, whatever that model is called."""
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(content_type=ContentType.ANIME), EnhanceOptions(scale=4)
    )
    assert plan.sr_model.task == Task.ANIME_RESTORATION


def test_a_small_image_gets_the_best_model_even_when_it_is_clean(fake_registry):
    """Quality first. On a small image the expensive model costs a second — spend it.

    The earlier policy matched capacity to *degradation* and gave a clean 0.3MP photo
    the cheap model, which is a bad trade: it was visibly worse (LPIPS 0.454 vs 0.391
    on the reference set) to save a fraction of a second on a laptop GPU. Cost should
    be weighed against the *work*, and on a small image there is barely any.
    """
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(width=640, height=480, quality_score=0.95), EnhanceOptions(scale=4)
    )
    assert plan.sr_model.id == "heavyweight-photo"


def test_degraded_input_earns_the_expensive_model(fake_registry):
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(quality_score=0.25, noise_level=0.7, blur_level=0.6),
        EnhanceOptions(scale=4),
    )
    assert plan.sr_model.id == "heavyweight-photo"


def test_a_large_image_switches_to_the_light_model(fake_registry):
    """A 24MP input through the heavy net is minutes of GPU and hundreds of tiles."""
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(width=6000, height=4000, quality_score=0.85), EnhanceOptions(scale=4)
    )
    assert plan.sr_model.id == "featherweight-photo"


def test_size_outweighs_damage_on_a_very_large_image(fake_registry):
    """Damage argues for capacity, size argues against, and on a 24MP input size wins.

    Otherwise a badly-degraded 24MP scan would silently commit the user to a multi-minute
    run they did not ask for. They can still ask, with --model.
    """
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(width=6000, height=4000, quality_score=0.2), EnhanceOptions(scale=4)
    )
    assert plan.sr_model.id == "featherweight-photo"


def test_the_crossover_is_monotonic_in_image_size(fake_registry):
    """Growing the image must never swing the choice back to the expensive model.

    Guards against a sign error or a bad weight making the policy non-monotonic, which
    would be baffling in use: a slightly bigger image suddenly taking ten times longer.
    """
    from visionsr.analysis.selector import ModelSelector

    selector = ModelSelector(fake_registry)
    chosen = [
        selector.select(
            make_analysis(width=w, height=w * 3 // 4, quality_score=0.85),
            EnhanceOptions(scale=4),
        ).sr_model.id
        for w in (320, 640, 1280, 2560, 4000, 6000, 8000)
    ]

    # Once it goes light, it must stay light.
    switched = [i for i, m in enumerate(chosen) if m == "featherweight-photo"]
    assert switched, f"never switched to the light model at any size: {chosen}"
    assert switched == list(range(switched[0], len(chosen))), (
        f"the choice flip-flopped with image size: {chosen}"
    )


def test_explicit_model_overrides_the_selector(fake_registry):
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(content_type=ContentType.ANIME),
        EnhanceOptions(scale=4, model_id="double-only"),
    )
    assert plan.sr_model.id == "double-only"


def test_unknown_content_type_still_upscales(fake_registry):
    """An image nothing claims must not fail — it falls back to a generalist."""
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(content_type=ContentType.SATELLITE), EnhanceOptions(scale=4)
    )
    assert plan.sr_model.task == Task.SUPER_RESOLUTION


def test_faces_trigger_face_restoration_automatically(fake_registry):
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(faces=[(100, 100, 200, 200)]), EnhanceOptions(scale=4)
    )
    assert plan.face_model is not None
    assert plan.face_model.task == Task.FACE_RESTORATION


def test_a_tiny_face_is_left_alone(fake_registry):
    """Restoring a 12px face invents a stranger. Below a size floor, don't."""
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(faces=[(10, 10, 12, 12)]), EnhanceOptions(scale=4)
    )
    assert plan.face_model is None


def test_face_restore_can_be_forced_off(fake_registry):
    from visionsr.analysis.selector import ModelSelector

    plan = ModelSelector(fake_registry).select(
        make_analysis(faces=[(100, 100, 200, 200)]),
        EnhanceOptions(scale=4, face_restore=False),
    )
    assert plan.face_model is None


# -- scale planning ---------------------------------------------------------


@pytest.mark.parametrize(
    ("model_scale", "target", "passes", "resample"),
    [
        (4, 4, 1, 1.0),  # exact
        (4, 8, 2, 0.5),  # two x4 passes then halve — beats one pass + bicubic
        (4, 16, 2, 1.0),  # exactly two passes
        (4, 2, 1, 0.5),  # one pass then downscale
        (2, 8, 3, 1.0),  # three x2 passes
        (4, 1, 0, 1.0),  # no SR at all
    ],
)
def test_scale_planning(model_scale, target, passes, resample):
    from visionsr.analysis.selector import _plan_scaling

    assert _plan_scaling(model_scale, target) == (passes, pytest.approx(resample))


def test_overshoot_then_downscale_never_upsamples_after_the_net(fake_registry):
    """The final resample must only ever shrink.

    Upsampling after inference would mean handing the user bicubic pixels the network
    never saw — the exact thing they are paying the GPU to avoid.
    """
    from visionsr.analysis.selector import ModelSelector

    for target in (2, 4, 8, 16):
        plan = ModelSelector(fake_registry).select(
            make_analysis(), EnhanceOptions(scale=target)  # type: ignore[arg-type]
        )
        assert plan.final_resample <= 1.0, f"x{target} would upsample after the model"
        assert plan.effective_scale == pytest.approx(target)
