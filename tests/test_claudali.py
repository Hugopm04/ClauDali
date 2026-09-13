"""Tests that need no GPU, no model weights and no network.

Everything up to the sampler is deterministic and cheap to check: the spec
contract, the prompt compiler, the procedural control maps, compositing,
postprocessing and the diagnostics. Those are also the parts most likely to
break silently -- a bad render is obvious, a subtly wrong prompt is not.

Run with:  .venv\\Scripts\\python -m pytest -q
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from claudali.compiler import _assemble, compile_spec, load_vocabulary
from claudali.compose import apply_overlays, apply_postprocess, quantize_to_palette
from claudali.control.maps import build_depth_map, build_edge_map, build_region_masks
from claudali.diagnostics import analyse
from claudali.engine import quiet
from claudali.engine.pipelines import _plan_cudnn, _should_upcast_vae
from claudali.engine.regional import grid_for
from claudali.spec import ASPECT_BUCKETS, Overlay, Postprocess, SceneSpec, load_spec
from claudali.tokens import CHUNK_CONTENT_TOKENS, count_content_tokens, estimate_tokens, pack

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.json"))


def minimal(**overrides) -> SceneSpec:
    data = {"subject": {"primary": "a stoneware coffee cup"}}
    data.update(overrides)
    return load_spec(data)


# ---------------------------------------------------------------------------
# Spec contract
# ---------------------------------------------------------------------------


def test_minimal_spec_is_valid():
    spec = minimal()
    assert spec.intent == "photoreal"
    assert spec.resolution() == (1024, 1024)


def test_unknown_field_is_rejected():
    """A misspelled section is always a bug; silence would hide it."""
    with pytest.raises(Exception) as caught:
        load_spec({"subject": {"primary": "x"}, "lightning": {"key": "neon"}})
    assert "lightning" in str(caught.value)


def test_aspect_resolves_to_a_training_bucket():
    for name, (width, height) in ASPECT_BUCKETS.items():
        spec = minimal(composition={"aspect": name})
        assert spec.resolution() == (width, height)


def test_explicit_size_overrides_the_aspect_bucket():
    spec = minimal(composition={"aspect": "1:1"}, render={"width": 768, "height": 512})
    assert spec.resolution() == (768, 512)


def test_bad_bbox_is_rejected():
    with pytest.raises(Exception):
        load_spec(
            {
                "subject": {"primary": "x"},
                "composition": {"layers": [{"role": "a", "bbox": [0.8, 0.1, 0.2, 0.9]}]},
            }
        )


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def test_intent_supplies_defaults():
    compiled = compile_spec(minimal(intent="painterly"))
    assert compiled.model == "dreamshaper-xl"
    assert compiled.cfg == 7.5
    assert "oil painting" in compiled.prompt


def test_explicit_value_beats_the_intent_default():
    """Regression: `steps` equal to the schema default must still count as set."""
    compiled = compile_spec(minimal(intent="painterly", render={"steps": 30}))
    assert compiled.steps == 30, "an explicit 30 was overwritten by the intent default"

    inferred = compile_spec(minimal(intent="painterly"))
    assert inferred.steps == 34


def test_subject_is_always_present():
    compiled = compile_spec(minimal())
    assert "stoneware coffee cup" in compiled.prompt


def test_unknown_vocabulary_key_warns_but_still_renders():
    compiled = compile_spec(minimal(style={"medium": "cyanotype"}))
    assert "cyanotype" in compiled.prompt
    assert any("cyanotype" in warning for warning in compiled.warnings)


def test_weights_use_compel_syntax():
    compiled = compile_spec(minimal(style={"medium": "pixel_art"}))
    assert ")1.2" in compiled.prompt or ")1.20" in compiled.prompt


def test_raw_prompt_replaces_and_says_so():
    compiled = compile_spec(minimal(raw={"prompt": "just this"}))
    assert compiled.prompt == "just this"
    assert any("raw.prompt" in warning for warning in compiled.warnings)


def test_prompt_has_no_duplicate_fragments():
    compiled = compile_spec(
        minimal(
            style={"descriptors": ["moody", "moody"]},
            subject={"primary": "a cup", "details": ["moody"]},
        )
    )
    fragments = [part.strip() for part in compiled.prompt.split(",")]
    assert len(fragments) == len(set(fragments))


def test_horizon_low_in_frame_means_sky_dominant():
    """Regression: `horizon` is a y coordinate, not a proportion of sky."""
    low = compile_spec(minimal(composition={"horizon": 0.8}))
    high = compile_spec(minimal(composition={"horizon": 0.2}))
    assert "sky dominant" in low.prompt
    assert "ground dominant" in high.prompt


def test_layers_without_control_are_not_dropped_silently():
    """Layers only reach the render through a ControlNet map; saying so is the rule."""
    layers = {"composition": {"layers": [{"role": "tree", "bbox": [0.1, 0.1, 0.4, 0.9]}]}}
    quiet = compile_spec(minimal(control={"mode": "depth"}, **layers))
    loud = compile_spec(minimal(control={"mode": "none"}, **layers))
    assert not any("composition.layers" in warning for warning in quiet.warnings)
    assert any("composition.layers" in warning for warning in loud.warnings)


def test_negatives_are_not_absurdly_long():
    """Past ~40 tokens SDXL's negative conditioning mostly adds noise."""
    for intent in load_vocabulary()["intents"]:
        compiled = compile_spec(minimal(intent=intent))
        assert len(compiled.negative_prompt.split()) < 60, intent


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_compile_without_warnings(path: Path):
    spec = load_spec(json.loads(path.read_text(encoding="utf-8")))
    compiled = compile_spec(spec)
    assert compiled.prompt
    assert compiled.warnings == [], compiled.warnings


# ---------------------------------------------------------------------------
# Control maps
# ---------------------------------------------------------------------------


def layered_spec() -> SceneSpec:
    return minimal(
        composition={
            "aspect": "3:2",
            "horizon": 0.62,
            "layers": [
                {"role": "ridge", "shape": "horizon", "bbox": [0.0, 0.52, 1.0, 0.64], "depth": 0.18},
                {"role": "tower", "shape": "column", "bbox": [0.62, 0.14, 0.74, 0.66], "depth": 0.75},
                {"role": "rock", "shape": "blob", "bbox": [0.0, 0.55, 0.55, 1.0], "depth": 0.92},
            ],
        },
        control={"mode": "depth"},
    )


def test_depth_map_orders_layers_by_depth():
    spec = layered_spec()
    array = np.asarray(build_depth_map(spec), dtype=np.float32)
    height, width = array.shape
    near = array[int(0.85 * height), int(0.25 * width)]   # the blob
    far = array[int(0.57 * height), int(0.90 * width)]    # the ridge
    sky = array[int(0.10 * height), int(0.20 * width)]    # nothing
    assert near > far > sky


def test_depth_map_matches_the_requested_resolution():
    spec = layered_spec()
    assert build_depth_map(spec).size == spec.resolution()


def test_blob_shape_is_stable_across_calls():
    """A blob's seed derives from its role, so it must not drift between runs."""
    spec = layered_spec()
    first = np.asarray(build_depth_map(spec))
    second = np.asarray(build_depth_map(spec))
    assert np.array_equal(first, second)


def test_edge_map_draws_something():
    edges = np.asarray(build_edge_map(layered_spec()))
    assert (edges > 128).sum() > 500


def test_region_masks_are_keyed_by_role():
    masks = build_region_masks(layered_spec())
    assert set(masks) == {"ridge", "tower", "rock"}
    for mask in masks.values():
        assert np.asarray(mask).max() == 255


# ---------------------------------------------------------------------------
# Compositing and postprocess
# ---------------------------------------------------------------------------


def swatch(color=(120, 120, 120), size=(256, 192)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_text_overlay_changes_pixels():
    base = swatch()
    out = apply_overlays(base, [Overlay(type="text", text="HELLO", size=48, color="#ffffff")])
    assert out.size == base.size
    assert not np.array_equal(np.asarray(out.convert("RGB")), np.asarray(base))


def test_gradient_direction_is_respected():
    """A bottom scrim needs `up`; getting this backwards was a real bug."""
    base = swatch(color=(255, 255, 255))
    up = np.asarray(
        apply_overlays(
            base, [Overlay(type="gradient", color="#000000", bbox=[0, 0, 1, 1], direction="up")]
        ).convert("L"),
        dtype=float,
    )
    assert up[-1, :].mean() < up[0, :].mean(), "direction='up' must darken the bottom edge"

    down = np.asarray(
        apply_overlays(
            base, [Overlay(type="gradient", color="#000000", bbox=[0, 0, 1, 1], direction="down")]
        ).convert("L"),
        dtype=float,
    )
    assert down[0, :].mean() < down[-1, :].mean()


def test_palette_lock_uses_only_the_given_colours():
    noisy = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB"
    )
    palette = ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]
    locked = np.asarray(quantize_to_palette(noisy, palette))
    unique = {tuple(color) for color in locked.reshape(-1, 3)}
    expected = {
        tuple(int(value[i : i + 2], 16) for i in (1, 3, 5)) for value in palette
    }
    assert unique <= expected


def test_pixelate_reduces_distinct_colours():
    noisy = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8), mode="RGB"
    )
    before = len({tuple(c) for c in np.asarray(noisy).reshape(-1, 3)})
    after_image = apply_postprocess(noisy, Postprocess(pixelate=16))
    after = len({tuple(c) for c in np.asarray(after_image).reshape(-1, 3)})
    assert after < before


def test_seamless_keeps_the_image_size():
    image = swatch(size=(256, 256))
    assert apply_postprocess(image, Postprocess(seamless=True)).size == (256, 256)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_black_frame_is_flagged_as_the_vae_failure():
    """The specific failure mode this hardware is prone to must name itself."""
    black = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    flags = analyse(black)["flags"]
    assert any("black" in flag and "VAE" in flag for flag in flags)


def test_normal_image_has_no_failure_flags():
    gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    image = Image.fromarray(np.dstack([gradient] * 3), mode="RGB")
    assert analyse(image)["flags"] == []


def test_diagnostics_report_every_section():
    image = Image.fromarray(
        np.random.default_rng(2).integers(0, 255, (96, 96, 3), dtype=np.uint8), mode="RGB"
    )
    report = analyse(image)
    assert set(report) >= {"size", "exposure", "detail", "color", "palette", "composition", "flags"}
    assert abs(sum(entry["coverage_pct"] for entry in report["palette"]) - 100) < 1.5


def test_composition_balance_finds_an_off_centre_subject():
    array = np.zeros((256, 256, 3), dtype=np.uint8)
    # High-contrast detail in the bottom-left quadrant only.
    array[160:240, 20:100] = np.random.default_rng(3).integers(
        0, 255, (80, 80, 3), dtype=np.uint8
    )
    balance = analyse(Image.fromarray(array, mode="RGB"))["composition"]
    assert balance["center_x"] < 0.5 and balance["center_y"] > 0.5
    assert balance["quadrants"]["bottom_left"] > 50


# ---------------------------------------------------------------------------
# Engine decisions that need no GPU
# ---------------------------------------------------------------------------


def test_vae_upcast_auto_follows_the_measurement():
    """A regression lock: a GTX 1660 Ti renders solid black without this."""
    assert _should_upcast_vae("auto", is_fp16=True, probe=lambda: True) is True
    assert _should_upcast_vae("auto", is_fp16=True, probe=lambda: False) is False


def test_vae_upcast_modes_skip_the_measurement():
    """'always' and 'never' have already decided, so the probe must not run."""

    def probe():
        raise AssertionError("the card must not be measured once the mode decides")

    assert _should_upcast_vae("always", is_fp16=True, probe=probe) is True
    assert _should_upcast_vae("never", is_fp16=True, probe=probe) is False


def test_vae_upcast_is_pointless_outside_fp16():
    def probe():
        raise AssertionError("fp32 cannot hit an fp16 fault")

    assert _should_upcast_vae("auto", is_fp16=False, probe=probe) is False
    assert _should_upcast_vae("always", is_fp16=False, probe=probe) is False


def _cudnn_plan(mode, results):
    """Drive _plan_cudnn with a scripted sequence of measurements."""
    measurements = iter(results)
    switches = []
    status = _plan_cudnn(mode, lambda: next(measurements), switches.append)
    return status, switches


def test_cudnn_is_left_alone_on_a_sound_machine():
    status, switches = _cudnn_plan("auto", [False])
    assert (status.broken, status.disabled, status.survives_workaround) == (False, False, False)
    assert switches == []


def test_cudnn_is_disabled_when_it_returns_nans():
    """A regression lock: a GTX 1660 Ti renders solid black at 1344x768 without this."""
    status, switches = _cudnn_plan("auto", [True, False])
    assert (status.broken, status.disabled, status.survives_workaround) == (True, True, False)
    assert switches == [False]
    assert any("cuDNN is disabled for this process" in note for note in status.notes)


def test_cudnn_is_restored_when_disabling_it_does_not_help():
    """Paying for a fallback that fixes nothing would be worse than saying so."""
    status, switches = _cudnn_plan("auto", [True, True])
    assert (status.broken, status.disabled, status.survives_workaround) == (True, False, True)
    assert switches == [False, True]


def test_cudnn_off_skips_the_measurement():
    def measure():
        raise AssertionError("the mode has already decided")

    status = _plan_cudnn("off", measure, lambda flag: None)
    assert status.disabled is True


def test_cudnn_on_reports_the_fault_it_was_told_to_keep():
    status, switches = _cudnn_plan("on", [True])
    assert (status.broken, status.disabled, status.survives_workaround) == (True, False, True)
    assert switches == []


def test_vae_upcast_covers_what_the_cudnn_workaround_could_not():
    """The fp32 decode is the second line of defence, not the first.

    Once cuDNN is off the fault is gone, so paying for an fp32 decode as well
    would be cost with no benefit; if it survives, the decode is all that is left.
    """
    cleared, _ = _cudnn_plan("auto", [True, False])
    survived, _ = _cudnn_plan("auto", [True, True])
    assert _should_upcast_vae("auto", True, lambda: cleared.survives_workaround) is False
    assert _should_upcast_vae("auto", True, lambda: survived.survives_workaround) is True


# ---------------------------------------------------------------------------
# CLIP chunking and the subject anchor
#
# All of these are regression locks on one measured failure. A spec asking for
# "an ancient forest with multiple small ethereal fairies" compiled to 306
# tokens; the word "fairies" appeared only at tokens 46, 58 and 73, so four of
# the five chunks CLIP read described a forest with nothing in it, and all four
# rendered images were empty forests.
# ---------------------------------------------------------------------------


def fairy_spec(**overrides) -> SceneSpec:
    """The spec that exposed the chunk-dilution bug, trimmed to essentials."""
    data = {
        "subject": {
            "primary": "an ancient forest with multiple small ethereal fairies",
            "anchor": "small ethereal fairies",
            "details": [
                "glowing gossamer wings",
                "naturalistic human forms",
                "bioluminescent fungi on the bark",
                "sparkling pixie dust trails",
            ],
            "action": "hovering around glowing flowers among mossy old-growth trees",
        },
        "scene": {
            "setting": "deep in a temperate rainforest, dense undergrowth",
            "time": "golden hour",
            "weather": "humid, with drifting mist",
        },
        "style": {
            "medium": "cinematic_still",
            "movement": "magic_realism",
            "descriptors": ["ethereal", "hyper-detailed", "atmospheric", "enchanted"],
            "detail": "intricate",
        },
        "camera": {"shot": "scene", "lens": "50mm", "angle": "eye_level"},
        "lighting": {"key": "god_rays"},
        "palette": {"name": "forest", "contrast": "high"},
        "composition": {"aspect": "16:9", "rule": "thirds"},
    }
    data.update(overrides)
    return load_spec(data)


def test_the_subject_appears_in_every_clip_chunk():
    """The bug, locked. No stretch of the prompt may run a whole chunk without it.

    Checked as gaps between mentions rather than by slicing at chunk offsets,
    because compel's window boundaries are not ours. A gap no larger than one
    chunk is what guarantees every window contains a mention, wherever it starts.
    """
    compiled = compile_spec(fairy_spec())
    assert compiled.tokens is not None
    assert compiled.tokens.chunks >= 3, "the fixture must be long enough to chunk"
    gaps = compiled.prompt.lower().split("small ethereal fairies")
    assert len(gaps) > 2, "the anchor was not restated at all"
    for gap in gaps:
        assert count_content_tokens(gap)[0] <= CHUNK_CONTENT_TOKENS, gap[:80]


def test_a_short_prompt_is_left_alone():
    """Anchoring a single-chunk prompt would repeat the subject for nothing.

    Tested on the assembler rather than on a spec, because the shortest real
    photoreal prompt lands within a token or two of the chunk boundary and a
    fixture pinned there would flip on any vocabulary edit.
    """
    prompt, anchors, warnings = _assemble(["a red fox", "in deep snow", "soft light"], "a red fox")
    assert anchors == []
    assert warnings == []
    assert prompt == "a red fox, in deep snow, soft light"


def test_the_explicit_anchor_is_preferred_to_the_subject_line():
    """`subject.primary` restated per chunk reinforces the setting along with it."""
    explicit = compile_spec(fairy_spec())
    inferred = compile_spec(
        fairy_spec(
            subject={
                "primary": "an ancient forest with multiple small ethereal fairies",
                "details": ["glowing gossamer wings"],
            }
        )
    )
    assert explicit.anchors and set(explicit.anchors) == {"small ethereal fairies"}
    assert "ancient forest" not in " ".join(explicit.anchors)
    assert any("subject.anchor" in warning for warning in inferred.warnings)


def test_a_short_subject_line_needs_no_anchor_advice():
    """Repeating three tokens per chunk is free, so saying so would be noise."""
    compiled = compile_spec(
        minimal(
            subject={"primary": "a red fox"},
            style={"descriptors": ["misty", "backlit", "painterly", "soft", "windswept"]},
        )
    )
    assert not any("subject.anchor" in warning for warning in compiled.warnings)


def test_the_token_count_says_whether_it_was_measured():
    """A number that is sometimes exact and sometimes guessed must say which."""
    compiled = compile_spec(fairy_spec())
    assert compiled.tokens.source in {"clip-tokenizer", "estimate"}
    assert compiled.tokens.exact == (compiled.tokens.source == "clip-tokenizer")
    assert compiled.tokens.chunks == -(-compiled.tokens.tokens // CHUNK_CONTENT_TOKENS)


def test_the_estimator_tracks_the_real_tokenizer():
    """Regression: words times 1.3 read 212 for a 304-token prompt, so the guard never fired.

    The reference numbers were measured with CLIP's own tokenizer. The estimator
    is fitted, so drift is a regression rather than a tuning preference.
    """
    cases = [
        (
            "(cinematic film still, anamorphic, colour graded, shallow depth of field, "
            "movie frame)1.10, (magic realism, ordinary world with one impossible detail)1.10",
            38,
        ),
        ("jpeg artifacts, compression noise, banding, moire, oversharpened halos", 15),
        ("a hand-thrown stoneware coffee cup", 8),
    ]
    for text, measured in cases:
        estimate = estimate_tokens(text)
        assert abs(estimate - measured) <= max(2, measured * 0.15), (text[:40], estimate, measured)


def test_packing_never_exceeds_one_chunk():
    """The invariant the whole guarantee rests on."""
    pieces = [f"fragment number {index} with some filler words in it" for index in range(40)]
    for group in pack(pieces, anchor="small ethereal fairies"):
        assert count_content_tokens(", ".join(group))[0] <= CHUNK_CONTENT_TOKENS


# ---------------------------------------------------------------------------
# Layer prompts, framing, and the negatives that suppressed the subject
# ---------------------------------------------------------------------------


def test_layer_prompts_are_no_longer_read_by_nothing():
    """`composition.layers[].prompt` used to vanish with no warning at all."""
    compiled = compile_spec(
        minimal(
            composition={
                "layers": [
                    {
                        "role": "fairies",
                        "shape": "blob",
                        "bbox": [0.05, 0.1, 0.3, 0.5],
                        "prompt": "three tiny winged fairies",
                        "depth": 0.8,
                    }
                ]
            }
        )
    )
    assert "three tiny winged fairies" in compiled.prompt
    assert "in the left third" in compiled.prompt
    assert "in the foreground" in compiled.prompt


def test_regional_takes_the_layer_prompts_instead_of_the_text():
    """With masking on, the same text must not also be added globally."""
    compiled = compile_spec(
        minimal(
            composition={
                "regional": {"enabled": True},
                "layers": [
                    {
                        "role": "fairies",
                        "bbox": [0.05, 0.1, 0.3, 0.5],
                        "prompt": "three tiny winged fairies",
                    }
                ],
            }
        )
    )
    assert "three tiny winged fairies" not in compiled.prompt
    assert [region["prompt"] for region in compiled.regions] == ["three tiny winged fairies"]
    assert any("composition.regional" in note for note in compiled.notes)


def test_a_body_crop_warns_when_there_is_no_body():
    """Regression: `medium` on a forest read as a macro shot of undergrowth."""
    compiled = compile_spec(
        minimal(subject={"primary": "an ancient mossy forest"}, camera={"shot": "medium"})
    )
    assert any("crops a human body" in warning for warning in compiled.warnings)


def test_an_occupation_still_counts_as_a_person():
    """Regression: the first word list called `full_body` on an alchemist a mistake."""
    for subject in ("a wandering alchemist with a lantern", "a blacksmith at the anvil"):
        compiled = compile_spec(
            minimal(subject={"primary": subject}, camera={"shot": "full_body"})
        )
        assert not any("crops a human body" in w for w in compiled.warnings), subject


def test_one_body_framing_warns_when_several_were_asked_for():
    compiled = compile_spec(
        minimal(subject={"primary": "multiple small ethereal fairies"}, camera={"shot": "medium"})
    )
    assert any("frames one body" in warning for warning in compiled.warnings)


def test_photoreal_no_longer_bans_the_rendered_look():
    """Regression: "cgi, 3d render" in the negatives deleted the fairies.

    SDXL's idea of a photorealistic fairy lives in illustrative and rendered
    space. Pushing that space away removes the subject and leaves the setting,
    which is exactly what four empty forests looked like. The plastic-skin half
    of the preset is still wanted, and is still there.
    """
    compiled = compile_spec(minimal(subject={"primary": "a small ethereal fairy"}))
    assert "3d render" not in compiled.negative_prompt
    assert "video game screenshot" not in compiled.negative_prompt
    assert "plastic skin" in compiled.negative_prompt


def test_anatomy_negatives_follow_whether_there_is_anatomy():
    """Kept for a fairy, which has hands; dropped for a cup, which does not."""
    fairy = compile_spec(minimal(subject={"primary": "a small ethereal fairy"}))
    cup = compile_spec(minimal(subject={"primary": "a stoneware coffee cup"}))
    assert "extra fingers" in fairy.negative_prompt
    assert "extra fingers" not in cup.negative_prompt
    assert any("anatomy" in note for note in cup.notes)


def test_an_explicit_anatomy_preset_beats_the_inference():
    """Explicit beats inferred: a preset the caller named is theirs to keep."""
    compiled = compile_spec(
        minimal(subject={"primary": "a stoneware coffee cup"}, negative={"presets": ["anatomy"]})
    )
    assert "extra fingers" in compiled.negative_prompt


def test_region_grids_recover_the_latent_shape():
    """Masks are resized per attention block, so a wrong grid scrambles them."""
    assert grid_for(96 * 168, 1344, 768) == (96, 168)
    assert grid_for(48 * 84, 1344, 768) == (48, 84)
    assert grid_for(24 * 42, 1344, 768) == (24, 42)
    assert grid_for(1024, 1024, 1024) == (32, 32)
    # An unrecognisable length must be declined, not guessed: that block is then
    # left unmasked, which is weaker rather than wrong.
    assert grid_for(4095, 1344, 768) is None


# ---------------------------------------------------------------------------
# Silenced library warnings
#
# Three specific messages are dropped and everything else must still print.
# The texts below are copied from the lines that emit them.
# ---------------------------------------------------------------------------


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def emitted_by(name: str):
    """What one named logger lets through, collected on that logger itself.

    Not caplog: transformers can switch propagation off on its loggers once any
    test has imported it, and a filter is only proven by a handler it guards.
    """
    logger = logging.getLogger(name)
    handler, level = _Collect(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def test_the_empty_float32_list_is_silenced_but_a_real_one_is_not():
    quiet.install()
    quiet.install()  # idempotent: a second install must not stack filters
    name = "diffusers.models.modeling_utils"
    template = (
        "There are modules in UNet2DConditionModel that should be kept in float32: {}. "
        "Casting directly with `to()` can lead to inconsistent results; set `torch_dtype` "
        "in `from_pretrained()` instead to keep these modules in float32."
    )
    with emitted_by(name) as messages:
        logging.getLogger(name).warning(template.format([]))
        logging.getLogger(name).warning(template.format(["norm_out"]))
    assert messages == [template.format(["norm_out"])]
    assert len(logging.getLogger(name).filters) == 1


def test_the_siglip_rename_is_silenced_and_nothing_else_on_that_logger():
    quiet.install()
    name = "transformers.utils.import_utils"
    with emitted_by(name) as messages:
        logging.getLogger(name).warning(
            "`Siglip2ImageProcessorFast` is deprecated. The `Fast` suffix for image processors "
            "has been removed; use `Siglip2ImageProcessor` instead."
        )
        logging.getLogger(name).warning("`OtherThingFast` is deprecated.")
    assert messages == ["`OtherThingFast` is deprecated."]


def test_the_long_sequence_warning_is_silenced_only_while_compel_encodes():
    """Outside compel the same message is true, so it must still print there."""
    name = "transformers.tokenization_utils_base"
    text = (
        "Token indices sequence length is longer than the specified maximum sequence length "
        "for this model (131 > 77). Running this sequence through the model will result in "
        "indexing errors"
    )
    logger = logging.getLogger(name)
    with emitted_by(name) as messages:
        logger.warning(text)
        with quiet.compel_tokenization():
            with quiet.compel_tokenization():  # re-entrant
                logger.warning(text)
            logger.warning(text)
        logger.warning(text)
    assert messages == [text, text]
    assert logger.filters == []
