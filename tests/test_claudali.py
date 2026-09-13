"""Tests that need no GPU, no model weights and no network.

Everything up to the sampler is deterministic and cheap to check: the spec
contract, the prompt compiler, the procedural control maps, compositing,
postprocessing and the diagnostics. Those are also the parts most likely to
break silently -- a bad render is obvious, a subtly wrong prompt is not.

Run with:  .venv\\Scripts\\python -m pytest -q
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from claudali import registry
from claudali.__main__ import pause_on_interrupt
from claudali.bundle import BundleWriter, read_bundle
from claudali.compiler import CompiledPrompt, _assemble, compile_spec, load_vocabulary
from claudali.compose import apply_overlays, apply_postprocess, quantize_to_palette
from claudali.control.maps import build_depth_map, build_edge_map, build_region_masks
from claudali.diagnostics import analyse
from claudali.engine import quiet
from claudali.engine.checkpoint import (
    RenderController,
    RenderPaused,
    ResumeMismatch,
    ResumeState,
    StepPaused,
    StepState,
    capture_scheduler_state,
    check_fingerprint,
    load_checkpoint,
    restore_scheduler_state,
    save_checkpoint,
)
from claudali.engine.decode import _run_on_cpu, cpu_need_gb, plan_vae_decode
from claudali.engine.pipelines import _plan_cudnn, _plan_offload, _should_upcast_vae
from claudali.engine.render import RenderedImage, RenderResult
from claudali.jobs import JobQueue, JobStatus, QueueFull
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


def inpaint_layer_spec(tmp_path: Path, **init) -> SceneSpec:
    """The layered spec without control, inpainting over a source image on disk."""
    source = tmp_path / "source.png"
    Image.new("RGB", (1216, 832), (90, 110, 130)).save(source)
    data = layered_spec().model_dump(mode="json", exclude_unset=True)
    data.pop("control")
    data["init"] = {"image": str(source), **init}
    return load_spec(data)


def test_init_mask_layer_inpaints_the_named_layer(tmp_path):
    """`build_region_masks` promised this for months and was read by nothing but a test."""
    from claudali.engine.render import _load_init_images, _select_task

    spec = inpaint_layer_spec(tmp_path, mask_layer="tower", mask_blur=0)
    assert _select_task(spec) == "inpaint"
    image, mask = _load_init_images(spec)
    assert image.size == mask.size == spec.resolution()
    assert np.array_equal(np.asarray(mask), np.asarray(build_region_masks(spec)["tower"]))
    # The layers are what the mask is made of, so they are not unused here.
    assert not any("composition.layers" in warning for warning in compile_spec(spec).warnings)


def test_init_mask_layer_must_name_exactly_one_layer(tmp_path):
    with pytest.raises(Exception, match="names no layer"):
        inpaint_layer_spec(tmp_path, mask_layer="lighthouse")
    with pytest.raises(Exception, match="alternatives"):
        inpaint_layer_spec(tmp_path, mask_layer="tower", mask=str(tmp_path / "source.png"))

    data = inpaint_layer_spec(tmp_path, mask_layer="rock").model_dump(mode="json", exclude_unset=True)
    data["composition"]["layers"][0]["role"] = "rock"
    with pytest.raises(Exception, match="matches 2 layers"):
        load_spec(data)


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
    assert status.warnings == (), "a workaround that worked is a note, not a warning"


def test_cudnn_is_restored_when_disabling_it_does_not_help():
    """Paying for a fallback that fixes nothing would be worse than saying so."""
    status, switches = _cudnn_plan("auto", [True, True])
    assert (status.broken, status.disabled, status.survives_workaround) == (True, False, True)
    assert switches == [False, True]
    assert any("left on" in warning for warning in status.warnings) and status.notes == ()


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


def test_plain_prompts_say_how_much_was_truncated():
    """Without compel, diffusers keeps 77 tokens of each prompt and silently drops the rest."""
    from claudali.engine.render import _encode_prompts

    no_compel = SimpleNamespace(compel=None)
    long = compile_spec(fairy_spec())
    kwargs, warnings = _encode_prompts(no_compel, long)
    assert kwargs == {"prompt": long.prompt, "negative_prompt": long.negative_prompt}
    assert any("truncated" in warning and str(long.tokens.tokens) in warning for warning in warnings)

    one_chunk = {**long.tokens.to_dict(), "chunks": 1}
    short = CompiledPrompt.from_dict({**long.to_dict(), "tokens": one_chunk, "negative_tokens": one_chunk})
    assert _encode_prompts(no_compel, short)[1] == []


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


# ---------------------------------------------------------------------------
# Pausing and resuming
# ---------------------------------------------------------------------------


def test_the_server_modules_do_not_import_torch():
    """CLAUDE.md's import-cost rule, which a render refactor breaks most easily."""
    code = "import sys, claudali.api, claudali.jobs, claudali.bundle; print('torch' in sys.modules)"
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert run.stdout.strip() == "False", run.stderr[-2000:]


def test_a_compiled_prompt_survives_the_trip_through_state_json():
    """A resume uses the prompt frozen when the job started, so it must come back whole."""
    compiled = compile_spec(fairy_spec())
    frozen = json.loads(json.dumps(compiled.to_dict()))
    assert CompiledPrompt.from_dict(frozen).to_dict() == compiled.to_dict()


def test_a_changed_environment_refuses_to_resume_unless_forced():
    saved = {"diffusers": "0.40.0", "cudnn_enabled": False, "steps": 32}
    assert check_fingerprint(saved, dict(saved), force=False) == []

    current = {**saved, "diffusers": "0.41.0", "cudnn_enabled": True}
    with pytest.raises(ResumeMismatch) as caught:
        check_fingerprint(saved, current, force=False)
    assert caught.value.differences == [
        "cudnn_enabled: False -> True",
        "diffusers: '0.40.0' -> '0.41.0'",
    ]

    notes = check_fingerprint(saved, current, force=True)
    assert len(notes) == 1 and "diffusers: '0.40.0' -> '0.41.0'" in notes[0]


def test_scheduler_state_keeps_what_steps_and_copies_it():
    torch = pytest.importorskip("torch")

    class FakeScheduler:
        def __init__(self) -> None:
            self.config = {"solver_order": 2}
            self._internal_dict = {"solver_order": 2}
            self._step_index = 3
            self.lower_order_nums = 2
            self.model_outputs = [None, torch.arange(4.0)]
            self.solver = object()

    source = FakeScheduler()
    state = capture_scheduler_state(source)
    assert set(state) == {"_step_index", "lower_order_nums", "model_outputs"}

    source.model_outputs[1].add_(100)  # a later step must not reach into the snapshot
    target = FakeScheduler()
    target._step_index, target.model_outputs = None, [None, None]
    restore_scheduler_state(target, state)
    assert target._step_index == 3
    assert torch.equal(target.model_outputs[1], torch.arange(4.0))


def test_a_checkpoint_round_trips_through_a_weights_only_load(tmp_path):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    torch.randn(3, generator=generator)
    compiled = compile_spec(minimal()).to_dict()
    state = ResumeState(
        seeds=[11, 12],
        variation=1,
        completed=[0],
        compiled=compiled,
        fingerprint={"torch": "x"},
        reason="pause",
        step=StepState(
            next_step=2,
            latents=torch.randn(1, 4, 8, 8).half(),
            scheduler={"_step_index": 2, "model_outputs": [None, {"tensor": torch.ones(2), "device": "cpu"}]},
            generator=generator.get_state(),
        ),
    )
    save_checkpoint(tmp_path, state, job_id="abc")

    meta = json.loads((tmp_path / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    assert (meta["has_step_state"], meta["next_step"], meta["job_id"]) == (True, 2, "abc")

    loaded = load_checkpoint(tmp_path)  # torch.load(weights_only=True) inside
    assert (loaded.seeds, loaded.completed, loaded.next_step) == ([11, 12], [0], 2)
    assert loaded.step.latents.dtype == torch.float16
    assert torch.equal(loaded.step.latents, state.step.latents)
    assert torch.equal(loaded.step.generator, state.step.generator)

    # A variation-boundary checkpoint written later must not leave stale tensors trusted.
    save_checkpoint(
        tmp_path, ResumeState(seeds=[11, 12], variation=1, completed=[0], compiled=compiled, fingerprint={})
    )
    assert not (tmp_path / "checkpoint" / "state.pt").exists()
    assert load_checkpoint(tmp_path).step is None


def _tiny_sdxl(sampler: str):
    """A random SDXL pipeline small enough to run on the CPU in well under a second.

    Prompt embeddings are passed in directly, so no tokenizer, text encoder,
    weights or network is needed. The time-embedding input is 6 * 8 + 32.
    """
    diffusers = pytest.importorskip("diffusers")
    from claudali.engine.pipelines import _build_scheduler

    unet, vae = _tiny_unet_and_vae(0)
    pipe = diffusers.StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        unet=unet,
        scheduler=diffusers.EulerDiscreteScheduler(),
        add_watermarker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    assert _build_scheduler(pipe, sampler) == []
    return pipe


def _tiny_unet_and_vae(seed: int):
    """A tiny SDXL UNet and VAE, random but the same for the same ``seed``."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")

    torch.manual_seed(seed)
    unet = diffusers.UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=2,
        sample_size=16,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,
        cross_attention_dim=64,
        norm_num_groups=1,
    )
    vae = diffusers.AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 2,
        up_block_types=["UpDecoderBlock2D"] * 2,
        latent_channels=4,
    )
    return unet, vae


def _tiny_call() -> dict:
    import torch

    embeds = torch.Generator().manual_seed(123)
    return {
        "prompt_embeds": torch.randn(1, 8, 64, generator=embeds),
        "pooled_prompt_embeds": torch.randn(1, 32, generator=embeds),
        "negative_prompt_embeds": torch.randn(1, 8, 64, generator=embeds),
        "negative_pooled_prompt_embeds": torch.randn(1, 32, generator=embeds),
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "height": 32,
        "width": 32,
    }


@pytest.mark.parametrize(
    "sampler, pause_after",
    [("dpmpp_2m_karras", 2), ("euler_a", 2), ("dpmpp_2m_karras", 5)],
    ids=["multistep-history", "ancestral-noise", "after-the-last-step"],
)
def test_a_paused_render_resumes_bit_for_bit(tmp_path, sampler, pause_after):
    """The whole promise of pause and resume, on the production denoise helper.

    dpmpp_2m_karras carries a model-output history from step to step; euler_a
    draws fresh noise from the generator at every step. Both must continue
    exactly, and so must a pause after the last step. The one pipeline serves
    every call, as the cached one does in the server, so a hook left installed
    would show up as a wrong final run.

    This also locks the diffusers loop shape the resume relies on: if an upgrade
    moves `_interrupt` or `prepare_latents`, it fails here. Fix the hooks in
    engine/checkpoint.py; do not loosen this.
    """
    torch = pytest.importorskip("torch")
    from claudali.engine.render import denoise

    pipe = _tiny_sdxl(sampler)
    reference = denoise(pipe, _tiny_call(), seed=42)

    controller = RenderController()

    def on_step(step: int) -> None:
        if step == pause_after:
            controller.pause_requested = True

    with pytest.raises(StepPaused) as caught:
        denoise(pipe, _tiny_call(), seed=42, controller=controller, on_step=on_step)
    assert caught.value.step.next_step == pause_after

    # Through the disk, as a real pause goes.
    save_checkpoint(
        tmp_path,
        ResumeState(seeds=[42], variation=0, completed=[], compiled={}, fingerprint={}, step=caught.value.step),
    )

    # Equal latents alone would also come from quietly re-running every step with
    # the same seed, so count the UNet calls: only the steps after the pause may run.
    unet_calls: list[int] = []
    hook = pipe.unet.register_forward_pre_hook(lambda _module, _args: unet_calls.append(1))
    try:
        resumed = denoise(pipe, _tiny_call(), seed=42, resume=load_checkpoint(tmp_path).step)
    finally:
        hook.remove()
    assert len(unet_calls) == 5 - pause_after
    assert torch.equal(resumed, reference)
    assert torch.equal(denoise(pipe, _tiny_call(), seed=42), reference)


def test_every_sampler_builds_with_the_installed_packages():
    """Regression: `lms` needed scipy, which no requirement installs, and failed at load."""
    diffusers = pytest.importorskip("diffusers")
    from claudali.engine.pipelines import SAMPLERS

    base = diffusers.EulerDiscreteScheduler()
    for class_name, kwargs in SAMPLERS.values():
        getattr(diffusers, class_name).from_config(base.config, **kwargs)


def test_a_sampler_is_built_from_the_checkpoint_scheduler_not_the_last_one():
    """Regression: `euler` after `dpmpp_2m_karras` on the cached pipeline inherited Karras sigmas."""
    diffusers = pytest.importorskip("diffusers")
    from claudali.engine.pipelines import _build_scheduler

    base = diffusers.EulerDiscreteScheduler()
    pipe = SimpleNamespace(scheduler=base)
    assert _build_scheduler(pipe, "dpmpp_2m_karras", base) == []
    assert pipe.scheduler.config.use_karras_sigmas
    assert _build_scheduler(pipe, "euler", base) == []
    assert not pipe.scheduler.config.use_karras_sigmas

    warnings = _build_scheduler(pipe, "no_such_sampler", base)
    assert len(warnings) == 1 and "unknown sampler" in warnings[0]
    assert type(pipe.scheduler) is type(base) and pipe.scheduler is not base


def test_first_ctrl_c_pauses_and_the_second_aborts():
    controller = RenderController()
    previous = signal.getsignal(signal.SIGINT)
    try:
        handler = pause_on_interrupt(controller, out=io.StringIO())
        signal.signal(signal.SIGINT, handler)
        handler(signal.SIGINT, None)
        assert controller.pause_requested
        second = signal.getsignal(signal.SIGINT)
        assert second is signal.default_int_handler
        with pytest.raises(KeyboardInterrupt):
            second(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGINT, previous)


def _queue(**options) -> JobQueue:
    return JobQueue(**{"max_size": 8, "retention": 50, "autostart": False, **options})


def test_a_resumed_job_goes_to_the_front_of_the_queue():
    queue = _queue()
    first, _second, third = (queue.submit(minimal()) for _ in range(3))
    queue.pause(third.id)
    assert third.status is JobStatus.PAUSED
    queue.resume(third.id)
    assert queue._take_next(timeout=0) is third
    assert queue._take_next(timeout=0) is first


def test_pause_and_hold_keeps_the_next_job_waiting_until_that_job_resumes():
    queue = _queue()
    running, waiting = queue.submit(minimal()), queue.submit(minimal())
    assert queue._take_next(timeout=0) is running

    queue.pause(running.id, hold_queue=True)
    assert running.status is JobStatus.PAUSING and running.controller.pause_requested
    running.status = JobStatus.PAUSED  # what the worker does once the checkpoint lands
    queue._settle(running)
    assert queue._take_next(timeout=0) is None
    assert queue.stats()["held"]

    queue.resume(running.id)
    assert not queue.stats()["held"]
    assert queue._take_next(timeout=0) is running
    assert queue._take_next(timeout=0) is waiting


def test_a_plain_pause_lets_the_next_job_run():
    queue = _queue()
    running, waiting = queue.submit(minimal()), queue.submit(minimal())
    queue._take_next(timeout=0)
    queue.pause(running.id)
    running.status = JobStatus.PAUSED
    queue._settle(running)
    assert queue._take_next(timeout=0) is waiting


def test_releasing_the_queue_does_not_resume_the_job_that_held_it():
    queue = _queue()
    held, waiting = queue.submit(minimal()), queue.submit(minimal())
    queue.pause(held.id, hold_queue=True)
    assert queue._take_next(timeout=0) is None
    queue.release()
    assert queue._take_next(timeout=0) is waiting
    assert held.status is JobStatus.PAUSED


def test_a_resume_refused_for_a_mismatch_can_be_forced():
    queue = _queue()
    job = queue.submit(minimal())
    queue._take_next(timeout=0)
    job.status, job.error_type, job.mismatch = JobStatus.ERROR, "resume_mismatch", ["gpu: 'a' -> 'b'"]
    queue._settle(job)
    queue.resume(job.id, force=True)
    assert (job.status, job.force_resume, job.mismatch) == (JobStatus.QUEUED, True, [])
    assert queue._take_next(timeout=0) is job


def test_the_queue_comes_back_held_after_a_restart(tmp_path):
    queue = _queue()
    paused = queue.submit(minimal())
    waiting = queue.submit(minimal(subject={"primary": "a cast iron teapot"}))
    queue.pause(paused.id)
    path = tmp_path / "queue.json"
    assert queue.save(path) == 2

    restarted = _queue()
    assert restarted.restore(path) == 2
    assert not path.exists(), "a file left behind would restore the same jobs twice"
    assert restarted.get(paused.id).status is JobStatus.PAUSED
    assert restarted.get(waiting.id).status is JobStatus.QUEUED
    assert restarted.stats()["held"] and restarted._take_next(timeout=0) is None

    restarted.release()
    assert restarted._take_next(timeout=0).spec.subject.primary == "a cast iron teapot"


def test_a_full_queue_says_so_instead_of_blocking():
    queue = _queue(max_size=1)
    queue.submit(minimal())
    with pytest.raises(QueueFull):
        queue.submit(minimal())


def test_a_bundle_is_written_as_each_image_lands(tmp_path):
    spec = minimal(render={"variations": 3, "seed": 5})
    compiled = compile_spec(spec)
    writer = BundleWriter.create(spec, compiled, "job12345", tmp_path / "bundle")
    manifest = read_bundle(writer.directory)
    assert (manifest["status"], manifest["variations"]) == ("running", [])

    boundary = ResumeState(seeds=[5, 6, 7], variation=0, completed=[], compiled=compiled.to_dict(), fingerprint={})
    writer.record_checkpoint(boundary)
    for index in (0, 1):
        writer.add_variation(RenderedImage(image=swatch(color=(40 * index, 90, 160)), seed=5 + index, index=index))
        assert len(read_bundle(writer.directory)["variations"]) == index + 1
    assert Path(read_bundle(writer.directory)["contact_sheet"]).is_file()

    state = ResumeState(seeds=[5, 6, 7], variation=2, completed=[0, 1], compiled=compiled.to_dict(), fingerprint={})
    writer.pause(
        RenderPaused(
            state,
            RenderResult(images=[], compiled=compiled, warnings=["a warning"], notes=["a note"], duration_s=12.5),
        )
    )
    manifest = read_bundle(writer.directory)
    assert manifest["status"] == "paused"
    assert manifest["checkpoint"]["variation"] == 2 and manifest["checkpoint"]["reason"] == "pause"

    reopened = BundleWriter.open(writer.directory)
    assert reopened.completed_indices() == [0, 1]
    reopened.add_variation(RenderedImage(image=swatch(), seed=7, index=2))
    reopened.complete(
        RenderResult(images=[], compiled=compiled, warnings=["a warning"], notes=["a note"], duration_s=3.0)
    )
    manifest = read_bundle(writer.directory)
    assert (manifest["status"], manifest["duration_s"], len(manifest["variations"])) == ("done", 15.5, 3)
    # Two channels, each holding one copy of what both sessions of the job said.
    assert (manifest["warnings"], manifest["notes"]) == (["a warning"], ["a note"])
    assert not (writer.directory / "checkpoint").exists()


def test_a_saved_spec_loads_back_with_what_was_inferred(tmp_path):
    """Regression: saved specs wrote every schema default out as if it had been chosen.

    Loaded back -- from history in the UI, or from runs/queue.json after a restart --
    `steps: 30` and `cfg: 6.5` then beat the painterly intent's 34 and 7.5, and the
    saved 1024x1024 beat whatever aspect was chosen afterwards.
    """
    spec = minimal(intent="painterly", render={"variations": 2, "seed": 9})
    compiled = compile_spec(spec)
    writer = BundleWriter.create(spec, compiled, "job12345", tmp_path / "bundle")
    saved = json.loads((writer.directory / "spec.json").read_text(encoding="utf-8"))
    again = compile_spec(load_spec(saved))
    assert (again.steps, again.cfg, again.sampler) == (compiled.steps, compiled.cfg, compiled.sampler)
    assert (again.steps, again.cfg) == (34, 7.5)

    saved["composition"] = {"aspect": "16:9"}
    assert load_spec(saved).resolution() == (1344, 768)

    queue = _queue()
    queue.submit(spec)
    queue.save(tmp_path / "queue.json")
    restarted = _queue()
    restarted.restore(tmp_path / "queue.json")
    assert compile_spec(restarted.list()[0].spec).steps == 34


# ---------------------------------------------------------------------------
# Quality options: the VAE decode policy, precision, the refiner and the hi-res pass
# ---------------------------------------------------------------------------


@pytest.fixture
def quality_models(monkeypatch):
    """Pretend the refiner and Real-ESRGAN are installed, or not, without touching models/."""
    state = {"refiner": True, "upscaler": True}
    monkeypatch.setattr(
        registry,
        "refiner_problem",
        lambda model_id: None if state["refiner"] else f"the refiner '{model_id}' is not installed",
    )
    monkeypatch.setattr(
        registry,
        "upscaler_problem",
        lambda model_id="realesrgan-x4": None if state["upscaler"] else "it is not installed",
    )
    return state


def test_auto_decode_goes_to_the_cpu_only_with_the_ram_for_it():
    """The agreed default: CPU fp32 untiled when free RAM covers it, GPU tiled otherwise."""
    need = cpu_need_gb(1344, 768)
    assert need == round(5.6 * 1344 * 768 / 1e6 + 1.0, 1)

    roomy = plan_vae_decode("auto", 1344, 768, free_gb=need + 0.5)
    assert (roomy.device, roomy.tiled) == ("cpu", False)

    # 1.3 GB is what was free on the target laptop after a pipeline load and a render.
    tight = plan_vae_decode("auto", 1344, 768, free_gb=1.3)
    assert (tight.device, tight.tiled) == ("gpu", True)
    assert tight.warnings == [], "a fallback auto chose is a note, not a warning"
    assert any("GPU with tiling" in note for note in tight.notes)


def test_unmeasured_ram_never_picks_the_cpu_decode():
    """An unknown must not choose the path that can run the machine out of memory."""
    plan = plan_vae_decode("auto", 1024, 1024, free_gb=None)
    assert (plan.device, plan.tiled) == ("gpu", True)
    assert any("could not be measured" in note for note in plan.notes)


def test_forced_decode_modes_ignore_the_ram():
    forced = plan_vae_decode("cpu", 1344, 768, free_gb=0.5)
    assert (forced.device, forced.tiled) == ("cpu", False)
    assert any("paged" in note for note in forced.notes)
    assert (plan_vae_decode("gpu", 1344, 768, 99.0).device, plan_vae_decode("gpu", 1344, 768, 99.0).tiled) == ("gpu", False)
    assert plan_vae_decode("gpu_tiled", 1344, 768, 99.0).tiled is True

    odd = plan_vae_decode("fastest", 1024, 1024, free_gb=99.0)
    assert odd.mode == "auto" and odd.device == "cpu" and len(odd.warnings) == 1


def test_a_cpu_decode_that_runs_out_of_ram_retries_tiled_and_says_so():
    """Forced, the fallback is a warning; chosen by auto, a note. Both keep fp32 on the CPU."""

    class FakeVae:
        use_tiling = False

        def enable_tiling(self):
            self.use_tiling = True

        def disable_tiling(self):
            self.use_tiling = False

    for mode, channel in (("cpu", "warnings"), ("auto", "notes")):
        vae = FakeVae()

        def run(vae=vae):
            if not vae.use_tiling:
                raise RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 9 GB")
            return "decoded"

        plan = plan_vae_decode(mode, 1344, 768, free_gb=99.0)
        warnings, notes = [], []
        assert _run_on_cpu(run, vae, plan, "decode", "1344x768", warnings, notes) == ("decoded", True)
        said = warnings if channel == "warnings" else notes
        assert any("ran out of RAM" in message for message in said), mode
        assert vae.use_tiling is False, "the CPU VAE is left untiled for the next image"

    with pytest.raises(ValueError):
        _run_on_cpu(lambda: (_ for _ in ()).throw(ValueError("not memory")), FakeVae(),
                    plan_vae_decode("cpu", 64, 64, 1.0), "decode", "64x64", [], [])


def test_latents_waiting_on_the_cpu_decode_on_the_gpu_path():
    """Regression: staged latents reached vae.decode on the CPU and in the wrong dtype.

    The hi-res pass decodes latents the base stage left on the CPU. Under offload
    the hook moves the VAE, not its input, so the GPU path must move them itself.
    Found on the GPU as "Input type (float) and bias type (Half)"; a float64 VAE
    reproduces the dtype half of it on the CPU.
    """
    torch = pytest.importorskip("torch")
    from diffusers.image_processor import VaeImageProcessor

    from claudali.engine.decode import decode

    _unet, vae = _tiny_unet_and_vae(0)
    vae = vae.double()
    pipe = SimpleNamespace(
        vae=vae, image_processor=VaeImageProcessor(vae_scale_factor=2), vae_scale_factor=2,
        watermark=None, _execution_device=torch.device("cpu"),
    )
    latents = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(0))  # float32
    image, record, _warnings, _notes = decode(pipe, latents, plan_vae_decode("gpu", 32, 32, None), lambda: None)
    assert image.size == (32, 32)
    assert (record["device"], record["precision"], record["tiled"]) == ("gpu", "float64", False)


def test_fp32_on_a_small_card_switches_to_sequential_offload():
    mode, warnings, notes = _plan_offload("model", "float32", 6.0)
    assert mode == "sequential" and warnings == []
    assert any("sequential CPU offload" in note and "measured" in note for note in notes)
    assert _plan_offload("model", "float16", 6.0) == ("model", [], [])
    assert _plan_offload("model", "float32", 24.0)[0] == "model"

    mode, warnings, _notes = _plan_offload("fast", "float16", 6.0)
    assert mode == "model" and len(warnings) == 1


def test_the_max_preset_turns_on_every_quality_option(quality_models):
    compiled = compile_spec(minimal(render={"quality": "max", "model": "sdxl-base"}))
    assert (compiled.precision, compiled.vae_decode) == ("float32", "cpu")
    assert compiled.refiner == {
        "model": "sdxl-refiner", "handoff": 0.8, "aesthetic_score": 6.0, "negative_aesthetic_score": 2.5,
    }
    assert compiled.hires["upscaler"] == "realesrgan-x4"
    assert (compiled.hires["width"], compiled.hires["height"]) == (1536, 1536)
    assert compiled.stages == ["base", "refiner", "hires"]
    assert not any("refiner" in warning or "hi-res" in warning for warning in compiled.warnings)
    assert any("render.quality 'max' turned on" in note for note in compiled.notes)

    # A preset for quality, not for sampling.
    standard = compile_spec(minimal(render={"model": "sdxl-base"}))
    assert (compiled.steps, compiled.cfg, compiled.sampler) == (standard.steps, standard.cfg, standard.sampler)


def test_explicit_quality_fields_beat_the_max_preset(quality_models):
    compiled = compile_spec(
        minimal(
            render={"quality": "max", "model": "sdxl-base", "precision": "fp16", "vae_decode": "gpu_tiled"},
            refiner={"enabled": False},
            hires={"upscaler": "lanczos", "scale": 1.25},
        )
    )
    assert (compiled.precision, compiled.vae_decode, compiled.refiner) == ("float16", "gpu_tiled", None)
    assert compiled.hires["upscaler"] == "lanczos" and compiled.hires["width"] == 1280
    turned_on = next(note for note in compiled.notes if "turned on" in note)
    assert "hi-res" in turned_on and "fp32" not in turned_on and "refiner" not in turned_on


def test_the_max_preset_says_what_it_could_not_turn_on(quality_models, tmp_path):
    quality_models.update(refiner=False, upscaler=False)
    compiled = compile_spec(minimal(render={"quality": "max", "model": "sdxl-base"}))
    assert compiled.refiner is None and compiled.hires["upscaler"] == "lanczos"
    assert any("would add the refiner" in warning for warning in compiled.warnings)
    assert any("Lanczos" in note for note in compiled.notes)

    quality_models["refiner"] = True
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(source)
    with_init = compile_spec(minimal(render={"quality": "max"}, init={"image": str(source)}))
    assert with_init.refiner is None
    assert any("left the refiner off" in note for note in with_init.notes), "a decision, so a note"


def test_a_refiner_that_cannot_run_is_refused_not_dropped(quality_models, tmp_path):
    """Explicit beats inferred, and nothing is silently dropped: the render stops before loading."""
    from claudali.engine.render import render

    quality_models["refiner"] = False
    missing = minimal(render={"model": "sdxl-base"}, refiner={"enabled": True})
    compiled = compile_spec(missing)
    assert compiled.refiner is not None
    assert any("rendering this spec stops" in warning for warning in compiled.warnings)
    with pytest.raises(FileNotFoundError, match="not installed"):
        render(missing)

    quality_models["refiner"] = True
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(source)
    with pytest.raises(ValueError, match="init images"):
        render(minimal(render={"model": "sdxl-base"}, refiner={"enabled": True}, init={"image": str(source)}))
    with pytest.raises(ValueError, match="0 steps"):
        render(minimal(hires={"enabled": True, "steps": 2, "strength": 0.2}))


def test_standard_quality_follows_the_server_settings(monkeypatch):
    from claudali.config import SETTINGS

    monkeypatch.setattr(SETTINGS, "vae_decode", "gpu")
    compiled = compile_spec(minimal())
    assert (compiled.quality, compiled.vae_decode, compiled.refiner, compiled.hires) == ("standard", "gpu", None, None)
    assert compiled.stages == ["base"]
    assert CompiledPrompt.from_dict(json.loads(json.dumps(compiled.to_dict()))).to_dict() == compiled.to_dict()

    monkeypatch.setattr(SETTINGS, "vae_decode", "sideways")
    assert any("CLAUDALI_VAE_DECODE" in warning for warning in compile_spec(minimal()).warnings)

    # A prompt frozen before these options existed decoded on the GPU tiled, and resumes so.
    old = {key: value for key, value in compiled.to_dict().items() if key not in {"quality", "precision", "vae_decode", "refiner", "hires", "stages"}}
    assert CompiledPrompt.from_dict(old).vae_decode == "gpu_tiled"


def test_a_saved_max_spec_keeps_only_what_was_set():
    """The preset resolves in the compiler, so a saved spec does not come back with it spelled out."""
    saved = minimal(render={"quality": "max"}).model_dump(mode="json", exclude_unset=True)
    assert saved["render"] == {"quality": "max"}
    assert "refiner" not in saved and "hires" not in saved


def test_quality_models_are_optional_and_never_a_base_model():
    for profile in registry.PROFILES.values():
        assert not set(profile) & set(registry.QUALITY_MODELS)
    assert {registry.get(model_id).kind for model_id in registry.QUALITY_MODELS} == {"refiner", "upscaler"}
    with pytest.raises(ValueError, match="not a base checkpoint"):
        registry.resolve_checkpoint("sdxl-refiner")


def test_upscaler_tiles_cover_the_image_and_blend_back_whole():
    torch = pytest.importorskip("torch")
    from claudali.engine.upscale import OVERLAP, TILE, _ramp, _starts

    for size in (100, 256, 257, 1344):
        starts = _starts(size, TILE, OVERLAP)
        assert starts[0] == 0 and min(size, starts[-1] + TILE) == size
        assert all(later - earlier <= TILE - OVERLAP for earlier, later in zip(starts, starts[1:]))

    # Constant tiles blended by their ramps must give the constant back everywhere.
    height, width = 300, 530
    output, weight = torch.zeros(1, 1, height, width), torch.zeros(1, 1, height, width)
    for top in _starts(height, TILE, OVERLAP):
        for left in _starts(width, TILE, OVERLAP):
            rows, columns = min(TILE, height - top), min(TILE, width - left)
            ramp = _ramp(rows, columns, OVERLAP)
            assert ramp.min() > 0
            output[:, :, top : top + rows, left : left + columns] += 0.7 * ramp
            weight[:, :, top : top + rows, left : left + columns] += ramp
    assert torch.allclose(output / weight, torch.full_like(output, 0.7))


def test_latents_waiting_between_stages_survive_the_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    compiled = compile_spec(minimal()).to_dict()
    waiting = {0: ("refiner", torch.randn(1, 4, 8, 8).half()), 1: ("base", torch.randn(1, 4, 8, 8).half())}
    save_checkpoint(
        tmp_path,
        ResumeState(
            seeds=[1, 2, 3], variation=2, completed=[], compiled=compiled, fingerprint={},
            stage="base", staged=waiting, stage_steps=26,
        ),
    )
    meta = json.loads((tmp_path / "checkpoint" / "state.json").read_text(encoding="utf-8"))
    assert meta["format"] == 2 and meta["has_step_state"] is False
    assert meta["staged"] == [{"variation": 0, "stage": "refiner"}, {"variation": 1, "stage": "base"}]

    loaded = load_checkpoint(tmp_path)  # torch.load(weights_only=True) inside
    assert (loaded.step, loaded.stage_steps, sorted(loaded.staged)) == (None, 26, [0, 1])
    for index, (stage, latents) in waiting.items():
        assert loaded.staged[index][0] == stage and torch.equal(loaded.staged[index][1], latents)

    # A format-1 checkpoint, from before quality stages, still reads.
    save_checkpoint(tmp_path, ResumeState(seeds=[1], variation=0, completed=[], compiled=compiled, fingerprint={}))
    path = tmp_path / "checkpoint" / "state.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    old["format"] = 1
    for key in ("staged", "stage_steps"):
        old.pop(key)
    path.write_text(json.dumps(old), encoding="utf-8")
    assert load_checkpoint(tmp_path).staged == {}


def _tiny_loaded(kind: str, seed: int):
    """A tiny base or refiner pipeline, wrapped as the loader would return it."""
    diffusers = pytest.importorskip("diffusers")
    from claudali.engine.pipelines import LoadedPipeline

    unet, vae = _tiny_unet_and_vae(seed)
    components = dict(
        vae=vae, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
        unet=unet, scheduler=diffusers.EulerDiscreteScheduler(), add_watermarker=False,
    )
    if kind == "base":
        pipe = diffusers.StableDiffusionXLPipeline(**components)
    else:
        # The real refiner asks for aesthetic scores; the tiny UNet's time embedding has no room for them.
        pipe = diffusers.StableDiffusionXLImg2ImgPipeline(**components, requires_aesthetics_score=False)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()  # as _apply_memory_strategy does for every load
    return LoadedPipeline(
        model_id=f"tiny-{kind}", controlnet_id=None, pipe=pipe, base_scheduler=pipe.scheduler, kind=kind
    )


def test_a_render_paused_between_stages_resumes_bit_for_bit(tmp_path, monkeypatch, quality_models):
    """The refiner and hi-res stages, paused inside the refiner and resumed from disk.

    render() itself runs, with tiny random pipelines standing in for the loaded
    models. So what this locks is the stage loop: every base stage before the
    refiner loads, latents carried between stages through the checkpoint, the
    decode, upscale and encode around the hi-res pass, and a resume that runs
    nothing twice. The refiner has never run on hardware; this is its only lock.
    """
    torch = pytest.importorskip("torch")
    import dataclasses

    from claudali.engine import render as engine

    base, refiner = _tiny_loaded("base", 0), _tiny_loaded("refiner", 1)
    call = _tiny_call()
    embeds = {key: call[key] for key in call if key.endswith("embeds")}
    monkeypatch.setattr(engine, "load_pipeline", lambda *args, **kwargs: base)
    monkeypatch.setattr(engine, "load_refiner", lambda *args, **kwargs: refiner)
    monkeypatch.setattr(engine, "_encode_prompts", lambda loaded, compiled: (dict(embeds), []))
    monkeypatch.setattr(engine, "_encode_refiner_prompts", lambda loaded, compiled: (dict(embeds), []))
    monkeypatch.setattr(engine, "_fingerprint", lambda *args, **kwargs: {"pipelines": "tiny"})
    monkeypatch.setattr(engine, "device_report", lambda: {})
    monkeypatch.setattr(engine, "resident", lambda: None)

    spec = minimal(
        render={"model": "sdxl-base", "seed": 11, "variations": 2, "steps": 4, "sampler": "euler", "vae_decode": "gpu"},
        refiner={"enabled": True, "handoff": 0.5},
        hires={"enabled": True, "strength": 0.5},
    )
    # The tiny UNet works at 16x16 latents, far below the smallest size a spec allows.
    compiled = compile_spec(spec)
    compiled = dataclasses.replace(compiled, width=32, height=32, hires={**compiled.hires, "width": 48, "height": 48})
    assert compiled.stages == ["base", "refiner", "hires"]

    def run(**options):
        images = {}
        result = engine.render(
            spec, compiled=compiled,
            on_variation=lambda rendered: images.__setitem__(rendered.index, np.asarray(rendered.image)),
            **options,
        )
        return images, result

    seen = []
    reference, _result = run(progress=lambda update: seen.append((update.stage, update.variation, update.done, update.total)))
    assert sorted(reference) == [0, 1] and reference[0].shape == (48, 48, 3)
    # Batched: both base stages, then both refiner stages, then both hi-res passes.
    order = [(stage, variation) for stage, variation, _done, _total in seen]
    assert list(dict.fromkeys(order)) == [
        ("base", 0), ("base", 1), ("refiner", 0), ("refiner", 1), ("hires", 0), ("hires", 1),
    ]
    done = [entry[2] for entry in seen]
    assert done == sorted(done) and done[-1] == seen[-1][3], "progress only rises, and ends full"

    controller = RenderController()

    def pause_in_refiner(update):
        if (update.stage, update.variation, update.step) == ("refiner", 1, 1):
            controller.pause_requested = True

    with pytest.raises(RenderPaused) as caught:
        run(controller=controller, progress=pause_in_refiner)
    state = caught.value.state
    assert (state.stage, state.variation, state.next_step) == ("refiner", 1, 1)
    assert {index: stage for index, (stage, _latents) in state.staged.items()} == {0: "refiner", 1: "base"}
    save_checkpoint(tmp_path, state)

    calls = {"base": 0, "refiner": 0}
    hooks = [
        pipe.unet.register_forward_pre_hook(lambda _m, _a, name=name: calls.__setitem__(name, calls[name] + 1))
        for name, pipe in (("base", base.pipe), ("refiner", refiner.pipe))
    ]
    try:
        resumed, result = run(resume=load_checkpoint(tmp_path))
    finally:
        for hook in hooks:
            hook.remove()

    # euler over 4 steps visits timesteps 999, 666, 333, 0. The handoff at 0.5 gives
    # the base 2 and the refiner 2; the hi-res pass at strength 0.5 runs 2.
    assert calls == {"refiner": 1, "base": 2 * 2}, "a resume must not re-run finished stages"
    for index in (0, 1):
        assert np.array_equal(resumed[index], reference[index]), index
    assert any("refiner stage of variation 2 of 2 at step 1" in note for note in result.notes)
