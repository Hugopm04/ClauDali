"""Tests that need no GPU, no model weights and no network.

Everything up to the sampler is deterministic and cheap to check: the spec
contract, the prompt compiler, the procedural control maps, compositing,
postprocessing and the diagnostics. Those are also the parts most likely to
break silently -- a bad render is obvious, a subtly wrong prompt is not.

Run with:  .venv\\Scripts\\python -m pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from claudali.compiler import compile_spec, load_vocabulary
from claudali.compose import apply_overlays, apply_postprocess, quantize_to_palette
from claudali.control.maps import build_depth_map, build_edge_map, build_region_masks
from claudali.diagnostics import analyse
from claudali.spec import ASPECT_BUCKETS, Overlay, Postprocess, SceneSpec, load_spec

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
