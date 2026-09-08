"""Procedural ControlNet hints built from the scene spec.

This module is the answer to the obvious objection about prompt-driven image
generation: that composition is a suggestion the model may ignore. A caller
describes the frame as a stack of :class:`~claudali.spec.Layer` shapes with
depths, and these functions turn that stack into a depth map or an edge map.
ControlNet then treats it as a hard constraint, so "the lighthouse is on the
right third, the cliff fills the lower left" becomes geometry rather than hope.

Everything here is numpy and Pillow. No model is loaded, no network is touched,
and the same spec always produces the same map.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..spec import Layer, SceneSpec


def _layer_seed(layer: Layer, salt: int = 0) -> int:
    """A stable per-layer seed, so a blob keeps its shape across renders."""
    digest = hashlib.sha256(f"{layer.role}:{layer.shape}:{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _px(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Normalised bbox -> integer pixel box, clamped to the canvas."""
    x0 = int(round(bbox[0] * width))
    y0 = int(round(bbox[1] * height))
    x1 = int(round(bbox[2] * width))
    y1 = int(round(bbox[3] * height))
    x0, x1 = max(0, min(x0, width - 1)), max(1, min(x1, width))
    y0, y1 = max(0, min(y0, height - 1)), max(1, min(y1, height))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _radial_falloff(w: int, h: int, power: float = 1.6) -> np.ndarray:
    """A soft dome, 1.0 at the centre falling to 0 at the boundary.

    Painting a shape as a flat plate makes the depth ControlNet produce a flat
    cutout. A dome reads as volume, which is almost always what a caller means
    by "a rock in the foreground".
    """
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xs - cx) / max(cx, 1e-6)
    ny = (ys - cy) / max(cy, 1e-6)
    radius = np.sqrt(nx**2 + ny**2)
    return np.clip(1.0 - radius**power, 0.0, 1.0)


def _smooth_noise(w: int, h: int, seed: int, scale: int = 8) -> np.ndarray:
    """Low-frequency value noise in 0..1, used to make blobs organic."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((max(scale, 2), max(scale, 2)))
    noise = np.array(
        Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
        dtype=np.float32,
    )
    return noise / 255.0


def _blob_mask(w: int, h: int, seed: int) -> np.ndarray:
    """An irregular closed shape filling roughly the given box."""
    dome = _radial_falloff(w, h, power=2.0)
    noise = _smooth_noise(w, h, seed, scale=6)
    # Perturb the dome's radius with noise, then threshold. The 0.35 cut keeps
    # the shape solid rather than shredding it into islands.
    field = dome * (0.7 + 0.6 * noise)
    return (field > 0.35).astype(np.float32)


# ---------------------------------------------------------------------------
# Depth maps
# ---------------------------------------------------------------------------


def build_depth_map(spec: SceneSpec) -> Image.Image:
    """Compose a greyscale depth map: white is near the camera, black is far.

    The background is a ground plane receding to the horizon, which is what
    almost every outdoor scene needs and costs nothing to include. Layers are
    then painted back to front, so a near layer occludes a far one exactly as it
    would in a real scene.
    """
    width, height = spec.resolution()
    depth = np.zeros((height, width), dtype=np.float32)

    # Ground plane. Above the horizon is sky, which stays at zero (infinitely
    # far); below it, depth ramps toward the viewer at the bottom edge.
    horizon = spec.composition.horizon
    if horizon is not None:
        horizon_px = int(round(horizon * height))
        if horizon_px < height:
            ramp = np.linspace(0.0, 0.45, height - horizon_px, dtype=np.float32)
            depth[horizon_px:, :] = ramp[:, None]

    # Painter's algorithm: farthest first, so nearer layers overwrite.
    for layer in sorted(spec.composition.layers, key=lambda item: item.depth):
        x0, y0, x1, y1 = _px(layer.bbox, width, height)
        w, h = x1 - x0, y1 - y0
        patch = np.zeros((h, w), dtype=np.float32)

        if layer.shape == "rect":
            patch[:] = 1.0
        elif layer.shape == "ellipse":
            patch = _radial_falloff(w, h, power=1.4)
            patch = (patch > 0.0).astype(np.float32) * (0.6 + 0.4 * patch)
        elif layer.shape == "blob":
            mask = _blob_mask(w, h, _layer_seed(layer))
            patch = mask * (0.65 + 0.35 * _radial_falloff(w, h, power=1.2))
        elif layer.shape == "column":
            # A vertical form: full height of its box, rounded across x only.
            profile = _radial_falloff(w, 3, power=1.3)[1]
            patch = np.tile(profile, (h, 1))
        elif layer.shape == "horizon":
            # A distant band -- a mountain ridge or treeline.
            patch[:] = 1.0
        elif layer.shape == "line":
            thickness = max(1, h // 8)
            patch[h // 2 - thickness : h // 2 + thickness, :] = 1.0

        target = depth[y0:y1, x0:x1]
        # `patch` is a coverage mask scaled by shading; multiply by the layer's
        # depth so a nearer layer genuinely reads as nearer.
        contribution = patch * layer.depth
        depth[y0:y1, x0:x1] = np.where(contribution > 0, np.maximum(target, contribution), target)

    image = Image.fromarray((np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    # A light blur removes stair-stepping that ControlNet would otherwise try to
    # reproduce as literal geometry.
    return image.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) / 400)))


# ---------------------------------------------------------------------------
# Edge / scribble maps
# ---------------------------------------------------------------------------


def build_edge_map(spec: SceneSpec) -> Image.Image:
    """Draw layer outlines as white strokes on black, ready for canny ControlNet.

    SDXL has no widely available scribble ControlNet, so both ``canny`` and
    ``scribble`` modes route through the canny model. The difference is only how
    much of the drawing is filled in.
    """
    width, height = spec.resolution()
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    stroke = max(2, int(min(width, height) / 300))

    if spec.composition.horizon is not None:
        y = int(round(spec.composition.horizon * height))
        draw.line([(0, y), (width, y)], fill=255, width=stroke)

    for layer in sorted(spec.composition.layers, key=lambda item: item.depth):
        x0, y0, x1, y1 = _px(layer.bbox, width, height)
        if layer.shape in {"rect", "horizon"}:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=255, width=stroke)
        elif layer.shape in {"ellipse", "column"}:
            draw.ellipse([x0, y0, x1 - 1, y1 - 1], outline=255, width=stroke)
        elif layer.shape == "line":
            draw.line([(x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2)], fill=255, width=stroke)
        elif layer.shape == "blob":
            mask = _blob_mask(x1 - x0, y1 - y0, _layer_seed(layer))
            # Outline = the mask minus an eroded copy of itself.
            edge = mask - np.minimum(
                mask,
                np.array(
                    Image.fromarray((mask * 255).astype(np.uint8)).filter(
                        ImageFilter.MinFilter(size=max(3, stroke * 2 + 1))
                    ),
                    dtype=np.float32,
                )
                / 255.0,
            )
            patch = Image.fromarray((np.clip(edge, 0, 1) * 255).astype(np.uint8), mode="L")
            image.paste(patch, (x0, y0), patch)

    return image


def canny_from_image(path: str, low: int = 100, high: int = 200) -> Image.Image:
    """Run Canny edge detection on an existing image.

    Used when ``control.source == 'file'`` and the mode is canny or scribble:
    the caller supplies a reference photo or sketch and gets its edges.
    """
    import cv2

    array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise FileNotFoundError(f"could not read control image: {path}")
    edges = cv2.Canny(array, low, high)
    return Image.fromarray(edges, mode="L")


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------


def build_region_masks(spec: SceneSpec) -> dict[str, Image.Image]:
    """One white-on-black mask per layer, keyed by role.

    Handed to the inpainting path so a caller can say "regenerate the layer
    called 'sky'" without drawing a mask by hand.
    """
    width, height = spec.resolution()
    masks: dict[str, Image.Image] = {}
    for layer in spec.composition.layers:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        x0, y0, x1, y1 = _px(layer.bbox, width, height)
        if layer.shape in {"ellipse", "column", "blob"}:
            draw.ellipse([x0, y0, x1 - 1, y1 - 1], fill=255)
        else:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
        masks[layer.role] = mask
    return masks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_control_image(spec: SceneSpec) -> Optional[Image.Image]:
    """Produce the control image for a spec, or ``None`` when control is off.

    A file-sourced depth map is used verbatim: estimating depth from a photo
    would mean shipping a second model, and a caller who has a depth map already
    should not be made to download MiDaS to use it.
    """
    control = spec.control
    if control.mode == "none":
        return None

    width, height = spec.resolution()

    if control.source == "file":
        assert control.image is not None  # guaranteed by the spec validator
        if control.mode == "depth":
            image = Image.open(control.image).convert("L")
        else:
            image = canny_from_image(control.image)
        return image.resize((width, height), Image.BICUBIC)

    if control.mode == "depth":
        return build_depth_map(spec)
    return build_edge_map(spec)


def control_repo_for_mode(mode: str) -> Optional[str]:
    """Map a control mode to the registry id of the ControlNet it needs."""
    return {
        "depth": "controlnet-depth-sdxl",
        "canny": "controlnet-canny-sdxl",
        "scribble": "controlnet-canny-sdxl",
    }.get(mode)


__all__ = [
    "build_control_image",
    "build_depth_map",
    "build_edge_map",
    "build_region_masks",
    "canny_from_image",
    "control_repo_for_mode",
]
