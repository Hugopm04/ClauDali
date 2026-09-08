"""Post-diffusion compositing: legible text, exact shapes, deterministic finishing.

Diffusion models cannot spell. Asking SDXL for a poster with a title returns
beautiful lettering that says nothing, and no amount of prompting fixes it
because the model has no concept of a glyph. So ClauDali splits the problem:
diffusion makes the imagery, and Pillow draws anything that has to be *right*
on top of it.

The postprocess stage is separate and equally deliberate. Pixelation, palette
locking and seamless tiling are exact operations that a sampler can only
approximate, and a game asset usually needs them exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .config import FONTS_DIR
from .spec import Overlay, Postprocess, SceneSpec

# Where to look for a usable TrueType face, in order of preference. Pillow's
# built-in default font is a small bitmap that ignores `size` entirely, so
# falling back to it produces unreadably tiny text -- worth avoiding.
FONT_DIRS = [
    FONTS_DIR,
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path.home() / ".fonts",
]

FALLBACK_FONTS = [
    "Inter-Regular.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
    "Arial.ttf",
    "segoeui.ttf",
    "Helvetica.ttc",
    "LiberationSans-Regular.ttf",
    "NotoSans-Regular.ttf",
]

# Normalised anchor -> (x fraction, y fraction) of the text box to place at xy.
ANCHOR_OFFSETS = {
    "nw": (0.0, 0.0),
    "n": (0.5, 0.0),
    "ne": (1.0, 0.0),
    "w": (0.0, 0.5),
    "c": (0.5, 0.5),
    "e": (1.0, 0.5),
    "sw": (0.0, 1.0),
    "s": (0.5, 1.0),
    "se": (1.0, 1.0),
}


def _hex_to_rgba(value: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    text = value.lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    red, green, blue = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return red, green, blue, int(round(255 * max(0.0, min(1.0, opacity))))


def _find_font_file(name: Optional[str]) -> Optional[Path]:
    """Resolve a font name or path to a file on disk."""
    if name:
        candidate = Path(name)
        if candidate.is_file():
            return candidate
        for directory in FONT_DIRS:
            if not directory.is_dir():
                continue
            for suffix in ("", ".ttf", ".otf", ".ttc"):
                probe = directory / f"{name}{suffix}"
                if probe.is_file():
                    return probe

    for fallback in FALLBACK_FONTS:
        for directory in FONT_DIRS:
            probe = directory / fallback
            if probe.is_file():
                return probe

    # Last resort: any TrueType face in the system font directory.
    for directory in FONT_DIRS:
        if directory.is_dir():
            for probe in sorted(directory.glob("*.ttf"))[:1]:
                return probe
    return None


def load_font(name: Optional[str], size: int) -> ImageFont.ImageFont:
    path = _find_font_file(name)
    if path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def available_fonts(limit: int = 300) -> list[str]:
    """Font names the UI can offer. Deduplicated by stem, capped for sanity."""
    names: set[str] = set()
    for directory in FONT_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.tt[fc]"):
            names.add(path.stem)
        for path in directory.glob("*.otf"):
            names.add(path.stem)
        if len(names) >= limit:
            break
    return sorted(names)[:limit]


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------


def _draw_text(layer: Image.Image, overlay: Overlay, width: int, height: int) -> None:
    """Draw text, anchored by its own bounding box rather than Pillow's anchors.

    Measuring the box first and offsetting by hand keeps single-line and
    multi-line text on the same rules; Pillow's `anchor` argument does not
    support vertical anchoring for multi-line strings.
    """
    if not overlay.text:
        return

    font = load_font(overlay.font, overlay.size)
    draw = ImageDraw.Draw(layer)
    spacing = int(overlay.size * (overlay.line_spacing - 1.0))

    box = draw.multiline_textbbox(
        (0, 0), overlay.text, font=font, spacing=spacing, align=overlay.align
    )
    text_width, text_height = box[2] - box[0], box[3] - box[1]

    frac_x, frac_y = ANCHOR_OFFSETS[overlay.anchor]
    x = overlay.xy[0] * width - text_width * frac_x - box[0]
    y = overlay.xy[1] * height - text_height * frac_y - box[1]

    draw.multiline_text(
        (x, y),
        overlay.text,
        font=font,
        fill=_hex_to_rgba(overlay.color, overlay.opacity),
        spacing=spacing,
        align=overlay.align,
        stroke_width=overlay.stroke_width,
        stroke_fill=_hex_to_rgba(overlay.stroke, overlay.opacity) if overlay.stroke else None,
    )


def _shape_box(overlay: Overlay, width: int, height: int) -> tuple[int, int, int, int]:
    bbox = overlay.bbox or [0.1, 0.1, 0.9, 0.9]
    return (
        int(bbox[0] * width),
        int(bbox[1] * height),
        int(bbox[2] * width),
        int(bbox[3] * height),
    )


def _draw_shape(layer: Image.Image, overlay: Overlay, width: int, height: int) -> None:
    draw = ImageDraw.Draw(layer)
    box = _shape_box(overlay, width, height)
    fill = _hex_to_rgba(overlay.color, overlay.opacity) if overlay.fill else None
    outline = None if overlay.fill else _hex_to_rgba(overlay.color, overlay.opacity)

    if overlay.type == "rect":
        draw.rectangle(box, fill=fill, outline=outline, width=overlay.width)
    elif overlay.type == "ellipse":
        draw.ellipse(box, fill=fill, outline=outline, width=overlay.width)
    elif overlay.type == "line":
        draw.line(
            [(box[0], box[1]), (box[2], box[3])],
            fill=_hex_to_rgba(overlay.color, overlay.opacity),
            width=overlay.width,
        )


def _draw_gradient(layer: Image.Image, overlay: Overlay, width: int, height: int) -> None:
    """An alpha ramp in the overlay colour, falling off in a chosen direction.

    The workhorse of legible poster text: a dark scrim behind a title guarantees
    contrast no matter what the diffusion model put there. `direction` says
    which edge stays opaque -- 'up' keeps the bottom solid and clears upward,
    which is what a title along the lower edge needs.
    """
    box = _shape_box(overlay, width, height)
    box_w, box_h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    red, green, blue, alpha = _hex_to_rgba(overlay.color, overlay.opacity)

    if overlay.direction in {"down", "up"}:
        ramp = np.linspace(alpha, 0, box_h, dtype=np.float32)
        if overlay.direction == "up":
            ramp = ramp[::-1]
        alpha_plane = np.repeat(ramp[:, None], box_w, axis=1)
    else:
        ramp = np.linspace(alpha, 0, box_w, dtype=np.float32)
        if overlay.direction == "right":
            ramp = ramp[::-1]
        alpha_plane = np.repeat(ramp[None, :], box_h, axis=0)

    patch = np.zeros((box_h, box_w, 4), dtype=np.uint8)
    patch[..., 0], patch[..., 1], patch[..., 2] = red, green, blue
    patch[..., 3] = alpha_plane.astype(np.uint8)
    layer.alpha_composite(Image.fromarray(patch, mode="RGBA"), dest=(box[0], box[1]))


def _draw_vignette(layer: Image.Image, overlay: Overlay, width: int, height: int) -> None:
    ys, xs = np.mgrid[0:height, 0:width]
    nx = (xs - (width - 1) / 2) / ((width - 1) / 2)
    ny = (ys - (height - 1) / 2) / ((height - 1) / 2)
    radius = np.clip(np.sqrt(nx**2 + ny**2) / np.sqrt(2), 0, 1)
    red, green, blue, alpha = _hex_to_rgba(overlay.color, overlay.opacity)

    patch = np.zeros((height, width, 4), dtype=np.uint8)
    patch[..., 0], patch[..., 1], patch[..., 2] = red, green, blue
    patch[..., 3] = (radius**2 * alpha).astype(np.uint8)
    layer.alpha_composite(Image.fromarray(patch, mode="RGBA"))


def _draw_image(layer: Image.Image, overlay: Overlay, width: int, height: int) -> None:
    if not overlay.image:
        return
    source = Image.open(overlay.image).convert("RGBA")
    box = _shape_box(overlay, width, height)
    target = (max(1, box[2] - box[0]), max(1, box[3] - box[1]))
    source = source.resize(target, Image.LANCZOS)
    if overlay.opacity < 1.0:
        alpha = source.getchannel("A").point(lambda value: int(value * overlay.opacity))
        source.putalpha(alpha)
    layer.alpha_composite(source, dest=(box[0], box[1]))


def apply_overlays(image: Image.Image, overlays: list[Overlay]) -> Image.Image:
    """Composite every overlay onto a copy of the image, in order."""
    if not overlays:
        return image

    result = image.convert("RGBA")
    width, height = result.size

    for overlay in overlays:
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if overlay.type == "text":
            _draw_text(layer, overlay, width, height)
        elif overlay.type in {"rect", "ellipse", "line"}:
            _draw_shape(layer, overlay, width, height)
        elif overlay.type == "gradient":
            _draw_gradient(layer, overlay, width, height)
        elif overlay.type == "vignette":
            _draw_vignette(layer, overlay, width, height)
        elif overlay.type == "image":
            _draw_image(layer, overlay, width, height)

        if overlay.rotation:
            layer = layer.rotate(overlay.rotation, resample=Image.BICUBIC, center=(width / 2, height / 2))

        result = Image.alpha_composite(result, layer)

    return result


# ---------------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------------


def make_seamless(image: Image.Image, blend: float = 0.15) -> Image.Image:
    """Make an image tile edge to edge by cross-fading its own wrapped seams.

    The image is rolled by half its size, which moves the former edges into the
    middle, and the resulting cross is blended away against a mirrored copy.
    Good enough for organic textures -- stone, bark, fabric -- and visibly wrong
    for anything with structure, which is why it is opt-in.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    rolled = np.roll(np.roll(array, height // 2, axis=0), width // 2, axis=1)

    band_x = max(1, int(width * blend))
    band_y = max(1, int(height * blend))

    ramp_x = np.clip(np.linspace(0, 1, band_x * 2), 0, 1)[None, :, None]
    centre_x = width // 2
    left = centre_x - band_x
    seam = rolled[:, left : left + band_x * 2, :]
    mirrored = seam[:, ::-1, :]
    rolled[:, left : left + band_x * 2, :] = seam * ramp_x + mirrored * (1 - ramp_x)

    ramp_y = np.clip(np.linspace(0, 1, band_y * 2), 0, 1)[:, None, None]
    centre_y = height // 2
    top = centre_y - band_y
    seam = rolled[top : top + band_y * 2, :, :]
    mirrored = seam[::-1, :, :]
    rolled[top : top + band_y * 2, :, :] = seam * ramp_y + mirrored * (1 - ramp_y)

    return Image.fromarray(np.clip(rolled, 0, 255).astype(np.uint8), mode="RGB")


def cut_background(image: Image.Image, tolerance: int = 28) -> Image.Image:
    """Flood-fill the background from the corners into transparency.

    Deliberately a heuristic rather than a segmentation model: it needs no extra
    download, and the `game_asset` intent already asks for a plain background,
    which is exactly the case this handles well. On a busy background it will do
    a poor job, and the docs say so.
    """
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    flood = rgb.copy()

    for seed in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        cv2.floodFill(
            flood,
            mask,
            seed,
            (0, 0, 0),
            (tolerance,) * 3,
            (tolerance,) * 3,
            cv2.FLOODFILL_FIXED_RANGE | 4,
        )

    background = mask[1:-1, 1:-1].astype(bool)
    alpha = np.where(background, 0, 255).astype(np.uint8)
    # Soften the cut so the edge does not look like scissors work.
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)

    result = image.convert("RGBA")
    result.putalpha(Image.fromarray(alpha, mode="L"))
    return result


def quantize_to_palette(image: Image.Image, colors: list[str]) -> Image.Image:
    """Snap every pixel to the nearest colour in an explicit palette."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    targets = np.array([_hex_to_rgba(color)[:3] for color in colors], dtype=np.int16)
    distances = ((rgb[:, :, None, :] - targets[None, None, :, :]) ** 2).sum(axis=3)
    nearest = targets[np.argmin(distances, axis=2)]
    return Image.fromarray(nearest.astype(np.uint8), mode="RGB")


def apply_postprocess(
    image: Image.Image, post: Postprocess, palette_colors: Optional[list[str]] = None
) -> Image.Image:
    """Apply the deterministic finishing stage.

    Order is not arbitrary: geometry first, then colour, then grain last so the
    noise is not itself quantised or resampled.
    """
    result = image

    if post.pixelate and post.pixelate > 1:
        width, height = result.size
        small = result.resize(
            (max(1, width // post.pixelate), max(1, height // post.pixelate)), Image.NEAREST
        )
        result = small.resize((width, height), Image.NEAREST)

    if post.seamless:
        result = make_seamless(result)

    if post.contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(post.contrast)
    if post.saturation != 1.0:
        result = ImageEnhance.Color(result).enhance(post.saturation)
    if post.sharpen > 0:
        result = result.filter(
            ImageFilter.UnsharpMask(radius=2, percent=int(post.sharpen * 100), threshold=3)
        )

    if post.palette_lock and palette_colors:
        result = quantize_to_palette(result, palette_colors)
    elif post.palette_size:
        result = result.convert("RGB").quantize(colors=post.palette_size).convert("RGB")

    if post.grain > 0:
        array = np.asarray(result.convert("RGB"), dtype=np.float32)
        noise = np.random.default_rng(0).normal(0, post.grain * 32, array.shape)
        result = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")

    # Alpha last: cutting the background before quantisation would let the
    # palette step reintroduce opaque pixels around the edges.
    if post.transparent_bg:
        result = cut_background(result)

    return result


def finish(image: Image.Image, spec: SceneSpec) -> Image.Image:
    """Run the full post-diffusion stage for a spec: postprocess, then overlays."""
    result = apply_postprocess(image, spec.post, spec.palette.colors or None)
    result = apply_overlays(result, spec.overlays)
    if result.mode == "RGBA" and not spec.post.transparent_bg:
        result = result.convert("RGB")
    return result


__all__ = [
    "apply_overlays",
    "apply_postprocess",
    "available_fonts",
    "cut_background",
    "finish",
    "load_font",
    "make_seamless",
    "quantize_to_palette",
]
