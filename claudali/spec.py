"""The ClauDali scene spec: the structured prompt format.

A scene spec is a JSON document that describes an image in named parts rather
than one long sentence. It exists so that a caller -- a person in the web UI,
another program, or a language model -- can reason about *subject*, *lighting*
and *camera* independently, and so that a change to one part is a diff rather
than a rewrite.

The spec is deliberately declarative. It says what the image contains, not how
to sample it; :mod:`claudali.compiler` turns it into the weighted text prompts
SDXL actually responds to, and records exactly what it produced.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPEC_VERSION = "1"

# SDXL was trained on a fixed set of aspect buckets. Rendering off-bucket costs
# coherence -- limbs duplicate, horizons bend -- so named aspects always resolve
# to a real training resolution instead of arbitrary arithmetic.
ASPECT_BUCKETS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "5:4": (1152, 896),
    "4:5": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "21:9": (1536, 640),
    "9:21": (640, 1536),
}

Intent = Literal["photoreal", "painterly", "game_asset", "render3d", "graphic"]


class Base(BaseModel):
    """Strict base: unknown keys are rejected rather than silently ignored.

    A typo in a spec should fail loudly. Silent acceptance is how a caller ends
    up wondering why ``lightning`` had no effect on the image.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Subject(Base):
    """What the image is actually of."""

    primary: str = Field(..., min_length=1, description="The main subject, in plain words.")
    details: list[str] = Field(default_factory=list, description="Specific attributes to enforce.")
    secondary: list[str] = Field(default_factory=list, description="Supporting elements.")
    count: Optional[int] = Field(None, ge=1, le=99, description="How many of the primary subject.")
    action: Optional[str] = Field(None, description="What the subject is doing.")
    anchor: Optional[str] = Field(
        None,
        description=(
            "Two to four words naming the subject alone, restated in every CLIP chunk of a "
            "long prompt. Defaults to `primary`, which works but also re-states the setting."
        ),
    )


class Scene(Base):
    """Where and when the subject is."""

    setting: Optional[str] = None
    time: Optional[str] = Field(None, description="e.g. 'blue hour', 'high noon'.")
    weather: Optional[str] = None
    season: Optional[str] = None
    background: Optional[str] = None
    foreground: Optional[str] = None


class Style(Base):
    """How the image is rendered, as an artefact."""

    medium: Optional[str] = Field(None, description="Vocabulary key, e.g. 'oil_impasto'.")
    movement: Optional[str] = Field(None, description="Vocabulary key, e.g. 'surrealism'.")
    artists: list[str] = Field(default_factory=list, description="Free-text artist references.")
    descriptors: list[str] = Field(default_factory=list, description="Free-text style words.")
    detail: Optional[Literal["minimal", "moderate", "high", "intricate"]] = None


class Camera(Base):
    """Optical framing. Ignored for non-photographic media, by design."""

    shot: Optional[str] = Field(None, description="Vocabulary key, e.g. 'close_up'.")
    lens: Optional[str] = Field(None, description="Vocabulary key, e.g. '85mm'.")
    aperture: Optional[str] = Field(None, description="e.g. 'f/1.8'.")
    angle: Optional[str] = Field(None, description="Vocabulary key, e.g. 'low_angle'.")
    focus: Optional[str] = Field(None, description="Vocabulary key, e.g. 'shallow_dof'.")


class Lighting(Base):
    """Light as a first-class field: it drives mood more than any other knob."""

    key: Optional[str] = Field(None, description="Vocabulary key, e.g. 'rembrandt'.")
    mood: Optional[str] = None
    accents: list[str] = Field(default_factory=list)


class Palette(Base):
    """Colour direction, either a named vocabulary palette or explicit hex."""

    name: Optional[str] = Field(None, description="Vocabulary key, e.g. 'teal_amber'.")
    colors: list[str] = Field(default_factory=list, description="Hex colours, '#rrggbb'.")
    contrast: Optional[Literal["low", "normal", "high"]] = None

    @field_validator("colors")
    @classmethod
    def _validate_hex(cls, values: list[str]) -> list[str]:
        for value in values:
            if not (value.startswith("#") and len(value) in (4, 7)):
                raise ValueError(f"colour must be hex like '#1a2b3c', got {value!r}")
        return values


class Layer(Base):
    """One element placed in the frame, in normalised 0-1 coordinates.

    Layers are what let ClauDali build a ControlNet hint procedurally: each one
    contributes a shape at a known depth, so the stack becomes a depth map or a
    scribble that the model is then forced to obey. Without layers, composition
    is a wish; with them, it is a constraint.
    """

    role: str = Field(..., description="Label, e.g. 'subject', 'horizon', 'foreground_rock'.")
    shape: Literal["rect", "ellipse", "horizon", "column", "blob", "line"] = "rect"
    bbox: list[float] = Field(
        default_factory=lambda: [0.25, 0.25, 0.75, 0.75],
        description="[x0, y0, x1, y1] normalised to the frame.",
    )
    depth: float = Field(0.5, ge=0.0, le=1.0, description="0 = far, 1 = near the camera.")
    prompt: Optional[str] = Field(None, description="Optional per-region description.")

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox must have exactly 4 values: [x0, y0, x1, y1]")
        if any(not 0.0 <= v <= 1.0 for v in value):
            raise ValueError("bbox values must be normalised to 0..1")
        if value[0] >= value[2] or value[1] >= value[3]:
            raise ValueError("bbox must satisfy x0 < x1 and y0 < y1")
        return value


class Regional(Base):
    """Per-region text conditioning driven by ``composition.layers``.

    When off -- the default -- a layer's ``prompt`` is compiled into the text
    prompt as positional words ("in the left third, mid-ground"), which costs
    nothing and lands maybe half the time. When on, each layer's prompt is
    encoded separately and applied only inside that layer's mask, through the
    UNet's cross-attention.

    Orthogonal to ``control`` on purpose. ControlNet conditions geometry through
    a side network; this decides which text applies where. They act at different
    points in the UNet and can be combined, which is why this is a flag here
    rather than another ``control.mode``.

    **Untested on real hardware.** It patches diffusers' attention processors,
    an interface that has changed repeatedly. If installing the patch fails the
    render proceeds without it and says so in the notes, rather than dying.
    """

    enabled: bool = Field(False, description="Apply per-layer prompts as masked attention.")
    strength: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="How fully a region's own conditioning replaces the global one inside its mask.",
    )
    feather: int = Field(
        24, ge=0, le=256, description="Mask edge feather in pixels, to avoid hard region seams."
    )


class Composition(Base):
    """Framing rules and the layer stack that becomes a control map."""

    aspect: str = Field("1:1", description="One of the SDXL aspect buckets.")
    rule: Optional[Literal["thirds", "centered", "golden_spiral", "symmetry", "diagonal"]] = None
    layers: list[Layer] = Field(default_factory=list)
    horizon: Optional[float] = Field(None, ge=0.0, le=1.0, description="Horizon height, 0 = top.")
    regional: Regional = Field(default_factory=Regional)

    @field_validator("aspect")
    @classmethod
    def _validate_aspect(cls, value: str) -> str:
        if value not in ASPECT_BUCKETS:
            raise ValueError(f"aspect must be one of {sorted(ASPECT_BUCKETS)}, got {value!r}")
        return value


class Control(Base):
    """ControlNet configuration.

    ``source='procedural'`` is the interesting case: the control image is
    generated from ``composition.layers`` in numpy, so a caller that cannot draw
    can still dictate exactly where things sit in the frame.
    """

    mode: Literal["none", "depth", "canny", "scribble"] = "none"
    source: Literal["procedural", "file"] = "procedural"
    image: Optional[str] = Field(None, description="Path to a control image when source='file'.")
    strength: float = Field(0.6, ge=0.0, le=2.0, description="ControlNet conditioning scale.")
    start: float = Field(0.0, ge=0.0, le=1.0, description="Step fraction where control begins.")
    end: float = Field(1.0, ge=0.0, le=1.0, description="Step fraction where control ends.")

    @model_validator(mode="after")
    def _check_source(self) -> "Control":
        if self.mode != "none" and self.source == "file" and not self.image:
            raise ValueError("control.source='file' requires control.image")
        if self.end < self.start:
            raise ValueError("control.end must be >= control.start")
        return self


class InitImage(Base):
    """img2img / inpainting input. Drives the surgical part of the refine loop."""

    image: str = Field(..., description="Path to the source image.")
    strength: float = Field(0.55, ge=0.0, le=1.0, description="How far to depart from the source.")
    mask: Optional[str] = Field(None, description="Path to a mask; white = regenerate.")
    mask_layer: Optional[str] = Field(
        None,
        description=(
            "Inpaint the composition layer with this role instead of drawing a mask. Its bbox "
            "is normalised, so the spec must keep the original's layers and aspect."
        ),
    )
    mask_blur: int = Field(8, ge=0, le=128, description="Feather radius for the mask, in pixels.")

    @model_validator(mode="after")
    def _one_mask(self) -> "InitImage":
        if self.mask and self.mask_layer:
            raise ValueError("init.mask and init.mask_layer are alternatives; set only one")
        return self


class Render(Base):
    """Sampling parameters. Full-step by design: no distilled shortcuts."""

    model: Optional[str] = Field(None, description="Model id from the registry.")
    width: Optional[int] = Field(None, ge=256, le=2048)
    height: Optional[int] = Field(None, ge=256, le=2048)
    steps: int = Field(30, ge=1, le=150)
    cfg: float = Field(6.5, ge=0.0, le=30.0, description="Classifier-free guidance scale.")
    sampler: str = Field("dpmpp_2m_karras", description="Scheduler key.")
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1, description="None = random per variation.")
    variations: int = Field(1, ge=1, le=16, description="How many images to render.")
    clip_skip: Optional[int] = Field(None, ge=1, le=4)

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_eight(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value % 8 != 0:
            raise ValueError("width and height must be multiples of 8")
        return value


class Negative(Base):
    """What to steer away from."""

    presets: list[str] = Field(
        default_factory=lambda: ["artifacts"],
        description="Vocabulary keys, e.g. 'anatomy', 'text', 'artifacts'.",
    )
    extra: list[str] = Field(default_factory=list)
    disable_defaults: bool = Field(False, description="Skip the intent's implicit negatives.")


class Overlay(Base):
    """A compositing instruction applied after diffusion.

    Diffusion models cannot spell. Anything that must be *legible* -- a title, a
    label, a logo, a frame -- is drawn here with Pillow, on top of the generated
    image, where it is exact.
    """

    type: Literal["text", "rect", "line", "ellipse", "image", "gradient", "vignette"] = "text"
    text: Optional[str] = None
    font: Optional[str] = Field(None, description="Font family or path; falls back to a default.")
    size: int = Field(48, ge=1, le=1024, description="Font size in pixels.")
    color: str = "#ffffff"
    stroke: Optional[str] = Field(None, description="Outline colour.")
    stroke_width: int = Field(0, ge=0, le=64)
    xy: list[float] = Field(default_factory=lambda: [0.5, 0.5], description="Normalised anchor.")
    bbox: Optional[list[float]] = Field(None, description="Normalised box for shapes.")
    anchor: Literal["nw", "n", "ne", "w", "c", "e", "sw", "s", "se"] = "c"
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    rotation: float = Field(0.0, ge=-360.0, le=360.0)
    width: int = Field(2, ge=1, le=256, description="Stroke width for line/rect/ellipse.")
    fill: bool = False
    image: Optional[str] = Field(None, description="Path, for type='image'.")
    align: Literal["left", "center", "right"] = "left"
    line_spacing: float = Field(1.2, ge=0.5, le=4.0)
    direction: Literal["down", "up", "left", "right"] = Field(
        "down",
        description=(
            "For type='gradient': the direction opacity falls off in. A bottom scrim "
            "behind a title wants 'up' (opaque at the bottom edge, clearing upward)."
        ),
    )


class Postprocess(Base):
    """Deterministic finishing applied to every variation."""

    seamless: bool = Field(False, description="Make the result tile edge-to-edge.")
    transparent_bg: bool = Field(False, description="Cut the background to alpha.")
    grain: float = Field(0.0, ge=0.0, le=1.0)
    sharpen: float = Field(0.0, ge=0.0, le=2.0)
    saturation: float = Field(1.0, ge=0.0, le=3.0)
    contrast: float = Field(1.0, ge=0.0, le=3.0)
    palette_lock: bool = Field(False, description="Quantise to palette.colors.")
    palette_size: Optional[int] = Field(None, ge=2, le=256)
    pixelate: Optional[int] = Field(None, ge=1, le=64, description="Downscale factor for pixel art.")


class Raw(Base):
    """Full escape hatch.

    Any field here bypasses the compiler. Included because a rich vocabulary
    should never become a cage: when a caller already knows the exact prompt
    they want, they must be able to say it.
    """

    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    append: Optional[str] = Field(None, description="Appended after the compiled prompt.")
    prepend: Optional[str] = Field(None, description="Prepended before the compiled prompt.")


class SceneSpec(Base):
    """A complete image request."""

    version: str = SPEC_VERSION
    name: Optional[str] = Field(None, description="Human label; used in output filenames.")
    intent: Intent = Field("photoreal", description="Selects vocabulary defaults and negatives.")

    subject: Subject
    scene: Scene = Field(default_factory=Scene)
    style: Style = Field(default_factory=Style)
    camera: Camera = Field(default_factory=Camera)
    lighting: Lighting = Field(default_factory=Lighting)
    palette: Palette = Field(default_factory=Palette)
    composition: Composition = Field(default_factory=Composition)
    control: Control = Field(default_factory=Control)
    init: Optional[InitImage] = None
    render: Render = Field(default_factory=Render)
    negative: Negative = Field(default_factory=Negative)
    overlays: list[Overlay] = Field(default_factory=list)
    post: Postprocess = Field(default_factory=Postprocess)
    raw: Raw = Field(default_factory=Raw)

    notes: Optional[str] = Field(None, description="Free text; carried into the sidecar, unused.")

    @model_validator(mode="after")
    def _resolve_resolution(self) -> "SceneSpec":
        """Fill width/height from the aspect bucket when not given explicitly.

        The filled-in size is not recorded as set. A spec is saved with
        ``exclude_unset``, and a size saved as if the caller had chosen it would
        override the aspect of every spec loaded back from it.
        """
        if self.render.width is None or self.render.height is None:
            width, height = ASPECT_BUCKETS[self.composition.aspect]
            inferred = {"width", "height"} - self.render.model_fields_set
            self.render.width = self.render.width or width
            self.render.height = self.render.height or height
            self.render.model_fields_set.difference_update(inferred)
        return self

    @model_validator(mode="after")
    def _check_mask_layer(self) -> "SceneSpec":
        """``init.mask_layer`` must name exactly one layer: a mask cannot be a guess."""
        role = self.init.mask_layer if self.init is not None else None
        if role is None:
            return self
        roles = [layer.role for layer in self.composition.layers]
        if role not in roles:
            known = ", ".join(repr(name) for name in roles) or "there are none"
            raise ValueError(
                f"init.mask_layer {role!r} names no layer in composition.layers (roles: {known})"
            )
        if roles.count(role) > 1:
            raise ValueError(
                f"init.mask_layer {role!r} matches {roles.count(role)} layers; give the layer "
                "to inpaint a role of its own"
            )
        return self

    def resolution(self) -> tuple[int, int]:
        assert self.render.width is not None and self.render.height is not None
        return self.render.width, self.render.height

    def slug(self) -> str:
        """A filesystem-safe label for this spec."""
        source = self.name or self.subject.primary
        cleaned = "".join(char if char.isalnum() or char in "-_ " else "" for char in source)
        return "-".join(cleaned.lower().split())[:48] or "scene"


def load_spec(data: dict[str, Any]) -> SceneSpec:
    """Validate a raw dict into a :class:`SceneSpec`."""
    return SceneSpec.model_validate(data)


__all__ = [
    "SPEC_VERSION",
    "ASPECT_BUCKETS",
    "SceneSpec",
    "Subject",
    "Scene",
    "Style",
    "Camera",
    "Lighting",
    "Palette",
    "Composition",
    "Layer",
    "Regional",
    "Control",
    "InitImage",
    "Render",
    "Negative",
    "Overlay",
    "Postprocess",
    "Raw",
    "load_spec",
]
