"""Scene spec -> weighted SDXL prompts.

This is the translation layer that makes a structured spec worth having. It
turns named vocabulary keys into the phrasing SDXL was actually trained on,
orders the fragments so the important ones land in the tokens that matter, and
attaches attention weights in compel syntax.

Two rules govern everything here:

1. **Nothing is hidden.** Every compiled result carries the exact prompt, the
   exact negative, and a fragment-by-fragment account of where each phrase came
   from. A caller who dislikes the result can see precisely why it happened.
2. **Explicit beats inferred.** An intent supplies defaults; any field the
   caller actually set always wins, and unknown vocabulary keys pass through as
   free text with a warning rather than raising.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .spec import SceneSpec

VOCAB_DIR = Path(__file__).resolve().parent / "vocabulary"

# Media that describe a lens-based image. Used only to warn when a caller asks
# for an aperture on a woodcut -- the fragment is still emitted, because the
# caller asked for it and silently dropping input is worse than a strange image.
OPTICAL_MEDIA = {
    "photograph",
    "film_35mm",
    "polaroid",
    "cinematic_still",
    "render3d",
    "clay_render",
}

# Order defines prominence. CLIP attends most strongly to early tokens, so the
# medium anchors the whole image, the subject follows immediately, and the
# atmospheric material trails behind where dilution costs least.
FRAGMENT_ORDER = [
    "medium",
    "movement",
    "subject",
    "count",
    "action",
    "details",
    "secondary",
    "setting",
    "time",
    "weather",
    "season",
    "background",
    "foreground",
    "lighting",
    "lighting_mood",
    "lighting_accents",
    "shot",
    "lens",
    "aperture",
    "angle",
    "focus",
    "rule",
    "palette",
    "palette_contrast",
    "artists",
    "descriptors",
    "detail",
    "quality",
]


@dataclass
class Fragment:
    """One phrase in the compiled prompt, with its provenance."""

    source: str
    text: str
    weight: float = 1.0
    key: Optional[str] = None

    def render(self) -> str:
        """Emit compel attention syntax when the weight is not neutral."""
        text = self.text.strip().strip(",")
        if not text:
            return ""
        if abs(self.weight - 1.0) < 0.01:
            return text
        return f"({text}){self.weight:.2f}"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "key": self.key, "text": self.text, "weight": self.weight}


@dataclass
class CompiledPrompt:
    """Everything the renderer needs, plus the audit trail of how it got there."""

    prompt: str
    negative_prompt: str
    model: str
    steps: int
    cfg: float
    sampler: str
    width: int
    height: int
    fragments: list[Fragment] = field(default_factory=list)
    negative_fragments: list[Fragment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model": self.model,
            "steps": self.steps,
            "cfg": self.cfg,
            "sampler": self.sampler,
            "width": self.width,
            "height": self.height,
            "fragments": [f.to_dict() for f in self.fragments],
            "negative_fragments": [f.to_dict() for f in self.negative_fragments],
            "warnings": self.warnings,
        }


@functools.lru_cache(maxsize=1)
def load_vocabulary() -> dict[str, Any]:
    """Read and merge every YAML table in ``claudali/vocabulary/``.

    Cached: the tables are static at runtime, and re-reading six files per
    render would be pointless. Call ``load_vocabulary.cache_clear()`` after
    editing a table in a live process.
    """
    merged: dict[str, Any] = {}
    for path in sorted(VOCAB_DIR.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for key, value in data.items():
            if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def vocabulary_index() -> dict[str, list[str]]:
    """The available keys per table, for the API's discovery endpoint."""
    vocab = load_vocabulary()
    return {
        name: sorted(table)
        for name, table in vocab.items()
        if isinstance(table, dict) and name != "intents"
    }


class _Builder:
    """Accumulates fragments while compiling one spec."""

    def __init__(self, vocab: dict[str, Any]) -> None:
        self.vocab = vocab
        self.positive: list[Fragment] = []
        self.negative: list[Fragment] = []
        self.warnings: list[str] = []

    # -- fragment helpers -------------------------------------------------

    def add_text(self, source: str, text: Optional[str], weight: float = 1.0) -> None:
        """Add a free-text fragment, ignoring blanks."""
        if text and text.strip():
            self.positive.append(Fragment(source=source, text=text.strip(), weight=weight))

    def add_key(self, source: str, table_name: str, key: Optional[str]) -> None:
        """Look ``key`` up in a vocabulary table and add what it expands to.

        An unmatched key is not an error. Callers legitimately invent style
        words, and "cinematic" is a perfectly good prompt fragment even though
        it is not in the media table -- so it passes through, with a warning so
        the caller can tell the difference between a working key and a typo.
        """
        if not key:
            return
        table = self.vocab.get(table_name, {})
        entry = table.get(key) if isinstance(table, dict) else None
        if entry is None:
            self.warnings.append(
                f"{source}: '{key}' is not a known {table_name} key; used as free text"
            )
            self.add_text(source, key.replace("_", " "))
            return
        self.positive.append(
            Fragment(
                source=source,
                text=str(entry.get("prompt", "")),
                weight=float(entry.get("weight", 1.0)),
                key=key,
            )
        )
        implied = entry.get("negative")
        if implied:
            self.negative.append(Fragment(source=f"{source}:implied", text=str(implied), key=key))

    def add_negative_preset(self, key: str) -> None:
        table = self.vocab.get("negatives", {})
        entry = table.get(key)
        if entry is None:
            self.warnings.append(f"negative: '{key}' is not a known preset; used as free text")
            self.negative.append(Fragment(source="negative", text=key.replace("_", " ")))
            return
        self.negative.append(
            Fragment(source="negative", text=str(entry.get("prompt", "")), key=key)
        )


def _dedupe(fragments: list[Fragment]) -> list[Fragment]:
    """Drop repeated phrases, keeping the first (and so highest-priority) one.

    Repetition inside a prompt is not free: it consumes the token budget and
    reads to CLIP as emphasis the caller did not ask for.
    """
    seen: set[str] = set()
    result: list[Fragment] = []
    for fragment in fragments:
        text = fragment.text.strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(fragment)
    return result


def _order(fragments: list[Fragment]) -> list[Fragment]:
    """Sort by FRAGMENT_ORDER, keeping insertion order within a source.

    Python's sort is stable, so fragments sharing a source stay in the order the
    caller wrote them -- three subject details keep the caller's emphasis.
    """
    rank = {name: index for index, name in enumerate(FRAGMENT_ORDER)}
    return sorted(fragments, key=lambda f: rank.get(f.source.split(":")[0], len(rank)))


def compile_spec(spec: SceneSpec) -> CompiledPrompt:
    """Compile a :class:`~claudali.spec.SceneSpec` into renderer parameters."""
    vocab = load_vocabulary()
    intents = vocab.get("intents", {})
    intent = intents.get(spec.intent, {})
    builder = _Builder(vocab)

    # -- style anchor -----------------------------------------------------
    medium = spec.style.medium or intent.get("medium")
    builder.add_key("medium", "media", medium)
    builder.add_key("movement", "movements", spec.style.movement)

    # -- subject ----------------------------------------------------------
    # The subject carries a small weight bump. It is the one thing the image
    # must not lose, and it sits far enough into the prompt that a nudge helps.
    subject_text = spec.subject.primary
    if spec.subject.count and spec.subject.count > 1:
        subject_text = f"{spec.subject.count} {subject_text}"
    builder.add_text("subject", subject_text, weight=1.1)
    builder.add_text("action", spec.subject.action)
    for detail in spec.subject.details:
        builder.add_text("details", detail)
    for item in spec.subject.secondary:
        builder.add_text("secondary", item)

    # -- scene ------------------------------------------------------------
    scene_default = intent.get("scene_default")
    if scene_default and not any(
        [spec.scene.setting, spec.scene.background, spec.scene.foreground]
    ):
        builder.add_text("setting", scene_default)
    builder.add_text("setting", spec.scene.setting)
    builder.add_text("time", spec.scene.time)
    builder.add_text("weather", spec.scene.weather)
    builder.add_text("season", spec.scene.season)
    builder.add_text("background", spec.scene.background)
    builder.add_text("foreground", spec.scene.foreground)

    # -- lighting ---------------------------------------------------------
    lighting_key = spec.lighting.key or intent.get("lighting")
    builder.add_key("lighting", "lighting", lighting_key)
    builder.add_text("lighting_mood", spec.lighting.mood)
    for accent in spec.lighting.accents:
        builder.add_text("lighting_accents", accent)

    # -- camera -----------------------------------------------------------
    builder.add_key("shot", "shot", spec.camera.shot)
    builder.add_key("lens", "lens", spec.camera.lens)
    builder.add_text("aperture", spec.camera.aperture)
    builder.add_key("angle", "angle", spec.camera.angle)
    builder.add_key("focus", "focus", spec.camera.focus)

    if medium and medium not in OPTICAL_MEDIA:
        optical_asked = [spec.camera.lens, spec.camera.aperture, spec.camera.focus]
        if any(optical_asked):
            builder.warnings.append(
                f"camera: lens/aperture/focus specified with non-optical medium '{medium}'; "
                "kept as requested, but it may read as incoherent"
            )

    # -- composition ------------------------------------------------------
    rule = spec.composition.rule or intent.get("composition_rule")
    builder.add_key("rule", "rule", rule)
    if spec.composition.horizon is not None:
        # `horizon` is a y coordinate: 0 is the top edge, 1 the bottom. A
        # horizon near the top leaves most of the frame to the ground; a horizon
        # near the bottom leaves most of it to the sky.
        position = (
            "high horizon line, ground dominant"
            if spec.composition.horizon < 0.4
            else "low horizon line, sky dominant"
            if spec.composition.horizon > 0.6
            else "central horizon line"
        )
        builder.add_text("rule", position)

    # -- palette ----------------------------------------------------------
    if spec.palette.name:
        builder.add_key("palette", "palettes", spec.palette.name)
    if spec.palette.colors:
        builder.add_text(
            "palette", "colour palette of " + ", ".join(spec.palette.colors), weight=1.05
        )
    builder.add_key("palette_contrast", "contrast", spec.palette.contrast)

    # -- style extras -----------------------------------------------------
    for artist in spec.style.artists:
        builder.add_text("artists", f"in the style of {artist}")
    for descriptor in spec.style.descriptors:
        builder.add_text("descriptors", descriptor)
    builder.add_key("detail", "detail", spec.style.detail or intent.get("detail"))

    # -- quality tail -----------------------------------------------------
    builder.add_text("quality", intent.get("quality"))
    builder.add_text("quality", vocab.get("global_quality"))

    # -- negatives --------------------------------------------------------
    presets = list(spec.negative.presets)
    if not spec.negative.disable_defaults:
        for preset in intent.get("negatives", []):
            if preset not in presets:
                presets.append(preset)
    for preset in presets:
        builder.add_negative_preset(preset)
    for extra in spec.negative.extra:
        builder.negative.append(Fragment(source="negative:extra", text=extra))
    if not spec.negative.disable_defaults:
        builder.negative.append(
            Fragment(source="negative:global", text=str(vocab.get("global_negative", "")))
        )

    # -- assemble ---------------------------------------------------------
    positive_fragments = _dedupe(_order(builder.positive))
    negative_fragments = _dedupe(builder.negative)

    prompt = ", ".join(f.render() for f in positive_fragments if f.render())
    negative_prompt = ", ".join(f.render() for f in negative_fragments if f.render())

    if spec.raw.prepend:
        prompt = f"{spec.raw.prepend.strip()}, {prompt}"
    if spec.raw.append:
        prompt = f"{prompt}, {spec.raw.append.strip()}"
    if spec.raw.prompt:
        prompt = spec.raw.prompt
        builder.warnings.append("raw.prompt set: the compiled positive prompt was discarded")
    if spec.raw.negative_prompt:
        negative_prompt = spec.raw.negative_prompt
        builder.warnings.append("raw.negative_prompt set: compiled negatives were discarded")

    width, height = spec.resolution()

    # SDXL's text encoders take 77 tokens per chunk. compel concatenates extra
    # chunks rather than truncating, so two chunks are unremarkable and warning
    # about them would just train callers to ignore warnings. Past three chunks
    # the dilution is real and worth saying out loud.
    approx_tokens = len(prompt.split()) * 1.3
    if approx_tokens > 225:
        builder.warnings.append(
            f"prompt is long (~{int(approx_tokens)} tokens); later fragments will have "
            "little influence. Consider trimming details or descriptors."
        )

    # Pydantic records which fields the caller actually supplied, which is the
    # only reliable way to tell an explicit `steps: 30` from the schema default
    # of 30. Comparing against the default value would silently override a
    # caller who happened to choose the same number.
    was_set = spec.render.model_fields_set

    return CompiledPrompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=spec.render.model or intent.get("model") or "sdxl-base",
        steps=spec.render.steps if "steps" in was_set else int(intent.get("steps", spec.render.steps)),
        cfg=spec.render.cfg if "cfg" in was_set else float(intent.get("cfg", spec.render.cfg)),
        sampler=(
            spec.render.sampler
            if "sampler" in was_set
            else str(intent.get("sampler", spec.render.sampler))
        ),
        width=width,
        height=height,
        fragments=positive_fragments,
        negative_fragments=negative_fragments,
        warnings=builder.warnings,
    )


__all__ = [
    "CompiledPrompt",
    "Fragment",
    "compile_spec",
    "load_vocabulary",
    "vocabulary_index",
    "VOCAB_DIR",
]
