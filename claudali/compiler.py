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

from . import tokens
from .spec import Layer, SceneSpec

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
    "layer",
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
    # Two channels, because they ask different things of the reader. A *warning*
    # means the spec probably wants changing: a field that cannot take effect, a
    # shot that contradicts the subject. A *note* means the compiler decided
    # something on the caller's behalf and is saying so. Collapsing them trained
    # the eye to skip both.
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tokens: Optional[tokens.TokenCount] = None
    negative_tokens: Optional[tokens.TokenCount] = None
    # One entry per CLIP chunk after the first, naming the anchor restated at
    # its head. Empty when the prompt fits in a single chunk.
    anchors: list[str] = field(default_factory=list)
    # Per-layer prompts encoded separately, when composition.regional is on.
    regions: list[dict[str, Any]] = field(default_factory=list)

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
            "notes": self.notes,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "negative_tokens": self.negative_tokens.to_dict() if self.negative_tokens else None,
            "anchors": self.anchors,
            "regions": self.regions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompiledPrompt":
        """Rebuild a compiled prompt exactly as it was, without recompiling.

        A resumed render must use the prompt frozen when it started: compiling
        the spec again would pick up any vocabulary edit made in between.
        """

        def count(value: Optional[dict[str, Any]]) -> Optional[tokens.TokenCount]:
            return tokens.TokenCount(**value) if value else None

        return cls(
            prompt=data["prompt"],
            negative_prompt=data["negative_prompt"],
            model=data["model"],
            steps=data["steps"],
            cfg=data["cfg"],
            sampler=data["sampler"],
            width=data["width"],
            height=data["height"],
            fragments=[Fragment(**item) for item in data.get("fragments", [])],
            negative_fragments=[Fragment(**item) for item in data.get("negative_fragments", [])],
            warnings=list(data.get("warnings", [])),
            notes=list(data.get("notes", [])),
            tokens=count(data.get("tokens")),
            negative_tokens=count(data.get("negative_tokens")),
            anchors=list(data.get("anchors", [])),
            regions=list(data.get("regions", [])),
        )


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
        self.notes: list[str] = []

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


def _subject_words(spec: SceneSpec) -> set[str]:
    """Every lowercased word the caller used to describe the subject."""
    source = " ".join(
        [
            spec.subject.primary,
            spec.subject.action or "",
            *spec.subject.details,
            *spec.subject.secondary,
        ]
    ).lower()
    return {word.strip(".,;:!?()'\"-") for word in source.split()} - {""}


def _names_a_person(spec: SceneSpec, vocab: dict[str, Any]) -> bool:
    """Whether the subject description mentions anything with a body.

    The word lists live in ``vocabulary/subjects.yaml`` because they are data and
    permanently incomplete: every trade is a person and no list of occupations
    is ever finished. A subject this misjudges is fixed by adding a word there,
    with no code change. The suffix rule catches the ``-smith`` and ``-keeper``
    tail; it is deliberately not extended to ``-ist`` or ``-er``, which would
    make people out of "mist" and "water".
    """
    words = _subject_words(spec)
    if words & set(vocab.get("person_words", [])):
        return True
    suffixes = tuple(vocab.get("person_suffixes", []))
    return bool(suffixes) and any(word.endswith(suffixes) for word in words)


def _names_several(spec: SceneSpec, vocab: dict[str, Any]) -> bool:
    """Whether the caller asked for more than one of the subject."""
    if spec.subject.count and spec.subject.count > 1:
        return True
    return bool(_subject_words(spec) & set(vocab.get("plural_words", [])))


def _shot_flag(vocab: dict[str, Any], key: Optional[str], flag: str) -> bool:
    """Read an optional boolean off a shot vocabulary entry."""
    entry = vocab.get("shot", {}).get(key) if key else None
    return bool(entry.get(flag)) if isinstance(entry, dict) else False


def describe_position(layer: Layer) -> str:
    """Turn a layer's bbox and depth into the words SDXL has some chance of using.

    A nudge, not a constraint. SDXL follows positional language weakly -- expect
    it to land roughly half the time -- but it costs nothing at render time and
    it is the honest alternative to reading ``layer.prompt`` and doing nothing
    with it. ``composition.regional`` is the version that actually enforces
    placement.
    """
    x0, y0, x1, y1 = layer.bbox
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    parts: list[str] = []
    if (x1 - x0) > 0.8 and (y1 - y0) > 0.8:
        parts.append("filling the frame")
    else:
        parts.append(
            "in the left third" if x < 0.34
            else "in the right third" if x > 0.66
            else "centred in frame"
        )
        if y < 0.34:
            parts.append("high in the frame")
        elif y > 0.66:
            parts.append("low in the frame")
    # `depth` is 0 = far, 1 = near, matching the depth map the same numbers build.
    parts.append(
        "in the foreground" if layer.depth > 0.66
        else "in the far distance" if layer.depth < 0.34
        else "in the mid-ground"
    )
    return ", ".join(parts)


def _assemble(pieces: list[str], anchor: str) -> tuple[str, list[str], list[str]]:
    """Join fragments into one prompt, restating ``anchor`` once per CLIP chunk.

    This is the fix for the failure that motivated the whole module. SDXL reads
    75 content tokens at a time and compel concatenates the chunks, so a long
    prompt is really several prompts whose embeddings are averaged by
    cross-attention. A subject named only in the first chunk is outvoted by
    every later chunk, all of which describe a scene with nothing in it: a
    306-token forest-with-fairies prompt mentioned fairies only in tokens 46 to
    73 and rendered four empty forests.

    Restating the subject at the head of each chunk fixes that. The packing in
    :func:`claudali.tokens.pack` reserves room for the anchor and never exceeds
    a chunk, which is what makes the guarantee hold even though these chunk
    boundaries do not line up with compel's -- see that function for why.

    Returns the prompt, the anchors that were inserted, and any warnings.
    """
    warnings: list[str] = []
    if not pieces:
        return "", [], warnings
    groups = tokens.pack(pieces, anchor=anchor)
    if len(groups) <= 1 or not anchor.strip():
        return ", ".join(pieces), [], warnings

    oversize = [
        piece
        for piece in pieces
        if tokens.count_content_tokens(piece)[0] > tokens.CHUNK_CONTENT_TOKENS
    ]
    if oversize:
        warnings.append(
            f"a single fragment is longer than one CLIP chunk ({oversize[0][:40]}...); the "
            "subject cannot be restated inside it, so part of the prompt will not mention "
            "the subject. Split it into shorter fragments."
        )

    # A chunk boundary can fall right after the subject fragment itself, which
    # would print the subject twice in a row. Skipping the anchor there is safe:
    # the guarantee is that every chunk mentions the subject, and that chunk
    # plainly does.
    needle = anchor.lower()
    parts = [", ".join(groups[0])]
    anchors: list[str] = []
    for previous, group in zip(groups, groups[1:]):
        adjacent = f"{previous[-1]} {group[0]}".lower()
        if needle in adjacent:
            parts.append(", ".join(group))
            continue
        anchors.append(anchor)
        parts.append(", ".join([anchor, *group]))
    return ", ".join(parts), anchors, warnings


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

    if _shot_flag(vocab, spec.camera.shot, "body") and not _names_a_person(spec, vocab):
        builder.warnings.append(
            f"camera.shot '{spec.camera.shot}' crops a human body, and the subject names "
            "nobody, so SDXL has no body to crop and will resolve it as a tight close-up of "
            "whatever is nearest. Use 'scene', 'tableau', 'wide' or 'extreme_wide' to frame a "
            "subject inside an environment."
        )
    if _shot_flag(vocab, spec.camera.shot, "single") and _names_several(spec, vocab):
        builder.warnings.append(
            f"camera.shot '{spec.camera.shot}' frames one body, but the subject asks for "
            "several. Use 'group' to keep them all in frame, or 'scene' to place them in "
            "their surroundings."
        )

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

    # A layer's `prompt` used to be read by nothing at all: the field existed,
    # callers could fill it in, and it vanished without a warning. Now it either
    # becomes positional words here, or -- with composition.regional on -- is
    # encoded separately and applied inside the layer's own mask.
    regional = spec.composition.regional
    described = [layer for layer in spec.composition.layers if layer.prompt]
    regions: list[dict[str, Any]] = []
    if described and regional.enabled:
        regions = [
            {
                "role": layer.role,
                "prompt": layer.prompt,
                "bbox": list(layer.bbox),
                "shape": layer.shape,
                "depth": layer.depth,
            }
            for layer in described
        ]
        builder.notes.append(
            f"composition.regional is on, so {len(regions)} layer prompts are conditioned "
            "through masked cross-attention rather than added to the text prompt. That path "
            "is untested on real hardware; if it fails the render falls back and says so."
        )
    else:
        for layer in described:
            builder.add_text("layer", f"{layer.prompt}, {describe_position(layer)}")

    if spec.composition.layers and spec.control.mode == "none":
        # Geometry still only reaches the render through a ControlNet map, so a
        # carefully placed stack is still a wish without one. Saying so is the
        # rule; what changed is that the prompts are no longer lost with it.
        honoured = (
            f" The {len(described)} layer prompts were compiled as positional text instead."
            if described and not regional.enabled
            else ""
        )
        builder.warnings.append(
            f"composition.layers: {len(spec.composition.layers)} layers were given but "
            "control.mode is 'none', so their geometry is not used. Set control.mode to "
            f"'depth' or 'canny' to constrain it, or drop the layers.{honoured}"
        )
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
        # `anatomy` lists hand and face failures. On a subject with no body it
        # spends a third of the negative budget on flaws that cannot occur, and
        # steers away from figures in a scene that should have them. Dropped
        # only from the intent's *defaults*: a preset the caller listed by name
        # is theirs, and explicit beats inferred.
        inherited = list(intent.get("negatives", []))
        if "anatomy" in inherited and not _names_a_person(spec, vocab):
            inherited.remove("anatomy")
            builder.notes.append(
                f"intent '{spec.intent}' would add the 'anatomy' negatives, but the subject "
                "names no person, so they were left out. Add 'anatomy' to negative.presets to "
                "force them."
            )
        for preset in inherited:
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

    pieces = [rendered for f in positive_fragments if (rendered := f.render())]
    if spec.raw.prepend:
        pieces.insert(0, spec.raw.prepend.strip())
    if spec.raw.append:
        pieces.append(spec.raw.append.strip())

    # The anchor is whatever names the subject on its own. `subject.primary`
    # works and is the safe default, but it usually carries the setting with it
    # -- restating "ancient forest with fairies" reinforces the forest just as
    # hard as the fairies, and the forest was already winning. A two-word
    # anchor repeats only the half that is losing.
    anchor = (spec.subject.anchor or spec.subject.primary).strip()
    prompt, anchors, assembly_warnings = _assemble(pieces, anchor)
    builder.warnings.extend(assembly_warnings)

    negative_prompt = ", ".join(f.render() for f in negative_fragments if f.render())

    if spec.raw.prompt:
        prompt = spec.raw.prompt
        anchors = []
        builder.warnings.append(
            "raw.prompt set: the compiled positive prompt was discarded, and with it the "
            "per-chunk subject anchoring. A raw prompt past 75 tokens is on its own."
        )
    if spec.raw.negative_prompt:
        negative_prompt = spec.raw.negative_prompt
        builder.warnings.append("raw.negative_prompt set: compiled negatives were discarded")

    width, height = spec.resolution()

    prompt_tokens = tokens.count(prompt)
    negative_count = tokens.count(negative_prompt)

    # Only worth saying when the fallback anchor is long enough that repeating it
    # costs something. A three-token subject restated per chunk is free and
    # exactly right; warning about it would train callers to ignore warnings.
    anchor_tokens = tokens.count_content_tokens(anchor)[0]
    if anchors and not spec.subject.anchor and anchor_tokens > 6:
        builder.warnings.append(
            f"the prompt spans {prompt_tokens.chunks} CLIP chunks, so the subject was restated "
            f"in each one using subject.primary ({anchor_tokens} tokens, {len(anchors)} times). "
            "Set subject.anchor to a short phrase naming the subject alone: restating the "
            "setting along with it reinforces the setting too."
        )
    # Anchoring keeps every chunk on-subject, so length is no longer the danger
    # it was; the remaining cost is that each chunk's non-subject material gets
    # thinner. Five chunks is where that starts to show.
    if prompt_tokens.chunks > 4:
        approx = "" if prompt_tokens.exact else "~"
        builder.warnings.append(
            f"prompt is {approx}{prompt_tokens.tokens} tokens across {prompt_tokens.chunks} "
            "CLIP chunks. The subject is restated in each, but everything else is thinly "
            "spread. Consider trimming details or descriptors."
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
        notes=builder.notes,
        tokens=prompt_tokens,
        negative_tokens=negative_count,
        anchors=anchors,
        regions=regions,
    )


__all__ = [
    "CompiledPrompt",
    "describe_position",
    "Fragment",
    "compile_spec",
    "load_vocabulary",
    "vocabulary_index",
    "VOCAB_DIR",
]
