"""Building the ordered benchmark list.

Four phases, and the order between them is the whole design:

1. **features** -- every code path that has never run on this GPU, smoke-sized
   first. Failures surface in hour two rather than week five.
2. **settings** -- the quality options, on three fixed scenes. Runs before the
   survey because its findings decide the settings the survey then holds still.
3. **accuracy** -- prompt adherence on things that score themselves.
4. **survey** -- 90 sub-cases x 3 prompts x 2 spec styles x 3 models, in twelve
   waves of increasing replication depth.

Within the first three phases items are ordered by estimated time, shortest
first, as asked. Inside the survey every item costs the same, so the ordering
is by information instead: each wave is a coherent result on its own, and
stopping between waves never leaves a half-filled matrix.

Stdlib plus PyYAML. Importing this must not import torch or the compiler.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from .model import Calibration, Item

TOPICS_DIR = Path(__file__).resolve().parent / "topics"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

MODELS = ("juggernaut-xl", "dreamshaper-xl", "sdxl-base")

# Which checkpoint each intent picks, from vocabulary/intents.yaml. The survey
# shows every topic on its intent's own model first, so a family's first
# verdict is the model you would actually have reached for.
INTENT_MODEL = {
    "photoreal": "juggernaut-xl",
    "painterly": "dreamshaper-xl",
    "game_asset": "sdxl-base",
    "render3d": "sdxl-base",
    "graphic": "sdxl-base",
}

# Every spec in the survey and the accuracy battery renders at SDXL's native
# resolution with these seeds, so any two of them are comparable.
SURVEY_SEED = 20260919
SURVEY_VARIATIONS = 4

# The decode path is PINNED everywhere except the axis that tests it, and this
# is not a detail. `auto` chooses between a CPU fp32 untiled decode and a GPU
# tiled one by looking at free RAM at that moment, so with `auto` two items
# rendered on different evenings are not comparable -- one of them silently
# used a better decoder. Worse, measured here: at 640x640 with 8.45 GB free,
# `auto` picked the CPU, drove free RAM to 0.0 GB, and the render took 686 s
# against 100 s for the same thing decoded on the GPU.
#
# `gpu_tiled` is the predictable one: fast, low memory, unaffected by what else
# is open. The settings phase measures what the untiled CPU decode is actually
# worth (CLAUDE.md puts it at 50.2 dB against 43.6 dB), and if that turns out
# to justify the cost, change this and re-plan.
PINNED_VAE_DECODE = "gpu_tiled"

# Keys a `rich` block may set, and where each one lands in a SceneSpec.
RICH_KEYS = {
    "medium": ("style", "medium"),
    "movement": ("style", "movement"),
    "detail": ("style", "detail"),
    "shot": ("camera", "shot"),
    "lens": ("camera", "lens"),
    "angle": ("camera", "angle"),
    "focus": ("camera", "focus"),
    "lighting": ("lighting", "key"),
    "palette": ("palette", "name"),
    "contrast": ("palette", "contrast"),
    "aspect": ("composition", "aspect"),
    "rule": ("composition", "rule"),
    "horizon": ("composition", "horizon"),
}


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------


def load_topics(directory: Path = TOPICS_DIR) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every family file merged, plus the defaults block. Ordered by filename."""
    families: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        defaults.update(data.pop("defaults", {}))
        families.update(data)
    return families, defaults


def _apply_rich(spec: dict[str, Any], rich: dict[str, Any]) -> None:
    """Merge one level of the style stack. ``null`` removes an inherited key.

    Removal matters: a family sets ``lens: 85mm`` because most of its
    sub-cases are photographs, and the one painterly sub-case must be able to
    take it off again. Leaving it on earns "lens specified with non-optical
    medium" from the compiler, which is the compiler correctly reporting that
    the spec asks for two incompatible things.
    """
    for key, value in rich.items():
        if key not in RICH_KEYS:
            raise KeyError(f"unknown rich key {key!r}; add it to RICH_KEYS or fix the topic file")
        section, field = RICH_KEYS[key]
        if value is None:
            spec.get(section, {}).pop(field, None)
            continue
        spec.setdefault(section, {})[field] = value


def _survey_spec(
    family: dict[str, Any],
    subcase: dict[str, Any],
    prompt: dict[str, Any],
    *,
    variant: str,
    model: str,
) -> dict[str, Any]:
    """One spec. ``rich`` carries the style stack, ``minimal`` is the control."""
    intent = prompt.get("intent") or subcase.get("intent") or family.get("intent", "photoreal")
    spec: dict[str, Any] = {
        "intent": intent,
        "subject": {"primary": prompt["subject"]},
        "render": {
            "model": model,
            "seed": SURVEY_SEED,
            "variations": SURVEY_VARIATIONS,
            "vae_decode": PINNED_VAE_DECODE,
        },
    }
    if variant == "minimal":
        # Nothing but the subject and the intent. Square, because an aspect is
        # a compositional choice and this variant makes none.
        spec["composition"] = {"aspect": "1:1"}
        return spec

    for key in ("details", "action", "anchor", "count"):
        if prompt.get(key):
            spec["subject"][key] = prompt[key]
    scene = {key: prompt[key] for key in ("setting", "time", "weather", "season") if prompt.get(key)}
    if scene:
        spec["scene"] = scene

    spec["composition"] = {"aspect": "1:1"}
    for level in (family.get("rich"), subcase.get("rich"), prompt.get("rich")):
        if level:
            _apply_rich(spec, level)
    return spec


def survey_items(directory: Path = TOPICS_DIR) -> list[Item]:
    """The topic survey, in twelve waves.

    Wave order is what makes an interrupted run still worth something:

    ==========  ====================================================
    w01         prompt 1, rich, the intent's own model
    w02         prompt 1, minimal -- the vocabulary control
    w03, w04    prompts 2 and 3, rich then minimal: sub-case depth
    w05..w08    the same four waves on the second model
    w09..w12    and on the third
    ==========  ====================================================

    So w01 alone is a complete family-level survey, w01+w02 says what the
    vocabulary was worth, and w01..w04 is a defensible sub-case survey of one
    model. Nothing is ever half-finished.
    """
    families, _defaults = load_topics(directory)
    items: list[Item] = []

    # (wave number, prompt indices, variant)
    schedule = [
        (1, [0], "rich"),
        (2, [0], "minimal"),
        (3, [1, 2], "rich"),
        (4, [1, 2], "minimal"),
    ]

    for rank in range(3):
        for wave_offset, indices, variant in schedule:
            wave = f"w{rank * 4 + wave_offset:02d}"
            for family_name, family in families.items():
                for subcase_name, subcase in family["subcases"].items():
                    intent = subcase.get("intent") or family.get("intent", "photoreal")
                    default = INTENT_MODEL.get(intent, "juggernaut-xl")
                    order = [default] + [m for m in MODELS if m != default]
                    model = order[rank]
                    for index in indices:
                        if index >= len(subcase["prompts"]):
                            continue
                        prompt = subcase["prompts"][index]
                        spec = _survey_spec(
                            family, subcase, prompt, variant=variant, model=model
                        )
                        items.append(
                            Item(
                                id=f"survey/{wave}/{family_name}-{subcase_name}"
                                   f"-p{index + 1}-{variant}-{model}",
                                phase="survey",
                                wave=wave,
                                title=f"{subcase['title']} #{index + 1} ({variant}, {model})",
                                why=str(subcase.get("why") or family.get("why", "")).strip(),
                                spec=spec,
                                model=model,
                                group=f"survey/{family_name}/{subcase_name}/p{index + 1}",
                                tags={
                                    "family": family_name,
                                    "subcase": subcase_name,
                                    "prompt_index": str(index + 1),
                                    "variant": variant,
                                    "model": model,
                                    "model_rank": str(rank),
                                    "intent": spec["intent"],
                                },
                            )
                        )
    return items


# ---------------------------------------------------------------------------
# The three fixed scenes the settings benchmark varies options against
# ---------------------------------------------------------------------------

SCENES: dict[str, dict[str, Any]] = {
    "portrait": {
        "title": "Portrait with hands",
        "why": "Faces and hands: the content the refiner and the anatomy negatives exist for.",
        "spec": {
            "intent": "photoreal",
            "subject": {
                "primary": "a luthier examining an unfinished violin body",
                "details": ["both hands holding the instrument", "rolled shirtsleeves"],
                "anchor": "the luthier",
            },
            "scene": {"setting": "a workshop bench under a window", "time": "late afternoon"},
            "style": {"medium": "photograph", "detail": "high"},
            "camera": {"shot": "medium", "lens": "85mm", "focus": "shallow_dof"},
            "lighting": {"key": "softbox"},
            "palette": {"name": "earth_tones", "contrast": "normal"},
            "composition": {"aspect": "1:1", "rule": "thirds"},
        },
    },
    "landscape": {
        "title": "Wide landscape with smooth sky",
        "why": "Large flat gradients: where VAE tile seams, banding and fp16 noise are visible.",
        "spec": {
            "intent": "photoreal",
            "subject": {"primary": "a lone farmhouse on an empty plain under a vast clear sky"},
            "scene": {"setting": "high plains", "time": "blue hour", "weather": "still and clear"},
            "style": {"medium": "photograph", "detail": "high"},
            "camera": {"shot": "extreme_wide", "lens": "24mm", "focus": "deep_focus"},
            "lighting": {"key": "blue_hour"},
            "palette": {"name": "teal_amber", "contrast": "low"},
            "composition": {"aspect": "1:1", "rule": "thirds", "horizon": 0.72},
        },
    },
    "macro": {
        "title": "Mechanical macro",
        "why": "The densest legitimate detail available: what a hi-res pass and an upscaler act on.",
        "spec": {
            "intent": "photoreal",
            "subject": {
                "primary": "the exposed movement of a mechanical wristwatch",
                "details": ["ruby jewels", "engraved bridges", "a visible balance wheel"],
                "anchor": "the watch movement",
            },
            "style": {"medium": "photograph", "detail": "intricate"},
            "camera": {"shot": "macro", "lens": "macro_lens", "focus": "tack_sharp"},
            "lighting": {"key": "studio_three_point"},
            "palette": {"contrast": "high"},
            "composition": {"aspect": "1:1", "rule": "centered"},
        },
    },
}

SETTINGS_SEED = 20260920


def _scene_spec(scene: str, **render: Any) -> dict[str, Any]:
    spec = copy.deepcopy(SCENES[scene]["spec"])
    spec["render"] = {
        "seed": SETTINGS_SEED,
        "variations": 1,
        "model": "juggernaut-xl",
        # Pinned so that every other axis is measured against one decode path;
        # the vae_decode axis overrides it, which is the point of that axis.
        "vae_decode": PINNED_VAE_DECODE,
        **render,
    }
    return spec


# axis name -> (what it is testing, [(value label, render overrides, extra spec)])
def _settings_axes() -> list[tuple[str, str, list[tuple[str, dict[str, Any], dict[str, Any]]]]]:
    return [
        (
            "baseline",
            "The reference every other item in this phase is compared against.",
            [("default", {}, {})],
        ),
        (
            "vae_decode",
            "Where the VAE decode runs. CLAUDE.md measures 25.5 s GPU-tiled against "
            "29.0 s CPU-untiled, and 43.6 dB against 50.2 dB of decoder quality. "
            "This is the axis that decides the survey's decode setting.",
            [
                ("auto", {"vae_decode": "auto"}, {}),
                ("gpu_tiled", {"vae_decode": "gpu_tiled"}, {}),
                ("gpu", {"vae_decode": "gpu"}, {}),
                ("cpu", {"vae_decode": "cpu"}, {}),
            ],
        ),
        (
            "sampler",
            "Every scheduler in SAMPLERS, same seed and steps. Both image and time.",
            [
                (name, {"sampler": name}, {})
                for name in (
                    "ddim", "dpmpp_2m", "dpmpp_2m_karras", "dpmpp_sde_karras",
                    "euler", "euler_a", "heun", "unipc",
                )
            ],
        ),
        (
            "steps",
            "Where more steps stop paying. The tool defaults to 32 for photoreal.",
            [(f"{n}", {"steps": n}, {}) for n in (16, 24, 32, 48, 64)],
        ),
        (
            "cfg",
            "Guidance scale, from under-guided to fried.",
            [(str(value).replace(".", "_"), {"cfg": value}, {}) for value in (3.0, 5.5, 8.0, 12.0)],
        ),
        (
            "model",
            "The three installed checkpoints on identical specs.",
            [(name, {"model": name}, {}) for name in MODELS],
        ),
        (
            "aspect",
            "Resolution and shape. Also the timing ladder the ETA calibrates on, and "
            "the axis that caught the cuDNN fault: it passed at 1024x1024 and failed "
            "at 1344x768.",
            [
                (aspect.replace(":", "-"), {}, {"composition": {"aspect": aspect}})
                for aspect in ("1:1", "5:4", "3:2", "16:9", "9:16", "21:9")
            ],
        ),
        (
            "clip_skip",
            "Whether skipping the last CLIP layer changes anything on SDXL.",
            [("none", {}, {}), ("2", {"clip_skip": 2}, {})],
        ),
        (
            "batching",
            "Four variations in one job against one. Measures how much of a render "
            "is the model load, which decides whether the survey should batch.",
            [("one", {"variations": 1}, {}), ("four", {"variations": 4}, {})],
        ),
        (
            "precision",
            "fp16 against fp32. fp32 selects sequential offload on a 6 GB card and "
            "has never been run; it may page hard or fail outright.",
            [("fp16", {"precision": "fp16"}, {}), ("fp32", {"precision": "fp32"}, {})],
        ),
        (
            "refiner",
            "The SDXL refiner at three handoff points. Never run on this hardware.",
            [
                ("off", {}, {"refiner": {"enabled": False}}),
                ("on_0_70", {}, {"refiner": {"enabled": True, "handoff": 0.7}}),
                ("on_0_80", {}, {"refiner": {"enabled": True, "handoff": 0.8}}),
                ("on_0_90", {}, {"refiner": {"enabled": True, "handoff": 0.9}}),
            ],
        ),
        (
            "hires",
            "The hi-res pass: both upscalers at two scales. Real-ESRGAN has never run, "
            "and 2016x1152 has never been reached.",
            [
                ("off", {}, {"hires": {"enabled": False}}),
                ("lanczos_1_25", {}, {"hires": {"enabled": True, "scale": 1.25, "upscaler": "lanczos"}}),
                ("lanczos_1_50", {}, {"hires": {"enabled": True, "scale": 1.5, "upscaler": "lanczos"}}),
                ("esrgan_1_50", {}, {"hires": {"enabled": True, "scale": 1.5, "upscaler": "realesrgan-x4"}}),
                ("esrgan_2_00", {}, {"hires": {"enabled": True, "scale": 2.0, "upscaler": "realesrgan-x4"}}),
            ],
        ),
        (
            "hires_strength",
            "How far the hi-res img2img pass departs from the upscale. Too low and the "
            "pass is decoration; too high and it reinvents the picture.",
            [
                (f"{value}".replace(".", "_"), {},
                 {"hires": {"enabled": True, "scale": 1.5, "strength": value}})
                for value in (0.15, 0.3, 0.45)
            ],
        ),
        (
            "quality_preset",
            "render.quality 'max' end to end: fp32, CPU decode, refiner and hi-res "
            "together. Expected to be the slowest single item in the benchmark.",
            [("standard", {}, {}), ("max", {"quality": "max"}, {})],
        ),
    ]


def settings_items() -> list[Item]:
    """Every axis on all three scenes, ordered shortest-estimate first."""
    calibration = Calibration()
    items: list[Item] = []
    for axis, why, values in _settings_axes():
        for label, render, extra in values:
            for scene_name in SCENES:
                spec = _scene_spec(scene_name, **render)
                for section, payload in extra.items():
                    if isinstance(payload, dict):
                        spec.setdefault(section, {}).update(payload)
                    else:
                        spec[section] = payload
                items.append(
                    Item(
                        id=f"settings/{axis}/{scene_name}-{label}",
                        phase="settings",
                        wave=axis,
                        title=f"{axis} = {label} on {SCENES[scene_name]['title'].lower()}",
                        why=why,
                        spec=spec,
                        model=spec["render"].get("model", "juggernaut-xl"),
                        group=f"settings/{axis}/{scene_name}",
                        tags={"axis": axis, "value": label, "scene": scene_name},
                    )
                )
    items.sort(key=lambda item: (calibration.estimate(item), item.id))
    return items


# ---------------------------------------------------------------------------
# Prompt accuracy: the things that score themselves
# ---------------------------------------------------------------------------


def _accuracy_spec(subject: dict[str, Any], **extra: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "intent": extra.pop("intent", "photoreal"),
        "subject": subject,
        "composition": {"aspect": "1:1"},
        "render": {
            "model": "juggernaut-xl",
            "seed": SURVEY_SEED,
            "variations": SURVEY_VARIATIONS,
            "vae_decode": PINNED_VAE_DECODE,
        },
    }
    spec.update(extra)
    return spec


ACCURACY: list[tuple[str, str, str, dict[str, Any]]] = [
    # (id, title, what a pass looks like, spec)
    (
        "count-apples", "Counting: five apples",
        "Exactly five apples. Count them; there is no taste involved.",
        _accuracy_spec({"primary": "exactly five red apples arranged on a white plate", "count": 5}),
    ),
    (
        "count-chairs", "Counting: three chairs",
        "Exactly three chairs, all alike.",
        _accuracy_spec({"primary": "three identical wooden chairs in a row against a wall", "count": 3}),
    ),
    (
        "count-candles", "Counting: seven candles",
        "Seven candles. Higher counts usually fail where low ones pass.",
        _accuracy_spec({"primary": "a birthday cake with exactly seven lit candles", "count": 7}),
    ),
    (
        "spatial-stack", "Spatial: on top of",
        "The blue cube above the red sphere, not beside it.",
        _accuracy_spec({"primary": "a blue cube resting on top of a red sphere"}, intent="render3d"),
    ),
    (
        "spatial-between", "Spatial: left, right and between",
        "Cat left, dog right, ball between. Three relations in one sentence.",
        _accuracy_spec({"primary": "a cat on the left, a dog on the right, and a ball between them"}),
    ),
    (
        "spatial-under", "Spatial: under and beside",
        "Book under the lamp, mug beside it.",
        _accuracy_spec({"primary": "a book lying under a desk lamp, with a mug beside the book"}),
    ),
    (
        "colour-bind-two", "Colour binding: two objects",
        "The cube blue and the sphere red, not swapped and not both purple.",
        _accuracy_spec({"primary": "a blue cube and a red sphere on a grey table"}, intent="render3d"),
    ),
    (
        "colour-bind-worn", "Colour binding: worn items",
        "Yellow coat, green umbrella. Binding fails most when both attach to one person.",
        _accuracy_spec({"primary": "a woman in a yellow raincoat holding a green umbrella"}),
    ),
    (
        "negation-cars", "Negation: no cars",
        "An empty street. Diffusion has no NOT, so this usually produces cars.",
        _accuracy_spec({"primary": "a city street with no cars anywhere"}),
    ),
    (
        "negation-people", "Negation: no people",
        "An empty restaurant. Compare against putting it in negative.extra instead.",
        _accuracy_spec({"primary": "the interior of a restaurant with no people in it"}),
    ),
    (
        "negation-proper", "Negation done properly",
        "The same empty restaurant, with the exclusion in negative.extra where it belongs. "
        "This is the control that shows the right way to ask.",
        _accuracy_spec(
            {"primary": "the interior of an empty restaurant, chairs on the tables"},
            negative={"extra": ["people, person, diners, waiter, crowd"]},
        ),
    ),
    (
        "text-short", "Text: one short word",
        "A sign reading OPEN. Almost certainly wrong; the question is how wrong.",
        _accuracy_spec({"primary": "a shop window with a sign reading OPEN"}),
    ),
    (
        "text-short-overlay", "Text: the same word, composited",
        "The control. compose.py draws real lettering after diffusion, so this is "
        "correct by construction. If it is not, that is a compositing bug.",
        _accuracy_spec(
            {"primary": "a shop window with a blank white sign hanging in it"},
            overlays=[{
                "type": "text", "text": "OPEN", "size": 96, "color": "#111111",
                "xy": [0.5, 0.5], "anchor": "c",
            }],
        ),
    ),
    (
        "text-long", "Text: a whole phrase",
        "CLOSED FOR WINTER. Long strings degrade faster than short ones.",
        _accuracy_spec({"primary": "a wooden sign on a gate reading CLOSED FOR WINTER"}),
    ),
    (
        "text-long-overlay", "Text: the same phrase, composited",
        "The control for the phrase, same as above.",
        _accuracy_spec(
            {"primary": "a blank wooden sign on a gate in a snowy field"},
            overlays=[{
                "type": "text", "text": "CLOSED FOR WINTER", "size": 64, "color": "#2a1a0e",
                "xy": [0.5, 0.5], "anchor": "c", "align": "center",
            }],
        ),
    ),
    (
        "multi-subject", "Three named animals together",
        "A fox, an owl and a badger, all three present and all three correct.",
        _accuracy_spec({"primary": "a fox, an owl and a badger sitting together on a log"}),
    ),
    (
        "small-subject-plain", "Small subject in a large scene",
        "The failure CLAUDE.md documents: fairies in a forest, with no anchor. "
        "Expected to render an empty forest. This is a regression check on a real bug.",
        _accuracy_spec(
            {"primary": "small ethereal fairies hovering in an ancient sunlit forest", "count": 3},
            style={"detail": "high"},
            camera={"shot": "scene", "lens": "85mm"},
            lighting={"key": "god_rays"},
        ),
    ),
    (
        "small-subject-anchored", "The same scene, anchored",
        "Identical but for subject.anchor. The documented fix; the pair is the evidence.",
        _accuracy_spec(
            {
                "primary": "small ethereal fairies hovering in an ancient sunlit forest",
                "count": 3, "anchor": "small ethereal fairies",
            },
            style={"detail": "high"},
            camera={"shot": "scene", "lens": "85mm"},
            lighting={"key": "god_rays"},
        ),
    ),
    (
        "relative-size", "Relative size",
        "A mouse and an elephant at their real relative sizes.",
        _accuracy_spec({"primary": "a tiny mouse standing next to an adult elephant's foot"}),
    ),
    (
        "attribute-bind", "Attribute binding across two people",
        "Tall and thin on one, short and round on the other, not averaged across both.",
        _accuracy_spec({"primary": "a tall thin man standing beside a short round man"}),
    ),
    (
        "compound", "Compound: count, colour and place",
        "Two cats, one black one white, and a blue sofa. Three constraints at once.",
        _accuracy_spec({"primary": "two cats on a blue sofa, one cat black and one cat white", "count": 2}),
    ),
]


def accuracy_items() -> list[Item]:
    return [
        Item(
            id=f"accuracy/{name}",
            phase="accuracy",
            wave="accuracy",
            title=title,
            why=why,
            spec=spec,
            model=spec["render"]["model"],
            group=f"accuracy/{name.rsplit('-', 1)[0]}",
            tags={"test": name},
        )
        for name, title, why, spec in ACCURACY
    ]


# ---------------------------------------------------------------------------
# Feature paths: everything that has never run on this GPU
# ---------------------------------------------------------------------------

INIT_SOURCE = ASSETS_DIR / "init-source.png"

_LAYERS = [
    {"role": "sky", "shape": "horizon", "bbox": [0.0, 0.0, 1.0, 0.55], "depth": 0.05},
    {"role": "tower", "shape": "column", "bbox": [0.60, 0.18, 0.74, 0.80], "depth": 0.85,
     "prompt": "a stone lighthouse tower with a lit lamp room"},
    {"role": "rocks", "shape": "blob", "bbox": [0.0, 0.66, 1.0, 1.0], "depth": 0.6,
     "prompt": "wet black basalt boulders"},
]


def _feature_spec(scale: str, **extra: Any) -> dict[str, Any]:
    """A small, fast spec for the smoke pass and a full one for the real pass."""
    small = scale == "smoke"
    spec: dict[str, Any] = {
        "intent": "photoreal",
        "subject": {
            "primary": "a lighthouse on a basalt headland above a cold sea",
            "anchor": "the lighthouse",
        },
        "scene": {"setting": "a north atlantic coast", "time": "blue hour"},
        "style": {"medium": "photograph", "detail": "high"},
        "camera": {"shot": "wide", "lens": "35mm", "focus": "deep_focus"},
        "lighting": {"key": "blue_hour"},
        "composition": {"aspect": "1:1", "rule": "thirds", "horizon": 0.55},
        "render": {
            "model": "juggernaut-xl",
            "seed": SETTINGS_SEED,
            "variations": 1,
            "vae_decode": PINNED_VAE_DECODE,
            "steps": 6 if small else 32,
            "width": 640 if small else None,
            "height": 640 if small else None,
        },
    }
    # The full pass takes its size from the aspect instead of pinning it.
    spec["render"] = {key: value for key, value in spec["render"].items() if value is not None}
    for section, payload in extra.items():
        if isinstance(payload, dict) and isinstance(spec.get(section), dict):
            spec[section].update(payload)
        else:
            spec[section] = payload
    return spec


FEATURES: list[tuple[str, str, str, dict[str, Any], dict[str, str]]] = [
    (
        "control-depth", "ControlNet: procedural depth map",
        "composition.layers compiled into a depth map and fed to the depth ControlNet.",
        {"composition": {"layers": _LAYERS}, "control": {"mode": "depth", "strength": 0.6}},
        {},
    ),
    (
        "control-canny", "ControlNet: canny edges",
        "The same layers as edges instead of depth.",
        {"composition": {"layers": _LAYERS}, "control": {"mode": "canny", "strength": 0.6}},
        {},
    ),
    (
        "control-scribble", "ControlNet: scribble",
        "The third control mode, which shares the canny checkpoint.",
        {"composition": {"layers": _LAYERS}, "control": {"mode": "scribble", "strength": 0.6}},
        {},
    ),
    (
        "overlays", "Compositing: every overlay type",
        "Text, rect, line, ellipse, gradient and vignette drawn after diffusion. "
        "Pure Pillow, so a failure here is not the model's.",
        {
            "overlays": [
                {"type": "gradient", "color": "#000000", "opacity": 0.55, "direction": "up",
                 "bbox": [0.0, 0.62, 1.0, 1.0]},
                {"type": "rect", "bbox": [0.06, 0.06, 0.40, 0.17], "color": "#ffffff",
                 "opacity": 0.85, "fill": True},
                {"type": "text", "text": "TEMPUS", "size": 72, "color": "#f4f1ea",
                 "xy": [0.5, 0.88], "anchor": "c", "stroke": "#000000", "stroke_width": 2},
                {"type": "line", "bbox": [0.10, 0.80, 0.90, 0.80], "color": "#f4f1ea", "width": 3},
                {"type": "ellipse", "bbox": [0.80, 0.06, 0.94, 0.20], "color": "#e8b04b", "width": 4},
                {"type": "vignette", "opacity": 0.35},
            ]
        },
        {},
    ),
    (
        "post-adjust", "Postprocessing: grain, sharpen, saturation, contrast",
        "The exact, non-model half of compose.py.",
        {"post": {"grain": 0.25, "sharpen": 0.6, "saturation": 1.15, "contrast": 1.1}},
        {},
    ),
    (
        "post-seamless", "Postprocessing: seamless tiling",
        "Documented as a cross-fade heuristic, good for organic and visibly wrong for "
        "structured. This renders an organic texture so the claim can be checked fairly.",
        {
            "subject": {"primary": "a mossy cobblestone ground texture, seen from directly above"},
            "camera": {"shot": "macro", "angle": "top_down", "lens": "50mm", "focus": "deep_focus"},
            "composition": {"aspect": "1:1", "rule": "centered", "horizon": None},
            "post": {"seamless": True},
        },
        {},
    ),
    (
        "post-transparent", "Postprocessing: background to alpha",
        "A corner flood fill, not segmentation. Needs a plain ground to have a chance.",
        {
            "intent": "game_asset",
            "subject": {"primary": "a single brass ship's bell, isolated on a plain white background"},
            "camera": {"shot": "tight", "lens": "50mm", "focus": "tack_sharp"},
            "composition": {"aspect": "1:1", "rule": "centered", "horizon": None},
            "post": {"transparent_bg": True},
        },
        {},
    ),
    (
        "regional", "Regional prompting",
        "Per-layer prompts as masked cross-attention. CLAUDE.md: written to request, "
        "opt-in, never run on a GPU, and every failure path falls back with a note. "
        "This is the item that finds out.",
        {
            "composition": {
                "layers": _LAYERS,
                "regional": {"enabled": True, "strength": 1.0, "feather": 24},
            }
        },
        {},
    ),
    (
        "init-img2img", "init: img2img from a rendered source",
        "Re-renders an earlier bundle's image at strength 0.55.",
        {"init": {"image": str(INIT_SOURCE), "strength": 0.55}},
        {"needs": "init-source"},
    ),
    (
        "init-inpaint", "init: inpainting by layer name",
        "Masks one named layer and regenerates only that region.",
        {
            "composition": {"layers": _LAYERS},
            "init": {"image": str(INIT_SOURCE), "strength": 0.85,
                     "mask_layer": "tower", "mask_blur": 12},
        },
        {"needs": "init-source"},
    ),
]


def feature_items() -> list[Item]:
    """Every feature path twice: smoke-sized first, then at full quality.

    The smoke pass is 640x640 at 6 steps, about a minute each, and exists so a
    path that cannot run at all says so in the first hour of the benchmark
    rather than in its fifth week.
    """
    items: list[Item] = []
    for scale in ("smoke", "full"):
        # The source image the init and inpaint items consume is produced by
        # the first plain render of each pass, so both passes are self-contained.
        base = _feature_spec(scale)
        items.append(
            Item(
                id=f"features/{scale}/init-source",
                phase="features",
                wave=f"features-{scale}",
                title=f"Source image for the init and inpaint items ({scale})",
                why="A plain txt2img whose first image the init and inpaint items load.",
                spec=base,
                model="juggernaut-xl",
                group=f"features/{scale}/init-source",
                tags={"scale": scale, "provides": "init-source"},
            )
        )
        for name, title, why, extra, tags in FEATURES:
            items.append(
                Item(
                    id=f"features/{scale}/{name}",
                    phase="features",
                    wave=f"features-{scale}",
                    title=f"{title} ({scale})",
                    why=why,
                    spec=_feature_spec(scale, **extra),
                    model="juggernaut-xl",
                    group=f"features/{name}",
                    tags={"scale": scale, "feature": name, **tags},
                )
            )
    return items


# ---------------------------------------------------------------------------
# The whole list
# ---------------------------------------------------------------------------


def build_plan(phases: Optional[Iterable[str]] = None) -> list[Item]:
    """Every item, in the order they should run.

    Phases run in this order for a reason. Features first because they are
    cheap and they fail. Settings next because what it finds decides the
    survey's fixed options, and running the survey first would risk 1,400
    hours on a setting the benchmark then shows was the wrong one. Accuracy
    before the survey because it is short and it is the part with objective
    answers. The survey last, and longest.
    """
    wanted = set(phases) if phases else {"features", "settings", "accuracy", "survey"}
    items: list[Item] = []
    if "features" in wanted:
        items += feature_items()
    if "settings" in wanted:
        items += settings_items()
    if "accuracy" in wanted:
        items += accuracy_items()
    if "survey" in wanted:
        items += survey_items()
    return items


__all__ = [
    "ASSETS_DIR",
    "INIT_SOURCE",
    "MODELS",
    "SCENES",
    "accuracy_items",
    "build_plan",
    "feature_items",
    "load_topics",
    "settings_items",
    "survey_items",
]
