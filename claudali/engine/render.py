"""Turning a compiled spec into pixels.

The rendering path picks one of four tasks from the spec -- text to image,
img2img, inpainting, or any of those under ControlNet -- runs the sampler, and
returns images alongside the seeds that produced them. Seeds matter more than
they look: they are what makes "that one, but with warmer light" a possible
request instead of a reroll.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PIL import Image, ImageFilter

from ..compiler import CompiledPrompt, compile_spec
from ..control.maps import build_control_image, control_repo_for_mode
from ..spec import SceneSpec
from . import quiet, regional
from .pipelines import derive_pipeline, device_report, load_pipeline

# Called as (step, total_steps, variation_index, total_variations).
ProgressCallback = Callable[[int, int, int, int], None]

MAX_SEED = 2**32 - 1


@dataclass
class RenderedImage:
    """One generated image and the seed that produced it."""

    image: Image.Image
    seed: int
    index: int


@dataclass
class RenderResult:
    """Everything a caller needs to judge, reproduce, or refine a render."""

    images: list[RenderedImage]
    compiled: CompiledPrompt
    control_image: Optional[Image.Image] = None
    notes: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    task: str = "txt2img"
    device: dict[str, Any] = field(default_factory=dict)


def _seeds_for(spec: SceneSpec) -> list[int]:
    """One seed per variation.

    A fixed seed produces consecutive seeds across variations rather than the
    same image repeated, so `seed: 42, variations: 4` explores a neighbourhood
    that can be returned to exactly.
    """
    count = spec.render.variations
    if spec.render.seed is None:
        return [random.randint(0, MAX_SEED) for _ in range(count)]
    return [(spec.render.seed + offset) % (MAX_SEED + 1) for offset in range(count)]


def _encode_prompts(loaded: Any, compiled: CompiledPrompt) -> tuple[dict[str, Any], list[str]]:
    """Build the prompt kwargs, using compel for attention weights when possible."""
    notes: list[str] = []
    if loaded.compel is None:
        return (
            {"prompt": compiled.prompt, "negative_prompt": compiled.negative_prompt},
            notes,
        )
    try:
        # Both prompts go in together: SDXL requires the positive and negative
        # embeddings to be the same length, and the wrapper pads them rather than
        # truncating, which is why prompts longer than 77 tokens work at all.
        with quiet.compel_tokenization():
            conditioning = loaded.compel(compiled.prompt, negative_prompt=compiled.negative_prompt)
        return (
            {
                "prompt_embeds": conditioning.embeds,
                "pooled_prompt_embeds": conditioning.pooled_embeds,
                "negative_prompt_embeds": conditioning.negative_embeds,
                "negative_pooled_prompt_embeds": conditioning.negative_pooled_embeds,
            },
            notes,
        )
    except Exception as exc:  # noqa: BLE001 - never fail a render over weighting
        notes.append(
            f"compel encoding failed ({type(exc).__name__}: {exc}); fell back to plain "
            "prompts, so attention weights had no effect"
        )
        return (
            {"prompt": compiled.prompt, "negative_prompt": compiled.negative_prompt},
            notes,
        )


def _load_init_images(spec: SceneSpec) -> tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Load and size the img2img source and, when inpainting, its mask."""
    if spec.init is None:
        return None, None
    width, height = spec.resolution()
    init_image = Image.open(spec.init.image).convert("RGB").resize((width, height), Image.LANCZOS)

    mask_image = None
    if spec.init.mask:
        mask_image = Image.open(spec.init.mask).convert("L").resize((width, height), Image.LANCZOS)
        if spec.init.mask_blur > 0:
            # A hard mask edge leaves a visible seam where the regenerated
            # region meets the original. Feathering hides the join.
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(spec.init.mask_blur))
    return init_image, mask_image


def _select_task(spec: SceneSpec) -> str:
    if spec.init is not None and spec.init.mask:
        return "inpaint"
    if spec.init is not None:
        return "img2img"
    return "txt2img"


def render(
    spec: SceneSpec,
    progress: Optional[ProgressCallback] = None,
    compiled: Optional[CompiledPrompt] = None,
) -> RenderResult:
    """Render every variation of a spec.

    Raises rather than guessing when a request is unsupported: silently dropping
    a field the caller set is worse than an error that says what to change.
    """
    import torch

    started = time.time()
    compiled = compiled or compile_spec(spec)
    notes = [*compiled.warnings, *compiled.notes]

    task = _select_task(spec)
    controlnet_id = control_repo_for_mode(spec.control.mode) if spec.control.mode != "none" else None

    if controlnet_id is not None and task != "txt2img":
        raise ValueError(
            "ControlNet combined with init/inpainting is not supported yet. "
            "Use control for the initial composition, then refine with init on a later pass."
        )

    loaded = load_pipeline(spec.render.model or compiled.model, controlnet_id, compiled.sampler)
    notes.extend(loaded.notes)

    pipe = derive_pipeline(loaded, task)
    prompt_kwargs, encode_notes = _encode_prompts(loaded, compiled)
    notes.extend(encode_notes)

    # Regional conditioning patches the UNet's cross-attention in place, so it
    # has to be installed after the pipeline is derived and taken off again
    # whatever happens -- the pipeline is cached, and a processor left behind
    # would apply this job's regions to the next job's render.
    regional_handle, regional_notes = regional.install(pipe, loaded, spec, compiled)
    notes.extend(regional_notes)
    if regional_handle is not None and task != "txt2img":
        notes.append(
            f"composition.regional was applied to a {task} render. The masks are in latent "
            "space so this should work, but the combination has never been run; check the "
            "result against regional.enabled=false"
        )

    control_image = build_control_image(spec) if controlnet_id else None
    init_image, mask_image = _load_init_images(spec)

    base_kwargs: dict[str, Any] = dict(prompt_kwargs)
    base_kwargs.update(
        {
            "num_inference_steps": compiled.steps,
            "guidance_scale": compiled.cfg,
        }
    )
    if spec.render.clip_skip is not None:
        base_kwargs["clip_skip"] = spec.render.clip_skip

    if controlnet_id is not None:
        base_kwargs.update(
            {
                "image": control_image,
                "controlnet_conditioning_scale": spec.control.strength,
                "control_guidance_start": spec.control.start,
                "control_guidance_end": spec.control.end,
                "width": compiled.width,
                "height": compiled.height,
            }
        )
    elif task == "txt2img":
        base_kwargs.update({"width": compiled.width, "height": compiled.height})
    elif task == "img2img":
        assert spec.init is not None
        base_kwargs.update({"image": init_image, "strength": spec.init.strength})
    elif task == "inpaint":
        assert spec.init is not None
        base_kwargs.update(
            {
                "image": init_image,
                "mask_image": mask_image,
                "strength": spec.init.strength,
                "width": compiled.width,
                "height": compiled.height,
            }
        )

    seeds = _seeds_for(spec)
    images: list[RenderedImage] = []

    # img2img and inpainting start partway along the schedule, so they run
    # `steps * strength` iterations rather than `steps`. Reporting the nominal
    # count would leave progress stalled at ~55% and then jump straight to done.
    effective_steps = compiled.steps
    if task in {"img2img", "inpaint"} and spec.init is not None:
        effective_steps = max(1, int(compiled.steps * spec.init.strength))

    try:
        for index, seed in enumerate(seeds):
            generator = torch.Generator(device="cpu").manual_seed(seed)

            step_callback = None
            if progress is not None:

                def step_callback(  # noqa: F811 - rebound per variation on purpose
                    _pipe: Any,
                    step: int,
                    _timestep: Any,
                    callback_kwargs: dict[str, Any],
                    _index: int = index,
                ) -> dict[str, Any]:
                    progress(step + 1, effective_steps, _index, len(seeds))
                    return callback_kwargs

            output = pipe(
                generator=generator,
                callback_on_step_end=step_callback,
                **base_kwargs,
            )
            images.append(RenderedImage(image=output.images[0], seed=seed, index=index))
    finally:
        if regional_handle is not None:
            regional_handle.remove()

    return RenderResult(
        images=images,
        compiled=compiled,
        control_image=control_image,
        notes=notes,
        duration_s=round(time.time() - started, 2),
        task=task,
        device=device_report(),
    )


__all__ = ["RenderResult", "RenderedImage", "render", "MAX_SEED"]
