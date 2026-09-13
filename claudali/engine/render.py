"""Turning a compiled spec into pixels.

The rendering path picks one of four tasks from the spec -- text to image,
img2img, inpainting, or any of those under ControlNet -- runs the sampler, and
returns images alongside the seeds that produced them. Seeds matter more than
they look: they are what makes "that one, but with warmer light" a possible
request instead of a reroll.

A render can be paused at any sampler step and resumed later, bit for bit
(``checkpoint.py`` explains how). This module only works out *where* a render
stands -- the seeds, which variations are finished, which step the current one
reached -- and hands that to its caller, which owns the disk.
"""

from __future__ import annotations

import contextlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PIL import Image, ImageFilter

from ..compiler import CompiledPrompt, compile_spec
from ..config import SETTINGS
from ..control.maps import build_control_image, control_repo_for_mode
from ..spec import SceneSpec
from . import quiet, regional
from .checkpoint import (
    RenderAborted,
    RenderController,
    RenderPaused,
    ResumeState,
    StepAborted,
    StepPaused,
    StepState,
    capture_scheduler_state,
    check_fingerprint,
    file_digest,
    image_digest,
    package_versions,
    resume_into,
)
from .pipelines import apply_cudnn_workaround, derive_pipeline, device_report, load_pipeline

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


def _free_offload_hooks(pipe: Any) -> None:
    """Park every component back on the CPU after a call that did not finish.

    A pipeline only does this at the end of a completed call. Raising out of the
    loop leaves the UNet on the GPU, and the next job's text encoders would then
    be loaded next to it on a card with no room for both.
    """
    free = getattr(pipe, "maybe_free_model_hooks", None)
    if callable(free):
        try:
            free()
        except Exception:  # noqa: BLE001 - cleanup must not mask the real exception
            pass


def denoise(
    pipe: Any,
    call_kwargs: dict[str, Any],
    seed: int,
    *,
    controller: Optional[RenderController] = None,
    resume: Optional[StepState] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Any:
    """Run the sampler for one variation and return its final latents.

    ``on_step(n)`` is called after step ``n`` finishes. A pause requested on the
    controller raises :class:`StepPaused` carrying the exact state after the step
    in progress; an abort raises :class:`StepAborted`. With ``resume``, the call
    continues from that state instead of starting afresh.

    Decoding is left to :func:`decode_latents`, so that a render paused after its
    last step, or handed on to another stage, needs no second path.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def step_end(
        step_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        if on_step is not None:
            on_step(step + 1)
        if controller is not None:
            if controller.abort_requested:
                raise StepAborted()
            if controller.pause_requested:
                raise StepPaused(
                    StepState(
                        next_step=step + 1,
                        latents=callback_kwargs["latents"].detach().to("cpu", copy=True),
                        scheduler=capture_scheduler_state(step_pipe.scheduler),
                        generator=generator.get_state(),
                    )
                )
        return callback_kwargs

    resuming = resume_into(pipe, resume, generator) if resume is not None else contextlib.nullcontext()
    try:
        with resuming:
            output = pipe(
                generator=generator,
                callback_on_step_end=step_end,
                output_type="latent",
                **call_kwargs,
            )
    except Exception:
        _free_offload_hooks(pipe)
        raise
    return output.images


def decode_latents(pipe: Any, latents: Any) -> list[Image.Image]:
    """Decode latents to images exactly as the SDXL pipelines do after their loop.

    A line-for-line mirror of ``pipeline_stable_diffusion_xl.py`` (0.40, lines
    1260-1303), which the img2img, inpainting and ControlNet pipelines repeat:
    the same fp32 upcast rule, the same latent denormalisation, the VAE tiling
    that was configured on it, the watermark when there is one, the same
    postprocess, and the offload hooks reset afterwards. ``vae.decode`` carries
    diffusers' ``apply_forward_hook``, so under model offload it still moves the
    VAE onto the GPU.
    """
    import torch

    vae = pipe.vae
    with torch.no_grad():
        needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
        if needs_upcasting:
            vae.to(dtype=torch.float32)
            latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != vae.dtype and torch.backends.mps.is_available():
            pipe.vae = vae = vae.to(latents.dtype)

        has_mean = getattr(vae.config, "latents_mean", None) is not None
        has_std = getattr(vae.config, "latents_std", None) is not None
        if has_mean and has_std:
            mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents = latents * std / vae.config.scaling_factor + mean
        else:
            latents = latents / vae.config.scaling_factor

        image = vae.decode(latents, return_dict=False)[0]
        if needs_upcasting:
            vae.to(dtype=torch.float16)

        watermark = getattr(pipe, "watermark", None)
        if watermark is not None:
            image = watermark.apply_watermark(image)
        images = pipe.image_processor.postprocess(image, output_type="pil")

    _free_offload_hooks(pipe)
    return images


def _model_identity(model_id: Optional[str]) -> Optional[str]:
    """What identifies a model's weights on disk, for the resume fingerprint.

    A catalogue model is identified by its install manifest, which changes only
    when it is reinstalled; a custom checkpoint by its size and mtime.
    """
    if model_id is None:
        return None
    from ..registry import CATALOG, get, resolve_checkpoint

    if model_id in CATALOG:
        manifest = get(model_id).local_dir / "claudali-manifest.json"
        return f"manifest {file_digest(manifest)}" if manifest.is_file() else "not installed"
    path, _layout = resolve_checkpoint(model_id)
    stat = path.stat()
    return f"size {stat.st_size} mtime {int(stat.st_mtime)}"


def _fingerprint(
    spec: SceneSpec,
    compiled: CompiledPrompt,
    task: str,
    model_id: str,
    controlnet_id: Optional[str],
    images: dict[str, Optional[Image.Image]],
) -> dict[str, Any]:
    """Everything that could make a resumed step differ from an uninterrupted one.

    Offload is left out: it moves weights between devices and changes no number.
    """
    import torch

    apply_cudnn_workaround()  # the probe decides cuDNN's state, so it runs first
    return {
        **package_versions(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model": model_id,
        "model_files": _model_identity(model_id),
        "controlnet": controlnet_id,
        "controlnet_files": _model_identity(controlnet_id),
        "dtype": SETTINGS.dtype,
        "fp16_vae_fix": SETTINGS.fp16_vae_fix,
        "vae_upcast": SETTINGS.vae_upcast,
        "vae_tiling": SETTINGS.vae_tiling,
        "attention_slicing": SETTINGS.attention_slicing,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "task": task,
        "sampler": compiled.sampler,
        "steps": compiled.steps,
        "cfg": compiled.cfg,
        "width": compiled.width,
        "height": compiled.height,
        "clip_skip": spec.render.clip_skip,
        **{name: image_digest(image) for name, image in images.items()},
    }


def _describe_resume(resume: ResumeState, variations: int) -> str:
    where = f"variation {resume.variation + 1} of {variations}"
    if resume.step is not None:
        return f"resumed {where} at step {resume.next_step} from a checkpoint"
    if resume.reason == "pause":
        return f"resumed at the start of {where}"
    return (
        f"resumed at {where}, which had no step checkpoint (stopped by '{resume.reason}'), "
        "so it restarted from step 0 with its original seed"
    )


def render(
    spec: SceneSpec,
    progress: Optional[ProgressCallback] = None,
    compiled: Optional[CompiledPrompt] = None,
    *,
    controller: Optional[RenderController] = None,
    resume: Optional[ResumeState] = None,
    force: bool = False,
    on_variation: Optional[Callable[[RenderedImage], None]] = None,
    on_checkpoint: Optional[Callable[[ResumeState], None]] = None,
) -> RenderResult:
    """Render every variation of a spec, or the rest of a paused one.

    Raises rather than guessing when a request is unsupported: silently dropping
    a field the caller set is worse than an error that says what to change.

    ``on_variation`` receives each image as soon as it is decoded, so the caller
    can save it before the next variation starts. ``on_checkpoint`` receives a
    step-free resume state before each variation begins; saving it is what lets
    even a killed process resume at a variation boundary. A pause raises
    :class:`RenderPaused` and an abort :class:`RenderAborted`, each carrying the
    resume state and the images finished in this call.

    A resume always uses the compiled prompt frozen in its checkpoint and never
    recompiles: a vocabulary edit made between pause and resume must not change
    the image. It is refused with :class:`ResumeMismatch` if anything that
    shapes the pixels changed, unless ``force`` is set.
    """
    started = time.time()
    if resume is not None:
        compiled = CompiledPrompt.from_dict(resume.compiled)
    compiled = compiled or compile_spec(spec)
    notes = [*compiled.warnings, *compiled.notes]

    task = _select_task(spec)
    controlnet_id = control_repo_for_mode(spec.control.mode) if spec.control.mode != "none" else None

    if controlnet_id is not None and task != "txt2img":
        raise ValueError(
            "ControlNet combined with init/inpainting is not supported yet. "
            "Use control for the initial composition, then refine with init on a later pass."
        )

    # Everything read from disk or derived from the spec comes before the model
    # load, so a resume that has to be refused is refused in seconds rather than
    # after a minute of loading weights.
    control_image = build_control_image(spec) if controlnet_id else None
    init_image, mask_image = _load_init_images(spec)
    model_id = spec.render.model or compiled.model
    fingerprint = _fingerprint(
        spec,
        compiled,
        task,
        model_id,
        controlnet_id,
        {"control_image": control_image, "init_image": init_image, "mask_image": mask_image},
    )

    seeds = list(resume.seeds) if resume is not None else _seeds_for(spec)
    completed: set[int] = set(resume.completed) if resume is not None else set()
    if resume is not None:
        notes.extend(check_fingerprint(resume.fingerprint, fingerprint, force))
        if completed >= set(range(len(seeds))):
            # The process died after saving the last image but before marking the
            # job done. There is nothing left to render, so skip the model load.
            notes.append("resumed a job whose every variation was already saved")
            return RenderResult(
                images=[],
                compiled=compiled,
                control_image=control_image,
                notes=notes,
                duration_s=round(time.time() - started, 2),
                task=task,
                device=device_report(),
            )
        notes.append(_describe_resume(resume, len(seeds)))

    loaded = load_pipeline(model_id, controlnet_id, compiled.sampler)
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

    images: list[RenderedImage] = []

    # img2img and inpainting start partway along the schedule, so they run
    # `steps * strength` iterations rather than `steps`. Reporting the nominal
    # count would leave progress stalled at ~55% and then jump straight to done.
    effective_steps = compiled.steps
    if task in {"img2img", "inpaint"} and spec.init is not None:
        effective_steps = max(1, int(compiled.steps * spec.init.strength))

    def state_at(index: int, step: Optional[StepState] = None, reason: str = "running") -> ResumeState:
        return ResumeState(
            seeds=seeds,
            variation=index,
            completed=sorted(completed),
            compiled=compiled.to_dict(),
            fingerprint=fingerprint,
            reason=reason,
            step=step,
        )

    def finished(images_so_far: list[RenderedImage]) -> RenderResult:
        return RenderResult(
            images=list(images_so_far),
            compiled=compiled,
            control_image=control_image,
            notes=notes,
            duration_s=round(time.time() - started, 2),
            task=task,
            device=device_report(),
        )

    try:
        for index, seed in enumerate(seeds):
            if index in completed:
                continue
            step_resume = resume.step if resume is not None and resume.variation == index else None

            # A variation resuming mid-image keeps its step checkpoint on disk
            # until it passes it; overwriting it with a boundary state here would
            # turn a crash during the resume into a restart from step 0.
            if step_resume is None and on_checkpoint is not None:
                on_checkpoint(state_at(index))
            if controller is not None and controller.abort_requested:
                raise RenderAborted(state_at(index, reason="abort"), finished(images))
            if controller is not None and controller.pause_requested:
                raise RenderPaused(state_at(index, step_resume, reason="pause"), finished(images))

            def on_step(step: int, _index: int = index) -> None:
                if progress is not None:
                    progress(step, effective_steps, _index, len(seeds))

            try:
                latents = denoise(
                    pipe, base_kwargs, seed, controller=controller, resume=step_resume, on_step=on_step
                )
            except StepPaused as paused:
                raise RenderPaused(state_at(index, paused.step, reason="pause"), finished(images)) from None
            except StepAborted:
                raise RenderAborted(state_at(index, reason="abort"), finished(images)) from None

            rendered = RenderedImage(image=decode_latents(pipe, latents)[0], seed=seed, index=index)
            images.append(rendered)
            completed.add(index)
            if on_variation is not None:
                on_variation(rendered)
    finally:
        if regional_handle is not None:
            regional_handle.remove()

    return finished(images)


__all__ = [
    "MAX_SEED",
    "RenderResult",
    "RenderedImage",
    "decode_latents",
    "denoise",
    "render",
]
