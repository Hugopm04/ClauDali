"""Turning a compiled spec into pixels.

The rendering path picks one of four tasks from the spec -- text to image,
img2img, inpainting, or any of those under ControlNet -- runs the sampler, and
returns images alongside the seeds that produced them. Seeds matter more than
they look: they are what makes "that one, but with warmer light" a possible
request instead of a reroll.

A variation goes through up to three sampler stages (``CompiledPrompt.stages``):
the base model, the optional SDXL refiner, and an optional hi-res pass. The
refiner is a second model, so with it the stages are batched -- every
variation's base stage, then the base model is unloaded and every refiner stage
runs, then the base weights come back for the hi-res passes -- and only one UNet
is ever in memory. Without the refiner each variation goes straight from its
base stage to its hi-res pass, so its image lands sooner.

A render can be paused at any sampler step of any stage and resumed later, bit
for bit (``checkpoint.py`` explains how). This module only works out *where* a
render stands -- the seeds, which variations are finished, which stage and step
the current one reached, and the latents waiting between stages -- and hands
that to its caller, which owns the disk.
"""

from __future__ import annotations

import contextlib
import random
import time
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Iterator, Optional

from PIL import Image, ImageFilter

from .. import registry, sysinfo
from ..compiler import CompiledPrompt, compile_spec
from ..config import SETTINGS
from ..control.maps import build_control_image, build_region_masks, control_repo_for_mode
from ..spec import SceneSpec
from ..tokens import CHUNK_CONTENT_TOKENS
from . import quiet, regional
from .checkpoint import (
    STAGE_NAMES,
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
from .decode import decode, decode_latents, encode, plan_vae_decode
from .pipelines import (
    apply_cudnn_workaround,
    apply_sampler,
    cpu_vae,
    derive_pipeline,
    device_report,
    free_offload_hooks,
    hires_pipeline,
    load_pipeline,
    load_refiner,
    resident,
)
from .upscale import upscale

MAX_SEED = 2**32 - 1


@dataclass
class Progress:
    """Where a render stands, for a progress bar and an ETA.

    ``step`` of ``steps`` counts the sampler steps of one stage of one variation.
    ``done`` and ``total`` count the whole job in base-step units: a hi-res step
    counts ``scale`` squared, since it samples that many times the pixels.
    ``resumed_from`` is where ``done`` stood when this call started, so an ETA
    can be based only on the steps this session ran.
    """

    stage: str
    step: int
    steps: int
    variation: int
    variations: int
    done: float
    total: float
    resumed_from: float = 0.0


ProgressCallback = Callable[[Progress], None]


@dataclass
class RenderedImage:
    """One generated image and the seed that produced it."""

    image: Image.Image
    seed: int
    index: int
    # Where the final VAE decode ran and why, from engine/decode.py.
    decode: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    """Everything a caller needs to judge, reproduce, or refine a render."""

    images: list[RenderedImage]
    compiled: CompiledPrompt
    control_image: Optional[Image.Image] = None
    # Two channels, as in CompiledPrompt: a warning says something asked for did
    # not happen, a note says something was decided on the caller's behalf.
    warnings: list[str] = field(default_factory=list)
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


def _plain_prompts(compiled: CompiledPrompt) -> tuple[dict[str, Any], list[str]]:
    """The prompts as text, for the pipeline to encode itself. Returns warnings.

    compel is what lets a prompt run past one CLIP window. Without it diffusers
    truncates each prompt to 77 tokens, so everything after the first chunk --
    the subject restated at the head of later chunks included -- never reaches
    the model. That loss is reported with its size.
    """
    warnings: list[str] = []
    for label, count in (("prompt", compiled.tokens), ("negative prompt", compiled.negative_tokens)):
        if count is not None and count.chunks > 1:
            about = "" if count.exact else "~"
            warnings.append(
                f"without compel the {label} is truncated to CLIP's 77-token window: it is "
                f"{about}{count.tokens} tokens, and everything after the first "
                f"{CHUNK_CONTENT_TOKENS} was dropped"
            )
    return {"prompt": compiled.prompt, "negative_prompt": compiled.negative_prompt}, warnings


def _encode_prompts(loaded: Any, compiled: CompiledPrompt) -> tuple[dict[str, Any], list[str]]:
    """Build the prompt kwargs, using compel for attention weights when possible. Returns warnings."""
    if loaded.compel is None:
        return _plain_prompts(compiled)
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
            [],
        )
    except Exception as exc:  # noqa: BLE001 - never fail a render over weighting
        kwargs, warnings = _plain_prompts(compiled)
        warnings.insert(
            0,
            f"compel encoding failed ({type(exc).__name__}: {exc}); fell back to plain "
            "prompts, so attention weights had no effect",
        )
        return kwargs, warnings


def _encode_refiner_prompts(loaded: Any, compiled: CompiledPrompt) -> tuple[dict[str, Any], list[str]]:
    """The refiner's prompt kwargs. Returns warnings.

    The refiner has only SDXL's second text encoder, so its compel is a
    single-encoder ``Compel``. Given a list, it pads the positive and negative to
    the same length itself and stacks them.
    """
    if loaded.compel is None:
        return _plain_prompts(compiled)
    try:
        with quiet.compel_tokenization():
            embeds, pooled = loaded.compel([compiled.prompt, compiled.negative_prompt])
        return (
            {
                "prompt_embeds": embeds[0:1],
                "pooled_prompt_embeds": pooled[0:1],
                "negative_prompt_embeds": embeds[1:2],
                "negative_pooled_prompt_embeds": pooled[1:2],
            },
            [],
        )
    except Exception as exc:  # noqa: BLE001 - never fail a render over weighting
        kwargs, warnings = _plain_prompts(compiled)
        warnings.insert(
            0,
            f"compel encoding failed for the refiner ({type(exc).__name__}: {exc}); fell back to "
            "plain prompts, so attention weights had no effect in the refiner stage",
        )
        return kwargs, warnings


def _load_init_images(spec: SceneSpec) -> tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Load and size the img2img source and, when inpainting, its mask."""
    if spec.init is None:
        return None, None
    width, height = spec.resolution()
    init_image = Image.open(spec.init.image).convert("RGB").resize((width, height), Image.LANCZOS)

    mask_image = None
    if spec.init.mask:
        mask_image = Image.open(spec.init.mask).convert("L").resize((width, height), Image.LANCZOS)
    elif spec.init.mask_layer:
        # A layer's bbox is normalised, so this covers the layer at this spec's
        # size -- the original image's region only while the spec keeps the
        # original's layers and aspect.
        mask_image = build_region_masks(spec)[spec.init.mask_layer]
    if mask_image is not None and spec.init.mask_blur > 0:
        # A hard mask edge leaves a visible seam where the regenerated
        # region meets the original. Feathering hides the join.
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(spec.init.mask_blur))
    return init_image, mask_image


def _select_task(spec: SceneSpec) -> str:
    if spec.init is not None and (spec.init.mask or spec.init.mask_layer):
        return "inpaint"
    if spec.init is not None:
        return "img2img"
    return "txt2img"


def denoise(
    pipe: Any,
    call_kwargs: dict[str, Any],
    seed: int,
    *,
    controller: Optional[RenderController] = None,
    resume: Optional[StepState] = None,
    on_step: Optional[Callable[[int], None]] = None,
) -> Any:
    """Run the sampler for one stage of one variation and return its final latents.

    ``on_step(n)`` is called after step ``n`` finishes. A pause requested on the
    controller raises :class:`StepPaused` carrying the exact state after the step
    in progress; an abort raises :class:`StepAborted`. With ``resume``, the call
    continues from that state instead of starting afresh.

    Decoding is left to ``engine/decode.py``, so that a render paused after its
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
        free_offload_hooks(pipe)
        raise
    return output.images


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


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


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
    refiner = compiled.refiner
    esrgan = compiled.hires is not None and compiled.hires["upscaler"] == "realesrgan-x4"
    return {
        **package_versions(),
        "spandrel": _package_version("spandrel") if esrgan else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model": model_id,
        "model_files": _model_identity(model_id),
        "controlnet": controlnet_id,
        "controlnet_files": _model_identity(controlnet_id),
        "precision": compiled.precision,
        "fp16_vae_fix": SETTINGS.fp16_vae_fix,
        "vae_upcast": SETTINGS.vae_upcast,
        "vae_decode": compiled.vae_decode,
        "attention_slicing": SETTINGS.attention_slicing,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "task": task,
        "sampler": compiled.sampler,
        "steps": compiled.steps,
        "cfg": compiled.cfg,
        "width": compiled.width,
        "height": compiled.height,
        "clip_skip": spec.render.clip_skip,
        "refiner": refiner,
        "refiner_files": _model_identity(refiner["model"]) if refiner else None,
        "hires": compiled.hires,
        "upscaler_files": _model_identity("realesrgan-x4") if esrgan else None,
        **{name: image_digest(image) for name, image in images.items()},
    }


def _describe_resume(resume: ResumeState, variations: int) -> str:
    where = f"variation {resume.variation + 1} of {variations}"
    if resume.stage != "base":
        where = f"the {STAGE_NAMES.get(resume.stage, resume.stage)} stage of {where}"
    if resume.step is not None:
        return f"resumed {where} at step {resume.next_step} from a checkpoint"
    if resume.reason == "pause":
        return f"resumed at the start of {where}"
    return (
        f"resumed at {where}, which had no step checkpoint (stopped by '{resume.reason}'), "
        "so that stage restarted from its first step with its original seed"
    )


def _check_quality_stages(compiled: CompiledPrompt, task: str) -> None:
    """Refuse a quality stage that cannot run, before anything is read or loaded.

    The compiler already warned; this is where a spec that asked explicitly stops,
    rather than rendering without what it asked for.
    """
    refiner = compiled.refiner
    if refiner is not None:
        if task != "txt2img":
            raise ValueError(
                "the refiner does not support init images or inpainting yet. Set "
                "refiner.enabled to false, or render without init and refine on a later pass."
            )
        problem = registry.refiner_problem(refiner["model"])
        if problem is not None:
            raise FileNotFoundError(f"the refiner stage cannot run: {problem}")

    hires = compiled.hires
    if hires is not None:
        if int(hires["steps"] * hires["strength"]) < 1:
            raise ValueError(
                f"the hi-res pass would run {hires['steps']} steps x strength {hires['strength']} "
                "= 0 steps; raise hires.steps or hires.strength"
            )
        if hires["upscaler"] == "realesrgan-x4":
            problem = registry.upscaler_problem()
            if problem is not None:
                raise FileNotFoundError(f"hires.upscaler is 'realesrgan-x4', but {problem}")


def _stage_plan(
    compiled: CompiledPrompt, task: str, strength: float
) -> tuple[dict[str, int], dict[str, float]]:
    """Sampler steps and progress weight per stage, known before any stage runs.

    img2img and inpainting start partway along the schedule, so they run
    ``steps * strength`` iterations rather than ``steps``; reporting the nominal
    count would leave progress stalled at ~55% and then jump straight to done.
    With a refiner diffusers cuts the schedule at the handoff timestep, which
    this matches to within a step; the progress bar reads the exact count from
    the pipeline once a stage starts.
    """
    steps = compiled.steps
    counts = {"base": steps if task == "txt2img" else max(1, int(steps * strength))}
    weights = {"base": 1.0, "refiner": 1.0}
    if compiled.refiner:
        counts["base"] = max(1, round(steps * compiled.refiner["handoff"]))
        counts["refiner"] = max(1, steps - counts["base"])
    if compiled.hires:
        counts["hires"] = int(compiled.hires["steps"] * compiled.hires["strength"])
        weights["hires"] = float(compiled.hires["scale"]) ** 2
    return counts, weights


def _phases(stages: list[str]) -> list[list[str]]:
    """Stages grouped by the weights they need, in the order they run.

    With the refiner every stage is a phase of its own, run for every variation
    before the next begins, because each switch of model costs a load. Without it
    the base weights serve both stages and each variation runs them back to back.
    """
    if "refiner" in stages:
        return [[stage] for stage in stages]
    return [list(stages)]


@contextlib.contextmanager
def _one_head_at_a_time(pipe: Any) -> Iterator[None]:
    """Slice attention to one head for the hi-res pass, and restore it afterwards.

    At 2016x1152 the first transformer level holds 9072 tokens, ~165 MB of
    attention per head in fp16, next to a 5 GB UNet on a 6 GB card. Slicing
    computes the same attention in smaller pieces; only memory changes.
    """
    setter = getattr(pipe, "set_attention_slice", None)
    if not callable(setter):
        yield
        return
    setter(1)
    try:
        yield
    finally:
        if SETTINGS.attention_slicing:
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()


class _Run:
    """One call of :func:`render`: the state that moves as stages run."""

    def __init__(
        self,
        *,
        spec: SceneSpec,
        compiled: CompiledPrompt,
        task: str,
        controlnet_id: Optional[str],
        control_image: Optional[Image.Image],
        init_image: Optional[Image.Image],
        mask_image: Optional[Image.Image],
        fingerprint: dict[str, Any],
        seeds: list[int],
        completed: set[int],
        resume: Optional[ResumeState],
        controller: Optional[RenderController],
        progress: Optional[ProgressCallback],
        on_variation: Optional[Callable[[RenderedImage], None]],
        on_checkpoint: Optional[Callable[[ResumeState], None]],
        warnings: list[str],
        notes: list[str],
        started: float,
    ) -> None:
        self.spec = spec
        self.compiled = compiled
        self.task = task
        self.controlnet_id = controlnet_id
        self.control_image = control_image
        self.init_image = init_image
        self.mask_image = mask_image
        self.fingerprint = fingerprint
        self.seeds = seeds
        self.completed = completed
        self.staged: dict[int, tuple[str, Any]] = dict(resume.staged) if resume is not None else {}
        self.controller = controller
        self.progress = progress
        self.on_variation = on_variation
        self.on_checkpoint = on_checkpoint
        self.warnings = warnings
        self.notes = notes
        self.started = started
        self.images: list[RenderedImage] = []

        # A step checkpoint applies once, to the stage and variation it was taken in.
        self.step_resume = resume.step if resume is not None else None
        self.resume_at = (resume.stage, resume.variation) if resume is not None else None

        self.stages = compiled.stages
        strength = spec.init.strength if spec.init is not None else 1.0
        self.counts, self.weights = _stage_plan(compiled, task, strength)
        per_variation = self._units_through(self.stages[-1])
        self.total = per_variation * len(seeds)
        self.done = len(completed) * per_variation + sum(
            self._units_through(stage)
            for index, (stage, _latents) in self.staged.items()
            if index not in completed
        )
        partial = 0.0
        if self.step_resume is not None and resume is not None:
            partial = min(self.step_resume.next_step, self.counts[resume.stage]) * self.weights[resume.stage]
        self.resumed_from = self.done + partial

        # What the current phase has loaded. Cleared before the next phase loads,
        # or the old pipeline stays referenced and two UNets share the RAM.
        self.loaded: Any = None
        self.prompt_kwargs: dict[str, Any] = {}
        self.base_pipe: Any = None
        self.hires_pipe: Any = None
        self.regional_handle: Any = None

    # -- bookkeeping --------------------------------------------------------

    def _units_through(self, stage: str) -> float:
        included = self.stages[: self.stages.index(stage) + 1]
        return sum(self.counts[name] * self.weights[name] for name in included)

    @staticmethod
    def _say(channel: list[str], messages: list[str]) -> None:
        # A reloaded pipeline and a per-variation decode repeat themselves.
        for message in messages:
            if message not in channel:
                channel.append(message)

    def _finished(self, index: int, stage: str) -> bool:
        if index in self.completed:
            return True
        held = self.staged.get(index)
        return held is not None and self.stages.index(held[0]) >= self.stages.index(stage)

    def state_at(
        self, index: int, stage: str, step: Optional[StepState] = None, reason: str = "running"
    ) -> ResumeState:
        return ResumeState(
            seeds=self.seeds,
            variation=index,
            completed=sorted(self.completed),
            compiled=self.compiled.to_dict(),
            fingerprint=self.fingerprint,
            stage=stage,
            reason=reason,
            step=step,
            staged=dict(self.staged),
            stage_steps=self.counts[stage],
        )

    def result(self) -> RenderResult:
        return RenderResult(
            images=list(self.images),
            compiled=self.compiled,
            control_image=self.control_image,
            warnings=self.warnings,
            notes=self.notes,
            duration_s=round(time.time() - self.started, 2),
            task=self.task,
            device=device_report(),
        )

    # -- the loop -----------------------------------------------------------

    def run(self) -> RenderResult:
        try:
            for phase in _phases(self.stages):
                todo = [
                    index
                    for index in range(len(self.seeds))
                    if not all(self._finished(index, stage) for stage in phase)
                ]
                if not todo:
                    continue  # a resume past this phase needs none of its weights
                self._acquire(phase)
                for index in todo:
                    for stage in phase:
                        if not self._finished(index, stage):
                            self._run_stage(stage, index)
        finally:
            self._drop_regional()
        return self.result()

    def _acquire(self, phase: list[str]) -> None:
        """Load the weights a phase needs and prepare everything built from them."""
        self._drop_regional()
        self.loaded = self.base_pipe = self.hires_pipe = None
        self.prompt_kwargs = {}

        if phase[0] == "refiner":
            assert self.compiled.refiner is not None
            loaded = load_refiner(self.compiled.refiner["model"], self.compiled.precision)
            encoder = _encode_refiner_prompts
        else:
            held = resident()
            if (
                held is not None
                and (held["kind"], held["model"]) == ("base", self.compiled.model)
                and held["precision"] != self.compiled.precision
            ):
                self._say(self.notes, [
                    f"'{self.compiled.model}' was loaded in {held['precision']} for an earlier job, "
                    f"so it was reloaded in {self.compiled.precision}"
                ])
            loaded = load_pipeline(self.compiled.model, self.controlnet_id, self.compiled.precision)
            encoder = _encode_prompts
        self._say(self.warnings, loaded.warnings)
        self._say(self.notes, loaded.notes)
        # Before deriving: a derived pipeline shares the scheduler object it finds.
        self._say(self.warnings, apply_sampler(loaded, self.compiled.sampler))
        self.loaded = loaded

        self.prompt_kwargs, encode_warnings = encoder(loaded, self.compiled)
        self._say(self.warnings, encode_warnings)

        if "base" in phase:
            self.base_pipe = derive_pipeline(loaded, self.task)
            # Regional conditioning patches the UNet's cross-attention in place, so
            # it is installed after deriving and always taken off again: the
            # pipeline is cached, and a processor left behind would apply this
            # job's regions to the next job's render.
            handle, regional_warnings, regional_notes = regional.install(
                self.base_pipe, loaded, self.spec, self.compiled
            )
            self._say(self.warnings, regional_warnings)
            self._say(self.notes, regional_notes)
            if handle is not None and self.task != "txt2img":
                self._say(self.notes, [
                    f"composition.regional was applied to a {self.task} render. The masks are in "
                    "latent space so this should work, but the combination has never been run; "
                    "check the result against regional.enabled=false"
                ])
            self.regional_handle = handle
        if "hires" in phase:
            self.hires_pipe = hires_pipeline(loaded)

    def _drop_regional(self) -> None:
        if self.regional_handle is not None:
            self.regional_handle.remove()
            self.regional_handle = None

    def _shared_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {**self.prompt_kwargs, "guidance_scale": self.compiled.cfg}
        if self.spec.render.clip_skip is not None:
            kwargs["clip_skip"] = self.spec.render.clip_skip
        return kwargs

    def _base_call(self) -> tuple[Any, dict[str, Any]]:
        compiled, spec = self.compiled, self.spec
        kwargs = {**self._shared_kwargs(), "num_inference_steps": compiled.steps}
        if self.controlnet_id is not None:
            kwargs.update(
                {
                    "image": self.control_image,
                    "controlnet_conditioning_scale": spec.control.strength,
                    "control_guidance_start": spec.control.start,
                    "control_guidance_end": spec.control.end,
                    "width": compiled.width,
                    "height": compiled.height,
                }
            )
        elif self.task == "txt2img":
            kwargs.update({"width": compiled.width, "height": compiled.height})
        elif self.task == "img2img":
            assert spec.init is not None
            kwargs.update({"image": self.init_image, "strength": spec.init.strength})
        elif self.task == "inpaint":
            assert spec.init is not None
            kwargs.update(
                {
                    "image": self.init_image,
                    "mask_image": self.mask_image,
                    "strength": spec.init.strength,
                    "width": compiled.width,
                    "height": compiled.height,
                }
            )
        if compiled.refiner is not None:
            kwargs["denoising_end"] = float(compiled.refiner["handoff"])
        return self.base_pipe, kwargs

    def _refiner_call(self, index: int) -> tuple[Any, dict[str, Any]]:
        refiner = self.compiled.refiner
        assert refiner is not None
        _stage, latents = self.staged[index]
        kwargs = {
            **self._shared_kwargs(),
            # A 4-channel image is taken as latents: no encode, and with
            # denoising_start set, no fresh noise either.
            "image": latents,
            "denoising_start": float(refiner["handoff"]),
            "num_inference_steps": self.compiled.steps,
            "aesthetic_score": float(refiner["aesthetic_score"]),
            "negative_aesthetic_score": float(refiner["negative_aesthetic_score"]),
        }
        return self.loaded.pipe, kwargs

    def _hires_call(self, index: int) -> tuple[Any, dict[str, Any]]:
        """Decode, upscale and re-encode the variation, ready for its img2img pass."""
        hires = self.compiled.hires
        assert hires is not None
        _stage, latents = self.staged[index]
        image, _record = self._decode(self.hires_pipe, latents, self.compiled.width, self.compiled.height)

        model_path = None
        if hires["upscaler"] == "realesrgan-x4":
            entry = registry.get("realesrgan-x4")
            model_path = entry.local_dir / (entry.single_file or "")
        larger = upscale(image, hires["width"], hires["height"], hires["upscaler"], model_path)

        plan = self._plan(hires["width"], hires["height"], "encode")
        encoded, encode_warnings, encode_notes = encode(self.hires_pipe, larger, plan, self._cpu_vae)
        self._say(self.warnings, encode_warnings)
        self._say(self.notes, encode_notes)
        kwargs = {
            **self._shared_kwargs(),
            "image": encoded,
            "strength": float(hires["strength"]),
            "num_inference_steps": int(hires["steps"]),
        }
        return self.hires_pipe, kwargs

    @contextlib.contextmanager
    def _hires_context(self, pipe: Any) -> Iterator[None]:
        # Regional masks are sized for the base resolution; the pass runs without them.
        suspended = (
            self.regional_handle.suspended()
            if self.regional_handle is not None
            else contextlib.nullcontext()
        )
        with suspended, _one_head_at_a_time(pipe):
            yield

    def _run_stage(self, stage: str, index: int) -> None:
        step_resume = None
        if self.step_resume is not None and self.resume_at == (stage, index):
            step_resume, self.step_resume = self.step_resume, None

        # A stage resuming mid-image keeps its step checkpoint on disk until it
        # passes it; overwriting it with a boundary state here would turn a crash
        # during the resume into a restart from the stage's first step.
        if step_resume is None and self.on_checkpoint is not None:
            self.on_checkpoint(self.state_at(index, stage))
        if self.controller is not None and self.controller.abort_requested:
            raise RenderAborted(self.state_at(index, stage, reason="abort"), self.result())
        if self.controller is not None and self.controller.pause_requested:
            raise RenderPaused(self.state_at(index, stage, step_resume, reason="pause"), self.result())

        if stage == "base":
            pipe, kwargs = self._base_call()
        elif stage == "refiner":
            pipe, kwargs = self._refiner_call(index)
        else:
            pipe, kwargs = self._hires_call(index)

        count, weight = self.counts[stage], self.weights[stage]
        before = self.done

        def on_step(step: int) -> None:
            if self.progress is None:
                return
            self.progress(
                Progress(
                    stage=stage,
                    step=step,
                    steps=getattr(pipe, "_num_timesteps", None) or count,
                    variation=index,
                    variations=len(self.seeds),
                    done=before + min(step, count) * weight,
                    total=self.total,
                    resumed_from=self.resumed_from,
                )
            )

        context = self._hires_context(pipe) if stage == "hires" else contextlib.nullcontext()
        try:
            with context:
                latents = denoise(
                    pipe, kwargs, self.seeds[index],
                    controller=self.controller, resume=step_resume, on_step=on_step,
                )
        except StepPaused as paused:
            raise RenderPaused(
                self.state_at(index, stage, paused.step, reason="pause"), self.result()
            ) from None
        except StepAborted:
            raise RenderAborted(self.state_at(index, stage, reason="abort"), self.result()) from None
        self.done = before + count * weight

        if stage != self.stages[-1]:
            self.staged[index] = (stage, latents.detach().to("cpu"))
            return
        width, height = (
            (self.compiled.hires["width"], self.compiled.hires["height"])
            if stage == "hires" and self.compiled.hires
            else (self.compiled.width, self.compiled.height)
        )
        image, record = self._decode(pipe, latents, width, height)
        rendered = RenderedImage(image=image, seed=self.seeds[index], index=index, decode=record)
        self.images.append(rendered)
        self.completed.add(index)
        self.staged.pop(index, None)
        if self.on_variation is not None:
            self.on_variation(rendered)

    # -- the VAE --------------------------------------------------------------

    def _plan(self, width: int, height: int, what: str) -> Any:
        mode = self.compiled.vae_decode
        free = sysinfo.available_ram_gb() if mode in {"auto", "cpu"} else None
        return plan_vae_decode(mode, width, height, free, what=what)

    def _cpu_vae(self) -> Any:
        return cpu_vae(self.loaded)

    def _decode(self, pipe: Any, latents: Any, width: int, height: int) -> tuple[Image.Image, dict[str, Any]]:
        plan = self._plan(width, height, "decode")
        image, record, decode_warnings, decode_notes = decode(pipe, latents, plan, self._cpu_vae)
        self._say(self.warnings, decode_warnings)
        self._say(self.notes, decode_notes)
        return image, record


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
    can save it before the next one starts. ``on_checkpoint`` receives a
    step-free resume state before each stage of each variation begins; saving it
    is what lets even a killed process resume at a stage boundary. A pause raises
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
    warnings = list(compiled.warnings)
    notes = list(compiled.notes)

    task = _select_task(spec)
    controlnet_id = control_repo_for_mode(spec.control.mode) if spec.control.mode != "none" else None

    if controlnet_id is not None and task != "txt2img":
        raise ValueError(
            "ControlNet combined with init/inpainting is not supported yet. "
            "Use control for the initial composition, then refine with init on a later pass."
        )
    _check_quality_stages(compiled, task)

    # Everything read from disk or derived from the spec comes before the model
    # load, so a resume that has to be refused is refused in seconds rather than
    # after a minute of loading weights.
    control_image = build_control_image(spec) if controlnet_id else None
    init_image, mask_image = _load_init_images(spec)
    fingerprint = _fingerprint(
        spec,
        compiled,
        task,
        compiled.model,
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
                warnings=warnings,
                notes=notes,
                duration_s=round(time.time() - started, 2),
                task=task,
                device=device_report(),
            )
        notes.append(_describe_resume(resume, len(seeds)))

    return _Run(
        spec=spec,
        compiled=compiled,
        task=task,
        controlnet_id=controlnet_id,
        control_image=control_image,
        init_image=init_image,
        mask_image=mask_image,
        fingerprint=fingerprint,
        seeds=seeds,
        completed=completed,
        resume=resume,
        controller=controller,
        progress=progress,
        on_variation=on_variation,
        on_checkpoint=on_checkpoint,
        warnings=warnings,
        notes=notes,
        started=started,
    ).run()


__all__ = [
    "MAX_SEED",
    "Progress",
    "RenderResult",
    "RenderedImage",
    "decode_latents",
    "denoise",
    "render",
]
