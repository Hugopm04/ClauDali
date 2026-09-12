"""Loading, configuring and caching SDXL pipelines.

Everything awkward about running SDXL on a 6 GB consumer card lives here:

* **Model CPU offload.** SDXL's UNet alone is ~5 GB in fp16. Offloading moves
  each component to the GPU only while it runs, which is what makes 1024px
  generation fit at all on a 6 GB card.
* **The GTX 16-series VAE bug.** Turing GTX cards produce NaNs in the stock fp16
  VAE, and every image decodes to solid black. ClauDali swaps in the fp16-fix
  VAE by default, which is the standard remedy.
* **The cuDNN fp16 convolution fault.** On some driver and cuDNN builds an fp16
  convolution returns NaNs for a quarter of its values, from finite inputs and
  finite weights, turning every image solid black. Which shapes are hit is not
  predictable -- it depends on the batch, the channel counts *and* the spatial
  size together -- so the remedy is to stop using cuDNN for the whole process
  rather than to work around one shape. The card is measured once; if it fails,
  cuDNN is disabled and the measurement repeated to confirm the fallback is
  sound. An fp32 VAE decode remains as a second line of defence.
* **One resident pipeline.** Loading SDXL from disk costs 30-60 s. Pipelines for
  other tasks are derived with ``from_pipe``, which reuses the weights already
  in memory instead of reading them again.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

from ..config import SETTINGS
from ..registry import get as get_model
from ..registry import resolve_checkpoint

logger = logging.getLogger(__name__)

# Scheduler keys exposed in the spec, mapped to diffusers classes and the
# constructor kwargs that make them behave as the name promises.
SAMPLERS: dict[str, tuple[str, dict[str, Any]]] = {
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++"}),
    "dpmpp_2m_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpmpp_sde_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
}

_LOCK = threading.Lock()
_CACHE: "Optional[LoadedPipeline]" = None


@dataclass
class LoadedPipeline:
    """A resident base pipeline and the identity of what it holds."""

    model_id: str
    controlnet_id: Optional[str]
    pipe: Any
    compel: Any = None
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def device_report() -> dict[str, Any]:
    """What hardware ClauDali will actually use. Used by ``claudali doctor``."""
    import torch

    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "offload": SETTINGS.offload,
        "dtype": SETTINGS.dtype,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": properties.name,
                "vram_gb": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
        # Turing GTX cards (7.5, no tensor cores) are the ones that need the
        # fp16 VAE fix. Reporting it makes a black-image bug self-diagnosing.
        report["needs_fp16_vae_fix"] = properties.major == 7 and "GTX" in properties.name
        report["cudnn"] = torch.backends.cudnn.version()
        status = apply_cudnn_workaround()
        report["fp16_conv_broken"] = status.broken
        report["cudnn_disabled"] = status.disabled
        report["fp16_conv_broken_without_cudnn"] = status.survives_workaround
        report["vae_upcast"] = SETTINGS.vae_upcast
    return report


# (batch, in_channels, out_channels, height, width, kernel) shapes that have
# actually been measured returning NaN on this project's target machine. They
# are a *sample*, not a specification: the fault is not predictable from the
# shape, so a pass here is no proof the build is sound, while a single failure
# is proof it is not. Together they cost about 70 MB of VRAM and a few ms.
_CONV_PROBE_SHAPES = (
    (1, 256, 128, 256, 256, 3),  # VAE decoder's last block, at 1024px and above
    (2, 2560, 1280, 24, 42, 1),  # UNet up_blocks.0 shortcut, 1344x768 under CFG
    (2, 1920, 1280, 24, 42, 1),  # the second resnet of the same block
)


@dataclass(frozen=True)
class CudnnStatus:
    """What the fp16 convolution measurement found, and what was done about it."""

    broken: bool
    disabled: bool
    survives_workaround: bool
    notes: tuple[str, ...] = ()


def _conv_probe_fails() -> bool:
    """Does any probe shape return NaN with the current backend settings?

    Measured rather than inferred from the card's name. On a GTX 1660 Ti with
    driver 591.86 and cuDNN 9.1 these convolutions return NaN for exactly a
    quarter of their output from finite inputs and finite weights, while
    neighbouring shapes are clean: 1344x768 fails where 768x1344 and 1024x1024
    do not, and only at batch 2. So a "is this a Turing GTX card" test would
    both over- and under-fire, and so would any rule about kernel size or
    narrowing, which is why the whole table is tried.

    A failure to run the probe at all is reported as *not* broken: an
    unavailable measurement must not silently reconfigure the renderer.
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return False
    try:
        for batch, in_channels, out_channels, height, width, kernel in _CONV_PROBE_SHAPES:
            generator = torch.Generator(device="cuda").manual_seed(0)
            activations = torch.randn(batch, in_channels, height, width, device="cuda",
                                      dtype=torch.float16, generator=generator) * 4
            weights = torch.randn(out_channels, in_channels, kernel, kernel, device="cuda",
                                  dtype=torch.float16, generator=generator) * 0.02
            with torch.no_grad():
                result = F.conv2d(activations, weights, padding=kernel // 2)
            torch.cuda.synchronize()
            if bool(torch.isnan(result).any()):
                return True
            del activations, weights, result
            torch.cuda.empty_cache()
        return False
    except Exception as exc:  # noqa: BLE001 - a probe must never break a render
        logger.warning("fp16 convolution probe failed (%s); assuming the card is sound", exc)
        return False
    finally:
        torch.cuda.empty_cache()


def _plan_cudnn(
    mode: str, measure: Callable[[], bool], set_enabled: Callable[[bool], None]
) -> CudnnStatus:
    """Decide what to do about the fp16 convolution fault. Pure, so it is testable.

    ``measure()`` reports whether fp16 convolutions return NaN *with cuDNN in
    whatever state it is currently in*, and ``set_enabled()`` changes that
    state, so the two are called in turn rather than up front.
    """
    notes: list[str] = []
    if mode not in {"auto", "on", "off"}:
        notes.append(
            f"unknown CLAUDALI_CUDNN '{mode}'; expected auto, on or off. Falling back to auto."
        )
        mode = "auto"

    if mode == "off":
        set_enabled(False)
        notes.append("cuDNN is disabled by CLAUDALI_CUDNN=off")
        return CudnnStatus(False, True, False, tuple(notes))

    if not measure():
        return CudnnStatus(False, False, False, tuple(notes))

    if mode == "on":
        notes.append(
            "this machine's cuDNN returns NaN from some fp16 convolutions, which renders "
            "images solid black, but CLAUDALI_CUDNN=on keeps it enabled. Unset it to let "
            "ClauDali disable cuDNN instead."
        )
        return CudnnStatus(True, False, True, tuple(notes))

    set_enabled(False)
    if measure():
        # cuDNN was not the culprit. Put it back rather than paying for a
        # fallback that fixes nothing, and say so: the fp32 VAE decode is the
        # only remaining defence and it may not be enough either.
        set_enabled(True)
        notes.append(
            "this machine returns NaN from some fp16 convolutions even with cuDNN "
            "disabled, so cuDNN is not the cause and has been left on. Images may "
            "still come out solid black; set CLAUDALI_DTYPE=float32 to avoid fp16 "
            "convolutions altogether."
        )
        return CudnnStatus(True, False, True, tuple(notes))

    notes.append(
        "this machine's cuDNN returns NaN from some fp16 convolutions, which renders "
        "every image solid black; cuDNN is disabled for this process. On a card with "
        "no tensor cores this costs no measurable speed. Set CLAUDALI_CUDNN=on to override."
    )
    return CudnnStatus(True, True, False, tuple(notes))


@lru_cache(maxsize=1)
def apply_cudnn_workaround() -> CudnnStatus:
    """Turn cuDNN off for this process when its fp16 convolutions return NaN.

    Process-wide, idempotent, and safe to call from a report: the measurement
    runs once and the setting is global anyway, so there is nothing to undo.

    Disabling cuDNN is the right lever rather than a per-shape workaround
    because the fault moves with the shape in ways no rule predicts. PyTorch
    then convolves through its own cuBLAS path, which on a card with no tensor
    cores is not slower -- measured at 0.93x of cuDNN over a real sampler step.
    """
    import torch

    def set_enabled(flag: bool) -> None:
        torch.backends.cudnn.enabled = flag

    return _plan_cudnn(SETTINGS.cudnn, _conv_probe_fails, set_enabled)


def _should_upcast_vae(mode: str, is_fp16: bool, probe: Callable[[], bool]) -> bool:
    """Decide whether the VAE must decode in fp32. Pure, so it is testable.

    ``probe`` is only called for ``auto``, which keeps the GPU measurement out
    of the two modes that have already made the decision.
    """
    if not is_fp16 or mode == "never":
        return False
    if mode == "always":
        return True
    return probe()


def _plan_vae_precision(torch_dtype: Any) -> tuple[bool, list[str]]:
    """Work out the VAE's decode precision before any weights are loaded.

    In ``auto`` this is the *second* line of defence. Disabling cuDNN already
    clears the fp16 convolution fault wherever it can, so the fp32 decode is
    only worth its cost when the fault outlived that.
    """
    import torch

    notes: list[str] = []
    mode = SETTINGS.vae_upcast
    if mode not in {"auto", "always", "never"}:
        notes.append(
            f"unknown CLAUDALI_VAE_UPCAST '{mode}'; expected auto, always or never. "
            "Falling back to auto."
        )
        mode = "auto"

    status = apply_cudnn_workaround()
    upcast = _should_upcast_vae(
        mode, torch_dtype is torch.float16, lambda: status.survives_workaround
    )
    if upcast and mode == "auto":
        notes.append(
            "fp16 convolutions still return NaNs with cuDNN disabled, which decodes "
            "every image to solid black; the VAE will decode in fp32 instead. Renders "
            "are a little slower. Set CLAUDALI_VAE_UPCAST=never to override."
        )
    return upcast, notes


def _build_scheduler(pipe: Any, sampler: str) -> list[str]:
    """Swap the pipeline's scheduler for the requested sampler."""
    notes: list[str] = []
    if sampler not in SAMPLERS:
        notes.append(f"unknown sampler '{sampler}'; keeping the model's default")
        return notes

    import diffusers

    class_name, kwargs = SAMPLERS[sampler]
    scheduler_class = getattr(diffusers, class_name)
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config, **kwargs)
    return notes


def _variant_for(local_dir: Any) -> Optional[str]:
    """Return ``"fp16"`` when the downloaded files are fp16 variants.

    The installer keeps whichever variant a repo actually publishes, and repos
    differ: SDXL base ships fp16 weights for every component, while
    ``sdxl-vae-fp16-fix`` ships a single fp32 file (it is built to be *stable*
    in fp16, not stored in it). Passing ``variant="fp16"`` when no such file
    exists fails the load outright, so the variant is detected rather than
    assumed.
    """
    from pathlib import Path

    directory = Path(local_dir)
    if not directory.is_dir():
        return None
    return "fp16" if any(directory.rglob("*.fp16.safetensors")) else None


def _load_vae(torch_dtype: Any) -> tuple[Any, list[str]]:
    """Load the fp16-safe VAE when it is installed and enabled."""
    notes: list[str] = []
    if not SETTINGS.fp16_vae_fix or SETTINGS.dtype != "float16":
        return None, notes

    entry = get_model("sdxl-vae-fp16-fix")
    if not entry.is_installed():
        notes.append(
            "sdxl-vae-fp16-fix is not installed; on a GTX 16-series card images "
            "will very likely decode to solid black. Install it with: "
            "python -m installer models --add sdxl-vae-fp16-fix"
        )
        return None, notes

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        str(entry.local_dir),
        torch_dtype=torch_dtype,
        variant=_variant_for(entry.local_dir),
        local_files_only=True,
    )
    return vae, notes


def _apply_memory_strategy(pipe: Any) -> list[str]:
    """Configure offloading and slicing for the available VRAM."""
    notes: list[str] = []
    import torch

    if not torch.cuda.is_available():
        notes.append("CUDA is not available; rendering on CPU will take many minutes per image")
        return notes

    if SETTINGS.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        notes.append("sequential CPU offload: lowest VRAM, slowest")
    elif SETTINGS.offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
        notes.append("no offload: the whole pipeline is resident in VRAM")

    if SETTINGS.attention_slicing:
        pipe.enable_attention_slicing()
    if SETTINGS.vae_tiling:
        # Tiling decodes the latent in chunks. Without it, the VAE decode of a
        # 1024px image is often the single largest VRAM spike in the run.
        #
        # These helpers moved from the pipeline onto the VAE itself (they are
        # gone from the pipeline in diffusers 0.40), so try the current location
        # first and fall back for older versions that requirements.txt allows.
        notes.extend(_enable_vae_memory_savers(pipe))
    return notes


def _enable_vae_memory_savers(pipe: Any) -> list[str]:
    """Turn on VAE tiling and slicing, whichever API this diffusers exposes."""
    notes: list[str] = []
    vae = getattr(pipe, "vae", None)

    for vae_method, pipe_method in (
        ("enable_tiling", "enable_vae_tiling"),
        ("enable_slicing", "enable_vae_slicing"),
    ):
        target = getattr(vae, vae_method, None) or getattr(pipe, pipe_method, None)
        if target is None:
            notes.append(
                f"could not enable VAE {vae_method.split('_')[1]}; decoding a 1024px "
                "image may spike VRAM. Lower the resolution if you hit an OOM."
            )
            continue
        try:
            target()
        except Exception as exc:  # noqa: BLE001 - a memory hint must not fail a render
            notes.append(f"VAE {vae_method} failed ({type(exc).__name__}); continuing without it")
    return notes


def _apply_vae_precision(pipe: Any, upcast: bool) -> list[str]:
    """Route the decode through fp32 when the card cannot be trusted in fp16.

    This sets ``force_upcast``, which is diffusers' own switch: the pipeline
    then casts the VAE to fp32 for the decode and back afterwards. Doing it that
    way rather than casting the module here keeps img2img and inpainting, which
    share these weights through ``from_pipe``, covered by the same flag.

    Written through ``register_to_config`` rather than by assigning to
    ``vae.config.force_upcast``. A config is a ``FrozenDict``, and assigning to
    it sets an *attribute* while leaving the dict entry at its old value, so the
    two disagree: the pipeline happens to read the attribute today, but anything
    reading the entry would silently skip the upcast and decode black.
    """
    notes: list[str] = []
    if not upcast:
        return notes

    vae = getattr(pipe, "vae", None)
    config = getattr(vae, "config", None)
    if config is None:
        notes.append(
            "this card needs fp32 VAE decoding but the pipeline exposes no VAE config; "
            "images may decode to solid black. Set CLAUDALI_DTYPE=float32 to be safe."
        )
        return notes

    register = getattr(vae, "register_to_config", None)
    if callable(register):
        register(force_upcast=True)
    else:
        config.force_upcast = True
    return notes


def _execution_device(pipe: Any) -> Any:
    """The device this pipeline's components run on, offload hooks included."""
    import torch

    device = getattr(pipe, "_execution_device", None)
    if isinstance(device, torch.device):
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_compel(pipe: Any) -> tuple[Any, list[str]]:
    """Set up compel so ``(phrase)1.15`` attention weights actually take effect.

    Without compel, weight syntax is passed to CLIP as literal punctuation --
    the parentheses become tokens and the number does nothing. If compel cannot
    be constructed, the caller is told rather than silently losing every weight.

    ``CompelForSDXL`` is the wrapper compel expects to be used for two text
    encoders. Driving the bare ``Compel`` class with a list of encoders still
    works but its padding helper does not: it reaches for an attribute the
    multi-encoder provider has never had, so any prompt whose positive and
    negative differ in token length fails. The wrapper pads them itself.

    Construction has to happen with the text encoders on the execution device.
    Each provider captures the device of the encoder it was handed, once, at
    construction. Under CPU offload the encoders are parked on the CPU at that
    moment, so every provider records ``cpu`` and builds its token ids there,
    while the offload hook has moved the weights to the GPU by the time they are
    used. The mismatch raises at ``index_select`` and silently costs every
    attention weight -- the ``device`` argument alone does not fix it, because it
    is not passed down to the providers. Moving the encoders first, and back
    afterwards, is what makes them record the GPU. The 1.6 GB this needs is not
    left sitting on a 6 GB card: the encoders go straight back to where they were.
    """
    notes: list[str] = []
    try:
        from compel import CompelForSDXL

        encoders = [pipe.text_encoder, pipe.text_encoder_2]
        device = _execution_device(pipe)
        origins = [encoder.device for encoder in encoders]
        try:
            for encoder in encoders:
                encoder.to(device)
            compel = CompelForSDXL(pipe, device=str(device))
        finally:
            for encoder, origin in zip(encoders, origins):
                encoder.to(origin)
        return compel, notes
    except Exception as exc:  # noqa: BLE001 - compel failure must not be fatal
        notes.append(
            f"compel unavailable ({type(exc).__name__}); prompt attention weights "
            "will be ignored and the raw text sent to CLIP"
        )
        return None, notes


def load_pipeline(
    model_id: str, controlnet_id: Optional[str] = None, sampler: str = "dpmpp_2m_karras"
) -> LoadedPipeline:
    """Load (or reuse) the base pipeline for a model, optionally with ControlNet.

    Thread-safe and single-slot: only one checkpoint is held at a time, because
    two resident SDXL models would not fit in 16 GB of system RAM alongside the
    offload buffers.
    """
    global _CACHE

    with _LOCK:
        if (
            _CACHE is not None
            and _CACHE.model_id == model_id
            and _CACHE.controlnet_id == controlnet_id
        ):
            # Reuse the resident weights, but honour a different sampler. Load-time
            # notes are kept: a warning that compel is unavailable is still true on
            # the second render, and dropping it would hide a real problem.
            for note in _build_scheduler(_CACHE.pipe, sampler):
                if note not in _CACHE.notes:
                    _CACHE.notes.append(note)
            return _CACHE

        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLPipeline,
        )

        notes: list[str] = []
        torch_dtype = SETTINGS.torch_dtype
        path, layout = resolve_checkpoint(model_id)

        # Probe the card before any weights are resident: the measurement needs
        # its own VRAM, and with offload disabled there is none to spare later.
        notes.extend(apply_cudnn_workaround().notes)
        upcast_vae, upcast_notes = _plan_vae_precision(torch_dtype)
        notes.extend(upcast_notes)

        vae, vae_notes = _load_vae(torch_dtype)
        notes.extend(vae_notes)

        common: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
        }
        if vae is not None:
            common["vae"] = vae

        if _CACHE is not None:
            # Drop the previous model before allocating the next one, or the two
            # briefly coexist and the machine swaps itself to a standstill.
            _release_locked()

        if layout == "single_file":
            base_entry = get_model("sdxl-base")
            if not base_entry.is_installed():
                raise FileNotFoundError(
                    "single-file checkpoints need sdxl-base installed for their configs. "
                    "Run: python -m installer models --add sdxl-base"
                )
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(path), config=str(base_entry.local_dir), local_files_only=True, **common
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(path), variant=_variant_for(path), local_files_only=True, **common
            )

        if controlnet_id is not None:
            control_entry = get_model(controlnet_id)
            if not control_entry.is_installed():
                raise FileNotFoundError(
                    f"control mode needs '{controlnet_id}'. Run: "
                    f"python -m installer models --add {controlnet_id}"
                )
            controlnet = ControlNetModel.from_pretrained(
                str(control_entry.local_dir),
                torch_dtype=torch_dtype,
                variant=_variant_for(control_entry.local_dir),
                local_files_only=True,
            )
            pipe = StableDiffusionXLControlNetPipeline.from_pipe(pipe, controlnet=controlnet)

        notes.extend(_build_scheduler(pipe, sampler))
        notes.extend(_apply_memory_strategy(pipe))
        notes.extend(_apply_vae_precision(pipe, upcast_vae))
        pipe.set_progress_bar_config(disable=True)

        compel, compel_notes = _build_compel(pipe)
        notes.extend(compel_notes)

        _CACHE = LoadedPipeline(
            model_id=model_id,
            controlnet_id=controlnet_id,
            pipe=pipe,
            compel=compel,
            notes=notes,
        )
        return _CACHE


def derive_pipeline(loaded: LoadedPipeline, task: str) -> Any:
    """Get an img2img or inpainting pipeline sharing the loaded weights.

    ``from_pipe`` rebinds the same tensors into a different pipeline class, so
    inpainting costs no extra disk, no extra download and no second load. It is
    also why SDXL base can inpaint without a dedicated inpainting checkpoint.
    """
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting

    if task == "txt2img":
        return loaded.pipe
    if task == "img2img":
        return AutoPipelineForImage2Image.from_pipe(loaded.pipe)
    if task == "inpaint":
        return AutoPipelineForInpainting.from_pipe(loaded.pipe)
    raise ValueError(f"unknown task '{task}'")


def _release_locked() -> None:
    """Drop the cached pipeline. Caller must already hold ``_LOCK``."""
    global _CACHE
    if _CACHE is None:
        return
    _CACHE = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cleanup must never raise
        logger.debug("cache release cleanup failed", exc_info=True)


def release() -> None:
    """Unload the resident pipeline and free its memory."""
    with _LOCK:
        _release_locked()


__all__ = [
    "SAMPLERS",
    "LoadedPipeline",
    "derive_pipeline",
    "device_report",
    "load_pipeline",
    "release",
]
