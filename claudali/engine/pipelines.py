"""Loading, configuring and caching SDXL pipelines.

Everything awkward about running SDXL on a 6 GB consumer card lives here:

* **Model CPU offload.** SDXL's UNet alone is ~5 GB in fp16. Offloading moves
  each component to the GPU only while it runs, which is what makes 1024px
  generation fit at all on a 6 GB card.
* **The GTX 16-series VAE bug.** Turing GTX cards produce NaNs in the stock fp16
  VAE, and every image decodes to solid black. ClauDali swaps in the fp16-fix
  VAE by default, which is the standard remedy.
* **The cuDNN narrowing-convolution fault.** On some driver and cuDNN builds a
  fp16 3x3 convolution whose output is narrower than its input returns NaNs for
  a quarter of its values. The VAE decoder ends in exactly such a convolution,
  so this also turns every image black -- and the fp16-fix VAE makes it *more*
  likely, because that VAE declares it does not need upcasting. The card is
  measured once and the VAE decodes in fp32 when it is affected.
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
        report["fp16_narrowing_conv_broken"] = fp16_narrowing_conv_is_broken()
        report["vae_upcast"] = SETTINGS.vae_upcast
    return report


@lru_cache(maxsize=1)
def fp16_narrowing_conv_is_broken() -> bool:
    """Does this card return NaNs from a narrowing fp16 convolution?

    Measured rather than inferred from the card's name. On a GTX 1660 Ti with
    driver 591.86 and cuDNN 9.1, ``conv2d`` in fp16 with a 3x3 kernel, 256 input
    channels and 128 output channels returns NaN for exactly a quarter of its
    output, from finite inputs and finite weights. Widening convolutions, 1x1
    kernels, fp32 and images under 256px are all unaffected, so a generic "is
    this a Turing GTX card" test would both over- and under-fire.

    The probe costs about 50 MB of VRAM and runs once per process. A failure to
    run it at all is reported as *not* broken: an unavailable measurement must
    not silently switch the renderer into its slower path.
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return False
    try:
        generator = torch.Generator(device="cuda").manual_seed(0)
        activations = torch.randn(1, 256, 256, 256, device="cuda", dtype=torch.float16,
                                  generator=generator) * 4
        weights = torch.randn(128, 256, 3, 3, device="cuda", dtype=torch.float16,
                              generator=generator) * 0.02
        with torch.no_grad():
            result = F.conv2d(activations, weights, padding=1)
        torch.cuda.synchronize()
        return bool(torch.isnan(result).any())
    except Exception as exc:  # noqa: BLE001 - a probe must never break a render
        logger.warning("fp16 convolution probe failed (%s); assuming the card is sound", exc)
        return False
    finally:
        torch.cuda.empty_cache()


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
    """Work out the VAE's decode precision before any weights are loaded."""
    import torch

    notes: list[str] = []
    mode = SETTINGS.vae_upcast
    if mode not in {"auto", "always", "never"}:
        notes.append(
            f"unknown CLAUDALI_VAE_UPCAST '{mode}'; expected auto, always or never. "
            "Falling back to auto."
        )
        mode = "auto"

    upcast = _should_upcast_vae(mode, torch_dtype is torch.float16, fp16_narrowing_conv_is_broken)
    if upcast and mode == "auto":
        notes.append(
            "this card returns NaNs from narrowing fp16 convolutions, which decodes "
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
    """
    notes: list[str] = []
    if not upcast:
        return notes

    config = getattr(getattr(pipe, "vae", None), "config", None)
    if config is None:
        notes.append(
            "this card needs fp32 VAE decoding but the pipeline exposes no VAE config; "
            "images may decode to solid black. Set CLAUDALI_DTYPE=float32 to be safe."
        )
        return notes

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
