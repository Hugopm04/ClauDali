"""Loading, configuring and caching SDXL pipelines.

Everything awkward about running SDXL on a 6 GB consumer card lives here:

* **Model CPU offload.** SDXL's UNet alone is ~5 GB in fp16. Offloading moves
  each component to the GPU only while it runs, which is what makes 1024px
  generation fit at all on a 6 GB card.
* **The GTX 16-series VAE bug.** Turing GTX cards produce NaNs in the stock fp16
  VAE, and every image decodes to solid black. ClauDali swaps in the fp16-fix
  VAE by default, which is the standard remedy.
* **One resident pipeline.** Loading SDXL from disk costs 30-60 s. Pipelines for
  other tasks are derived with ``from_pipe``, which reuses the weights already
  in memory instead of reading them again.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

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
    return report


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
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
    return notes


def _build_compel(pipe: Any) -> tuple[Any, list[str]]:
    """Set up compel so ``(phrase)1.15`` attention weights actually take effect.

    Without compel, weight syntax is passed to CLIP as literal punctuation --
    the parentheses become tokens and the number does nothing. If compel cannot
    be constructed, the caller is told rather than silently losing every weight.
    """
    notes: list[str] = []
    try:
        from compel import Compel, ReturnedEmbeddingsType

        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
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
