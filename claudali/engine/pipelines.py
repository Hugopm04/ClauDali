"""Loading, configuring and caching SDXL pipelines.

Everything awkward about running SDXL on a 6 GB consumer card lives here:

* **Model CPU offload.** SDXL's UNet alone is ~5 GB in fp16. Offloading moves
  each component to the GPU only while it runs, which is what makes 1024px
  generation fit at all on a 6 GB card.
* **Precision is per job.** fp16 by default; fp32 when a spec asks for it, or
  under ``render.quality: "max"``. An fp32 UNet is ~10 GB, which no amount of
  model offload fits on a small card, so fp32 switches such a card to
  sequential offload. The pipeline cache is keyed by precision, so switching
  reloads.
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
  in memory instead of reading them again. The SDXL refiner is a second model
  and takes the same single slot, so loading it unloads the base.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..config import SETTINGS
from ..registry import get as get_model
from ..registry import resolve_checkpoint
from ..sysinfo import memory_gb
from . import quiet

logger = logging.getLogger(__name__)

# Before any diffusers import, which happens inside the functions below: one of
# the silenced warnings is printed the moment diffusers imports its pipelines.
quiet.install()

# Scheduler keys exposed in the spec, mapped to diffusers classes and the
# constructor kwargs that make them behave as the name promises. Every class
# must construct with requirements.txt alone, which a test checks: "lms" was
# dropped because LMSDiscreteScheduler needs scipy.
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
    "unipc": ("UniPCMultistepScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
}

OFFLOAD_MODES = ("model", "sequential", "none")
# Below this much VRAM fp32 SDXL does not fit under model offload: its UNet
# alone is ~10 GB, and model offload holds a whole component on the GPU.
FP32_MODEL_OFFLOAD_MIN_VRAM_GB = 12.0
# The key prefixes diffusers' own single-file VAE converter looks for.
_SINGLE_FILE_VAE_PREFIXES = ("first_stage_model.", "vae.")

_LOCK = threading.Lock()
_CACHE: "Optional[LoadedPipeline]" = None


@dataclass
class LoadedPipeline:
    """A resident pipeline and the identity of what it holds.

    ``warnings`` and ``notes`` are what loading it had to say. They belong to the
    load, so every render that reuses the pipeline repeats them: compel being
    unavailable is still true the second time.
    """

    model_id: str
    controlnet_id: Optional[str]
    pipe: Any
    # The checkpoint's own scheduler as loaded, which every sampler is built from.
    base_scheduler: Any = None
    compel: Any = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    kind: str = "base"  # "base" | "refiner"
    precision: str = "float16"
    offload: str = "model"
    # Where the weights came from, to read the checkpoint's own VAE again for the CPU.
    source: Optional[Path] = None
    layout: str = "diffusers"
    cpu_vae: Any = None  # that VAE in fp32 on the CPU, once something needed it


def torch_dtype_for(precision: str) -> Any:
    """The torch dtype for a precision name: float16, bfloat16 or float32."""
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[precision]


def device_report() -> dict[str, Any]:
    """What hardware ClauDali will actually use. Used by ``claudali doctor``."""
    import torch

    memory = memory_gb()
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "offload": SETTINGS.offload,
        "dtype": SETTINGS.dtype,
        "vae_decode": SETTINGS.vae_decode,
        "ram_total_gb": round(memory[1], 1) if memory else None,
        "ram_available_gb": round(memory[0], 1) if memory else None,
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
    warnings: tuple[str, ...] = ()


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

    A fault left in place is a warning; a workaround applied is a note.
    """
    warnings: list[str] = []
    notes: list[str] = []

    def status(broken: bool, disabled: bool, survives: bool) -> CudnnStatus:
        return CudnnStatus(broken, disabled, survives, tuple(notes), tuple(warnings))

    if mode not in {"auto", "on", "off"}:
        warnings.append(
            f"unknown CLAUDALI_CUDNN '{mode}'; expected auto, on or off. Falling back to auto."
        )
        mode = "auto"

    if mode == "off":
        set_enabled(False)
        notes.append("cuDNN is disabled by CLAUDALI_CUDNN=off")
        return status(False, True, False)

    if not measure():
        return status(False, False, False)

    if mode == "on":
        warnings.append(
            "this machine's cuDNN returns NaN from some fp16 convolutions, which renders "
            "images solid black, but CLAUDALI_CUDNN=on keeps it enabled. Unset it to let "
            "ClauDali disable cuDNN instead."
        )
        return status(True, False, True)

    set_enabled(False)
    if measure():
        # cuDNN was not the culprit. Put it back rather than paying for a
        # fallback that fixes nothing, and say so: the fp32 VAE decode is the
        # only remaining defence and it may not be enough either.
        set_enabled(True)
        warnings.append(
            "this machine returns NaN from some fp16 convolutions even with cuDNN "
            "disabled, so cuDNN is not the cause and has been left on. Images may "
            "still come out solid black; render.precision 'fp32' avoids fp16 "
            "convolutions altogether, at the cost of sequential offload on a small card."
        )
        return status(True, False, True)

    notes.append(
        "this machine's cuDNN returns NaN from some fp16 convolutions, which renders "
        "every image solid black; cuDNN is disabled for this process. On a card with "
        "no tensor cores this costs no measurable speed. Set CLAUDALI_CUDNN=on to override."
    )
    return status(True, True, False)


@lru_cache(maxsize=1)
def apply_cudnn_workaround() -> CudnnStatus:
    """Turn cuDNN off for this process when its fp16 convolutions return NaN.

    Process-wide, idempotent, and safe to call from a report: the measurement
    runs once and the setting is global anyway, so there is nothing to undo.

    Disabling cuDNN is the right lever rather than a per-shape workaround
    because the fault moves with the shape in ways no rule predicts. PyTorch
    then convolves through its own cuBLAS path, which on a card with no tensor
    cores is not slower -- measured at 0.93x of cuDNN over a real sampler step.
    The probe measures fp16 shapes only; the process-wide switch it sets
    applies to fp32 convolutions as well.
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


def _plan_vae_precision(torch_dtype: Any) -> tuple[bool, list[str], list[str]]:
    """Work out the VAE's decode precision before any weights are loaded.

    In ``auto`` this is the *second* line of defence. Disabling cuDNN already
    clears the fp16 convolution fault wherever it can, so the fp32 decode is
    only worth its cost when the fault outlived that. Returns
    ``(upcast, warnings, notes)``.
    """
    import torch

    warnings: list[str] = []
    notes: list[str] = []
    mode = SETTINGS.vae_upcast
    if mode not in {"auto", "always", "never"}:
        warnings.append(
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
    return upcast, warnings, notes


def _build_scheduler(pipe: Any, sampler: str, base: Any = None) -> list[str]:
    """Give the pipeline a fresh scheduler for the requested sampler. Returns warnings.

    Built from ``base``, the checkpoint's own scheduler as loaded, and never from
    whichever one the previous render left on a cached pipeline. ``from_config``
    carries over every setting the new class accepts, so ``dpmpp_2m_karras``
    followed by ``euler`` used to hand Karras sigmas to a sampler that never
    asked for them: the same spec and seed rendered differently depending on
    the job before it.
    """
    base = base if base is not None else pipe.scheduler
    if sampler not in SAMPLERS:
        pipe.scheduler = type(base).from_config(base.config)
        return [
            f"unknown sampler '{sampler}'; used the model's default scheduler "
            f"({type(base).__name__}). Known samplers: {', '.join(sorted(SAMPLERS))}"
        ]

    import diffusers

    class_name, kwargs = SAMPLERS[sampler]
    scheduler_class = getattr(diffusers, class_name)
    pipe.scheduler = scheduler_class.from_config(base.config, **kwargs)
    return []


def _variant_for(local_dir: Any) -> Optional[str]:
    """Return ``"fp16"`` when the downloaded files are fp16 variants.

    The installer keeps whichever variant a repo actually publishes, and repos
    differ: SDXL base ships fp16 weights for every component, while
    ``sdxl-vae-fp16-fix`` ships a single fp32 file (it is built to be *stable*
    in fp16, not stored in it). Passing ``variant="fp16"`` when no such file
    exists fails the load outright, so the variant is detected rather than
    assumed.
    """
    directory = Path(local_dir)
    if not directory.is_dir():
        return None
    return "fp16" if any(directory.rglob("*.fp16.safetensors")) else None


def _load_vae(torch_dtype: Any) -> tuple[Any, list[str]]:
    """Load the fp16-safe VAE when it is installed and enabled. Returns ``(vae, warnings)``."""
    import torch

    warnings: list[str] = []
    if not SETTINGS.fp16_vae_fix or torch_dtype is not torch.float16:
        return None, warnings

    entry = get_model("sdxl-vae-fp16-fix")
    if not entry.is_installed():
        warnings.append(
            "sdxl-vae-fp16-fix is not installed; on a GTX 16-series card images "
            "will very likely decode to solid black. Install it with: "
            "python -m installer models --add sdxl-vae-fp16-fix"
        )
        return None, warnings

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        str(entry.local_dir),
        torch_dtype=torch_dtype,
        variant=_variant_for(entry.local_dir),
        local_files_only=True,
    )
    return vae, warnings


def _plan_offload(
    setting: str, precision: str, vram_gb: Optional[float]
) -> tuple[str, list[str], list[str]]:
    """Choose the offload mode for a load. Pure, so it is testable.

    Returns ``(mode, warnings, notes)``. fp32 on a card too small for model
    offload switches to sequential offload, which is a decision taken on the
    caller's behalf and so a note.
    """
    warnings: list[str] = []
    notes: list[str] = []
    if setting not in OFFLOAD_MODES:
        warnings.append(
            f"unknown CLAUDALI_OFFLOAD '{setting}'; expected model, sequential or none. Using model."
        )
        setting = "model"
    if (
        precision == "float32"
        and setting != "sequential"
        and vram_gb is not None
        and vram_gb < FP32_MODEL_OFFLOAD_MIN_VRAM_GB
    ):
        notes.append(
            f"fp32 weights do not fit this {vram_gb:.0f} GB card under {setting} offload, so "
            "this load uses sequential CPU offload: each layer visits the GPU only while it "
            "runs. It keeps ~13-14 GB of weights in RAM and is much slower; neither has been "
            "measured on a 6 GB card."
        )
        return "sequential", warnings, notes
    if setting == "sequential":
        notes.append("sequential CPU offload: lowest VRAM, slowest")
    elif setting == "none":
        notes.append("no offload: the whole pipeline is resident in VRAM")
    return setting, warnings, notes


def _apply_memory_strategy(pipe: Any, precision: str) -> tuple[str, list[str], list[str]]:
    """Configure offloading and slicing. Returns ``(offload mode, warnings, notes)``."""
    import torch

    if not torch.cuda.is_available():
        return "none", ["CUDA is not available; rendering on CPU will take many minutes per image"], []

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    offload, warnings, notes = _plan_offload(SETTINGS.offload, precision, vram_gb)
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if SETTINGS.attention_slicing:
        pipe.enable_attention_slicing()
    # Tiling is left on by default for the encodes the img2img and inpainting
    # pipelines run themselves, where an untiled pass is the largest VRAM spike
    # in the run. ClauDali's own decodes and hi-res encodes set tiling per image
    # (engine/decode.py) and put it back afterwards.
    #
    # These helpers moved from the pipeline onto the VAE itself (they are gone
    # from the pipeline in diffusers 0.40), so try the current location first and
    # fall back for older versions that requirements.txt allows.
    warnings.extend(_enable_vae_memory_savers(pipe))
    return offload, warnings, notes


def _enable_vae_memory_savers(pipe: Any) -> list[str]:
    """Turn on VAE tiling and slicing, whichever API this diffusers exposes. Returns warnings."""
    warnings: list[str] = []
    vae = getattr(pipe, "vae", None)

    for vae_method, pipe_method in (
        ("enable_tiling", "enable_vae_tiling"),
        ("enable_slicing", "enable_vae_slicing"),
    ):
        target = getattr(vae, vae_method, None) or getattr(pipe, pipe_method, None)
        if target is None:
            warnings.append(
                f"could not enable VAE {vae_method.split('_')[1]}; decoding a 1024px "
                "image may spike VRAM. Lower the resolution if you hit an OOM."
            )
            continue
        try:
            target()
        except Exception as exc:  # noqa: BLE001 - a memory hint must not fail a render
            warnings.append(f"VAE {vae_method} failed ({type(exc).__name__}); continuing without it")
    return warnings


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
    warnings: list[str] = []
    if not upcast:
        return warnings

    vae = getattr(pipe, "vae", None)
    config = getattr(vae, "config", None)
    if config is None:
        warnings.append(
            "this card needs fp32 VAE decoding but the pipeline exposes no VAE config; "
            "images may decode to solid black. render.vae_decode 'cpu' decodes in fp32 "
            "on the CPU instead."
        )
        return warnings

    register = getattr(vae, "register_to_config", None)
    if callable(register):
        register(force_upcast=True)
    else:
        config.force_upcast = True
    return warnings


def execution_device(pipe: Any) -> Any:
    """The device this pipeline's components run on, offload hooks included."""
    import torch

    device = getattr(pipe, "_execution_device", None)
    if isinstance(device, torch.device):
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_offload_hooks(pipe: Any) -> None:
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


@contextlib.contextmanager
def _encoders_on(encoders: list[Any], device: Any, offload: str) -> Iterator[None]:
    """Hold text encoders on ``device`` while compel is constructed. See :func:`_build_compel`.

    Under sequential offload the weights are placeholders that the hooks fill in
    layer by layer and cannot be moved, so there the encoders stay put and the
    ``device`` handed to compel is what counts.
    """
    if offload == "sequential":
        yield
        return
    origins = [encoder.device for encoder in encoders]
    try:
        for encoder in encoders:
            encoder.to(device)
        yield
    finally:
        for encoder, origin in zip(encoders, origins):
            encoder.to(origin)


def _build_compel(pipe: Any, offload: str) -> tuple[Any, list[str]]:
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
    attention weight. Moving the encoders first, and back afterwards, is what
    was verified on the GPU under model offload; the 1.6 GB this needs is not
    left sitting on a 6 GB card. Under sequential offload the encoders cannot
    move, and the explicit ``device`` is relied on instead: compel 2.4 does pass
    it down to each provider (``embeddings_provider.py``, line 76). That case is
    unmeasured.
    """
    warnings: list[str] = []
    try:
        from compel import CompelForSDXL

        device = execution_device(pipe)
        with _encoders_on([pipe.text_encoder, pipe.text_encoder_2], device, offload):
            compel = CompelForSDXL(pipe, device=str(device))
        return compel, warnings
    except Exception as exc:  # noqa: BLE001 - compel failure must not be fatal
        warnings.append(
            f"compel unavailable ({type(exc).__name__}); prompt attention weights "
            "will be ignored and the raw text sent to CLIP"
        )
        return None, warnings


def _build_refiner_compel(pipe: Any, offload: str) -> tuple[Any, list[str]]:
    """compel for the refiner, which has only SDXL's second text encoder.

    ``CompelForSDXL`` expects both encoders, so this is a single-encoder
    ``Compel`` set up the way that wrapper sets up its second half: penultimate
    hidden states, pooled output, no truncation. The encoder is held on the
    execution device during construction for the reason in :func:`_build_compel`.
    """
    warnings: list[str] = []
    try:
        from compel import Compel, ReturnedEmbeddingsType

        device = execution_device(pipe)
        with _encoders_on([pipe.text_encoder_2], device, offload):
            compel = Compel(
                tokenizer=pipe.tokenizer_2,
                text_encoder=pipe.text_encoder_2,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=True,
                truncate_long_prompts=False,
                device=str(device),
            )
        return compel, warnings
    except Exception as exc:  # noqa: BLE001 - compel failure must not be fatal
        warnings.append(
            f"compel unavailable for the refiner ({type(exc).__name__}); its prompt attention "
            "weights will be ignored and the raw text sent to CLIP"
        )
        return None, warnings


def _prepare_load(precision: str) -> tuple[Any, bool, Any, list[str], list[str]]:
    """What every load does before reading weights: probe, plan the VAE, load the fix VAE.

    Returns ``(torch_dtype, upcast_vae, vae, warnings, notes)``. The probe runs
    before any weights are resident: it needs its own VRAM, and with offload
    disabled there is none to spare later.
    """
    warnings: list[str] = []
    notes: list[str] = []
    torch_dtype = torch_dtype_for(precision)
    cudnn = apply_cudnn_workaround()
    warnings.extend(cudnn.warnings)
    notes.extend(cudnn.notes)
    upcast_vae, upcast_warnings, upcast_notes = _plan_vae_precision(torch_dtype)
    warnings.extend(upcast_warnings)
    notes.extend(upcast_notes)
    vae, vae_warnings = _load_vae(torch_dtype)
    warnings.extend(vae_warnings)
    return torch_dtype, upcast_vae, vae, warnings, notes


def _finish_load(pipe: Any, precision: str, upcast_vae: bool, warnings: list[str], notes: list[str]) -> str:
    """What every load does after reading weights. Returns the offload mode used."""
    offload, memory_warnings, memory_notes = _apply_memory_strategy(pipe, precision)
    warnings.extend(memory_warnings)
    notes.extend(memory_notes)
    warnings.extend(_apply_vae_precision(pipe, upcast_vae))
    pipe.set_progress_bar_config(disable=True)
    return offload


def _is_cached(kind: str, model_id: str, controlnet_id: Optional[str], precision: str) -> bool:
    cached = _CACHE
    return cached is not None and (
        cached.kind, cached.model_id, cached.controlnet_id, cached.precision
    ) == (kind, model_id, controlnet_id, precision)


def resident() -> Optional[dict[str, Any]]:
    """What the single cache slot holds right now, or None."""
    cached = _CACHE
    if cached is None:
        return None
    return {
        "kind": cached.kind,
        "model": cached.model_id,
        "controlnet": cached.controlnet_id,
        "precision": cached.precision,
    }


def load_pipeline(
    model_id: str, controlnet_id: Optional[str] = None, precision: Optional[str] = None
) -> LoadedPipeline:
    """Load (or reuse) the base pipeline for a model, optionally with ControlNet.

    Thread-safe and single-slot: only one checkpoint is held at a time, because
    two resident SDXL models would not fit in 16 GB of system RAM alongside the
    offload buffers. ``precision`` defaults to ``CLAUDALI_DTYPE``; a different
    precision is a different pipeline. The sampler is set per render, by
    :func:`apply_sampler`.
    """
    global _CACHE

    precision = precision or SETTINGS.dtype
    with _LOCK:
        if _is_cached("base", model_id, controlnet_id, precision):
            assert _CACHE is not None
            return _CACHE

        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLPipeline,
        )

        path, layout = resolve_checkpoint(model_id)
        torch_dtype, upcast_vae, vae, warnings, notes = _prepare_load(precision)

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

        offload = _finish_load(pipe, precision, upcast_vae, warnings, notes)
        compel, compel_warnings = _build_compel(pipe, offload)
        warnings.extend(compel_warnings)

        _CACHE = LoadedPipeline(
            model_id=model_id,
            controlnet_id=controlnet_id,
            pipe=pipe,
            base_scheduler=pipe.scheduler,
            compel=compel,
            warnings=warnings,
            notes=notes,
            kind="base",
            precision=precision,
            offload=offload,
            source=Path(path),
            layout=layout,
        )
        return _CACHE


def load_refiner(model_id: str, precision: Optional[str] = None) -> LoadedPipeline:
    """Load (or reuse) the SDXL refiner as an img2img pipeline, in the single cache slot.

    Loading it unloads whatever base model is resident, which is the point of
    running every variation's base stage first. Its config asks for aesthetic
    scores and it has no first text encoder. Under fp16 it takes the fp16-fix
    VAE for the same reason the base does. **Untested on hardware.**
    """
    global _CACHE

    precision = precision or SETTINGS.dtype
    with _LOCK:
        if _is_cached("refiner", model_id, None, precision):
            assert _CACHE is not None
            return _CACHE

        from diffusers import StableDiffusionXLImg2ImgPipeline

        entry = get_model(model_id)
        if entry.kind != "refiner":
            raise ValueError(f"'{model_id}' is a {entry.kind}, not a refiner")
        if not entry.is_installed():
            raise FileNotFoundError(
                f"the refiner '{model_id}' is not installed. Run: "
                f"python -m installer models --add {model_id}"
            )

        torch_dtype, upcast_vae, vae, warnings, notes = _prepare_load(precision)
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
            "variant": _variant_for(entry.local_dir),
            "local_files_only": True,
        }
        if vae is not None:
            kwargs["vae"] = vae

        if _CACHE is not None:
            _release_locked()
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(str(entry.local_dir), **kwargs)

        offload = _finish_load(pipe, precision, upcast_vae, warnings, notes)
        compel, compel_warnings = _build_refiner_compel(pipe, offload)
        warnings.extend(compel_warnings)

        _CACHE = LoadedPipeline(
            model_id=model_id,
            controlnet_id=None,
            pipe=pipe,
            base_scheduler=pipe.scheduler,
            compel=compel,
            warnings=warnings,
            notes=notes,
            kind="refiner",
            precision=precision,
            offload=offload,
            source=entry.local_dir,
            layout="diffusers",
        )
        return _CACHE


def cpu_vae(loaded: LoadedPipeline) -> Any:
    """The checkpoint's own VAE in fp32 on the CPU, loaded on first use and kept with it.

    This is the reference decode: fp32 throughout, with the VAE the checkpoint
    shipped with rather than the fp16-fix VAE the GPU path swaps in. The weights
    on disk are fp16 variants, so it is fp32 computation on upcast weights.

    A single-file checkpoint carries its VAE inside the one file. Only those
    tensors are read, lazily through safetensors, and handed to diffusers' own
    single-file converter: loading the whole 7 GB file for 160 MB of VAE would
    not fit beside a loaded pipeline.
    """
    with _LOCK:
        if loaded.cpu_vae is not None:
            return loaded.cpu_vae

        import torch
        from diffusers import AutoencoderKL

        if loaded.source is None:
            raise ValueError("this pipeline does not record where its weights came from")
        source = Path(loaded.source)
        if loaded.layout == "single_file":
            from safetensors import safe_open

            with safe_open(str(source), framework="pt", device="cpu") as handle:
                weights = {
                    key: handle.get_tensor(key)
                    for key in handle.keys()
                    if key.startswith(_SINGLE_FILE_VAE_PREFIXES)
                }
            vae = AutoencoderKL.from_single_file(
                weights,
                config=str(get_model("sdxl-base").local_dir),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                str(source),
                subfolder="vae",
                variant=_variant_for(source / "vae"),
                torch_dtype=torch.float32,
                local_files_only=True,
            )
        vae.to("cpu")
        vae.eval()
        loaded.cpu_vae = vae
        return vae


def apply_sampler(loaded: LoadedPipeline, sampler: str) -> list[str]:
    """Put a render's sampler on the loaded pipeline. Returns warnings.

    Per render rather than per load, so a warning about an unknown sampler goes
    to the job that asked for it, instead of staying on the cached pipeline and
    repeating on every later render.
    """
    with _LOCK:
        return _build_scheduler(loaded.pipe, sampler, loaded.base_scheduler)


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
        derived = AutoPipelineForImage2Image.from_pipe(loaded.pipe)
    elif task == "inpaint":
        derived = AutoPipelineForInpainting.from_pipe(loaded.pipe)
    else:
        raise ValueError(f"unknown task '{task}'")
    # from_pipe does not carry the progress bar setting over, and a tqdm bar
    # fights the CLI's own progress line.
    derived.set_progress_bar_config(disable=True)
    return derived


def hires_pipeline(loaded: LoadedPipeline) -> Any:
    """An img2img pipeline on the loaded base weights, for the hi-res pass.

    Named by class rather than through AutoPipeline: from a ControlNet pipeline,
    AutoPipeline builds ControlNet img2img, which wants a control image at the new
    size. ``from_pipe`` into plain img2img leaves the ControlNet out, so the pass
    runs uncontrolled, as the compiled notes say.
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline

    derived = StableDiffusionXLImg2ImgPipeline.from_pipe(loaded.pipe)
    derived.set_progress_bar_config(disable=True)
    return derived


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
    "apply_sampler",
    "cpu_vae",
    "derive_pipeline",
    "device_report",
    "execution_device",
    "free_offload_hooks",
    "hires_pipeline",
    "load_pipeline",
    "load_refiner",
    "release",
    "resident",
    "torch_dtype_for",
]
