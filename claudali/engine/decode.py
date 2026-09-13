"""Latents to pixels and pixels to latents, on whichever device does it best right now.

The VAE pass is where quality and memory trade most directly on a 6 GB card.
Measured on the target laptop for one 1344x768 image (a random latent, cuDNN
disabled as ClauDali does on that card):

| Decode | Time | Memory | Result |
|---|---|---|---|
| GPU fp16, tiled, fp16-fix VAE | 25.5 s | 1.76 GB VRAM | ok |
| GPU fp16, untiled, fp16-fix VAE | 18.2 s | 6.39 GB "VRAM" | over the card; shared memory absorbed it |
| GPU fp32, untiled, checkpoint VAE | -- | wanted 8.86 GiB | out of memory |
| CPU fp32, untiled, checkpoint VAE | 29.0 s | ~5.8 GB of RAM | ok |

Tiling decodes overlapping tiles and blends them, and GroupNorm statistics
differ from tile to tile, so it is a compromise -- and on the GPU path it
applies to every SDXL size: diffusers tiles above the VAE's ``sample_size``,
which is 512 px for the fp16-fix VAE (1024 px for SDXL base's own). Without
cuDNN, PyTorch's GPU convolutions go through im2col, whose column buffer for a
single 3x3 convolution at 1344x768 is 4.75 GB in fp16 and 9.5 GB in fp32: that
is the out-of-memory above, and why the CPU is no slower. On the CPU the limit
is RAM, about 5.6 GB per megapixel.

Hence :func:`plan_vae_decode`. ``auto`` decodes on the CPU in fp32 without
tiling, with the checkpoint's own VAE, when free RAM covers the estimate, and on
the GPU with tiling otherwise. ``cpu`` forces the CPU and accepts paging, ``gpu``
tries the GPU untiled first, and ``gpu_tiled`` is what every render did before
this existed. A fallback is a note when ``auto`` chose and a warning when a mode
was forced.

Stdlib at import time: the API reads the constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .pipelines import execution_device, free_offload_hooks

# Measured once: an untiled fp32 decode at 1344x768 peaked ~5.8 GB above the
# process's baseline. Linear in pixels is an extrapolation from that point.
CPU_GB_PER_MEGAPIXEL = 5.6
# Headroom for the estimate running low and for the rest of the process.
CPU_MARGIN_GB = 1.0
MODES = ("auto", "cpu", "gpu", "gpu_tiled")


@dataclass
class DecodePlan:
    """Where one VAE pass runs, and what deciding that had to say."""

    mode: str
    device: str  # "cpu" | "gpu"
    tiled: bool
    need_gb: float
    free_gb: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def cpu_need_gb(width: int, height: int) -> float:
    """Estimated free RAM an untiled fp32 VAE pass at this size needs on the CPU."""
    return round(CPU_GB_PER_MEGAPIXEL * width * height / 1e6 + CPU_MARGIN_GB, 1)


def plan_vae_decode(
    mode: str, width: int, height: int, free_gb: Optional[float], *, what: str = "decode"
) -> DecodePlan:
    """Decide where a VAE pass runs. Pure: the caller measures ``free_gb``.

    ``free_gb`` is None when RAM could not be measured, and ``auto`` treats that
    as too little: an unknown must not pick the path that can exhaust memory.
    The messages carry no measurement, so repeating them per variation dedupes;
    the numbers go in the plan, which the renderer records per image.
    """
    need = cpu_need_gb(width, height)
    size = f"{width}x{height}"
    plan = DecodePlan(mode=mode, device="gpu", tiled=True, need_gb=need, free_gb=free_gb)
    if mode not in MODES:
        plan.warnings.append(
            f"unknown VAE decode mode '{mode}'; expected auto, cpu, gpu or gpu_tiled. Using auto."
        )
        mode = plan.mode = "auto"

    if mode == "cpu":
        plan.device, plan.tiled = "cpu", False
        if free_gb is not None and free_gb < need:
            plan.notes.append(
                f"render.vae_decode 'cpu': the VAE {what} at {size} ran on the CPU with less "
                f"free RAM than its ~{need} GB estimate, so the machine paged"
            )
    elif mode == "gpu":
        plan.tiled = False
    elif mode == "auto":
        if free_gb is None:
            plan.notes.append(
                f"VAE {what} auto: free RAM could not be measured, so {size} ran on the GPU with "
                "tiling rather than risk the CPU running out of memory. Set render.vae_decode to "
                f"'cpu' to force the untiled fp32 {what}."
            )
        elif free_gb >= need:
            plan.device, plan.tiled = "cpu", False
            plan.notes.append(
                f"VAE {what} auto: {size} ran on the CPU in fp32 without tiling, since free RAM "
                f"covered its ~{need} GB estimate"
            )
        else:
            plan.notes.append(
                f"VAE {what} auto: {size} needs ~{need} GB of free RAM for the untiled fp32 {what} "
                "on the CPU and less was free, so it ran on the GPU with tiling. Set "
                f"render.vae_decode to 'cpu' to force the untiled fp32 {what} and accept paging."
            )
    return plan


def _set_tiling(vae: Any, tiled: bool) -> None:
    method = getattr(vae, "enable_tiling" if tiled else "disable_tiling", None)
    if callable(method):
        method()


def _really_tiled(vae: Any, latents: Any, tiling_on: bool) -> bool:
    """Whether a decode with tiling switched on actually split into tiles.

    diffusers only tiles latents larger than the VAE's ``tile_latent_min_size``
    on a side: 64 (512 px) for the fp16-fix VAE, 128 (1024 px) for SDXL base's
    VAE. The record says what happened, not what was switched on.
    """
    limit = getattr(vae, "tile_latent_min_size", None)
    if not tiling_on or limit is None:
        return tiling_on
    return latents.shape[-1] > limit or latents.shape[-2] > limit


def _is_out_of_memory(exc: BaseException) -> bool:
    """Both torch's CUDA and CPU allocators raise RuntimeErrors saying so."""
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(
        phrase in text
        for phrase in ("out of memory", "not enough memory", "defaultcpuallocator", "can't allocate memory")
    )


def _denormalise(latents: Any, config: Any) -> Any:
    import torch

    mean = getattr(config, "latents_mean", None)
    std = getattr(config, "latents_std", None)
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        std = torch.tensor(std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        return latents * std / config.scaling_factor + mean
    return latents / config.scaling_factor


def _normalise(latents: Any, config: Any) -> Any:
    import torch

    mean = getattr(config, "latents_mean", None)
    std = getattr(config, "latents_std", None)
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        std = torch.tensor(std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        return (latents - mean) * config.scaling_factor / std
    return latents * config.scaling_factor


def _size(pipe: Any, latents: Any) -> str:
    factor = getattr(pipe, "vae_scale_factor", 8)
    return f"{latents.shape[-1] * factor}x{latents.shape[-2] * factor}"


def decode_latents(pipe: Any, latents: Any) -> list[Any]:
    """Decode latents on the pipeline's own VAE exactly as the SDXL pipelines do.

    A line-for-line mirror of ``pipeline_stable_diffusion_xl.py`` (0.40, lines
    1260-1303), which the img2img, inpainting and ControlNet pipelines repeat:
    the same fp32 upcast rule, the same latent denormalisation, whatever tiling
    is set on the VAE, the watermark when there is one, the same postprocess, and
    the offload hooks reset afterwards. ``vae.decode`` carries diffusers'
    ``apply_forward_hook``, so under model offload it still moves the VAE onto
    the GPU. The one difference: the fp16 cast back also happens on an error, so
    an out-of-memory retry does not find the VAE left in fp32.
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

        latents = _denormalise(latents, vae.config)
        try:
            image = vae.decode(latents, return_dict=False)[0]
        finally:
            if needs_upcasting:
                vae.to(dtype=torch.float16)

        watermark = getattr(pipe, "watermark", None)
        if watermark is not None:
            image = watermark.apply_watermark(image)
        images = pipe.image_processor.postprocess(image, output_type="pil")

    free_offload_hooks(pipe)
    return images


def _gpu_precision(vae: Any) -> str:
    import torch

    if vae.dtype == torch.float16 and vae.config.force_upcast:
        return "float32"
    return str(vae.dtype).replace("torch.", "")


def _decode_on_gpu(pipe: Any, latents: Any, plan: DecodePlan, notes: list[str]) -> tuple[Any, bool]:
    import torch

    vae = pipe.vae
    # Latents held between stages wait on the CPU, and under offload the hook on
    # vae.decode moves the VAE to the GPU but not its input. From the pipeline's
    # own loop they are already there and this changes nothing.
    latents = latents.to(execution_device(pipe), dtype=vae.dtype)
    previous = bool(getattr(vae, "use_tiling", False))
    try:
        _set_tiling(vae, plan.tiled)
        try:
            return decode_latents(pipe, latents)[0], plan.tiled
        except Exception as exc:  # noqa: BLE001 - only an out-of-memory is retried
            if plan.tiled or not _is_out_of_memory(exc):
                raise
        free_offload_hooks(pipe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        notes.append(
            f"render.vae_decode 'gpu': the untiled decode at {_size(pipe, latents)} ran out of GPU "
            "memory, so it was decoded again with tiling"
        )
        _set_tiling(vae, True)
        return decode_latents(pipe, latents)[0], True
    finally:
        # The pipelines encode init images on this VAE themselves, with the
        # tiling configured at load; leave it as it was found.
        _set_tiling(vae, previous)


def _run_on_cpu(
    run: Callable[[], Any], vae: Any, plan: DecodePlan, what: str, size: str,
    warnings: list[str], notes: list[str],
) -> tuple[Any, bool]:
    """Run a CPU VAE pass untiled, and tiled if RAM runs out. Returns ``(output, tiled)``."""
    try:
        _set_tiling(vae, False)
        try:
            return run(), False
        except Exception as exc:  # noqa: BLE001 - only an out-of-memory is retried
            if not _is_out_of_memory(exc):
                raise
        message = (
            f"the untiled fp32 {what} at {size} ran out of RAM on the CPU, so it ran again "
            "on the CPU with tiling: still fp32 and the checkpoint's own VAE, but tiled"
        )
        if plan.mode == "cpu":
            warnings.append(f"render.vae_decode 'cpu': {message}")
        else:
            notes.append(f"VAE {what} auto: {message}")
        _set_tiling(vae, True)
        return run(), True
    finally:
        _set_tiling(vae, False)


def decode(
    pipe: Any, latents: Any, plan: DecodePlan, cpu_vae: Callable[[], Any]
) -> tuple[Any, dict[str, Any], list[str], list[str]]:
    """Decode one image's latents by ``plan``.

    ``cpu_vae`` returns the checkpoint's own VAE in fp32 on the CPU; it is only
    called when the plan needs it, because loading it costs RAM. Returns the
    image, a record of what ran, warnings and notes.
    """
    import torch

    warnings, notes = list(plan.warnings), list(plan.notes)
    if plan.device == "cpu":
        vae = cpu_vae()
        size = _size(pipe, latents)
        denormalised = _denormalise(latents.detach().to("cpu", torch.float32), vae.config)

        def run() -> Any:
            with torch.no_grad():
                return vae.decode(denormalised, return_dict=False)[0]

        output, tiled = _run_on_cpu(run, vae, plan, "decode", size, warnings, notes)
        tiled = _really_tiled(vae, latents, tiled)
        watermark = getattr(pipe, "watermark", None)
        if watermark is not None:
            output = watermark.apply_watermark(output)
        image = pipe.image_processor.postprocess(output, output_type="pil")[0]
        precision = "float32"
    else:
        precision = _gpu_precision(pipe.vae)
        image, tiled = _decode_on_gpu(pipe, latents, plan, notes)
        tiled = _really_tiled(pipe.vae, latents, tiled)

    record = {
        "mode": plan.mode,
        "device": plan.device,
        "precision": precision,
        "tiled": tiled,
        "need_gb": plan.need_gb,
        "free_gb": None if plan.free_gb is None else round(plan.free_gb, 1),
    }
    return image, record, warnings, notes


def encode(
    pipe: Any, image: Any, plan: DecodePlan, cpu_vae: Callable[[], Any]
) -> tuple[Any, list[str], list[str]]:
    """Encode an image to the scaled latents an img2img pipeline starts from.

    What img2img's own ``prepare_latents`` does, on the device the plan chose.
    The result goes to the pipeline as a 4-channel ``image``, which it takes as
    latents without encoding again. It uses the distribution's mean rather than
    a sample: the upscaled image is what the pass should start from, and a
    sample would add noise before the pass adds its own. Returns ``(latents,
    warnings, notes)``, the latents on the CPU in fp32.
    """
    import torch

    warnings, notes = list(plan.warnings), list(plan.notes)
    pixels = pipe.image_processor.preprocess(image)
    size = f"{image.width}x{image.height}"

    if plan.device == "cpu":
        vae = cpu_vae()

        def run() -> Any:
            with torch.no_grad():
                return vae.encode(pixels.to("cpu", torch.float32)).latent_dist.mode()

        latents, _tiled = _run_on_cpu(run, vae, plan, "encode", size, warnings, notes)
    else:
        vae = pipe.vae
        upcast = vae.dtype == torch.float16 and vae.config.force_upcast
        dtype = torch.float32 if upcast else vae.dtype
        previous = bool(getattr(vae, "use_tiling", False))
        try:
            if upcast:
                vae.to(dtype=torch.float32)
            _set_tiling(vae, plan.tiled)
            try:
                with torch.no_grad():
                    latents = vae.encode(pixels.to(execution_device(pipe), dtype)).latent_dist.mode()
            except Exception as exc:  # noqa: BLE001 - only an out-of-memory is retried
                if plan.tiled or not _is_out_of_memory(exc):
                    raise
                free_offload_hooks(pipe)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                notes.append(
                    f"render.vae_decode 'gpu': the untiled encode at {size} ran out of GPU memory, "
                    "so it was encoded again with tiling"
                )
                _set_tiling(vae, True)
                with torch.no_grad():
                    latents = vae.encode(pixels.to(execution_device(pipe), dtype)).latent_dist.mode()
        finally:
            _set_tiling(vae, previous)
            if upcast:
                vae.to(dtype=torch.float16)
            free_offload_hooks(pipe)

    latents = _normalise(latents.to(torch.float32), vae.config)
    return latents.detach().to("cpu"), warnings, notes


__all__ = [
    "CPU_GB_PER_MEGAPIXEL",
    "CPU_MARGIN_GB",
    "MODES",
    "DecodePlan",
    "cpu_need_gb",
    "decode",
    "decode_latents",
    "encode",
    "plan_vae_decode",
]
