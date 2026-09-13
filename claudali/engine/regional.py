"""Per-region text conditioning through masked cross-attention.

**Untested on real hardware.** It was written to a request, it is opt-in behind
``composition.regional.enabled``, and every failure path falls back to a normal
render with a note saying what did not happen. Treat the cost figures below as
reasoning, not measurement, until somebody runs it.

## What it does

A normal render conditions every pixel on the same text. This gives each layer
in ``composition.layers`` that carries a ``prompt`` its own conditioning,
applied only inside that layer's mask, so "fairies" can be asked for in one part
of the frame and "mossy trunk" in another.

## How

SDXL's UNet reads text in cross-attention only, at every transformer block, via
modules named ``attn2``. This replaces their processors with one that:

1. runs the original processor to get the global result,
2. runs it again per region with that region's embeddings,
3. blends the outputs with the region masks, resized to that block's own
   spatial resolution.

Blending the *outputs* rather than patching the attention maps is what makes it
version-tolerant: the original processor is called, whatever class it is, so
``AttnProcessor2_0`` and its successors work without this module knowing them.
It is also why the blend is correct. Each processor adds its own residual and
applies ``rescale_output_factor`` itself, and the per-region weights sum to
exactly one at every pixel, so a convex combination adds the residual once.

## Cost

Cross-attention is a small share of the UNet's work next to self-attention and
the convolution stacks, so N regions should cost well under N times the render.
Two or three regions ought to land somewhere near 1.2 to 1.5 times. Nobody has
timed it.

## Known limitations

- A region's prompt is encoded alone, without the compiled style fragments, so
  style can drift inside a mask. A ``strength`` below 1.0 leaves the global
  conditioning mixed in, which is what keeps that in check.
- A block whose sequence length does not factor to the frame's aspect is left
  unmasked rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..compiler import CompiledPrompt
from ..spec import SceneSpec
from . import quiet


@dataclass
class Region:
    """One layer's conditioning and where it applies."""

    role: str
    prompt: str
    mask: np.ndarray  # float32, (h, w), 0..1, at latent resolution
    states: Any = None  # torch.Tensor (batch, tokens, channels)


@dataclass
class RegionalHandle:
    """What was installed, so it can be taken back off again."""

    modules: list[Any] = field(default_factory=list)
    originals: list[Any] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)

    def remove(self) -> None:
        """Restore the processors this replaced.

        Always called from a ``finally``. The pipeline is cached and reused
        across jobs, so a processor left behind would silently apply one job's
        regions to the next job's render.
        """
        for module, original in zip(self.modules, self.originals):
            try:
                module.set_processor(original)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self.modules = []
        self.originals = []


def _region_mask(spec: SceneSpec, bbox: list[float], shape: str, feather: int) -> np.ndarray:
    """A soft 0..1 mask for one layer, at the latent resolution the UNet works in.

    Built at latent scale rather than image scale because every consumer is a
    further downsample of it, and feathering an already-small mask twice softens
    it away to nothing.
    """
    width, height = spec.resolution()
    latent_w, latent_h = max(1, width // 8), max(1, height // 8)
    mask = Image.new("L", (latent_w, latent_h), 0)
    draw = ImageDraw.Draw(mask)
    left, top = int(bbox[0] * latent_w), int(bbox[1] * latent_h)
    box = [
        left,
        top,
        max(left + 1, int(bbox[2] * latent_w) - 1),
        max(top + 1, int(bbox[3] * latent_h) - 1),
    ]
    if shape in {"ellipse", "column", "blob"}:
        draw.ellipse(box, fill=255)
    else:
        draw.rectangle(box, fill=255)
    if feather > 0:
        # The feather is given in image pixels; this mask is eight times smaller.
        mask = mask.filter(ImageFilter.GaussianBlur(max(1, feather // 8)))
    return np.asarray(mask, dtype=np.float32) / 255.0


def _encode(loaded: Any, pipe: Any, text: str, negative: str) -> Any:
    """Embeddings for one region prompt, stacked to match a CFG batch.

    The uncond half is the render's own negative prompt, unchanged. That is
    deliberate and it is what keeps the blend honest: with the same uncond
    everywhere, a region's contribution to the unconditional pass is identical
    to the global one, so compositing is a no-op there and only the conditional
    pass is steered.
    """
    import torch

    if loaded.compel is not None:
        with quiet.compel_tokenization():
            conditioning = loaded.compel(text, negative_prompt=negative)
        return torch.cat([conditioning.negative_embeds, conditioning.embeds], dim=0)

    positive, negative_embeds, _, _ = pipe.encode_prompt(
        prompt=text,
        negative_prompt=negative,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return torch.cat([negative_embeds, positive], dim=0)


def grid_for(sequence_length: int, width: int, height: int) -> Optional[tuple[int, int]]:
    """The (h, w) grid a flattened attention sequence came from, or None.

    Every attention block sees the latent grid at some power-of-two downsample,
    so the aspect ratio is preserved and the factorisation is recoverable. It is
    checked rather than assumed: returning None leaves that block unmasked,
    which is a weaker render, where a wrong guess would be a scrambled one.
    """
    latent_w, latent_h = max(1, width // 8), max(1, height // 8)
    for factor in (1, 2, 4, 8, 16, 32, 64):
        rows = -(-latent_h // factor)
        columns = -(-latent_w // factor)
        if rows * columns == sequence_length:
            return rows, columns
    return None


class _RegionalProcessor:
    """Wraps an attention processor and blends per-region results into its output."""

    def __init__(
        self, original: Any, regions: list[Region], width: int, height: int, strength: float
    ) -> None:
        self.original = original
        self.regions = regions
        self.width = width
        self.height = height
        self.strength = strength
        self._weights: dict[tuple[int, int], list[Any]] = {}

    def _weight_maps(self, grid: tuple[int, int], device: Any, dtype: Any) -> list[Any]:
        """Per-region blend weights at one grid size, with the global remainder first.

        Cached per grid. There are only a handful of distinct resolutions in the
        UNet, and this is called once per block per step.
        """
        import torch
        import torch.nn.functional as functional

        cached = self._weights.get(grid)
        if cached is None:
            masks = []
            for region in self.regions:
                tensor = torch.from_numpy(region.mask)[None, None]
                resized = functional.interpolate(
                    tensor, size=grid, mode="bilinear", align_corners=False
                )
                masks.append(resized.reshape(1, grid[0] * grid[1], 1).clamp(0.0, 1.0) * self.strength)

            # Overlapping regions could otherwise sum past one and invert the
            # global weight. Scale the stack down where they do, keeping ratios.
            total = torch.zeros_like(masks[0])
            for mask in masks:
                total = total + mask
            scale = torch.where(total > 1.0, 1.0 / total.clamp(min=1e-6), torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            remainder = torch.ones_like(total)
            for mask in masks:
                remainder = remainder - mask
            cached = [remainder.clamp(0.0, 1.0), *masks]
            self._weights[grid] = cached
        return [tensor.to(device=device, dtype=dtype) for tensor in cached]

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        base = self.original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Self-attention carries no text, and a 4-d output means this block was
        # never flattened to a token grid. Neither is ours to touch.
        if encoder_hidden_states is None or getattr(base, "ndim", 0) != 3:
            return base

        grid = grid_for(base.shape[1], self.width, self.height)
        if grid is None:
            return base

        batch = base.shape[0]
        outputs = [base]
        for region in self.regions:
            states = region.states
            if states.shape[0] != batch:
                if batch == 1:
                    states = states[-1:]
                elif batch % states.shape[0] == 0:
                    states = states.repeat(batch // states.shape[0], 1, 1)
                else:
                    return base
            outputs.append(
                self.original(
                    attn,
                    hidden_states,
                    encoder_hidden_states=states.to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    ),
                    attention_mask=None,
                    **kwargs,
                )
            )

        weights = self._weight_maps(grid, base.device, base.dtype)
        blended = outputs[0] * weights[0]
        for output, weight in zip(outputs[1:], weights[1:]):
            blended = blended + output * weight
        return blended


def install(
    pipe: Any, loaded: Any, spec: SceneSpec, compiled: CompiledPrompt
) -> tuple[Optional[RegionalHandle], list[str]]:
    """Patch the pipeline's cross-attention blocks for this spec's regions.

    Returns ``(None, notes)`` on any failure. Nothing here is worth killing a
    render for: the caller still gets an image, conditioned globally, and a note
    saying the regions were not applied.
    """
    notes: list[str] = []
    settings = spec.composition.regional
    if not settings.enabled or not compiled.regions:
        return None, notes

    try:
        width, height = spec.resolution()
        regions = [
            Region(
                role=entry["role"],
                prompt=entry["prompt"],
                mask=_region_mask(spec, entry["bbox"], entry["shape"], settings.feather),
            )
            for entry in compiled.regions
        ]
        for region in regions:
            region.states = _encode(loaded, pipe, region.prompt, compiled.negative_prompt)

        handle = RegionalHandle(regions=regions)
        for name, module in pipe.unet.named_modules():
            # attn2 is cross-attention; attn1 is self-attention and sees no text.
            if not name.endswith("attn2") or not hasattr(module, "set_processor"):
                continue
            handle.modules.append(module)
            handle.originals.append(module.get_processor())
            module.set_processor(
                _RegionalProcessor(
                    handle.originals[-1], regions, width, height, settings.strength
                )
            )

        if not handle.modules:
            notes.append(
                "composition.regional: no cross-attention blocks were found on this UNet, so "
                "per-region prompts were not applied and the render used the global prompt only"
            )
            return None, notes

        notes.append(
            f"composition.regional: {len(regions)} regions applied across "
            f"{len(handle.modules)} cross-attention blocks at strength {settings.strength}. "
            "This path is untested on real hardware; compare it against "
            "regional.enabled=false before trusting it."
        )
        return handle, notes
    except Exception as exc:  # noqa: BLE001 - a render must survive this
        notes.append(
            f"composition.regional failed to install ({type(exc).__name__}: {exc}); the render "
            "continued with the global prompt only, so per-layer prompts had no effect"
        )
        return None, notes


__all__ = ["Region", "RegionalHandle", "grid_for", "install"]
