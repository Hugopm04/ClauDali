"""Enlarging an image before the hi-res pass re-samples it.

Two upscalers, picked by ``hires.upscaler``:

* ``lanczos`` needs nothing. It enlarges faithfully and invents no detail, which
  leaves all of the new detail to the img2img pass that follows.
* ``realesrgan-x4`` runs Real-ESRGAN x4plus through ``spandrel`` and then
  Lanczos-downsamples the 4x result to the target size. A sharper start, at
  the risk of the texture a GAN invents. The model and the package are both
  optional installs.

**Real-ESRGAN is untested on hardware**: neither the model nor spandrel was
installed when this was written, so the spandrel calls follow its README
(``ModelLoader().load_from_file``, the descriptor called on an RGB tensor in
0..1) rather than a run.

It works in overlapping 256-pixel tiles, blended with a linear ramp. 256 rather
than the customary 512 because with cuDNN disabled every GPU convolution goes
through im2col: the network's 64-channel 3x3 convolutions at the 4x output of a
512 tile would want a ~2.4 GB column buffer in fp32 on a 6 GB card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image

TILE = 256
OVERLAP = 16


def upscale(
    image: Image.Image, width: int, height: int, upscaler: str, model_path: Optional[Path] = None
) -> Image.Image:
    """Enlarge ``image`` to exactly ``width`` x ``height``."""
    if upscaler == "lanczos":
        return image.resize((width, height), Image.LANCZOS)
    if upscaler == "realesrgan-x4":
        if model_path is None:
            raise ValueError("the realesrgan-x4 upscaler needs its model file")
        return _esrgan(image, model_path).resize((width, height), Image.LANCZOS)
    raise ValueError(f"unknown upscaler '{upscaler}'; expected lanczos or realesrgan-x4")


def _starts(size: int, tile: int, overlap: int) -> list[int]:
    """Tile origins along one axis: overlapping, and the last one flush with the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile, tile - overlap))
    starts.append(size - tile)
    return starts


def _ramp(height: int, width: int, overlap: int) -> Any:
    """Blend weights for one tile: rising over ``overlap`` pixels at each edge, never zero.

    Never zero, because a tile at the image border is the only one covering its
    outer pixels, and dividing by the summed weights has to give them back whole.
    """
    import torch

    def axis(length: int) -> Any:
        ramp = torch.ones(length)
        edge = min(overlap, length // 2)
        if edge > 0:
            rise = torch.arange(1, edge + 1, dtype=torch.float32) / (edge + 1)
            ramp[:edge] = rise
            ramp[-edge:] = rise.flip(0)
        return ramp

    return (axis(height)[:, None] * axis(width)[None, :])[None, None]


def _esrgan(image: Image.Image, model_path: Path) -> Image.Image:
    try:
        from spandrel import ModelLoader
    except ImportError as exc:
        raise RuntimeError(
            "hires.upscaler 'realesrgan-x4' needs the spandrel package. Re-run the installer, "
            "or: .venv\\Scripts\\python -m pip install spandrel"
        ) from exc
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptor = ModelLoader().load_from_file(str(model_path))
    descriptor.to(device)
    descriptor.eval()
    scale = int(descriptor.scale)

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    source = torch.from_numpy(array).permute(2, 0, 1)[None]
    _, _, height, width = source.shape
    output = torch.zeros(1, 3, height * scale, width * scale)
    weight = torch.zeros(1, 1, height * scale, width * scale)
    try:
        for top in _starts(height, TILE, OVERLAP):
            for left in _starts(width, TILE, OVERLAP):
                patch = source[:, :, top : top + TILE, left : left + TILE]
                with torch.no_grad():
                    result = descriptor(patch.to(device)).float().cpu()
                rows, columns = result.shape[-2:]
                ramp = _ramp(rows, columns, OVERLAP * scale)
                y, x = top * scale, left * scale
                output[:, :, y : y + rows, x : x + columns] += result * ramp
                weight[:, :, y : y + rows, x : x + columns] += ramp
    finally:
        del descriptor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    blended = (output / weight.clamp(min=1e-6)).clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((blended * 255.0 + 0.5).astype(np.uint8))


__all__ = ["upscale"]
