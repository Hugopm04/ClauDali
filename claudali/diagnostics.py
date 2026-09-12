"""Measurements that make a render judgeable without opening it.

These exist for the refine loop. An agent looking at a downscaled preview can
see that an image is muddy, but not that its highlights are clipped on 4% of
pixels or that all its visual weight sits in the bottom-left corner. Numbers
catch what a small preview hides, and they are cheap: every metric here is a
numpy or OpenCV pass over an already-decoded image.

Nothing here decides anything. The metrics are reported, never enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 relative luminance, which tracks perceived brightness."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def exposure_stats(rgb: np.ndarray) -> dict[str, Any]:
    """Brightness distribution and how much of it is clipped."""
    luma = _luminance(rgb)
    p01, p50, p99 = np.percentile(luma, [1, 50, 99])
    shadow_clip = float((luma <= 2).mean() * 100)
    highlight_clip = float((luma >= 253).mean() * 100)
    return {
        "mean": round(float(luma.mean()), 1),
        "std": round(float(luma.std()), 1),
        "p01": round(float(p01), 1),
        "median": round(float(p50), 1),
        "p99": round(float(p99), 1),
        "dynamic_range": round(float(p99 - p01), 1),
        "shadow_clip_pct": round(shadow_clip, 2),
        "highlight_clip_pct": round(highlight_clip, 2),
    }


def sharpness(rgb: np.ndarray) -> dict[str, Any]:
    """Laplacian variance: the standard cheap focus measure.

    Absolute values are only comparable between images of similar size and
    content, so this is most useful for ranking variations of one spec.
    """
    import cv2

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    edges = cv2.Canny(grey, 100, 200)
    return {
        "laplacian_variance": round(float(laplacian.var()), 1),
        "edge_density_pct": round(float((edges > 0).mean() * 100), 2),
    }


def colorfulness(rgb: np.ndarray) -> dict[str, Any]:
    """Hasler-Susstrunk colourfulness, plus mean saturation.

    Distinguishes "deliberately muted" from "washed out": a low-saturation image
    with a wide dynamic range is a colour choice, one with a narrow range is a
    flat render.
    """
    import cv2

    red, green, blue = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(
        np.float32
    ), rgb[..., 2].astype(np.float32)
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "colorfulness": round(float(std_root + 0.3 * mean_root), 1),
        "mean_saturation": round(float(hsv[..., 1].mean()), 1),
    }


def dominant_palette(rgb: np.ndarray, clusters: int = 5) -> list[dict[str, Any]]:
    """The image's main colours by k-means, as hex with coverage percentages.

    Downsampled first: clustering a full 1024x1024 image is slow and tells you
    nothing extra.
    """
    import cv2

    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    samples = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        samples, clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=clusters)
    total = counts.sum()

    palette = []
    for index in np.argsort(-counts):
        red, green, blue = (int(round(value)) for value in centers[index])
        palette.append(
            {
                "hex": f"#{red:02x}{green:02x}{blue:02x}",
                "rgb": [red, green, blue],
                "coverage_pct": round(float(counts[index] / total * 100), 1),
            }
        )
    return palette


def composition_balance(rgb: np.ndarray) -> dict[str, Any]:
    """Where the visual weight sits, using local contrast as the weighting.

    Luminance alone would call a bright empty sky the subject. Local contrast
    tracks detail, which is much closer to where the eye actually goes.
    """
    import cv2

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, (0, 0), sigmaX=8)
    weight = np.abs(grey.astype(np.float32) - blurred.astype(np.float32))
    total = weight.sum()
    if total <= 0:
        return {"center_x": 0.5, "center_y": 0.5, "quadrants": {}, "off_center": 0.0}

    height, width = weight.shape
    ys, xs = np.mgrid[0:height, 0:width]
    center_x = float((weight * xs).sum() / total / max(width - 1, 1))
    center_y = float((weight * ys).sum() / total / max(height - 1, 1))

    half_h, half_w = height // 2, width // 2
    quadrants = {
        "top_left": weight[:half_h, :half_w].sum(),
        "top_right": weight[:half_h, half_w:].sum(),
        "bottom_left": weight[half_h:, :half_w].sum(),
        "bottom_right": weight[half_h:, half_w:].sum(),
    }
    return {
        "center_x": round(center_x, 3),
        "center_y": round(center_y, 3),
        "off_center": round(float(np.hypot(center_x - 0.5, center_y - 0.5)), 3),
        "quadrants": {key: round(float(value / total * 100), 1) for key, value in quadrants.items()},
    }


def failure_flags(rgb: np.ndarray, exposure: dict[str, Any]) -> list[str]:
    """Detect known, specific failure modes rather than judging taste.

    The black-frame check earns its place on this hardware: two separate fp16
    faults on GTX 16-series cards make every image decode to solid black, one in
    the stock VAE's numerics and one in cuDNN's fp16 convolutions. Without this
    flag that failure looks like a mysteriously bad render; with it, the cause
    names itself.
    """
    flags: list[str] = []

    if exposure["mean"] < 2.0 and exposure["std"] < 1.5:
        flags.append(
            "image is essentially solid black: something upstream produced NaNs, which "
            "on GTX 16-series cards means either the fp16 VAE or a cuDNN fp16 fault. "
            "Run 'claudali doctor': if fp16_conv_broken is true and cudnn_disabled is "
            "false, set CLAUDALI_CUDNN=off; if cuDNN is already off, set "
            "CLAUDALI_VAE_UPCAST=always; otherwise ensure sdxl-vae-fp16-fix is "
            "installed. CLAUDALI_DTYPE=float32 avoids all of them."
        )
    elif exposure["mean"] > 253 and exposure["std"] < 1.5:
        flags.append("image is essentially solid white: the sampler likely diverged")

    if not np.isfinite(rgb).all():
        flags.append("image contains non-finite values")

    if exposure["dynamic_range"] < 25:
        flags.append(
            f"very low dynamic range ({exposure['dynamic_range']}): the image is flat, "
            "often a sign of too few steps or a CFG that is too low"
        )
    if exposure["highlight_clip_pct"] > 8:
        flags.append(
            f"{exposure['highlight_clip_pct']}% of pixels are blown out; consider lowering CFG"
        )
    if exposure["shadow_clip_pct"] > 20:
        flags.append(f"{exposure['shadow_clip_pct']}% of pixels are crushed to black")
    return flags


def analyse(image: Image.Image) -> dict[str, Any]:
    """Run every metric over one image and return a single report."""
    rgb = _as_rgb_array(image)
    exposure = exposure_stats(rgb)
    report: dict[str, Any] = {
        "size": {"width": image.width, "height": image.height},
        "exposure": exposure,
        "detail": sharpness(rgb),
        "color": colorfulness(rgb),
        "palette": dominant_palette(rgb),
        "composition": composition_balance(rgb),
    }
    report["flags"] = failure_flags(rgb, exposure)
    return report


__all__ = [
    "analyse",
    "composition_balance",
    "colorfulness",
    "dominant_palette",
    "exposure_stats",
    "failure_flags",
    "sharpness",
]
