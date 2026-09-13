"""The model catalogue: what ClauDali can download, and where it lives on disk.

Weights are stored in a plain, readable layout under ``models/weights/<id>/``
rather than in the HuggingFace blob cache. Three reasons:

* ``du -sh models/`` tells the truth, and the uninstaller can report exactly how
  many gigabytes each model costs before deleting it.
* The HF cache uses symlinks, which on Windows require Developer Mode or admin
  rights and fail confusingly when they are missing.
* A caller can drop their own ``.safetensors`` into ``models/weights/custom/``
  and have it picked up without understanding a cache format.

File lists are resolved from the HuggingFace tree API at install time rather
than hardcoded, so a repo that reorganises its layout does not silently break
the installer.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

from .config import WEIGHTS_DIR

HF_ENDPOINT = "https://huggingface.co"

# Extensions we never want: duplicate formats of weights we already take as
# safetensors, framework exports ClauDali cannot use, and the sample images and
# documentation model repos often carry.
SKIP_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".msgpack",
    ".h5",
    ".onnx",
    ".pt",
    ".pth",
    ".onnx_data",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".md",
}
SKIP_NAMES = {".gitattributes"}
SKIP_DIRS = {"onnx", "openvino", "coreml", "flax", "tf_model"}

# Canonical diffusers weight filenames. Anything else ending in .safetensors is
# a single-file convenience copy of the whole model -- `sd_xl_base_1.0.safetensors`
# sitting beside the unet/vae folders that actually get loaded. Taking those
# would triple a 7 GB download for bytes that are never read.
WEIGHT_PREFIXES = ("diffusion_pytorch_model", "model")


@dataclass
class ModelEntry:
    """One downloadable model."""

    id: str
    # "refiner" and "upscaler" are kinds of their own so that nothing offers them
    # as a base model: the UI's model list shows only "checkpoint".
    kind: Literal["checkpoint", "vae", "controlnet", "lora", "refiner", "upscaler"]
    repo: str
    layout: Literal["diffusers", "single_file"]
    description: str
    approx_gb: float
    license: str
    required: bool = False
    single_file: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    homepage: Optional[str] = None

    @property
    def local_dir(self) -> Path:
        return WEIGHTS_DIR / self.id

    def is_installed(self) -> bool:
        """Installed means the manifest written at the end of a download exists.

        Checking for a marker rather than for individual files means a download
        interrupted halfway is correctly reported as *not* installed, instead of
        half-present and mysteriously broken at load time.
        """
        return (self.local_dir / "claudali-manifest.json").is_file()

    def size_on_disk(self) -> int:
        if not self.local_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self.local_dir.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

CATALOG: dict[str, ModelEntry] = {
    entry.id: entry
    for entry in [
        ModelEntry(
            id="sdxl-base",
            kind="checkpoint",
            repo="stabilityai/stable-diffusion-xl-base-1.0",
            layout="diffusers",
            description=(
                "SDXL 1.0 base. The general-purpose model and the config source every "
                "single-file checkpoint is loaded against, so it is always required."
            ),
            approx_gb=7.14,
            license="CreativeML Open RAIL++-M",
            required=True,
            tags=["general", "graphic", "render3d", "game_asset"],
            homepage="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        ),
        ModelEntry(
            id="sdxl-vae-fp16-fix",
            kind="vae",
            repo="madebyollin/sdxl-vae-fp16-fix",
            layout="diffusers",
            description=(
                "SDXL VAE rebuilt to stay numerically stable in fp16. Required on GTX "
                "16-series cards, where the stock fp16 VAE emits NaNs and every image "
                "decodes to solid black."
            ),
            approx_gb=0.33,
            license="MIT",
            required=True,
            tags=["vae", "fix"],
            homepage="https://huggingface.co/madebyollin/sdxl-vae-fp16-fix",
        ),
        ModelEntry(
            id="controlnet-depth-sdxl",
            kind="controlnet",
            repo="diffusers/controlnet-depth-sdxl-1.0-small",
            layout="diffusers",
            description=(
                "Depth ControlNet for SDXL, distilled 'small' variant. Consumes the "
                "procedural depth maps ClauDali builds from composition.layers."
            ),
            approx_gb=0.32,
            license="OpenRAIL++",
            tags=["control", "depth"],
            homepage="https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small",
        ),
        ModelEntry(
            id="controlnet-canny-sdxl",
            kind="controlnet",
            repo="diffusers/controlnet-canny-sdxl-1.0-small",
            layout="diffusers",
            description=(
                "Canny edge ControlNet for SDXL, distilled 'small' variant. Serves both "
                "the canny and scribble control modes."
            ),
            approx_gb=0.32,
            license="OpenRAIL++",
            tags=["control", "canny", "scribble"],
            homepage="https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small",
        ),
        ModelEntry(
            id="juggernaut-xl",
            kind="checkpoint",
            repo="RunDiffusion/Juggernaut-XL-v9",
            layout="single_file",
            single_file="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
            description=(
                "Photorealism fine-tune: people, objects, product and location "
                "photography. The default for intent='photoreal'."
            ),
            approx_gb=7.11,
            license="CreativeML Open RAIL++-M",
            tags=["photoreal"],
            homepage="https://huggingface.co/RunDiffusion/Juggernaut-XL-v9",
        ),
        ModelEntry(
            id="dreamshaper-xl",
            kind="checkpoint",
            repo="Lykon/dreamshaper-xl-1-0",
            layout="diffusers",
            description=(
                "Painterly and stylised fine-tune: illustration, surrealism, concept "
                "art. The default for intent='painterly'."
            ),
            approx_gb=7.09,
            license="CreativeML Open RAIL++-M",
            tags=["painterly", "surreal", "concept"],
            homepage="https://huggingface.co/Lykon/dreamshaper-xl-1-0",
        ),
        ModelEntry(
            id="sdxl-refiner",
            kind="refiner",
            repo="stabilityai/stable-diffusion-xl-refiner-1.0",
            layout="diffusers",
            description=(
                "SDXL 1.0 refiner. Finishes the last part of the denoising after the base "
                "model, when refiner.enabled is set or render.quality is 'max'. Optional."
            ),
            approx_gb=6.25,
            license="CreativeML Open RAIL++-M",
            tags=["quality", "refiner"],
            homepage="https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0",
        ),
        ModelEntry(
            id="realesrgan-x4",
            kind="upscaler",
            repo="Comfy-Org/Real-ESRGAN_repackaged",
            layout="single_file",
            single_file="RealESRGAN_x4plus.safetensors",
            description=(
                "Real-ESRGAN x4plus, a 4x photo upscaler for the hi-res pass "
                "(hires.upscaler 'realesrgan-x4'). Needs the spandrel package. Optional."
            ),
            approx_gb=0.07,
            license="BSD-3-Clause",
            tags=["quality", "upscaler"],
            homepage="https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged",
        ),
    ]
}

# Named install profiles, so the installer can offer a size rather than a list.
# Each is spelled out: `full` used to be `list(CATALOG)`, which would have grown
# silently with every entry added, the optional quality models included.
PROFILES: dict[str, list[str]] = {
    "minimal": ["sdxl-base", "sdxl-vae-fp16-fix"],
    "standard": [
        "sdxl-base",
        "sdxl-vae-fp16-fix",
        "controlnet-depth-sdxl",
        "controlnet-canny-sdxl",
    ],
    "full": [
        "sdxl-base",
        "sdxl-vae-fp16-fix",
        "controlnet-depth-sdxl",
        "controlnet-canny-sdxl",
        "juggernaut-xl",
        "dreamshaper-xl",
    ],
}

# Downloaded only when asked for (`installer install --with-quality-models`), on
# top of any profile. Used by refiner.enabled, hires.upscaler and render.quality.
QUALITY_MODELS: list[str] = ["sdxl-refiner", "realesrgan-x4"]


def profile_size_gb(profile: str) -> float:
    return round(sum(CATALOG[model_id].approx_gb for model_id in PROFILES[profile]), 1)


def quality_models_size_gb() -> float:
    return round(sum(CATALOG[model_id].approx_gb for model_id in QUALITY_MODELS), 1)


def get(model_id: str) -> ModelEntry:
    try:
        return CATALOG[model_id]
    except KeyError:
        raise KeyError(
            f"unknown model '{model_id}'. Known models: {', '.join(sorted(CATALOG))}"
        ) from None


def custom_checkpoints() -> list[Path]:
    """Single-file checkpoints the user dropped into ``models/weights/custom/``.

    This is the documented escape hatch for Civitai downloads and anything else
    the catalogue does not know about.
    """
    custom_dir = WEIGHTS_DIR / "custom"
    if not custom_dir.is_dir():
        return []
    return sorted(p for p in custom_dir.glob("*.safetensors") if p.is_file())


def resolve_checkpoint(model_id: str) -> tuple[Path, str]:
    """Return the on-disk path and layout for a checkpoint id.

    Accepts catalogue ids and, for custom checkpoints, the bare filename stem.
    """
    if model_id in CATALOG:
        entry = get(model_id)
        if entry.kind != "checkpoint":
            raise ValueError(
                f"'{model_id}' is a {entry.kind}, not a base checkpoint, so it cannot be "
                "render.model. The refiner is turned on with refiner.enabled."
            )
        if not entry.is_installed():
            raise FileNotFoundError(
                f"model '{model_id}' is not installed. Run: python -m installer models --add {model_id}"
            )
        if entry.layout == "single_file":
            assert entry.single_file
            return entry.local_dir / entry.single_file, "single_file"
        return entry.local_dir, "diffusers"

    for path in custom_checkpoints():
        if path.stem == model_id or path.name == model_id:
            return path, "single_file"

    known = sorted(CATALOG) + [p.stem for p in custom_checkpoints()]
    raise FileNotFoundError(f"unknown model '{model_id}'. Available: {', '.join(known)}")


def refiner_problem(model_id: str) -> Optional[str]:
    """Why ``model_id`` cannot serve as the refiner right now, or None when it can.

    Shared by the compiler, which warns, and the renderer, which refuses: both
    must agree on what "not available" means.
    """
    entry = CATALOG.get(model_id)
    if entry is None or entry.kind != "refiner":
        refiners = ", ".join(e.id for e in CATALOG.values() if e.kind == "refiner")
        return f"'{model_id}' is not a refiner in the catalogue (refiners: {refiners})"
    if not entry.is_installed():
        return (
            f"the refiner '{model_id}' is not installed. Install it with: "
            f"python -m installer models --add {model_id}"
        )
    return None


def upscaler_problem(model_id: str = "realesrgan-x4") -> Optional[str]:
    """Why the Real-ESRGAN upscaler cannot run right now, or None when it can. No imports."""
    import importlib.util

    entry = CATALOG[model_id]
    if not entry.is_installed():
        return (
            f"the upscaler '{model_id}' is not installed. Install it with: "
            f"python -m installer models --add {model_id}"
        )
    if importlib.util.find_spec("spandrel") is None:
        return (
            "the spandrel package that runs it is not installed. Re-run the installer, or: "
            ".venv\\Scripts\\python -m pip install spandrel"
        )
    return None


# ---------------------------------------------------------------------------
# HuggingFace file resolution
# ---------------------------------------------------------------------------


@dataclass
class RemoteFile:
    path: str
    size: int


def _http_json(url: str, timeout: int = 30, attempts: int = 4) -> object:
    """GET and parse JSON, retrying transient failures with backoff.

    HuggingFace drops connections often enough that a 30 GB install will hit one
    almost every time. Retrying here (and in the downloader) is the difference
    between an install that completes unattended and one that needs babysitting.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "claudali/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - urllib raises a wide variety
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"could not fetch {url} after {attempts} attempts: {last_error}")


def _should_skip(path: str) -> bool:
    lowered = Path(path.lower())
    if lowered.name in SKIP_NAMES:
        return True
    if any(part in SKIP_DIRS for part in lowered.parts[:-1]):
        return True
    if lowered.suffix in SKIP_EXTENSIONS:
        return True
    # Keep only canonically named weights; see WEIGHT_PREFIXES.
    if lowered.suffix == ".safetensors" and not lowered.name.startswith(WEIGHT_PREFIXES):
        return True
    return False


def _prefer_fp16(files: Iterable[RemoteFile]) -> list[RemoteFile]:
    """Within each directory, keep fp16 weight variants when they exist.

    SDXL repos ship both full and fp16 weights. Taking both would double a 7 GB
    download for files that are never loaded.
    """
    by_dir: dict[str, list[RemoteFile]] = {}
    for item in files:
        by_dir.setdefault(str(Path(item.path).parent), []).append(item)

    kept: list[RemoteFile] = []
    for group in by_dir.values():
        has_fp16 = any(".fp16." in Path(f.path).name for f in group)
        for item in group:
            name = Path(item.path).name
            if name.endswith(".safetensors") and has_fp16 and ".fp16." not in name:
                continue
            kept.append(item)
    return kept


def resolve_remote_files(entry: ModelEntry) -> list[RemoteFile]:
    """Ask HuggingFace what this model actually consists of.

    Returns the exact files to fetch with their byte sizes, which is what makes
    an honest download progress bar possible.
    """
    if entry.layout == "single_file":
        assert entry.single_file
        tree = _http_json(f"{HF_ENDPOINT}/api/models/{entry.repo}/tree/main?recursive=true")
        for node in tree:  # type: ignore[union-attr]
            if node.get("type") == "file" and node.get("path") == entry.single_file:
                return [RemoteFile(path=entry.single_file, size=int(node.get("size", 0)))]
        raise FileNotFoundError(
            f"{entry.repo} does not contain {entry.single_file!r}; the repo layout may have changed"
        )

    tree = _http_json(f"{HF_ENDPOINT}/api/models/{entry.repo}/tree/main?recursive=true")
    files = [
        RemoteFile(path=node["path"], size=int(node.get("size", 0)))
        for node in tree  # type: ignore[union-attr]
        if node.get("type") == "file" and not _should_skip(node["path"])
    ]
    return sorted(_prefer_fp16(files), key=lambda f: f.path)


def download_url(entry: ModelEntry, path: str) -> str:
    return f"{HF_ENDPOINT}/{entry.repo}/resolve/main/{path}"


__all__ = [
    "CATALOG",
    "PROFILES",
    "QUALITY_MODELS",
    "ModelEntry",
    "RemoteFile",
    "custom_checkpoints",
    "download_url",
    "get",
    "profile_size_gb",
    "quality_models_size_gb",
    "refiner_problem",
    "resolve_checkpoint",
    "resolve_remote_files",
    "upscaler_problem",
]
