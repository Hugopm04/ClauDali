"""Paths, environment and runtime settings.

Importing this module is the first thing ClauDali does, because it redirects
every HuggingFace cache variable into the project's own ``models/`` directory.
That redirection is what makes the uninstaller honest: nothing ClauDali
downloads ever lands in ``~/.cache``, so deleting ``models/`` genuinely
reclaims every byte it took.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

# claudali/config.py -> claudali/ -> project root
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("CLAUDALI_MODELS_DIR", ROOT / "models"))
OUTPUTS_DIR = Path(os.environ.get("CLAUDALI_OUTPUTS_DIR", ROOT / "outputs"))
RUNS_DIR = Path(os.environ.get("CLAUDALI_RUNS_DIR", ROOT / "runs"))
VENV_DIR = ROOT / ".venv"

# Where downloaded weights live, in a plain readable layout (not the HF blob
# cache). One directory per model id, so `du` and the uninstaller both make
# sense to a human.
WEIGHTS_DIR = MODELS_DIR / "weights"
HF_CACHE_DIR = MODELS_DIR / "hf-cache"
FONTS_DIR = ROOT / "assets" / "fonts"


def _redirect_hf_env() -> None:
    """Point every HuggingFace cache variable inside the project.

    Set before transformers/diffusers are imported anywhere, otherwise those
    libraries capture the default ``~/.cache/huggingface`` at import time.
    """
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
    # Keeps startup quiet and avoids a network round-trip per launch.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


_redirect_hf_env()


def ensure_dirs() -> None:
    """Create the writable directories ClauDali needs at runtime."""
    for path in (MODELS_DIR, WEIGHTS_DIR, HF_CACHE_DIR, OUTPUTS_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime knobs, all overridable by ``CLAUDALI_*`` environment variables."""

    host: str = os.environ.get("CLAUDALI_HOST", "127.0.0.1")
    port: int = _env_int("CLAUDALI_PORT", 8188)

    # Memory strategy. On a 6 GB card model CPU offload is what makes SDXL fit
    # at all: components move to the GPU only while they are actually running.
    offload: str = os.environ.get("CLAUDALI_OFFLOAD", "model")  # model|sequential|none
    dtype: str = os.environ.get("CLAUDALI_DTYPE", "float16")
    attention_slicing: bool = _env_bool("CLAUDALI_ATTENTION_SLICING", True)
    vae_tiling: bool = _env_bool("CLAUDALI_VAE_TILING", True)

    # GTX 16-series cards emit NaNs from the fp16 VAE, which surface as fully
    # black images. The fp16-fix VAE is the standard remedy and costs 335 MB.
    fp16_vae_fix: bool = _env_bool("CLAUDALI_FP16_VAE_FIX", True)

    # Some cuDNN builds return NaNs from a narrowing fp16 convolution, which is
    # the shape the VAE decoder ends on -- also a fully black image, and one the
    # fp16-fix VAE does not prevent. "auto" measures the card once per process
    # and decodes in fp32 only if it is affected; "always" and "never" skip the
    # measurement and decide outright.
    vae_upcast: str = os.environ.get("CLAUDALI_VAE_UPCAST", "auto")  # auto|always|never

    # Keep the last-used pipeline resident. Reloading SDXL from disk costs
    # 30-60 s, so this matters far more than the VRAM it holds.
    keep_pipeline_warm: bool = _env_bool("CLAUDALI_KEEP_WARM", True)

    default_model: str = os.environ.get("CLAUDALI_DEFAULT_MODEL", "sdxl-base")
    preview_max_side: int = _env_int("CLAUDALI_PREVIEW_MAX_SIDE", 512)
    max_queue: int = _env_int("CLAUDALI_MAX_QUEUE", 64)
    job_retention: int = _env_int("CLAUDALI_JOB_RETENTION", 200)

    extra: dict = field(default_factory=dict)

    @property
    def torch_dtype(self):  # pragma: no cover - trivial mapping
        import torch

        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            self.dtype
        ]


SETTINGS = Settings()

__all__ = [
    "ROOT",
    "MODELS_DIR",
    "WEIGHTS_DIR",
    "HF_CACHE_DIR",
    "OUTPUTS_DIR",
    "RUNS_DIR",
    "VENV_DIR",
    "FONTS_DIR",
    "SETTINGS",
    "Settings",
    "ensure_dirs",
]
