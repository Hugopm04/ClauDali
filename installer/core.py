"""The ClauDali installer.

Runs on the system Python using only the standard library, so it can report
progress on its own bootstrap and leaves nothing behind that the uninstaller
would have to know about. It creates a virtual environment inside the project,
selects a PyTorch build matched to the detected GPU, installs the runtime
dependencies, and downloads model weights into ``models/weights/``.

Everything it writes lives under the project directory. That is what lets the
uninstaller promise a complete removal and mean it.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claudali.config import MODELS_DIR, OUTPUTS_DIR, ROOT, RUNS_DIR, VENV_DIR, WEIGHTS_DIR
from claudali.registry import CATALOG, PROFILES, ModelEntry, download_url, resolve_remote_files

from .download import DownloadError, download_file
from .progress import Bar, Spinner, Steps, human_bytes, rule

# PyTorch wheel index for CUDA 12.4. Turing (sm_75, which is what a GTX 1660 Ti
# is) is included in these builds.
CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

MIN_PYTHON = (3, 10)
TORCH_PACKAGES = ["torch", "torchvision"]


class InstallError(RuntimeError):
    """A stage failed in a way that makes continuing pointless."""


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def detect_gpu() -> dict[str, object]:
    """Ask nvidia-smi what hardware is present.

    Deliberately does not import torch: torch is not installed yet at this
    point, and the answer determines which torch to install.
    """
    info: dict[str, object] = {"vendor": "none", "name": None, "vram_gb": None}
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return info

    if output.returncode != 0 or not output.stdout.strip():
        return info

    name, memory, compute = (part.strip() for part in output.stdout.strip().splitlines()[0].split(","))
    info.update(
        {
            "vendor": "nvidia",
            "name": name,
            "vram_gb": round(float(memory) / 1024, 1),
            "compute_capability": compute,
            # The GTX 16-series is the family that needs the fp16 VAE fix, and
            # it is also the family least likely to be recognised by generic
            # setup guides.
            "needs_fp16_vae_fix": "GTX 16" in name,
            "low_vram": float(memory) / 1024 < 8.5,
        }
    )
    return info


def venv_python(venv: Path = VENV_DIR) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def preflight(profile: str) -> dict[str, object]:
    """Check the machine can host the install, and report what it found."""
    rule("Preflight")

    if sys.version_info < MIN_PYTHON:
        raise InstallError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; this is {platform.python_version()}"
        )
    print(f"  python            {platform.python_version()} ({sys.executable})")
    print(f"  platform          {platform.system()} {platform.release()}")

    gpu = detect_gpu()
    if gpu["vendor"] == "nvidia":
        print(f"  gpu               {gpu['name']}  {gpu['vram_gb']} GB  sm_{gpu.get('compute_capability')}")
        if gpu.get("low_vram"):
            print("                    under 8.5 GB VRAM: model CPU offload will be used")
        if gpu.get("needs_fp16_vae_fix"):
            print("                    GTX 16-series detected: the fp16-fix VAE is required")
    else:
        print("  gpu               none detected — torch will be installed as a CPU build")

    required_gb = sum(CATALOG[model_id].approx_gb for model_id in PROFILES[profile]) + 6.0
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"  disk              {free_gb:.1f} GB free, ~{required_gb:.1f} GB needed for '{profile}'")
    if free_gb < required_gb:
        raise InstallError(
            f"not enough free space: {free_gb:.1f} GB available, ~{required_gb:.1f} GB needed. "
            f"Try a smaller profile (--profile minimal) or free some space."
        )
    return gpu


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def create_venv(recreate: bool = False) -> Path:
    """Create the project-local virtual environment."""
    python = venv_python()
    if python.is_file() and not recreate:
        print(f"  reusing existing environment at {VENV_DIR}")
        return python

    if VENV_DIR.exists() and recreate:
        spinner = Spinner("removing the previous environment")
        spinner.update(str(VENV_DIR))
        spinner.start_ticking()
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        spinner.finish("removed")

    spinner = Spinner("creating the virtual environment")
    spinner.update(str(VENV_DIR))
    spinner.start_ticking()
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise InstallError(f"could not create the virtual environment:\n{result.stderr}")
    spinner.finish(str(VENV_DIR))
    return venv_python()


def run_pip(python: Path, args: list[str], label: str) -> None:
    """Run pip, streaming its output into a live status line.

    pip's own progress is suppressed. It cannot be aggregated into a meaningful
    bar across a dependency tree it has not finished resolving, so the installer
    shows elapsed time and the current package instead of inventing a
    percentage.

    The child is forced into UTF-8 mode. Without it, pip emits paths in the
    Windows console codepage, and a project directory containing a non-ASCII
    character -- an accent in a folder name is enough -- comes back as mojibake
    that then crashes the installer's own error reporting.
    """
    spinner = Spinner(label)
    spinner.start_ticking()

    # `--disable-pip-version-check` is a general option and belongs before the
    # subcommand; `--progress-bar` belongs to `install` and is rejected outright
    # if placed before it. Suppressing pip's own bar matters because it would
    # otherwise fight the spinner for the same line.
    arguments = list(args)
    if arguments and arguments[0] in {"install", "download", "wheel"}:
        arguments[1:1] = ["--progress-bar", "off"]
    command = [str(python), "-m", "pip", "--disable-pip-version-check", *arguments]

    child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=child_env,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        del tail[:-40]
        spinner.update(line)

    process.wait()
    if process.returncode != 0:
        # Stop the ticker first, or it redraws over the error output.
        spinner.stop_ticking()
        sys.stdout.write("\n")
        print("\n".join(tail[-25:]))
        raise InstallError(f"{label} failed (pip exit code {process.returncode})")
    spinner.finish("installed")


def install_torch(python: Path, gpu: dict[str, object], force_cpu: bool = False) -> str:
    """Install the PyTorch build that matches the detected hardware."""
    use_cuda = gpu["vendor"] == "nvidia" and not force_cpu
    index = CUDA_INDEX if use_cuda else CPU_INDEX
    flavour = "CUDA 12.4" if use_cuda else "CPU"

    print(f"  target            PyTorch ({flavour}) from {index}")
    if use_cuda:
        print("  note              this is roughly 2.5 GB and takes a while on a home connection")

    run_pip(
        python,
        ["install", "--index-url", index, *TORCH_PACKAGES],
        f"installing PyTorch ({flavour})",
    )
    return flavour


def install_requirements(python: Path) -> None:
    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        raise InstallError(f"requirements.txt not found at {requirements}")
    run_pip(python, ["install", "-r", str(requirements)], "installing runtime dependencies")


def write_manifest(entry: ModelEntry, files: list[tuple[str, int]]) -> None:
    """Mark a model as fully installed.

    Written only after every file has landed, so an interrupted download is
    correctly reported as missing rather than as a broken installation.
    """
    manifest = {
        "id": entry.id,
        "repo": entry.repo,
        "layout": entry.layout,
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "files": [{"path": path, "size": size} for path, size in files],
        "total_bytes": sum(size for _, size in files),
    }
    (entry.local_dir / "claudali-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def download_model(entry: ModelEntry) -> bool:
    """Fetch every file for one model, with a byte-level progress bar."""
    if entry.is_installed():
        print(f"  {entry.id}: already installed ({entry.size_on_disk()/1e9:.2f} GB)")
        return False

    spinner = Spinner(f"resolving {entry.id}")
    spinner.update(entry.repo)
    spinner.start_ticking()
    files = resolve_remote_files(entry)
    total = sum(item.size for item in files)
    spinner.finish(f"{len(files)} files, {human_bytes(total)}")

    entry.local_dir.mkdir(parents=True, exist_ok=True)
    bar = Bar(entry.id, total=total, unit="bytes")

    for item in files:
        destination = entry.local_dir / item.path
        download_file(
            download_url(entry, item.path),
            destination,
            expected_size=item.size or None,
            on_progress=bar.advance,
        )

    bar.finish(f"-> {entry.local_dir.name}")
    write_manifest(entry, [(item.path, item.size) for item in files])
    return True


def download_models(model_ids: Iterable[str]) -> None:
    ids = list(model_ids)
    total_gb = sum(CATALOG[model_id].approx_gb for model_id in ids)
    print(f"  {len(ids)} model(s), about {total_gb:.1f} GB\n")

    for model_id in ids:
        entry = CATALOG[model_id]
        try:
            download_model(entry)
        except DownloadError as exc:
            raise InstallError(
                f"could not download '{model_id}': {exc}\n"
                f"  Re-running the installer resumes from where it stopped."
            ) from exc


def verify(python: Path) -> dict[str, object]:
    """Import the stack in the new environment and report what it found."""
    probe = (
        "import json, torch, diffusers, transformers;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'cuda': torch.cuda.is_available(),"
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'diffusers': diffusers.__version__,"
        "'transformers': transformers.__version__}))"
    )
    spinner = Spinner("verifying the installed stack")
    spinner.update("importing torch and diffusers")
    spinner.start_ticking()
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        spinner.stop_ticking()
        sys.stdout.write("\n")
        print(result.stderr.strip()[-2000:])
        raise InstallError("the installed environment could not import its own dependencies")

    report = json.loads(result.stdout.strip().splitlines()[-1])
    spinner.finish(
        f"torch {report['torch']}, diffusers {report['diffusers']}, "
        f"cuda={'yes, ' + str(report['gpu']) if report['cuda'] else 'no'}"
    )
    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def install(
    profile: str = "standard",
    recreate_venv: bool = False,
    force_cpu: bool = False,
    skip_torch: bool = False,
    skip_models: bool = False,
) -> int:
    if profile not in PROFILES:
        raise InstallError(f"unknown profile '{profile}'. Choose from: {', '.join(PROFILES)}")

    print("=" * 78)
    print("  ClauDali installer")
    print(f"  project: {ROOT}")
    print("=" * 78)

    stage_names = ["preflight", "environment", "pip", "pytorch", "dependencies", "models", "verify"]
    if skip_torch:
        stage_names.remove("pytorch")
    if skip_models:
        stage_names.remove("models")
    steps = Steps(stage_names)

    steps.start("Preflight checks")
    gpu = preflight(profile)

    steps.start("Creating the virtual environment")
    python = create_venv(recreate=recreate_venv)

    steps.start("Updating pip")
    run_pip(python, ["install", "--upgrade", "pip", "setuptools", "wheel"], "updating pip")

    if not skip_torch:
        steps.start("Installing PyTorch")
        install_torch(python, gpu, force_cpu=force_cpu)

    steps.start("Installing dependencies")
    install_requirements(python)

    if not skip_models:
        steps.start(f"Downloading models (profile: {profile})")
        for directory in (MODELS_DIR, WEIGHTS_DIR, OUTPUTS_DIR, RUNS_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        download_models(PROFILES[profile])

    steps.start("Verifying")
    report = verify(python)

    installed_gb = sum(entry.size_on_disk() for entry in CATALOG.values()) / 1e9
    rule("Done")
    print(f"  {steps.overall()}")
    print(f"  models on disk    {installed_gb:.2f} GB in {WEIGHTS_DIR}")
    print(f"  environment       {VENV_DIR}")
    if not report["cuda"]:
        print("\n  WARNING: CUDA is not available in the installed environment.")
        print("  Renders will run on the CPU and take many minutes each.")
    if gpu.get("needs_fp16_vae_fix"):
        print("\n  Your GPU is a GTX 16-series card. ClauDali uses the fp16-fix VAE")
        print("  automatically; without it every image would decode to solid black.")

    launcher = ".\\scripts\\start.ps1" if os.name == "nt" else "./scripts/start.sh"
    print(f"\n  Start it with:  {launcher}")
    print(f"  Then open:      http://127.0.0.1:8188\n")
    return 0


def add_models(model_ids: list[str]) -> int:
    unknown = [model_id for model_id in model_ids if model_id not in CATALOG]
    if unknown:
        raise InstallError(
            f"unknown model(s): {', '.join(unknown)}. Known: {', '.join(sorted(CATALOG))}"
        )
    rule("Downloading models")
    download_models(model_ids)
    return 0


__all__ = ["InstallError", "add_models", "detect_gpu", "install", "venv_python", "verify"]
