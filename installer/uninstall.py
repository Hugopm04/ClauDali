"""The ClauDali uninstaller.

Tiered on purpose. The overwhelmingly common reason to run this is to reclaim
the ~22 GB of model weights, and that has nothing to do with wanting to delete
generated images. So the default target is models only, generated work is never
touched unless explicitly named, and every run prints exactly what it will
delete and how much it will free before asking for confirmation.

Because the installer keeps everything inside the project directory -- weights,
the HuggingFace cache and the virtual environment alike -- removal here is
genuinely complete. Nothing is left in ``~/.cache`` or anywhere else.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claudali.config import MODELS_DIR, OUTPUTS_DIR, ROOT, RUNS_DIR, VENV_DIR

from .download import directory_size
from .progress import Bar, confirm, human_bytes, rule


@dataclass
class Target:
    """One removable component."""

    key: str
    path: Path
    label: str
    description: str

    @property
    def size(self) -> int:
        return directory_size(self.path)

    @property
    def exists(self) -> bool:
        return self.path.exists()


def targets() -> dict[str, Target]:
    return {
        "models": Target(
            key="models",
            path=MODELS_DIR,
            label="Model weights and HuggingFace cache",
            description="Every downloaded checkpoint, VAE and ControlNet. Re-downloadable.",
        ),
        "env": Target(
            key="env",
            path=VENV_DIR,
            label="Python virtual environment",
            description="PyTorch and the runtime dependencies. Rebuilt by the installer.",
        ),
        "outputs": Target(
            key="outputs",
            path=OUTPUTS_DIR,
            label="Generated images",
            description="Your rendered bundles. NOT recoverable once deleted.",
        ),
        "runs": Target(
            key="runs",
            path=RUNS_DIR,
            label="Uploads and scratch files",
            description="Init images and masks uploaded through the UI.",
        ),
    }


def _delete_with_progress(target: Target) -> int:
    """Delete a directory tree, advancing a bar by bytes actually removed.

    File-by-file rather than a single ``rmtree`` so the bar reflects real work.
    Deleting 22 GB takes long enough that a frozen screen looks like a hang.
    """
    files = [path for path in target.path.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files) or 1
    bar = Bar(target.key, total=total, unit="bytes")

    freed = 0
    for path in files:
        try:
            size = path.stat().st_size
            path.unlink()
            freed += size
            bar.advance(size)
        except OSError as exc:
            sys.stdout.write("\n")
            print(f"  could not delete {path}: {exc}")

    shutil.rmtree(target.path, ignore_errors=True)
    bar.finish(f"freed {human_bytes(freed)}")
    return freed


def plan(selected: list[str]) -> list[Target]:
    available = targets()
    chosen = []
    for key in selected:
        if key not in available:
            raise SystemExit(f"unknown target '{key}'. Choose from: {', '.join(available)}")
        target = available[key]
        if target.exists:
            chosen.append(target)
    return chosen


def show_status() -> int:
    """Report what exists and what each part costs, deleting nothing."""
    rule("ClauDali disk usage")
    print(f"  project: {ROOT}\n")
    total = 0
    for target in targets().values():
        size = target.size if target.exists else 0
        total += size
        state = human_bytes(size) if target.exists else "not present"
        print(f"  {target.key:9s} {state:>12s}   {target.label}")
        print(f"            {'':>12s}   {target.description}")
    print(f"\n  total     {human_bytes(total):>12s}")
    print("\n  Remove with:  python -m installer uninstall --models")
    print("                python -m installer uninstall --all")
    return 0


def uninstall(selected: list[str], assume_yes: bool = False, dry_run: bool = False) -> int:
    chosen = plan(selected)

    rule("Uninstall plan")
    if not chosen:
        print("  Nothing to remove: none of the selected targets exist.")
        return 0

    total = 0
    destructive = False
    for target in chosen:
        size = target.size
        total += size
        print(f"  DELETE  {target.path}")
        print(f"          {target.label} — {human_bytes(size)}")
        if target.key in {"outputs", "runs"}:
            destructive = True
            print("          ^ this is generated work and cannot be recovered")

    print(f"\n  Total to free: {human_bytes(total)}")

    if dry_run:
        print("\n  --dry-run: nothing was deleted.")
        return 0

    if not assume_yes:
        # Anything that destroys generated work demands a typed word; reclaiming
        # re-downloadable weights only needs a y/N.
        if destructive:
            if not confirm(
                "\n  This permanently deletes generated images.", expected="DELETE"
            ):
                print("  Cancelled.")
                return 1
        elif not confirm("\n  Proceed?"):
            print("  Cancelled.")
            return 1

    rule("Removing")
    freed = sum(_delete_with_progress(target) for target in chosen)

    rule("Done")
    print(f"  Freed {human_bytes(freed)}")
    remaining = sum(target.size for target in targets().values() if target.exists)
    print(f"  ClauDali now occupies {human_bytes(remaining)} (excluding source code)")
    if "models" in selected:
        print("\n  Reinstall models with: python -m installer install --skip-torch")
    return 0


__all__ = ["Target", "plan", "show_status", "targets", "uninstall"]
