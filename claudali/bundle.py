"""Writing a render to disk as a self-describing bundle.

A bundle is the unit the refine loop operates on. It holds the full-size images,
a small preview of each, a contact sheet when there is more than one variation,
the exact spec and seeds, and the diagnostics -- everything needed to judge a
result and produce the next spec from it.

The previews are not a convenience. A 1024x1024 PNG is expensive for an agent to
read and mostly redundant: composition, colour and gross errors are all legible
at 512px. The full-size file stays on disk for when it is actually wanted.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

from .compose import finish, load_font
from .config import OUTPUTS_DIR, SETTINGS
from .diagnostics import analyse
from .engine.render import RenderResult
from .spec import SceneSpec

CONTACT_SHEET_MAX_COLUMNS = 4
CONTACT_SHEET_CELL = 384
LABEL_HEIGHT = 26


@dataclass
class VariationRecord:
    """One rendered image as it exists on disk."""

    index: int
    seed: int
    image: str
    preview: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Bundle:
    """The on-disk result of one job."""

    job_id: str
    directory: str
    spec: dict[str, Any]
    compiled: dict[str, Any]
    variations: list[VariationRecord]
    contact_sheet: Optional[str] = None
    control_image: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    task: str = "txt2img"
    device: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variations"] = [asdict(v) if not isinstance(v, dict) else v for v in self.variations]
        return data


def _preview_path(image_path: Path, transparent: bool) -> Path:
    return image_path.with_suffix(".preview.png" if transparent else ".preview.jpg")


def _write_preview(image: Image.Image, path: Path) -> Path:
    """Downscale for cheap reading, preserving alpha only when it exists."""
    transparent = image.mode == "RGBA"
    target = _preview_path(path, transparent)

    scale = SETTINGS.preview_max_side / max(image.size)
    if scale < 1.0:
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        preview = image.resize(size, Image.LANCZOS)
    else:
        preview = image.copy()

    if transparent:
        preview.save(target, format="PNG", optimize=True)
    else:
        preview.convert("RGB").save(target, format="JPEG", quality=85, optimize=True)
    return target


def _build_contact_sheet(
    images: list[tuple[int, int, Image.Image]], path: Path
) -> Optional[Path]:
    """A labelled grid of every variation, for judging a batch in one look.

    Each cell carries its index and seed, which is what makes "render 3 again at
    higher steps" expressible without opening files to find the seed.
    """
    if len(images) < 2:
        return None

    columns = min(CONTACT_SHEET_MAX_COLUMNS, len(images))
    rows = math.ceil(len(images) / columns)
    cell_w = CONTACT_SHEET_CELL
    cell_h = CONTACT_SHEET_CELL + LABEL_HEIGHT

    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    font = load_font(None, 15)

    for position, (index, seed, image) in enumerate(images):
        column, row = position % columns, position // columns
        thumb = image.convert("RGB").copy()
        thumb.thumbnail((cell_w - 8, CONTACT_SHEET_CELL - 8), Image.LANCZOS)

        x = column * cell_w + (cell_w - thumb.width) // 2
        y = row * cell_h + (CONTACT_SHEET_CELL - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        draw.text(
            (column * cell_w + 6, row * cell_h + CONTACT_SHEET_CELL + 4),
            f"#{index + 1}  seed {seed}",
            fill=(200, 200, 205),
            font=font,
        )

    sheet.save(path, format="JPEG", quality=88, optimize=True)
    return path


def bundle_directory(spec: SceneSpec, job_id: str) -> Path:
    """A sortable, human-readable directory name for one job."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return OUTPUTS_DIR / f"{stamp}_{spec.slug()}_{job_id[:8]}"


def write_bundle(
    spec: SceneSpec, result: RenderResult, job_id: str, directory: Optional[Path] = None
) -> Bundle:
    """Finish every variation, write the bundle, and return its manifest."""
    directory = directory or bundle_directory(spec, job_id)
    directory.mkdir(parents=True, exist_ok=True)

    control_path: Optional[Path] = None
    if result.control_image is not None:
        control_path = directory / "control.png"
        result.control_image.save(control_path)

    records: list[VariationRecord] = []
    sheet_inputs: list[tuple[int, int, Image.Image]] = []

    for rendered in result.images:
        # Overlays and postprocessing happen here rather than in the renderer, so
        # the diagnostics describe the image the caller actually receives.
        final = finish(rendered.image, spec)

        image_path = directory / f"{rendered.index + 1:03d}_seed{rendered.seed}.png"
        final.save(image_path, format="PNG", optimize=True)
        preview_path = _write_preview(final, image_path)

        records.append(
            VariationRecord(
                index=rendered.index,
                seed=rendered.seed,
                image=str(image_path),
                preview=str(preview_path),
                diagnostics=analyse(final),
            )
        )
        sheet_inputs.append((rendered.index, rendered.seed, final))

    sheet_path = _build_contact_sheet(sheet_inputs, directory / "contact-sheet.jpg")

    bundle = Bundle(
        job_id=job_id,
        directory=str(directory),
        spec=spec.model_dump(mode="json"),
        compiled=result.compiled.to_dict(),
        variations=records,
        contact_sheet=str(sheet_path) if sheet_path else None,
        control_image=str(control_path) if control_path else None,
        notes=result.notes,
        duration_s=result.duration_s,
        task=result.task,
        device=result.device,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    (directory / "spec.json").write_text(
        json.dumps(bundle.spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "result.json").write_text(
        json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return bundle


def read_bundle(directory: Path) -> Optional[dict[str, Any]]:
    """Load a previously written bundle manifest."""
    manifest = Path(directory) / "result.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))


def list_bundles(limit: int = 50) -> list[dict[str, Any]]:
    """Recent bundles, newest first. Backs the gallery and the history endpoint."""
    if not OUTPUTS_DIR.is_dir():
        return []
    directories = sorted(
        (path for path in OUTPUTS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    bundles = []
    for directory in directories[:limit]:
        manifest = read_bundle(directory)
        if manifest:
            bundles.append(manifest)
    return bundles


__all__ = ["Bundle", "VariationRecord", "bundle_directory", "list_bundles", "read_bundle", "write_bundle"]
