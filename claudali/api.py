"""The HTTP API, and the web UI that is its first client.

Designed for agents as much as for people, which means:

* **Everything is discoverable.** ``/api/schema`` returns the full JSON Schema
  of a scene spec and ``/api/vocabulary`` every valid key, so a caller can
  learn the format without reading the docs.
* **Errors are machine-readable.** A rejected spec comes back as a structured
  list of field paths and messages, not prose.
* **Nothing is hidden.** ``/api/compile`` renders no pixels but returns the
  exact prompt, negatives and warnings a spec would produce, which makes the
  compiler debuggable without spending four minutes of GPU time to look at it.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import __version__
from .bundle import list_bundles
from .compiler import compile_spec, load_vocabulary, vocabulary_index
from .compose import available_fonts
from .config import OUTPUTS_DIR, RUNS_DIR, SETTINGS, ensure_dirs
from .control.maps import build_control_image
from .engine.pipelines import SAMPLERS
from .jobs import QUEUE
from .registry import CATALOG, PROFILES, custom_checkpoints, profile_size_gb
from .spec import ASPECT_BUCKETS, SceneSpec

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = RUNS_DIR / "uploads"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Prepare directories and start the render worker before serving."""
    ensure_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE.start()
    yield
    QUEUE.stop()


app = FastAPI(
    title="ClauDali",
    version=__version__,
    description=(
        "Structured-prompt image generation over local SDXL. Submit a scene spec, "
        "poll the job, read the bundle."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _error(status: int, kind: str, message: str, detail: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"type": kind, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def _field_errors(exc: ValidationError | RequestValidationError) -> list[dict[str, Any]]:
    """Flatten pydantic errors into field paths a caller can act on.

    The leading 'body' segment FastAPI adds is stripped, so the reported path
    matches the spec the caller actually wrote: `composition.aspect`, not
    `body.composition.aspect`.
    """
    errors = []
    for error in exc.errors():
        location = [str(part) for part in error["loc"]]
        if location and location[0] in {"body", "query", "path"}:
            location = location[1:]
        errors.append(
            {
                "field": ".".join(location) or "(root)",
                "message": error["msg"],
                "type": error["type"],
            }
        )
    return errors


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """The handler that actually fires for a malformed spec in a request body."""
    return _error(422, "invalid_spec", "the scene spec failed validation", _field_errors(exc))


@app.exception_handler(ValidationError)
async def _validation_handler(_request: Request, exc: ValidationError) -> JSONResponse:
    """Covers validation raised inside a handler rather than during parsing."""
    return _error(422, "invalid_spec", "the scene spec failed validation", _field_errors(exc))


@app.exception_handler(FileNotFoundError)
async def _missing_handler(_request: Request, exc: FileNotFoundError) -> JSONResponse:
    return _error(404, "not_found", str(exc))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness plus the hardware facts that explain how fast renders will be."""
    from .engine.pipelines import device_report

    try:
        device = device_report()
    except Exception as exc:  # noqa: BLE001 - torch may be absent before install
        device = {"error": f"{type(exc).__name__}: {exc}", "cuda_available": False}

    return {
        "status": "ok",
        "version": __version__,
        "device": device,
        "queue": QUEUE.stats(),
        "outputs_dir": str(OUTPUTS_DIR),
        "settings": {
            "offload": SETTINGS.offload,
            "dtype": SETTINGS.dtype,
            "fp16_vae_fix": SETTINGS.fp16_vae_fix,
            "default_model": SETTINGS.default_model,
        },
    }


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    """The JSON Schema for a scene spec, so a caller can validate before posting."""
    return SceneSpec.model_json_schema()


@app.get("/api/vocabulary")
def vocabulary() -> dict[str, Any]:
    """Every valid vocabulary key, with the prompt fragment each expands to."""
    vocab = load_vocabulary()
    return {
        "keys": vocabulary_index(),
        "intents": {
            name: {
                "description": data.get("description"),
                "medium": data.get("medium"),
                "model": data.get("model"),
                "cfg": data.get("cfg"),
                "steps": data.get("steps"),
            }
            for name, data in vocab.get("intents", {}).items()
        },
        "expansions": {
            table: {key: entry.get("prompt", "") for key, entry in values.items()}
            for table, values in vocab.items()
            if isinstance(values, dict) and table != "intents"
        },
        "samplers": sorted(SAMPLERS),
        "aspects": {name: list(size) for name, size in ASPECT_BUCKETS.items()},
        "fonts": available_fonts(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    """The catalogue, what is installed, and what each profile would cost."""
    return {
        "models": [
            {
                "id": entry.id,
                "kind": entry.kind,
                "repo": entry.repo,
                "description": entry.description,
                "approx_gb": entry.approx_gb,
                "license": entry.license,
                "required": entry.required,
                "tags": entry.tags,
                "installed": entry.is_installed(),
                "size_on_disk_gb": round(entry.size_on_disk() / 1e9, 2),
                "homepage": entry.homepage,
            }
            for entry in CATALOG.values()
        ],
        "custom": [path.stem for path in custom_checkpoints()],
        "profiles": {name: {"models": ids, "gb": profile_size_gb(name)} for name, ids in PROFILES.items()},
    }


# ---------------------------------------------------------------------------
# Compiling and rendering
# ---------------------------------------------------------------------------


@app.post("/api/compile")
def compile_only(spec: SceneSpec) -> dict[str, Any]:
    """Compile a spec to prompts without rendering.

    The cheapest possible feedback loop: it costs milliseconds and shows exactly
    what the model would be asked for, including every warning.
    """
    return compile_spec(spec).to_dict()


@app.post("/api/control-preview")
def control_preview(spec: SceneSpec) -> Response:
    """Render the procedural ControlNet map for a spec, as a PNG.

    Lets a caller check the composition before committing GPU time to it.
    """
    if spec.control.mode == "none":
        raise HTTPException(status_code=400, detail="control.mode is 'none'; nothing to preview")

    image = build_control_image(spec)
    if image is None:
        raise HTTPException(status_code=400, detail="no control image could be built")

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/api/render")
def submit_render(spec: SceneSpec) -> dict[str, Any]:
    """Queue a render. Returns immediately with a job id to poll."""
    job = QUEUE.submit(spec)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs(limit: int = 50) -> dict[str, Any]:
    return {
        "jobs": [job.to_dict(include_bundle=False) for job in QUEUE.list(limit=limit)],
        "stats": QUEUE.stats(),
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = QUEUE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> dict[str, Any]:
    if not QUEUE.cancel(job_id):
        raise HTTPException(status_code=409, detail="job is not cancellable")
    return {"cancelled": job_id}


@app.get("/api/history")
def history(limit: int = 30) -> dict[str, Any]:
    """Bundles previously written to disk, newest first."""
    return {"bundles": list_bundles(limit=limit)}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

_ALLOWED_ROOTS = (OUTPUTS_DIR, RUNS_DIR)


def _safe_path(raw: str) -> Path:
    """Resolve a path and refuse anything outside the output directories.

    The UI receives absolute paths in bundle manifests and asks for them back,
    so this endpoint has to exist -- and therefore has to be locked down, or it
    becomes an arbitrary file read on the machine.
    """
    candidate = Path(raw).resolve()
    for root in _ALLOWED_ROOTS:
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
        raise HTTPException(status_code=404, detail="file not found")
    raise HTTPException(status_code=403, detail="path is outside the ClauDali output directories")


@app.get("/api/file")
def get_file(path: str) -> FileResponse:
    return FileResponse(_safe_path(path))


class UploadResponse(BaseModel):
    path: str
    width: int
    height: int


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    """Store an init image or mask and return the path to put in a spec."""
    from PIL import Image

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(await file.read())

    try:
        with Image.open(target) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"not a readable image: {exc}") from exc

    return UploadResponse(path=str(target), width=width, height=height)


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = WEB_DIR / "index.html"
    if not page.is_file():
        return HTMLResponse("<h1>ClauDali</h1><p>Web UI not found.</p>", status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/examples")
def examples() -> dict[str, Any]:
    """The bundled example specs, so the UI can offer real starting points."""
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    found = []
    if examples_dir.is_dir():
        for path in sorted(examples_dir.glob("*.json")):
            try:
                found.append({"name": path.stem, "spec": json.loads(path.read_text(encoding="utf-8"))})
            except json.JSONDecodeError:
                logger.warning("example %s is not valid JSON", path)
    return {"examples": found}


__all__ = ["app"]
