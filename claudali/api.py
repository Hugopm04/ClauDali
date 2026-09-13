"""The HTTP API, and the web UI that is its first client.

Designed for agents as much as for people, which means:

* **Everything is discoverable.** ``/api/schema`` returns the full JSON Schema
  of a scene spec and ``/api/vocabulary`` every valid key, so a caller can
  learn the format without reading the docs.
* **Errors are machine-readable.** A rejected spec comes back as a structured
  list of field paths and messages, not prose.
* **Nothing is hidden.** ``/api/compile`` renders no pixels but returns the
  exact prompt, negatives and warnings a spec would produce, which makes the
  compiler debuggable without spending minutes of GPU time to look at it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import __version__
from .bundle import list_bundles, list_resumable
from .compiler import compile_spec, load_vocabulary, vocabulary_index
from .compose import available_fonts
from .config import OUTPUTS_DIR, RUNS_DIR, SETTINGS, ensure_dirs
from .control.maps import build_control_image
from .engine.pipelines import SAMPLERS
from .jobs import FINISHED, QUEUE, QUEUE_STATE_FILE, JobStateError, JobStatus, QueueFull
from .registry import CATALOG, PROFILES, custom_checkpoints, profile_size_gb
from .spec import ASPECT_BUCKETS, SceneSpec

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
UPLOAD_DIR = RUNS_DIR / "uploads"

# How long stopping the server waits for the running job's checkpoint. One step
# is ~25 s on a 6 GB card; the margin covers a model that is still loading.
SHUTDOWN_PAUSE_TIMEOUT_S = 300.0

# Set by ``claudali serve`` to read uvicorn's "Ctrl+C pressed twice" flag, so a
# second press stops the shutdown waiting. Under a bare ``uvicorn`` command
# nothing sets it, and the wait runs to its timeout.
force_exit_requested: Callable[[], bool] = lambda: False


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Restore the saved queue and start the worker; on the way out, pause and save it.

    Ctrl+C on the server means pause and hold: the running job checkpoints at
    its next step, and it and every job not yet started are written to
    ``runs/queue.json``, to come back held on the next start.
    """
    ensure_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    restored = QUEUE.restore(QUEUE_STATE_FILE)
    if restored:
        logger.warning(
            "Restored %d paused or queued job(s) from the last shutdown. The queue is held: "
            "resume a job or release the queue to carry on.",
            restored,
        )
    QUEUE.start()
    yield
    if QUEUE.current() is not None:
        logger.warning(
            "Pausing the running job after its current step, then saving the queue. "
            "Ctrl+C again to quit at once; the job then resumes from the start of its "
            "current variation instead."
        )
    saved = await asyncio.to_thread(
        QUEUE.shutdown, QUEUE_STATE_FILE, SHUTDOWN_PAUSE_TIMEOUT_S, lambda: force_exit_requested()
    )
    if saved:
        logger.warning(
            "Saved %d paused or queued job(s) to %s; they come back held on the next start.",
            saved,
            QUEUE_STATE_FILE,
        )


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
    try:
        job = QUEUE.submit(spec)
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    """Cancel. A paused job keeps its finished images and loses its checkpoint."""
    if not QUEUE.cancel(job_id):
        raise HTTPException(status_code=409, detail="job is not cancellable")
    return {"cancelled": job_id}


class PauseRequest(BaseModel):
    hold_queue: bool = False


class ResumeRequest(BaseModel):
    force: bool = False


class BundleResumeRequest(BaseModel):
    bundle: str
    force: bool = False


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str, body: Optional[PauseRequest] = None) -> dict[str, Any]:
    """Pause at the next step boundary. `hold_queue` also keeps the next job from starting."""
    try:
        job = QUEUE.pause(job_id, hold_queue=bool(body and body.hold_queue))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from exc
    except JobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.to_dict(include_bundle=False)


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str, body: Optional[ResumeRequest] = None) -> dict[str, Any]:
    """Requeue a paused job at the front. `force` resumes despite a changed environment."""
    try:
        job = QUEUE.resume(job_id, force=bool(body and body.force))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from exc
    except JobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.to_dict(include_bundle=False)


def _bundle_path(raw: str) -> Path:
    """A bundle directory under ``outputs/``, or an HTTP error."""
    candidate = Path(raw).resolve()
    try:
        candidate.relative_to(OUTPUTS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="bundles live under outputs/") from exc
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail="no such bundle directory")
    return candidate


@app.post("/api/resume")
def resume_bundle(body: BundleResumeRequest) -> dict[str, Any]:
    """Resume a paused bundle left on disk, for instance by the CLI or a crash."""
    try:
        job = QUEUE.resume_bundle(_bundle_path(body.bundle), force=body.force)
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except JobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.to_dict(include_bundle=False)


@app.get("/api/paused")
def paused() -> dict[str, Any]:
    """Paused jobs in this server, and resumable bundles on disk that no job holds."""
    jobs = QUEUE.list(limit=SETTINGS.job_retention)
    held_dirs = [
        Path(job.bundle_dir).resolve()
        for job in jobs
        if job.bundle_dir and (job.status not in FINISHED or job.resumable)
    ]
    return {
        "jobs": [
            job.to_dict(include_bundle=False)
            for job in jobs
            if job.resumable or job.status is JobStatus.PAUSING
        ],
        "bundles": [
            bundle
            for bundle in list_resumable()
            if Path(bundle["directory"]).resolve() not in held_dirs
        ],
        "held": QUEUE.stats()["held"],
    }


@app.post("/api/queue/release")
def release_queue() -> dict[str, Any]:
    """Let queued jobs start again without resuming the job that held the queue."""
    QUEUE.release()
    return {"held": False}


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
