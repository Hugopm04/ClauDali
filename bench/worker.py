"""Renders a run of benchmark items in one process, reporting on stdout.

The runner spawns this; you do not normally call it yourself. It exists rather
than the runner shelling out to ``claudali render`` for two reasons:

* **Telemetry.** The whole-run progress bar needs a step-by-step feed, and
  ``claudali render`` prints a human bar rather than something parseable. This
  emits one JSON object per line instead.
* **Model grouping.** A run of consecutive items sharing a checkpoint can be
  rendered without unloading it between them, which saves a minute per item
  over 1,600 items. The runner decides how long a run to hand over; this just
  renders what it is given, in order.

It is the only module in ``bench`` that imports torch, and it imports it
indirectly, inside functions, exactly as ``claudali/engine`` does.

**Ctrl+C.** The first one pauses the item in flight at the end of its current
sampler step, writes an exact checkpoint into its bundle, reports ``paused``
and exits 0 without starting the next item. The second aborts. Re-running the
benchmark resumes the paused item from its saved step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .model import Item, load_plan


def _use_utf8() -> None:
    """The runner reads this stdout as UTF-8; say so whatever the code page is."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def emit(event: str, **payload: Any) -> None:
    """One JSON object per line on stdout. The runner's only input."""
    sys.stdout.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class _Interrupted(Exception):
    """Ctrl+C reached us between items."""


def _install_pause(controller: Any) -> Any:
    """First Ctrl+C asks the render to pause; the second falls through to abort."""

    def handler(_signum: int, _frame: Any) -> None:
        controller.pause_requested = True
        emit("pausing")
        signal.signal(signal.SIGINT, signal.default_int_handler)

    return signal.signal(signal.SIGINT, handler)


def _provide_asset(item: Item, bundle: Any) -> None:
    """Copy an item's first image where a later item expects to load it.

    ``features/*/init-source`` renders a plain image so the img2img and
    inpainting items downstream have something to work from. Without this the
    feature phase could not test those paths without a file committed to the
    repository.
    """
    name = item.tags.get("provides")
    if not name or not bundle.variations:
        return
    from .catalog import ASSETS_DIR

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target = ASSETS_DIR / f"{name}.png"
    shutil.copyfile(bundle.variations[0].image, target)
    emit("asset", name=name, path=str(target))


def render_item(item: Item, *, force: bool = False) -> str:
    """Render or resume one item. Returns its final status."""
    from claudali.bundle import BundleWriter
    from claudali.compiler import compile_spec
    from claudali.config import ensure_dirs
    from claudali.engine.checkpoint import (
        RenderAborted,
        RenderController,
        RenderPaused,
        ResumeMismatch,
        load_checkpoint,
        read_state,
    )
    from claudali.engine.render import render
    from claudali.spec import load_spec

    ensure_dirs()
    directory = item.bundle_dir
    spec = load_spec(item.spec)

    resume = None
    compiled = None
    if read_state(directory) is not None:
        # An earlier session paused or was killed inside this item.
        writer = BundleWriter.open(directory)
        resume = load_checkpoint(directory)
        resume.completed = sorted(set(resume.completed) | set(writer.completed_indices()))
        emit(
            "resuming",
            item=item.id,
            stage=resume.stage,
            variation=resume.variation,
            done=len(resume.completed),
        )
    else:
        directory.mkdir(parents=True, exist_ok=True)
        compiled = compile_spec(spec)
        writer = BundleWriter.create(spec, compiled, uuid.uuid4().hex, directory)

    controller = RenderController()
    previous = _install_pause(controller)
    started = time.time()
    emit("start", item=item.id, title=item.title, model=item.model)

    def progress(update: Any) -> None:
        emit(
            "step",
            item=item.id,
            stage=update.stage,
            step=update.step,
            steps=update.steps,
            variation=update.variation + 1,
            variations=update.variations,
            done=update.done,
            total=update.total,
        )

    try:
        result = render(
            writer.spec,
            progress=progress,
            compiled=compiled,
            controller=controller,
            resume=resume,
            force=force,
            on_variation=writer.add_variation,
            on_checkpoint=writer.record_checkpoint,
        )
    except RenderPaused as stopped:
        bundle = writer.pause(stopped)
        emit("paused", item=item.id, bundle=str(directory), images=len(bundle.variations))
        raise _Interrupted from None
    except RenderAborted:
        bundle = writer.abort(duration_s=time.time() - started)
        emit("aborted", item=item.id, bundle=str(directory), images=len(bundle.variations))
        raise _Interrupted from None
    except ResumeMismatch as exc:
        emit("failed", item=item.id, error="resume refused: " + "; ".join(exc.differences))
        return "failed"
    except KeyboardInterrupt:
        bundle = writer.abort(duration_s=time.time() - started)
        emit("aborted", item=item.id, bundle=str(directory), images=len(bundle.variations))
        raise _Interrupted from None
    except Exception as exc:  # noqa: BLE001 - one bad item must not end the run
        writer.fail(f"{type(exc).__name__}: {exc}")
        emit("failed", item=item.id, error=f"{type(exc).__name__}: {exc}")
        return "failed"
    finally:
        signal.signal(signal.SIGINT, previous)

    bundle = writer.complete(result)
    _provide_asset(item, bundle)
    emit(
        "done",
        item=item.id,
        bundle=str(directory),
        images=len(bundle.variations),
        duration_s=bundle.duration_s,
        timing=bundle.timing,
        memory=bundle.memory,
        warnings=bundle.warnings,
        notes=bundle.notes,
        flags=sorted(
            {flag for record in bundle.variations for flag in record.diagnostics.get("flags", [])}
        ),
        decode=(bundle.variations[0].decode if bundle.variations else {}),
    )
    return "done"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.worker")
    parser.add_argument("ids", nargs="+", help="item ids to render, in order")
    parser.add_argument("--force", action="store_true", help="resume across a changed fingerprint")
    _use_utf8()
    args = parser.parse_args(argv)

    plan = {item.id: item for item in load_plan()}
    try:
        for item_id in args.ids:
            item = plan.get(item_id)
            if item is None:
                emit("failed", item=item_id, error="not in the plan; re-run 'bench plan'")
                continue
            render_item(item, force=args.force)
    except _Interrupted:
        emit("interrupted")
        return 0
    except KeyboardInterrupt:
        emit("interrupted")
        return 0
    emit("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
