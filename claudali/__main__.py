"""Command line entry point: ``python -m claudali <command>``.

The web UI and the HTTP API are the primary interfaces, but a CLI matters for
three things the browser is bad at: diagnosing a broken install, rendering a
spec file from a script, and checking what a spec compiles to without starting a
server.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

from . import __version__


def _load_spec_file(path: str) -> Any:
    from .spec import load_spec

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_spec(data)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from . import api
    from .config import SETTINGS, ensure_dirs

    ensure_dirs()
    host = args.host or SETTINGS.host
    port = args.port or SETTINGS.port
    print(f"ClauDali {__version__} -> http://{host}:{port}")
    print("   UI      /            API docs  /docs")
    print("   Ctrl+C pauses the running job and saves the queue; twice quits at once\n")

    server = uvicorn.Server(uvicorn.Config(api.app, host=host, port=port, log_level=args.log_level))
    # uvicorn turns a second Ctrl+C into this flag rather than an exception, so
    # the shutdown's wait for a checkpoint polls it to know when to give up.
    api.force_exit_requested = lambda: server.force_exit
    try:
        server.run()
    except KeyboardInterrupt:
        # uvicorn re-raises the Ctrl+C it absorbed once the server has stopped.
        return 130
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    from .compiler import compile_spec

    compiled = compile_spec(_load_spec_file(args.spec))
    if args.json:
        print(json.dumps(compiled.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"model    : {compiled.model}")
    print(f"size     : {compiled.width}x{compiled.height}")
    print(f"sampling : {compiled.steps} steps, cfg {compiled.cfg}, {compiled.sampler}")
    print(
        f"quality  : {compiled.quality}, {compiled.precision}, VAE decode {compiled.vae_decode}, "
        f"stages {' -> '.join(compiled.stages)}"
    )
    if compiled.tokens is not None:
        about = "" if compiled.tokens.exact else "~"
        anchored = (
            f", subject restated {len(compiled.anchors)}x" if compiled.anchors else ""
        )
        print(
            f"prompt   : {about}{compiled.tokens.tokens} tokens in "
            f"{compiled.tokens.chunks} CLIP chunk(s){anchored}"
        )
    print()
    print(f"PROMPT\n{compiled.prompt}\n")
    print(f"NEGATIVE\n{compiled.negative_prompt}\n")
    for warning in compiled.warnings:
        print(f"warning: {warning}")
    for note in compiled.notes:
        print(f"note   : {note}")
    return 0


def pause_on_interrupt(controller: Any, out: Optional[TextIO] = None) -> Callable[[int, Any], None]:
    """A SIGINT handler: the first Ctrl+C pauses, the second aborts.

    The first press only sets the pause flag, which the render honours when the
    current step ends. It then puts Python's default handler back, so a second
    press raises ``KeyboardInterrupt`` in the main thread, where the sampler
    runs, and stops it at the next bytecode.
    """

    def handler(_signum: int, _frame: Any) -> None:
        controller.pause_requested = True
        stream = out or sys.stdout
        stream.write("\n  Pausing after the current step... (Ctrl+C again to abort)\n")
        stream.flush()
        signal.signal(signal.SIGINT, signal.default_int_handler)

    return handler


def _print_variations(bundle: Any) -> None:
    print(f"  {bundle.directory}")
    for record in bundle.variations:
        print(f"  #{record.index + 1} seed {record.seed}  {Path(record.image).name}")
        for flag in record.diagnostics.get("flags", []):
            print(f"      ! {flag}")
    for warning in bundle.warnings:
        print(f"  warning: {warning}")
    for note in bundle.notes:
        print(f"  note   : {note}")


def _run_render(writer: Any, compiled: Any = None, resume: Any = None, force: bool = False) -> int:
    """Render into a bundle writer, turning Ctrl+C into a pause and a second one into an abort."""
    from .engine.checkpoint import STAGE_NAMES, RenderController, RenderPaused, ResumeMismatch
    from .engine.render import render

    def progress(update: Any) -> None:
        share = min(1.0, update.done / update.total) if update.total > 0 else 0.0
        bar = "#" * int(share * 30)
        stage = STAGE_NAMES.get(update.stage, update.stage)
        sys.stdout.write(
            f"\r  [{bar:<30}] {share*100:5.1f}%  {stage}  variation {update.variation+1}"
            f"/{update.variations}  step {update.step}/{update.steps}   "
        )
        sys.stdout.flush()

    controller = RenderController()
    previous = signal.signal(signal.SIGINT, pause_on_interrupt(controller))
    started = time.time()
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
        state = stopped.state
        steps = state.stage_steps or state.compiled.get("steps")
        stage = STAGE_NAMES.get(state.stage, state.stage)
        print(
            f"\n\nPaused in the {stage} stage at variation {state.variation + 1}/{len(state.seeds)}, "
            f"step {state.next_step}/{steps}. {len(bundle.variations)} image(s) saved."
        )
        _print_variations(bundle)
        print(f'\nResume with: python -m claudali resume "{bundle.directory}"')
        return 0
    except ResumeMismatch as exc:
        print("\nRefusing to resume: this would not continue identically. Changed since the pause:")
        for difference in exc.differences:
            print(f"  {difference}")
        print("Add --force to resume anyway; the bundle notes will record the differences.")
        return 2
    except KeyboardInterrupt:
        bundle = writer.abort(duration_s=time.time() - started)
        print(f"\n\nAborted. {len(bundle.variations)} finished image(s) kept in {bundle.directory}")
        if bundle.checkpoint is not None:
            print(
                "The image in progress was lost; resuming restarts it from step 0 with the same "
                f'seed:\n  python -m claudali resume "{bundle.directory}"'
            )
        return 130
    except Exception as exc:
        writer.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.signal(signal.SIGINT, previous)

    bundle = writer.complete(result)
    sys.stdout.write("\n")
    print(f"\n{len(bundle.variations)} image(s) in {bundle.duration_s}s")
    _print_variations(bundle)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render a spec file synchronously, reporting progress on one line."""
    from .bundle import BundleWriter
    from .compiler import compile_spec
    from .config import ensure_dirs

    ensure_dirs()
    spec = _load_spec_file(args.spec)
    compiled = compile_spec(spec)
    writer = BundleWriter.create(
        spec, compiled, uuid.uuid4().hex, Path(args.out) if args.out else None
    )
    return _run_render(writer, compiled=compiled)


def cmd_resume(args: argparse.Namespace) -> int:
    """Carry on with a paused, aborted or interrupted bundle."""
    from .bundle import BundleWriter
    from .engine.checkpoint import STAGE_NAMES, load_checkpoint, read_state

    directory = Path(args.bundle)
    if read_state(directory) is None:
        print(f"No checkpoint in {directory}: nothing to resume.", file=sys.stderr)
        return 1

    writer = BundleWriter.open(directory)
    resume = load_checkpoint(directory)
    resume.completed = sorted(set(resume.completed) | set(writer.completed_indices()))
    where = f"the {STAGE_NAMES.get(resume.stage, resume.stage)} stage of variation {resume.variation + 1}/{len(resume.seeds)}"
    if resume.step is not None:
        where += f", step {resume.next_step}/{resume.stage_steps or resume.compiled.get('steps')}"
    print(f"Resuming {where}. {len(resume.completed)} image(s) already done.")
    return _run_render(writer, resume=resume, force=args.force)


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Report everything that determines whether a render will work."""
    from .config import MODELS_DIR, OUTPUTS_DIR, ROOT
    from .registry import CATALOG

    print(f"ClauDali {__version__}")
    print(f"  root    : {ROOT}")
    print(f"  models  : {MODELS_DIR}")
    print(f"  outputs : {OUTPUTS_DIR}\n")

    print("Python packages")
    for module in ("torch", "diffusers", "transformers", "accelerate", "compel", "fastapi", "PIL", "cv2"):
        try:
            imported = __import__(module)
            print(f"  {module:14s} {getattr(imported, '__version__', 'present')}")
        except ImportError:
            print(f"  {module:14s} MISSING")

    print("\nHardware")
    try:
        from .engine.pipelines import device_report

        for key, value in device_report().items():
            print(f"  {key:26s} {value}")
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable: {type(exc).__name__}: {exc}")

    print("\nModels")
    for entry in CATALOG.values():
        mark = "installed" if entry.is_installed() else "-"
        size = f"{entry.size_on_disk()/1e9:.2f} GB" if entry.is_installed() else f"~{entry.approx_gb} GB"
        flag = " (required)" if entry.required else ""
        print(f"  {entry.id:24s} {mark:10s} {size}{flag}")
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    from .registry import (
        CATALOG,
        PROFILES,
        QUALITY_MODELS,
        custom_checkpoints,
        profile_size_gb,
        quality_models_size_gb,
    )

    for entry in CATALOG.values():
        state = "installed" if entry.is_installed() else "not installed"
        print(f"{entry.id:24s} {entry.kind:11s} {entry.approx_gb:5.2f} GB  {state}")
        print(f"  {entry.description}")
    custom = custom_checkpoints()
    if custom:
        print("\nCustom checkpoints:")
        for path in custom:
            print(f"  {path.stem:24s} {path.stat().st_size/1e9:5.2f} GB  {path}")
    print("\nProfiles:")
    for name in PROFILES:
        print(f"  {name:10s} {profile_size_gb(name):5.2f} GB  ({', '.join(PROFILES[name])})")
    print(
        f"  {'+quality':10s} {quality_models_size_gb():5.2f} GB  ({', '.join(QUALITY_MODELS)}; "
        "installer install --with-quality-models)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claudali", description="ClauDali image generation")
    parser.add_argument("--version", action="version", version=f"claudali {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the web UI and HTTP API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    compile_cmd = sub.add_parser("compile", help="compile a spec to prompts without rendering")
    compile_cmd.add_argument("spec")
    compile_cmd.add_argument("--json", action="store_true", help="emit the full compiled JSON")
    compile_cmd.set_defaults(func=cmd_compile)

    render_cmd = sub.add_parser(
        "render", help="render a spec file (Ctrl+C pauses, Ctrl+C twice aborts)"
    )
    render_cmd.add_argument("spec")
    render_cmd.add_argument("--out", help="output directory (default: a new one under outputs/)")
    render_cmd.set_defaults(func=cmd_render)

    resume_cmd = sub.add_parser("resume", help="continue a paused or interrupted bundle")
    resume_cmd.add_argument("bundle", help="the bundle directory the pause printed")
    resume_cmd.add_argument(
        "--force",
        action="store_true",
        help="resume even if the model, libraries or settings changed since the pause",
    )
    resume_cmd.set_defaults(func=cmd_resume)

    doctor = sub.add_parser("doctor", help="report hardware, packages and installed models")
    doctor.set_defaults(func=cmd_doctor)

    models = sub.add_parser("models", help="list the model catalogue")
    models.set_defaults(func=cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
