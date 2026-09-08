"""Command line entry point: ``python -m claudali <command>``.

The web UI and the HTTP API are the primary interfaces, but a CLI matters for
three things the browser is bad at: diagnosing a broken install, rendering a
spec file from a script, and checking what a spec compiles to without starting a
server.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from . import __version__


def _load_spec_file(path: str) -> Any:
    from .spec import load_spec

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_spec(data)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import SETTINGS, ensure_dirs

    ensure_dirs()
    host = args.host or SETTINGS.host
    port = args.port or SETTINGS.port
    print(f"ClauDali {__version__} -> http://{host}:{port}")
    print("   UI      /            API docs  /docs")
    print("   Ctrl+C to stop\n")
    uvicorn.run("claudali.api:app", host=host, port=port, log_level=args.log_level)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    from .compiler import compile_spec

    compiled = compile_spec(_load_spec_file(args.spec))
    if args.json:
        print(json.dumps(compiled.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"model    : {compiled.model}")
    print(f"size     : {compiled.width}x{compiled.height}")
    print(f"sampling : {compiled.steps} steps, cfg {compiled.cfg}, {compiled.sampler}\n")
    print(f"PROMPT\n{compiled.prompt}\n")
    print(f"NEGATIVE\n{compiled.negative_prompt}\n")
    for warning in compiled.warnings:
        print(f"warning: {warning}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render a spec file synchronously, reporting progress on one line."""
    from .bundle import write_bundle
    from .config import ensure_dirs
    from .engine.render import render

    ensure_dirs()
    spec = _load_spec_file(args.spec)

    def progress(step: int, total: int, variation: int, variations: int) -> None:
        share = (variation * total + step) / max(1, total * variations)
        bar = "#" * int(share * 30)
        sys.stdout.write(
            f"\r  [{bar:<30}] {share*100:5.1f}%  variation {variation+1}/{variations}"
            f"  step {step}/{total}"
        )
        sys.stdout.flush()

    result = render(spec, progress=progress)
    sys.stdout.write("\n")

    bundle = write_bundle(
        spec, result, uuid.uuid4().hex, Path(args.out) if args.out else None
    )
    print(f"\n{len(bundle.variations)} image(s) in {bundle.duration_s}s")
    print(f"  {bundle.directory}")
    for record in bundle.variations:
        flags = record.diagnostics.get("flags", [])
        print(f"  #{record.index + 1} seed {record.seed}  {Path(record.image).name}")
        for flag in flags:
            print(f"      ! {flag}")
    for note in bundle.notes:
        print(f"  note: {note}")
    return 0


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
            print(f"  {key:20s} {value}")
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
    from .registry import CATALOG, PROFILES, custom_checkpoints, profile_size_gb

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

    render_cmd = sub.add_parser("render", help="render a spec file")
    render_cmd.add_argument("spec")
    render_cmd.add_argument("--out", help="output directory (default: a new one under outputs/)")
    render_cmd.set_defaults(func=cmd_render)

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
