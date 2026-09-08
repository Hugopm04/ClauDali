"""``python -m installer <command>``.

The PowerShell scripts are thin wrappers over this module, so the same commands
work identically on any platform and there is only one implementation to keep
correct.
"""

from __future__ import annotations

import argparse
import sys

from .core import InstallError, add_models, install, venv_python, verify
from .uninstall import show_status, uninstall


def cmd_install(args: argparse.Namespace) -> int:
    return install(
        profile=args.profile,
        recreate_venv=args.recreate_venv,
        force_cpu=args.cpu,
        skip_torch=args.skip_torch,
        skip_models=args.skip_models,
    )


def cmd_models(args: argparse.Namespace) -> int:
    if args.list:
        from claudali.registry import CATALOG, PROFILES, profile_size_gb

        for entry in CATALOG.values():
            state = "installed" if entry.is_installed() else "-"
            print(f"  {entry.id:24s} {entry.approx_gb:5.2f} GB  {state:10s} {entry.kind}")
            print(f"    {entry.description}")
        print()
        for name in PROFILES:
            print(f"  profile {name:9s} {profile_size_gb(name):5.2f} GB")
        return 0

    if not args.add:
        print("Nothing to do. Use --add <model-id> or --list.")
        return 1
    return add_models(args.add)


def cmd_uninstall(args: argparse.Namespace) -> int:
    selected: list[str] = []
    if args.all:
        selected = ["models", "env", "outputs", "runs"]
    else:
        if args.models:
            selected.append("models")
        if args.env:
            selected.append("env")
        if args.outputs:
            selected += ["outputs", "runs"]

    if not selected:
        # No target named: show what exists rather than guessing at intent.
        return show_status()
    return uninstall(selected, assume_yes=args.yes, dry_run=args.dry_run)


def cmd_verify(_args: argparse.Namespace) -> int:
    python = venv_python()
    if not python.is_file():
        print(f"No virtual environment at {python}. Run: python -m installer install")
        return 1
    verify(python)
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    return show_status()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="installer", description="Install, extend or remove ClauDali"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    install_cmd = sub.add_parser("install", help="full installation")
    install_cmd.add_argument(
        "--profile",
        default="standard",
        choices=["minimal", "standard", "full"],
        help="minimal ~7.5 GB, standard ~8.1 GB (adds ControlNets), full ~22.3 GB (adds fine-tunes)",
    )
    install_cmd.add_argument("--recreate-venv", action="store_true", help="rebuild the environment")
    install_cmd.add_argument("--cpu", action="store_true", help="force the CPU build of PyTorch")
    install_cmd.add_argument("--skip-torch", action="store_true", help="leave PyTorch as it is")
    install_cmd.add_argument("--skip-models", action="store_true", help="do not download weights")
    install_cmd.set_defaults(func=cmd_install)

    models_cmd = sub.add_parser("models", help="add or list model weights")
    models_cmd.add_argument("--add", nargs="+", metavar="ID", help="model ids to download")
    models_cmd.add_argument("--list", action="store_true", help="show the catalogue")
    models_cmd.set_defaults(func=cmd_models)

    uninstall_cmd = sub.add_parser("uninstall", help="remove parts of the installation")
    uninstall_cmd.add_argument("--models", action="store_true", help="delete model weights")
    uninstall_cmd.add_argument("--env", action="store_true", help="delete the virtual environment")
    uninstall_cmd.add_argument(
        "--outputs", action="store_true", help="delete generated images (not recoverable)"
    )
    uninstall_cmd.add_argument("--all", action="store_true", help="delete everything above")
    uninstall_cmd.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    uninstall_cmd.add_argument("--dry-run", action="store_true", help="show the plan only")
    uninstall_cmd.set_defaults(func=cmd_uninstall)

    status_cmd = sub.add_parser("status", help="report what is installed and what it costs")
    status_cmd.set_defaults(func=cmd_status)

    verify_cmd = sub.add_parser("verify", help="check the installed environment imports correctly")
    verify_cmd.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except InstallError as exc:
        print(f"\nInstall failed: {exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Re-running resumes partial downloads.\n", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
