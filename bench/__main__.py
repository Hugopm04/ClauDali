"""The benchmark's command line.

    python -m bench plan            build the ordered item list and write the specs
    python -m bench run             run it; Ctrl+C stops, the same command resumes
    python -m bench status          where it stands, without touching anything
    python -m bench report          write the HTML gallery and the CSV
    python -m bench estimate        what is left, and how long it should take

Every command works from a plain terminal with no virtual environment
activated, as long as it is given the project's interpreter. See bench/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .model import (
    PLAN_FILE,
    SPECS_DIR,
    Calibration,
    State,
    load_plan,
    save_plan,
)
from .progress import human_duration

PHASES = ("features", "settings", "accuracy", "survey")


def cmd_plan(args: argparse.Namespace) -> int:
    """Build the item list and write every spec to disk."""
    from .catalog import build_plan

    items = build_plan(args.phase or None)
    if args.check:
        # Compiling every spec is the only way to know that a four-week run
        # will not stop on a bad vocabulary key in its third week.
        from claudali.compiler import compile_spec
        from claudali.spec import load_spec

        problems = 0
        interesting = 0
        for item in items:
            try:
                compiled = compile_spec(load_spec(item.spec))
            except Exception as exc:  # noqa: BLE001
                print(f"  INVALID {item.id}: {type(exc).__name__}: {exc}")
                problems += 1
                continue
            for warning in compiled.warnings:
                if "CLIP chunks" in warning:
                    continue
                interesting += 1
                if args.verbose:
                    print(f"  warn {item.id}: {warning[:150]}")
        print(f"checked {len(items)} specs: {problems} invalid, {interesting} warned")
        if problems:
            return 1

    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    for item in items:
        item.spec_path.parent.mkdir(parents=True, exist_ok=True)
        item.spec_path.write_text(
            json.dumps(item.spec, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    save_plan(items)

    calibration = Calibration.load()
    total = sum(calibration.estimate(item) for item in items)
    print(f"wrote {len(items)} items to {PLAN_FILE}")
    print(f"      {len(items)} specs under {SPECS_DIR}")
    for phase in PHASES:
        subset = [item for item in items if item.phase == phase]
        if not subset:
            continue
        hours = sum(calibration.estimate(item) for item in subset) / 3600
        print(f"  {phase:10s} {len(subset):5d} items  {hours:8.1f} h")
    print(f"  {'total':10s} {len(items):5d} items  {total / 3600:8.1f} h")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .progress import ProgressBar
    from .runner import Runner, prepare

    items = load_plan()
    if args.phase:
        items = [item for item in items if item.phase in args.phase]
    if args.wave:
        items = [item for item in items if item.wave in args.wave]

    state = State.load()
    calibration = Calibration.load()
    plan = prepare(items, state)
    if not plan.outstanding:
        print("nothing outstanding: every item in this selection is finished.")
        return 0

    runner = Runner(state, calibration, ProgressBar(), force=args.force, dry_run=args.dry_run)
    progress = runner.run(plan, limit=args.limit)
    state.save()
    calibration.save()

    print()
    if runner.interrupted:
        print("Stopped. Run the same command again to carry on from the saved checkpoint.")
    remaining = [item for item in items if not state.is_finished(item.id)]
    print(
        f"{progress.done_items}/{progress.total_items} items done "
        f"({progress.fraction * 100:.4f}%); {len(remaining)} left, "
        f"{human_duration(sum(calibration.estimate(i) for i in remaining))} estimated."
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .runner import reconcile

    items = load_plan()
    state = State.load()
    calibration = Calibration.load()
    # Read the bundles too, so status tells the truth after a crash. Without
    # saving: a run may be going on right now, and its runner owns that file.
    reconcile(items, state, save=False)

    counts: dict[str, int] = {}
    for item in items:
        counts[state.status(item.id)] = counts.get(state.status(item.id), 0) + 1
    total_units = sum(item.work_units() for item in items)
    done_units = sum(item.work_units() for item in items if state.is_finished(item.id))
    spent = sum(record.duration_s for record in state.records.values())
    remaining = [item for item in items if not state.is_finished(item.id)]

    print(f"{len(items)} items, {done_units / total_units * 100:.4f}% of the work done")
    for status in ("done", "paused", "failed", "running", "pending"):
        if counts.get(status):
            print(f"  {status:9s} {counts[status]}")
    print(f"\n  GPU time spent     {human_duration(spent)}")
    print(f"  estimated left     {human_duration(sum(calibration.estimate(i) for i in remaining))}")

    measured = {key: len(values) for key, values in calibration.per_size.items() if values}
    if measured:
        print("\n  measured s/step")
        for key in sorted(measured):
            width, height = key.split("|")[0].split("x")
            seconds = calibration.step_seconds(int(width), int(height), key.split("|")[1])
            print(f"    {key:22s} {seconds:6.2f} s   ({measured[key]} steps)")

    paused = [i for i in items if state.status(i.id) == "paused"]
    failed = [i for i in items if state.status(i.id) == "failed"]
    if paused:
        print(f"\n  paused, will resume on the next run:")
        for item in paused[:10]:
            print(f"    {item.id}")
    if failed:
        print(f"\n  failed:")
        for item in failed[: args.limit or 20]:
            print(f"    {item.id}: {state.records[item.id].error}")
    if remaining:
        print(f"\n  next up: {remaining[0].id}")
    return 0


def cmd_report(_args: argparse.Namespace) -> int:
    from .report import write_report
    from .runner import reconcile

    items = load_plan()
    state = State.load()
    # Same reasoning as status: the bundles are the truth, and a report that
    # left out finished work because the state file was lost would be worse
    # than useless. Read-only, in case a run is going on right now.
    reconcile(items, state, save=False)
    index = write_report(items, state)
    print(f"wrote {index}")
    print(f"      {index.parent / 'measurements.csv'}")
    print(f"\nOpen it with:  start \"\" \"{index}\"")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    items = load_plan()
    state = State.load()
    calibration = Calibration.load()
    if args.phase:
        items = [item for item in items if item.phase in args.phase]

    print(f"{'phase':10s} {'wave':8s} {'left':>6s} {'estimate':>12s}")
    grand = 0.0
    for phase in PHASES:
        for wave in sorted({i.wave for i in items if i.phase == phase}):
            left = [
                item for item in items
                if item.phase == phase and item.wave == wave and not state.is_finished(item.id)
            ]
            if not left:
                continue
            seconds = sum(calibration.estimate(item) for item in left)
            grand += seconds
            print(f"{phase:10s} {wave:8s} {len(left):6d} {human_duration(seconds):>12s}")
    print(f"{'total':10s} {'':8s} {'':6s} {human_duration(grand):>12s}")
    nights = grand / 3600 / 8
    print(f"\n{grand / 3600:.0f} GPU hours — about {nights:.0f} nights at 8 hours each.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="build the item list and write the specs")
    plan.add_argument("--phase", action="append", choices=PHASES)
    plan.add_argument("--check", action="store_true",
                      help="compile every spec first and refuse to write a broken plan")
    plan.add_argument("--verbose", action="store_true", help="print each compiler warning")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="run the outstanding items (Ctrl+C stops, re-run resumes)")
    run.add_argument("--phase", action="append", choices=PHASES)
    run.add_argument("--wave", action="append", help="only these survey waves, e.g. --wave w01")
    run.add_argument("--limit", type=int, help="stop after this many items")
    run.add_argument("--force", action="store_true",
                     help="resume items whose fingerprint changed (a library or model upgrade)")
    run.add_argument("--dry-run", action="store_true", help="list what would run, render nothing")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="where the run stands")
    status.add_argument("--limit", type=int)
    status.set_defaults(func=cmd_status)

    report = sub.add_parser("report", help="write the HTML gallery and the CSV")
    report.set_defaults(func=cmd_report)

    estimate = sub.add_parser("estimate", help="what is left, by phase and wave")
    estimate.add_argument("--phase", action="append", choices=PHASES)
    estimate.set_defaults(func=cmd_estimate)

    return parser


def _use_utf8() -> None:
    """Print UTF-8 whatever the console's code page is.

    Bundle directories carry the project's own path, which contains an accented
    character, and Windows consoles default to a code page that cannot encode
    it. Writing one such path would otherwise end a four-week run with a
    UnicodeEncodeError from a logging line.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[list[str]] = None) -> int:
    _use_utf8()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
