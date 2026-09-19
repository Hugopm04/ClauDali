"""Turning finished bundles into something a person can actually judge.

Produces, under ``bench/report/``:

* ``measurements.csv`` -- one row per item, every number the run recorded.
  Sortable in any spreadsheet, which is how the settings phase gets read.
* ``index.html`` and a page per phase -- contact sheets grouped so that the
  images meant to be compared sit next to each other, each with its timings,
  memory, decode path and diagnostics beside it, and a scoring box.

Everything is local and relative: the pages reference the images in
``bench/runs/`` by relative path and nothing is uploaded anywhere.

The scoring boxes save to the browser's localStorage as you type, and the
**Export scores** button writes them out as JSON. localStorage on a ``file://``
page is per-browser and easy to lose, so the export is the copy that counts:
press it before you close the tab.

Stdlib only.
"""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from .model import BENCH_DIR, Item, Record, State

REPORT_DIR = BENCH_DIR / "report"

CSV_COLUMNS = [
    "id", "phase", "wave", "group", "title", "model", "status",
    "images", "duration_s", "s_per_step", "decode",
    "ram_free_start_gb", "ram_free_min_gb", "vram_peak_gb",
    "width", "height", "steps", "cfg", "sampler", "variations",
    "precision", "vae_decode", "quality", "refiner", "hires",
    "family", "subcase", "prompt_index", "variant", "intent",
    "axis", "value", "scene", "feature", "scale", "test",
    "flags", "warnings", "error", "bundle_dir",
]


def _bundle(record: Record) -> dict[str, Any]:
    if not record.bundle_dir:
        return {}
    path = Path(record.bundle_dir) / "result.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a half-written bundle must not stop the report
        return {}


def _row(item: Item, record: Record, bundle: dict[str, Any]) -> dict[str, Any]:
    render = item.spec.get("render", {})
    width, height = item.size()
    compiled = bundle.get("compiled") or {}
    flags = sorted(
        {
            flag
            for variation in bundle.get("variations", [])
            for flag in (variation.get("diagnostics") or {}).get("flags", [])
        }
    )
    row: dict[str, Any] = {
        "id": item.id,
        "phase": item.phase,
        "wave": item.wave,
        "group": item.group,
        "title": item.title,
        "model": item.model,
        "status": record.status,
        "images": record.images,
        "duration_s": round(record.duration_s, 1) if record.duration_s else "",
        "s_per_step": record.s_per_step or "",
        "decode": record.decode or "",
        "ram_free_start_gb": record.ram_free_start_gb or "",
        "ram_free_min_gb": record.ram_free_min_gb or "",
        "vram_peak_gb": record.vram_peak_gb or "",
        "width": compiled.get("width") or width,
        "height": compiled.get("height") or height,
        "steps": compiled.get("steps") or render.get("steps") or "",
        "cfg": compiled.get("cfg") or render.get("cfg") or "",
        "sampler": compiled.get("sampler") or render.get("sampler") or "",
        "variations": item.variations,
        "precision": render.get("precision") or compiled.get("precision") or "",
        "vae_decode": render.get("vae_decode") or compiled.get("vae_decode") or "",
        "quality": render.get("quality") or "standard",
        "refiner": bool(item.spec.get("refiner", {}).get("enabled")),
        "hires": bool(item.spec.get("hires", {}).get("enabled")),
        "flags": "; ".join(flags),
        "warnings": " | ".join(bundle.get("warnings") or [])[:400],
        "error": record.error or "",
        "bundle_dir": record.bundle_dir or "",
    }
    for key in ("family", "subcase", "prompt_index", "variant", "intent",
                "axis", "value", "scene", "feature", "scale", "test"):
        row[key] = item.tags.get(key, "")
    return row


def write_csv(items: list[Item], state: State, path: Optional[Path] = None) -> Path:
    path = path or REPORT_DIR / "measurements.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            record = state.records.get(item.id)
            if record is None:
                continue
            writer.writerow(_row(item, record, _bundle(record)))
    return path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

STYLE = """
:root {
  --bg: #14161a; --card: #1d2026; --line: #2c3039; --ink: #e7e9ee;
  --dim: #9aa2b1; --warn: #e2b04a; --bad: #e2685a; --good: #6fbf85;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--ink);
  font: 14px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 10px; border-bottom: 1px solid var(--line);
  padding-bottom: 6px; }
h3 { font-size: 14px; margin: 0 0 2px; }
a { color: #8ab4f8; }
.sub { color: var(--dim); margin: 0 0 20px; }
.why { color: var(--dim); font-style: italic; margin: 0 0 12px; max-width: 80ch; }
.group { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px; margin-bottom: 18px; }
.row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
.shot { flex: 1 1 420px; min-width: 300px; }
.shot img { width: 100%; border-radius: 5px; display: block; background: #000; }
.meta { color: var(--dim); font-size: 12px; margin-top: 6px; }
.meta b { color: var(--ink); font-weight: 600; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
  border: 1px solid var(--line); margin-right: 5px; }
.flag { color: var(--warn); border-color: var(--warn); }
.err { color: var(--bad); border-color: var(--bad); }
.ok { color: var(--good); border-color: var(--good); }
textarea { width: 100%; min-height: 48px; background: #12141a; color: var(--ink);
  border: 1px solid var(--line); border-radius: 5px; padding: 7px; font: inherit;
  font-size: 12.5px; resize: vertical; }
.score { display: flex; gap: 6px; align-items: center; margin: 8px 0 4px; }
.score button { background: #232733; color: var(--dim); border: 1px solid var(--line);
  border-radius: 5px; width: 30px; height: 28px; cursor: pointer; font: inherit; }
.score button.on { background: #35507a; color: #fff; border-color: #4a6ea8; }
.bar { position: sticky; top: 0; background: var(--bg); padding: 10px 0 12px;
  border-bottom: 1px solid var(--line); margin-bottom: 18px; z-index: 5; }
.bar button { background: #2a3040; color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 7px 13px; cursor: pointer; font: inherit; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 9px; border-bottom: 1px solid var(--line); }
th { color: var(--dim); font-weight: 600; }
"""

SCRIPT = """
const KEY = 'claudali-bench-scores';
const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); }
                     catch (e) { return {}; } };
const save = (s) => { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {} };
let scores = load();

document.querySelectorAll('[data-score]').forEach(box => {
  const id = box.dataset.score;
  const entry = scores[id] || {};
  box.querySelectorAll('button').forEach(button => {
    if (String(entry.rating) === button.textContent) button.classList.add('on');
    button.onclick = () => {
      box.querySelectorAll('button').forEach(b => b.classList.remove('on'));
      button.classList.add('on');
      scores[id] = { ...(scores[id] || {}), rating: Number(button.textContent) };
      save(scores);
    };
  });
});

document.querySelectorAll('[data-note]').forEach(area => {
  const id = area.dataset.note;
  area.value = (scores[id] || {}).note || '';
  area.oninput = () => {
    scores[id] = { ...(scores[id] || {}), note: area.value };
    save(scores);
  };
});

document.getElementById('export').onclick = () => {
  const blob = new Blob([JSON.stringify(scores, null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'bench-scores.json';
  a.click();
};
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _measurements(record: Record, bundle: dict[str, Any]) -> str:
    bits: list[str] = []
    if record.duration_s:
        minutes, seconds = divmod(int(record.duration_s), 60)
        bits.append(f"<b>{minutes}m {seconds:02d}s</b>")
    if record.s_per_step:
        bits.append(f"{record.s_per_step:.1f} s/step")
    if record.decode:
        bits.append(f"decode {_esc(record.decode)}")
    if record.vram_peak_gb:
        bits.append(f"{record.vram_peak_gb:.2f} GB VRAM")
    if record.ram_free_start_gb is not None:
        low = f" → {record.ram_free_min_gb:.1f}" if record.ram_free_min_gb is not None else ""
        bits.append(f"RAM free {record.ram_free_start_gb:.1f}{low} GB")
    timing = bundle.get("timing") or {}
    for stage, entry in (timing.get("stages") or {}).items():
        if stage != "base":
            bits.append(f"{stage} {entry.get('total_s', 0):.0f}s")
    return " · ".join(bits)


def _shot_html(item: Item, record: Record, bundle: dict[str, Any]) -> str:
    sheet = bundle.get("contact_sheet")
    images = bundle.get("variations") or []
    source = sheet or (images[0].get("preview") or images[0].get("image") if images else None)
    if source:
        try:
            relative = Path(source).resolve().relative_to(BENCH_DIR.resolve())
            src = "../" + relative.as_posix()
        except ValueError:
            src = Path(source).as_uri()
        picture = f'<img loading="lazy" src="{_esc(src)}" alt="{_esc(item.title)}">'
    else:
        picture = f'<div class="meta">no image: {_esc(record.status)}</div>'

    pills = [f'<span class="pill">{_esc(item.model)}</span>']
    for key in ("variant", "value", "scale", "axis"):
        if item.tags.get(key):
            pills.append(f'<span class="pill">{_esc(item.tags[key])}</span>')
    flags = sorted(
        {
            flag
            for variation in images
            for flag in (variation.get("diagnostics") or {}).get("flags", [])
        }
    )
    pills += [f'<span class="pill flag">{_esc(flag)}</span>' for flag in flags]
    if record.status == "failed":
        pills.append(f'<span class="pill err">failed</span>')
    elif record.status == "done":
        pills.append('<span class="pill ok">done</span>')

    error = f'<div class="meta err">{_esc(record.error)}</div>' if record.error else ""
    return f"""<div class="shot">
  <h3>{_esc(item.title)}</h3>
  <div class="meta">{"".join(pills)}</div>
  {picture}
  <div class="meta">{_measurements(record, bundle)}</div>
  {error}
  <div class="score" data-score="{_esc(item.id)}">
    <span class="meta">score</span>
    {"".join(f"<button>{n}</button>" for n in range(1, 6))}
  </div>
  <textarea data-note="{_esc(item.id)}" placeholder="what is wrong with it, in a few words"></textarea>
</div>"""


def _page(title: str, subtitle: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title><style>{STYLE}</style></head>
<body>
<div class="bar"><button id="export">Export scores</button>
  <a href="index.html" style="margin-left:14px">index</a></div>
<h1>{_esc(title)}</h1>
<p class="sub">{subtitle}</p>
{body}
<script>{SCRIPT}</script>
</body></html>"""


def write_phase_page(
    phase: str, items: list[Item], state: State, directory: Optional[Path] = None
) -> Optional[Path]:
    directory = directory or REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.phase == phase and state.records.get(item.id):
            groups[item.group].append(item)
    if not groups:
        return None

    sections: list[str] = []
    for name, group_items in groups.items():
        shots = []
        why = ""
        for item in group_items:
            record = state.records[item.id]
            why = why or item.why
            shots.append(_shot_html(item, record, _bundle(record)))
        sections.append(
            f'<div class="group"><h2>{_esc(name)}</h2>'
            f'<p class="why">{_esc(why)}</p>'
            f'<div class="row">{"".join(shots)}</div></div>'
        )

    done = sum(1 for item in items if item.phase == phase and state.is_finished(item.id))
    total = sum(1 for item in items if item.phase == phase)
    page = _page(
        f"ClauDali benchmark — {phase}",
        f"{done} of {total} items finished · generated {datetime.now():%Y-%m-%d %H:%M}. "
        "Scores save in this browser as you type; press Export scores to keep them.",
        "".join(sections),
    )
    path = directory / f"{phase}.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_index(items: list[Item], state: State, directory: Optional[Path] = None) -> Path:
    directory = directory or REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for phase in ("features", "settings", "accuracy", "survey"):
        phase_items = [item for item in items if item.phase == phase]
        if not phase_items:
            continue
        done = sum(1 for item in phase_items if state.is_finished(item.id))
        failed = sum(1 for item in phase_items if state.status(item.id) == "failed")
        hours = sum(
            state.records[item.id].duration_s
            for item in phase_items
            if item.id in state.records
        ) / 3600
        rows.append(
            f"<tr><td><a href='{phase}.html'>{phase}</a></td>"
            f"<td>{done} / {len(phase_items)}</td>"
            f"<td>{failed or ''}</td><td>{hours:.1f} h</td></tr>"
        )

    waves = []
    for wave in sorted({item.wave for item in items if item.phase == "survey"}):
        wave_items = [item for item in items if item.wave == wave]
        done = sum(1 for item in wave_items if state.is_finished(item.id))
        waves.append(f"<tr><td>{wave}</td><td>{done} / {len(wave_items)}</td></tr>")

    body = f"""
<table><tr><th>phase</th><th>done</th><th>failed</th><th>GPU time</th></tr>
{"".join(rows)}</table>
<h2>Survey waves</h2>
<p class="why">Each wave is a complete result on its own. w01 is a family-level
survey; w01 and w02 together say what the style vocabulary was worth; w01 to w04
is a defensible sub-case survey of the first model.</p>
<table><tr><th>wave</th><th>done</th></tr>{"".join(waves)}</table>
<h2>Numbers</h2>
<p class="why">Every measurement is in
<a href="measurements.csv">measurements.csv</a>, one row per item.</p>
"""
    page = _page(
        "ClauDali benchmark",
        f"generated {datetime.now():%Y-%m-%d %H:%M}",
        body,
    )
    path = directory / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_report(items: list[Item], state: State, directory: Optional[Path] = None) -> Path:
    directory = directory or REPORT_DIR
    write_csv(items, state, directory / "measurements.csv")
    for phase in ("features", "settings", "accuracy", "survey"):
        write_phase_page(phase, items, state, directory)
    return write_index(items, state, directory)


__all__ = ["CSV_COLUMNS", "REPORT_DIR", "write_csv", "write_index", "write_phase_page", "write_report"]
