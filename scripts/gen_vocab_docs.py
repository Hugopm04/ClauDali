"""Regenerate docs/vocabulary.md from the YAML tables.

The vocabulary is the part of ClauDali most likely to be edited, and a
hand-maintained reference for it would be wrong within a week. Run this after
changing anything under ``claudali/vocabulary/``:

    python scripts/gen_vocab_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claudali.compiler import load_vocabulary  # noqa: E402
from claudali.spec import ASPECT_BUCKETS  # noqa: E402

# Table name -> (heading, the spec field that selects from it)
TABLES = [
    ("media", "Media", "style.medium"),
    ("movements", "Movements", "style.movement"),
    ("lighting", "Lighting", "lighting.key"),
    ("shot", "Shot size", "camera.shot"),
    ("lens", "Lens", "camera.lens"),
    ("angle", "Camera angle", "camera.angle"),
    ("focus", "Focus", "camera.focus"),
    ("rule", "Composition rule", "composition.rule"),
    ("palettes", "Palettes", "palette.name"),
    ("contrast", "Contrast", "palette.contrast"),
    ("detail", "Detail level", "style.detail"),
    ("negatives", "Negative presets", "negative.presets[]"),
]

HEADER = """# Vocabulary reference

Every key ClauDali understands, and the prompt fragment it expands to.

**This file is generated.** Edit `claudali/vocabulary/*.yaml` and re-run
`python scripts/gen_vocab_docs.py`.

Keys are optional everywhere. An unrecognised value is not an error: it is
passed through as free text and reported in the compiled result's `warnings`,
so `"medium": "cyanotype"` works even though there is no `cyanotype` entry — it
just gets no curated phrasing or implied negatives.

`weight` is the attention multiplier applied in compel syntax. Values stay
within roughly 0.9–1.25; past about 1.3 SDXL starts to distort colour.
"""


def escape(text: str) -> str:
    return text.replace("|", "\\|")


def main() -> int:
    vocab = load_vocabulary()
    lines = [HEADER]

    intents = vocab.get("intents", {})
    lines.append("\n## Intents\n")
    lines.append("`intent` is the one field worth choosing deliberately: it sets the")
    lines.append("default model, sampler, CFG, medium, lighting and negatives.\n")
    lines.append("| Intent | Model | Steps | CFG | Default medium | Purpose |")
    lines.append("|---|---|---|---|---|---|")
    for name, data in intents.items():
        lines.append(
            f"| `{name}` | `{data.get('model')}` | {data.get('steps')} | {data.get('cfg')} | "
            f"`{data.get('medium') or '—'}` | {escape(str(data.get('description', '')))} |"
        )

    lines.append("\n## Aspect buckets\n")
    lines.append("SDXL was trained on fixed aspect buckets. Named aspects always resolve")
    lines.append("to one, because rendering off-bucket costs coherence.\n")
    lines.append("| `composition.aspect` | Resolution |")
    lines.append("|---|---|")
    for name, (width, height) in ASPECT_BUCKETS.items():
        lines.append(f"| `{name}` | {width} × {height} |")

    for table_name, heading, field in TABLES:
        table = vocab.get(table_name)
        if not isinstance(table, dict):
            continue
        lines.append(f"\n## {heading}\n")
        lines.append(f"Spec field: `{field}` — {len(table)} keys\n")
        lines.append("| Key | Expands to | Weight |")
        lines.append("|---|---|---|")
        for key, entry in table.items():
            prompt = escape(str(entry.get("prompt", "")).strip()) or "—"
            weight = entry.get("weight", 1.0)
            lines.append(f"| `{key}` | {prompt} | {weight} |")

    lines.append("\n## Subject word lists\n")
    lines.append("Plain lists, not keys. They decide whether a subject description")
    lines.append("contains a body, and whether it asks for more than one thing. Nothing")
    lines.append("here edits a prompt: they gate a framing warning, and whether an")
    lines.append("intent's implicit `anatomy` negatives are inherited. Both are keyword")
    lines.append("tests, so both are wrong sometimes — **add a word to")
    lines.append("`claudali/vocabulary/subjects.yaml` when a subject of yours is")
    lines.append("misjudged**, no code change needed.\n")
    lines.append("| List | Entries | Used for |")
    lines.append("|---|---|---|")
    for name, purpose in [
        ("person_words", "has a face and hands, fantasy humanoids included"),
        ("person_suffixes", "matched as suffixes, for the occupation tail"),
        ("plural_words", "means more than one, so one body cannot frame it"),
    ]:
        entries = vocab.get(name, [])
        if entries:
            lines.append(f"| `{name}` | {len(entries)} | {purpose} |")

    palettes = vocab.get("palettes", {})
    lines.append("\n## Palette colours\n")
    lines.append("Used by `post.palette_lock` and available as explicit hex in")
    lines.append("`palette.colors`.\n")
    lines.append("| Palette | Colours |")
    lines.append("|---|---|")
    for key, entry in palettes.items():
        colors = " ".join(f"`{c}`" for c in entry.get("colors", []))
        lines.append(f"| `{key}` | {colors} |")

    target = ROOT / "docs" / "vocabulary.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(len(v) for k, v in vocab.items() if isinstance(v, dict) and k != "intents")
    print(f"wrote {target} ({total} keys across {len(TABLES)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
