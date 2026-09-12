# ClauDali

**Structured-prompt image generation over local SDXL.**

Claude can read images but cannot emit them. ClauDali is the infrastructure that
closes that gap: you describe an image as a *structured scene spec* — subject,
lighting, camera, palette, composition — and ClauDali compiles it into the
prompts Stable Diffusion XL actually responds to, renders it on your own GPU,
and hands back a bundle designed to be looked at and refined.

Everything runs locally. No API keys, no per-image cost, nothing leaves your
machine.

```json
{
  "intent": "painterly",
  "subject": { "primary": "a lighthouse built from stacked antique clocks" },
  "scene":   { "setting": "a black basalt cliff", "time": "blue hour" },
  "style":   { "medium": "oil_impasto", "movement": "surrealism" },
  "lighting": { "key": "stormlight" },
  "palette": { "name": "teal_amber" },
  "composition": { "aspect": "3:2", "rule": "thirds" },
  "render":  { "variations": 4 }
}
```

---

## Why a spec instead of a prompt

A prompt is one long sentence where every word competes with every other word.
Changing the light means rewriting the sentence and hoping nothing else moved.

A spec is named parts. `lighting.key` is independent of `camera.lens`, so a
change is a diff rather than a rewrite, a result is reproducible from a seed, and
a caller — a person, a script, or a language model — can reason about one aspect
of an image without disturbing the rest.

ClauDali then does the prompt engineering. `medium: "oil_impasto"` becomes
`(oil painting, thick impasto brushwork, visible palette knife strokes, canvas
weave)1.15`, with matching negatives, ordered so the fragments that matter land
in the tokens CLIP weighs most heavily. Every compiled prompt is returned in
full, so nothing is hidden and any field can be overridden with raw text.

## What it does that a prompt cannot

| Capability | Why it matters |
|---|---|
| **Procedural ControlNet maps** | Describe the frame as layered shapes with depths; ClauDali builds a depth or edge map in numpy and ControlNet *enforces* it. Composition becomes a constraint, not a wish. |
| **Real text and vector overlays** | Diffusion models cannot spell. Titles, labels, frames and logos are composited afterwards with Pillow, so they are exact and legible. |
| **Inpainting and img2img** | Fix the one wrong detail instead of rerolling the whole image and losing what worked. |
| **Diagnostics on every render** | Exposure, clipping, sharpness, dominant palette, where the visual weight sits — and automatic detection of known failure modes. |
| **Exact postprocessing** | Seamless tiling, palette locking, pixel-grid alignment, background cutout. Things a sampler can only approximate. |

---

## Requirements

- **Windows 10/11, Linux or macOS** — the installer is tested on Windows.
- **Python 3.10+**
- **An NVIDIA GPU with 6 GB VRAM or more.** ClauDali runs without one, but a CPU
  render takes tens of minutes per image.
- **Disk**: ~5 GB for PyTorch and dependencies, plus 7.5–22.3 GB of model
  weights depending on the profile you choose.

Everything is installed *inside this folder* — the virtual environment in
`.venv/`, the weights and HuggingFace cache in `models/`. Nothing is written to
your user profile, which is what lets the uninstaller reclaim every byte.

## Install

```powershell
.\install.ps1                      # standard profile, ~8.1 GB of models
.\install.ps1 -ModelProfile full   # adds both fine-tunes, ~22.3 GB
.\install.ps1 -ModelProfile minimal # SDXL base only, ~7.5 GB
```

On Linux or macOS, call the same logic directly:

```bash
python -m installer install --profile standard
```

| Profile | Size | Contents |
|---|---|---|
| `minimal` | 7.5 GB | SDXL base + the fp16-fix VAE. Everything works. |
| `standard` | 8.1 GB | Adds the depth and canny ControlNets. **Default.** |
| `full` | 22.3 GB | Adds Juggernaut XL (photoreal) and DreamShaper XL (painterly). |

The installer detects your GPU, picks a matching PyTorch build, shows a real
byte-level progress bar per model, and resumes partial downloads if it is
interrupted. Re-run it as often as you like; it skips what is already there.

## Run

```powershell
.\scripts\start.ps1          # starts the server and opens your browser
```

```bash
./scripts/start.sh           # Linux / macOS
```

Then open <http://127.0.0.1:8188>. The API documentation is at `/docs`.

The server binds to localhost only. There is no authentication, so if you pass
`-BindHost 0.0.0.0` to expose it to your network, do that only on a network you
trust.

---

## Three ways to use it

### 1. The web UI

A form for every spec field, with the vocabulary as dropdowns. **Compile only**
shows you the prompt without spending GPU time. **Preview control** shows the
composition map before you commit to a render. Results arrive with previews, a
contact sheet and diagnostics, and history is one click away.

### 2. The HTTP API

Renders take minutes, so the API is a job queue rather than a blocking call.

```bash
# Submit
curl -X POST http://127.0.0.1:8188/api/render \
     -H "Content-Type: application/json" \
     -d @examples/product-photo.json
# -> {"id": "a1b2c3...", "status": "queued", ...}

# Poll
curl http://127.0.0.1:8188/api/jobs/a1b2c3...
# -> {"status": "running", "progress": 0.42, "eta_s": 96.3, ...}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/render` | Queue a render, returns a job id |
| `GET /api/jobs/{id}` | Status, progress, ETA, and the bundle when finished |
| `DELETE /api/jobs/{id}` | Cancel a queued or running job |
| `POST /api/compile` | Compile a spec to prompts without rendering |
| `POST /api/control-preview` | The procedural control map as a PNG |
| `GET /api/schema` | JSON Schema for a scene spec |
| `GET /api/vocabulary` | Every valid key and what it expands to |
| `GET /api/models` | Catalogue and what is installed |
| `GET /api/history` | Previous bundles |
| `GET /api/health` | Hardware, queue state, settings |

Full reference: [docs/api.md](docs/api.md).

### 3. The command line

```powershell
.venv\Scripts\python -m claudali compile examples/poster.json   # see the prompt
.venv\Scripts\python -m claudali render  examples/poster.json   # render it
.venv\Scripts\python -m claudali doctor                         # diagnose the install
.venv\Scripts\python -m claudali models                         # list the catalogue
```

---

## The scene spec

A minimal spec is a subject and an intent:

```json
{ "intent": "photoreal", "subject": { "primary": "a stoneware coffee cup" } }
```

`intent` chooses the model, sampler, CFG, default medium and negatives, so a
two-line spec still produces a competent image. Everything else is optional and
overrides those defaults.

| Section | What it controls |
|---|---|
| `subject` | The primary subject, details, count, action |
| `scene` | Setting, time, weather, season, background |
| `style` | Medium, movement, artists, descriptors, detail level |
| `camera` | Shot size, lens, aperture, angle, focus |
| `lighting` | Key light, mood, accents |
| `palette` | Named palette or explicit hex colours, contrast |
| `composition` | Aspect, framing rule, horizon, and the **layer stack** |
| `control` | ControlNet mode, source and strength |
| `init` | img2img source, strength, mask for inpainting |
| `render` | Model, size, steps, CFG, sampler, seed, variations |
| `negative` | Negative presets and extras |
| `overlays` | Text, shapes, gradients, vignettes drawn after diffusion |
| `post` | Seamless, transparent background, pixelate, palette lock, grain |
| `raw` | Escape hatch: prepend, append or replace the compiled prompt |

Full field reference: [docs/scene-spec.md](docs/scene-spec.md).
All vocabulary keys: [docs/vocabulary.md](docs/vocabulary.md).

### Composition as a constraint

The interesting part. Describe the frame as layers:

```json
"composition": {
  "aspect": "3:2",
  "horizon": 0.62,
  "layers": [
    { "role": "far_ridge",  "shape": "horizon", "bbox": [0.0, 0.52, 1.0, 0.64], "depth": 0.18 },
    { "role": "lighthouse", "shape": "column",  "bbox": [0.62, 0.14, 0.74, 0.66], "depth": 0.75 },
    { "role": "cliff",      "shape": "blob",    "bbox": [0.0, 0.55, 0.55, 1.0],  "depth": 0.92 }
  ]
},
"control": { "mode": "depth", "source": "procedural", "strength": 0.55 }
```

ClauDali renders that stack into a depth map — sky far, ridge behind, lighthouse
as a shaded cylinder, cliff nearest — and ControlNet forces SDXL to obey it. You
decide where things are; the model decides what they look like.

`bbox` is `[x0, y0, x1, y1]` normalised to 0–1. `depth` is 0 (far) to 1 (near).

---

## Examples

| File | Shows |
|---|---|
| [`surreal-lighthouse.json`](examples/surreal-lighthouse.json) | Painterly render with a procedural depth map |
| [`product-photo.json`](examples/product-photo.json) | Photorealism with full camera control |
| [`game-texture.json`](examples/game-texture.json) | Seamless tiling texture |
| [`pixel-sprite.json`](examples/pixel-sprite.json) | Pixel-grid alignment, palette lock, transparent background |
| [`poster.json`](examples/poster.json) | Real typography composited over a generated image |
| [`render3d-scene.json`](examples/render3d-scene.json) | Canny control from a layer stack |

---

## Output

Every job writes a self-describing bundle to `outputs/`:

```
outputs/2026-09-08_143355_surreal-lighthouse_a1b2c3d4/
  001_seed1848.png            full resolution
  001_seed1848.preview.jpg    downscaled, cheap to read
  contact-sheet.jpg           all variations, labelled with seeds
  control.png                 the control map, when one was used
  spec.json                   exactly what was asked for
  result.json                 compiled prompt, seeds, diagnostics, timings, notes
```

`result.json` carries the diagnostics: exposure and clipping, dynamic range,
sharpness, dominant palette with coverage, where the visual weight sits, and
flags for known failure modes. That is what makes "which of these four is best,
and what should change" answerable without opening a single file.

---

## Performance, honestly

ClauDali is full-step SDXL. There is no distilled shortcut, because a shortcut
changes what the image looks like.

On a **GTX 1660 Ti (6 GB)**, expect roughly **2–4 minutes per 1024×1024 image**
at 30 steps. That card has no tensor cores, so fp16 buys memory headroom but no
speed. Model CPU offload is what makes SDXL fit in 6 GB at all.

The queue is built around this. Submit eight variations, walk away, and come
back to a contact sheet. Slowness costs wall-clock time, not exploration.

Cards with 12 GB or more can set `CLAUDALI_OFFLOAD=none` to keep the whole
pipeline resident, which is considerably faster.

---

## Configuration

Environment variables, all optional:

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDALI_HOST` / `CLAUDALI_PORT` | `127.0.0.1` / `8188` | Where the server listens |
| `CLAUDALI_OFFLOAD` | `model` | `model`, `sequential` (less VRAM, slower) or `none` |
| `CLAUDALI_DTYPE` | `float16` | `float32` uses more VRAM but avoids fp16 issues entirely |
| `CLAUDALI_FP16_VAE_FIX` | `1` | Use the fp16-safe VAE. Required on GTX 16-series cards |
| `CLAUDALI_VAE_UPCAST` | `auto` | Decode in fp32. `auto` measures the card, `always` and `never` decide outright |
| `CLAUDALI_PREVIEW_MAX_SIDE` | `512` | Preview size in the bundle |
| `CLAUDALI_MODELS_DIR` | `./models` | Move weights to another drive |
| `CLAUDALI_OUTPUTS_DIR` | `./outputs` | Where bundles are written |

## Adding your own models

Drop any SDXL `.safetensors` checkpoint into `models/weights/custom/` and it
appears in the model list under its filename. This is how you use a Civitai
download — ClauDali's catalogue sources everything from HuggingFace, so no
account or token is needed for the built-in models.

```json
{ "render": { "model": "my-downloaded-checkpoint" } }
```

## Uninstall

```powershell
.\uninstall.ps1                # show what exists and what it costs, delete nothing
.\uninstall.ps1 -Models        # free the weights (~22 GB), keep everything else
.\uninstall.ps1 -Env           # remove the virtual environment
.\uninstall.ps1 -All           # everything, including generated images
```

Generated images are **never** deleted unless you explicitly pass `-Outputs` or
`-All`, and those require typing `DELETE` to confirm. Every run prints the plan
and the space it will reclaim first.

---

## Troubleshooting

**Every image comes out solid black.** The VAE decoded to NaNs. Two unrelated
fp16 faults on GTX 16-series cards cause this, and the diagnostics flag the
symptom by name either way. The first is the stock VAE's numerics, which
`sdxl-vae-fp16-fix` prevents; ClauDali installs and uses it by default. The
second is a cuDNN fault where a narrowing fp16 convolution returns NaNs, which
the fix VAE does not prevent and in fact exposes. Run `python -m claudali
doctor`: if `fp16_narrowing_conv_broken` is true, ClauDali already decodes in
fp32 to work around it, and `CLAUDALI_VAE_UPCAST=always` forces that on a card
the probe cannot measure. `CLAUDALI_DTYPE=float32` avoids both at the cost of
speed and VRAM.

**Out of memory.** Try `CLAUDALI_OFFLOAD=sequential`, or render at a smaller
aspect bucket. Close other GPU applications — a browser with hardware
acceleration can hold several hundred MB.

**Renders are extremely slow.** Run `python -m claudali doctor`. If it reports
`cuda_available: False`, PyTorch installed as a CPU build. Reinstall with
`.\install.ps1 -RecreateVenv`.

**A download failed.** Re-run the installer. Downloads resume from where they
stopped.

**The text in my image is gibberish.** That is diffusion working as designed —
it has no concept of a glyph. Use `overlays` for anything that must be legible.

---

## Development

```bash
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
```

The test suite needs no GPU, no model weights and no network: it covers the spec
contract, the prompt compiler, the procedural control maps, compositing,
postprocessing and diagnostics. Regenerate the vocabulary reference after
editing any table under `claudali/vocabulary/`:

```bash
python scripts/gen_vocab_docs.py
```

## Documentation

- [docs/scene-spec.md](docs/scene-spec.md) — every field, with examples
- [docs/api.md](docs/api.md) — endpoint reference
- [docs/vocabulary.md](docs/vocabulary.md) — every vocabulary key
- [docs/refine-loop.md](docs/refine-loop.md) — the render → judge → adjust workflow
- [CLAUDE.md](CLAUDE.md) — architecture and conventions, for future Claude sessions

## Licence

MIT — see [LICENSE](LICENSE).

Model weights are downloaded from HuggingFace under their own licences
(CreativeML Open RAIL++-M for the SDXL checkpoints, MIT for the fp16-fix VAE,
OpenRAIL++ for the ControlNets). Those licences govern what you may do with the
images you generate; this project's MIT licence covers only its own code.
