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

It also counts. CLIP reads 75 tokens at a time and a longer prompt becomes
several, each one voting in cross-attention — so a subject named only at the
start is outvoted by chunk after chunk of setting and style, and quietly
disappears from the image. ClauDali measures the prompt with CLIP's own
tokenizer and restates the subject at the head of every chunk, which is the
difference between a forest with fairies in it and an empty forest. See
[docs/scene-spec.md](docs/scene-spec.md) on `subject.anchor`.

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
  weights depending on the profile you choose, and 6.3 GB more for the optional
  quality models.

Everything is installed *inside this folder* — the virtual environment in
`.venv/`, the weights and HuggingFace cache in `models/`. Nothing is written to
your user profile, which is what lets the uninstaller reclaim every byte.

## Install

```powershell
.\install.ps1                      # standard profile, ~8.1 GB of models
.\install.ps1 -ModelProfile full   # adds both fine-tunes, ~22.3 GB
.\install.ps1 -ModelProfile minimal # SDXL base only, ~7.5 GB
.\install.ps1 -QualityModels       # any profile, plus the refiner and Real-ESRGAN, +6.3 GB
```

On Linux or macOS, call the same logic directly:

```bash
python -m installer install --profile standard [--with-quality-models]
```

| Profile | Size | Contents |
|---|---|---|
| `minimal` | 7.5 GB | SDXL base + the fp16-fix VAE. Everything works. |
| `standard` | 8.1 GB | Adds the depth and canny ControlNets. **Default.** |
| `full` | 22.3 GB | Adds Juggernaut XL (photoreal) and DreamShaper XL (painterly). |
| quality models | +6.3 GB | The SDXL refiner (6.25 GB) and Real-ESRGAN x4 (0.07 GB), on top of any profile. Only for the [quality options](#quality-options). |

The quality models are never downloaded unless you ask: `-QualityModels`,
`--with-quality-models`, or `python -m installer models --add sdxl-refiner
realesrgan-x4` on an existing install.

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

**Ctrl+C stops the server without losing work.** The running job pauses at the
end of its current step, and it and every job not yet started are saved to
`runs/queue.json`. They come back on the next start, held until you resume one
or release the queue. Ctrl+C twice quits at once; the running job can then still
resume, from the start of the image it was on.

---

## Three ways to use it

### 1. The web UI

A form for the everyday spec fields, with the vocabulary as dropdowns. A spec
loaded from an example, from history or as raw JSON keeps every field the form
has no input for: the form lists them and the render sends them unchanged. **Compile only**
shows you the prompt without spending GPU time. **Preview control** shows the
composition map before you commit to a render. Results arrive with previews, a
contact sheet and diagnostics, and history is one click away. A running job can
be paused, either letting the next queued job run or holding the queue, and
resumed later exactly where it stopped. The **Quality** section picks the
quality preset, VAE decode, precision, refiner and hi-res pass, and says what
the VAE decode will do with the RAM free right now.

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
| `GET /api/jobs/{id}` | Status, progress, ETA, and the bundle as images land |
| `DELETE /api/jobs/{id}` | Cancel a queued, running or paused job |
| `POST /api/jobs/{id}/pause` | Pause at the next step, optionally holding the queue |
| `POST /api/jobs/{id}/resume` | Resume a paused job, at the front of the queue |
| `POST /api/resume` | Resume a paused bundle left on disk |
| `GET /api/paused` | Paused jobs and resumable bundles |
| `POST /api/queue/release` | Let a held queue run again |
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
.venv\Scripts\python -m claudali resume  outputs\<bundle>       # continue a paused render
.venv\Scripts\python -m claudali doctor                         # diagnose the install
.venv\Scripts\python -m claudali models                         # list the catalogue
```

Ctrl+C during `render` pauses at the end of the current step and saves what is
needed to continue bit for bit, even after a reboot; the command prints the
`resume` line to use. Ctrl+C twice aborts at once. Finished images are saved as
each one lands, so neither loses them. `resume` refuses if anything that shapes
the pixels changed since the pause (model file, library versions, precision,
cuDNN state and so on) and lists what; `--force` continues anyway and records
the differences in the bundle notes.

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
| `render` | Model, size, steps, CFG, sampler, seed, variations, and the quality preset, precision and VAE decode |
| `refiner` | The optional SDXL refiner stage |
| `hires` | The optional hi-res pass: upscale, then re-sample |
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
| [`forest-fairies.json`](examples/forest-fairies.json) | A small subject in a long prompt: `subject.anchor` and scene framing |
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
  result.json                 compiled prompt, seeds, diagnostics, timings, warnings, notes, status
  checkpoint/                 only while the job can still be resumed
```

The bundle is written as the render goes: each image lands the moment it is
decoded, and `result.json` carries a `status` of `running`, `paused`, `done`,
`aborted`, `cancelled` or `error`. `checkpoint/` holds `state.json` and, for a
job paused mid-image, `state.pt` with the latents and sampler state (about
0.5 MB). It is deleted when the job completes.

`result.json` carries the diagnostics: exposure and clipping, dynamic range,
sharpness, dominant palette with coverage, where the visual weight sits, and
flags for known failure modes. Each image also records where its VAE decode ran
and the free RAM that decided it. That is what makes "which of these four is best,
and what should change" answerable without opening a single file.

---

## Performance, honestly

ClauDali is full-step SDXL. There is no distilled shortcut, because a shortcut
changes what the image looks like.

Measured on a laptop with a **GTX 1660 Ti (6 GB)**: **about 13–14 minutes per
image**, or 23–26 s per step, rendering Juggernaut XL at 1344×768 and 32 steps
with model CPU offload, attention slicing and cuDNN disabled. Four variations
took 47–53 minutes. That card has no tensor cores, so fp16 buys memory headroom
but no speed. Model CPU offload is what makes SDXL fit in 6 GB at all.

The queue is built around this. Submit eight variations, walk away, and come
back to a contact sheet. Slowness costs wall-clock time, not exploration.

Cards with 12 GB or more can set `CLAUDALI_OFFLOAD=none` to keep the whole
pipeline resident, which is considerably faster.

## Quality options

Any render can trade time for quality, one option at a time or all at once
with `"render": {"quality": "max"}`. None is on by default, and whatever a spec
sets explicitly beats the preset.

| Option | What it buys | What it costs |
|---|---|---|
| `render.vae_decode: "cpu"` | The finished image decoded in fp32 with the checkpoint's own VAE, in one piece instead of blended tiles. On a real 768×768 image SDXL base's VAE reconstructed at 50.2 dB PSNR, the fp16-fix VAE the GPU uses at 43.6 dB | ~5.6 GB of free RAM per megapixel. Measured for 1344×768: 29.0 s on the CPU, against 25.5 s tiled on the GPU. Short of RAM it pages, and the next render can be slowed while the weights page back in |
| `render.precision: "fp32"` | No fp16 numerics anywhere | ~13–14 GB of RAM and sequential offload on a 6 GB card. Speed unmeasured |
| `refiner.enabled` | The SDXL refiner finishes the last 20% of the schedule | A 6.25 GB download, and a model swap between the base and refiner stages. Untested on hardware |
| `hires.enabled` | Upscale 1.5× and re-sample at low strength, for detail the base size cannot hold | Each hi-res step samples 2.25× the pixels. Full size unmeasured on 6 GB; Real-ESRGAN untested |

**On a 16 GB laptop, the default `auto` decode almost never runs on the CPU.**
`auto` decodes on the CPU only when free RAM covers the estimate. Measured on
the GTX 1660 Ti laptop with SDXL base loaded: 0.1 GB free at decode time for a
768×768 image that wants ~4.3 GB, so it fell back to the GPU. Forcing `cpu`
worked all the same, paging, with no black frames. In the web UI that is
**Force CPU fp32 untiled**, and the line under it says what `auto` would do with
the RAM free right now.

The full reference is under [quality options in the scene spec](docs/scene-spec.md#quality-options).
The refiner and Real-ESRGAN are optional downloads; see [Install](#install).

---

## Configuration

Environment variables, all optional:

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDALI_HOST` / `CLAUDALI_PORT` | `127.0.0.1` / `8188` | Where the server listens |
| `CLAUDALI_OFFLOAD` | `model` | `model`, `sequential` (less VRAM, slower) or `none` |
| `CLAUDALI_DTYPE` | `float16` | Default precision. `float32` avoids fp16 faults; on a card under 12 GB it switches to sequential offload and needs ~13–14 GB of RAM. A spec's `render.precision` wins |
| `CLAUDALI_VAE_DECODE` | `auto` | Default VAE decode: `auto`, `cpu`, `gpu` or `gpu_tiled`, see [Quality options](#quality-options). A spec's `render.vae_decode` wins |
| `CLAUDALI_FP16_VAE_FIX` | `1` | Use the fp16-safe VAE. Required on GTX 16-series cards |
| `CLAUDALI_CUDNN` | `auto` | `auto` disables cuDNN if its fp16 convolutions return NaNs; `on` and `off` decide outright |
| `CLAUDALI_VAE_UPCAST` | `auto` | Decode the VAE in fp32. `auto` only fires if the fault survives `CLAUDALI_CUDNN`; `always` and `never` decide outright |
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

**Every image comes out solid black.** Something upstream produced NaNs. Two
unrelated fp16 faults on GTX 16-series cards cause this, and the diagnostics
flag the symptom by name either way. The first is the stock VAE's numerics,
which `sdxl-vae-fp16-fix` prevents; ClauDali installs and uses it by default.
The second is a cuDNN fault where some fp16 convolutions return NaNs, in the
UNet and the VAE alike, which the fix VAE does not prevent. Run `python -m
claudali doctor`: if `fp16_conv_broken` is true and `cudnn_disabled` is also
true, ClauDali has already worked around it, and this render is black for some
other reason. If `cudnn_disabled` is false, force it with `CLAUDALI_CUDNN=off`.
If cuDNN is off and images are still black, add `CLAUDALI_VAE_UPCAST=always`, or
decode on the CPU with `render.vae_decode: "cpu"`, which is fp32 throughout.
`render.precision: "fp32"` avoids all of them. On a 6 GB card it switches to
sequential offload and needs ~13–14 GB of RAM for the weights; how slow that is
has not been measured.

Which shapes the cuDNN fault hits is unpredictable, so a working render at one
size is no guarantee at another: on the machine this was measured on, 1024×1024
is clean and 1344×768 is not.

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
postprocessing, diagnostics, the engine's device decisions, pause and resume,
and the quality stages on tiny random pipelines. What only a real GPU and a
person looking can check is written up as a checklist with its specs in
[tests/manual/](tests/manual/README.md). Regenerate the vocabulary reference
after editing any table under `claudali/vocabulary/`:

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
(CreativeML Open RAIL++-M for the SDXL checkpoints and the refiner, MIT for the
fp16-fix VAE, OpenRAIL++ for the ControlNets, BSD-3-Clause for Real-ESRGAN). Those licences govern what you may do with the
images you generate; this project's MIT licence covers only its own code.
