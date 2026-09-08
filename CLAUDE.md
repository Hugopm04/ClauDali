# CLAUDE.md — working on ClauDali

Orientation for future Claude sessions in this repo. Read this before changing
anything; it records decisions whose reasons are not visible in the code.

## What this is

ClauDali gives a language model a way to produce images. The model writes a
**structured scene spec** (JSON); ClauDali compiles it into weighted SDXL
prompts, optionally builds a ControlNet map from the spec's layer stack, renders
locally, composites real text over the result, and writes a bundle containing
previews and diagnostics designed to be read back and critiqued.

The whole point is the loop: **spec → render → look → adjust the spec**. Design
choices that make a render easier to *judge* are worth more here than choices
that make it marginally prettier.

## The hardware this was built for

The target machine has a **GTX 1660 Ti, 6 GB VRAM, Turing sm_75**. Three
consequences shaped the architecture, and none of them are negotiable:

1. **No tensor cores.** fp16 buys memory headroom, not speed. Expect ~2–4
   minutes per 1024×1024 image at 30 steps. This is why the API is a job queue
   and not a blocking call, and why the queue has exactly one worker.
2. **6 GB VRAM.** SDXL's UNet alone is ~5 GB in fp16. `enable_model_cpu_offload()`
   is what makes it fit. Only one pipeline is resident at a time; loading a
   second checkpoint before releasing the first will swap the machine to death.
3. **The GTX 16-series fp16 VAE bug.** These cards emit NaNs from the stock fp16
   VAE and every image decodes to solid black. `sdxl-vae-fp16-fix` is installed
   as a *required* model and swapped in automatically. `diagnostics.failure_flags`
   detects the symptom by name, so if it ever regresses it is self-diagnosing.

Do not "optimise" any of these away without checking `claudali doctor` output on
the actual machine.

## Module map

Dependencies point downward. Nothing below imports anything above it.

```
claudali/
  config.py       Paths and settings. Redirects HF_HOME into models/ AT IMPORT
                  TIME -- must stay import-safe and stdlib-only.
  spec.py         The SceneSpec pydantic models. The contract. extra="forbid".
  compiler.py     spec -> weighted prompts. Reads vocabulary/*.yaml.
  vocabulary/     YAML tables: media, movements, lighting, camera, palettes,
                  negatives, intents. Data, not code -- edit freely.
  control/maps.py Procedural depth/edge/region maps from composition.layers.
                  numpy + Pillow only, no model loading.
  registry.py     Model catalogue; resolves file lists from the HF tree API.
  engine/
    pipelines.py  Loads, configures and caches diffusers pipelines. All the
                  VRAM and fp16 handling lives here.
    render.py     Runs the sampler. Chooses txt2img / img2img / inpaint.
  diagnostics.py  Measurements over a finished image. Reports, never enforces.
  compose.py      Post-diffusion: Pillow overlays and exact postprocessing.
  bundle.py       Writes the output bundle (images, previews, contact sheet,
                  sidecar, diagnostics).
  jobs.py         Single-worker queue with progress, ETA and cancellation.
  api.py          FastAPI app. Also serves web/index.html.
  web/index.html  The UI: one file, vanilla JS, no build step.
  __main__.py     CLI: serve / compile / render / doctor / models.

installer/
  progress.py     Dependency-free progress bars. Stdlib only, deliberately.
  download.py     Resumable HTTP downloads with byte-level progress.
  core.py         Install stages: venv, torch, deps, models, verify.
  uninstall.py    Tiered removal with confirmation and reclaimed-space reporting.
```

**Import-cost rule:** importing `claudali.api`, `claudali.jobs` or
`claudali.bundle` must never pull in torch. Torch is imported *inside functions*
in `engine/`. This keeps the server start, the CLI, and every test that does not
render fast. If you add a top-level `import torch` anywhere outside a function,
you have broken this.

## Data flow

```
SceneSpec
   |
   |-- compiler.compile_spec ------> CompiledPrompt (prompt, negatives, model,
   |                                  steps, cfg, sampler, size, warnings)
   |-- control.maps.build_control_image -> depth / canny PNG   (optional)
   |
   v
engine.render.render  ---> RenderResult (images + seeds + notes + device)
   |
   v
compose.finish        ---> postprocess, then overlays
   |
   v
bundle.write_bundle   ---> outputs/<stamp>_<slug>_<jobid>/
                             images, previews, contact sheet, spec.json,
                             result.json (with diagnostics)
```

## Design rules that must hold

1. **Nothing is silently dropped.** If a caller sets a field ClauDali cannot
   honour, either honour it, or raise with a message saying what to change.
   Never ignore it quietly. (See `render()` raising on control + init.)
2. **Every compiled prompt is returned in full**, with per-fragment provenance
   and warnings. The vocabulary must never become a black box.
3. **Unknown vocabulary keys pass through as free text with a warning.** Callers
   legitimately invent style words. A typo and a deliberate invention look the
   same, so warn rather than fail.
4. **Unknown *spec* keys are rejected** (`extra="forbid"`). A misspelled section
   name is always a bug, and silent acceptance hides it.
5. **Explicit beats inferred.** Intent supplies defaults; anything the caller
   actually set wins. Use `model_fields_set` to tell the difference — comparing
   against the default value cannot distinguish an explicit `steps: 30`.
6. **Diagnostics report, never gate.** No auto-retry, no quality thresholds
   blocking a render. Taste is the caller's.
7. **Everything downloaded lives under the project directory.** This is what
   makes the uninstaller's promise true. Never write to `~/.cache`.

## Common tasks

**Add a vocabulary key** — edit the relevant `claudali/vocabulary/*.yaml`. Each
entry takes `prompt`, optional `weight` (keep within 0.9–1.25; above ~1.3 SDXL
fries colours) and optional `negative` for implied negatives. No code change;
`/api/vocabulary` and the UI dropdowns pick it up automatically. Call
`load_vocabulary.cache_clear()` in a live process.

**Add a model** — add a `ModelEntry` to `CATALOG` in `registry.py`. Use
`layout="diffusers"` for a folder repo, `layout="single_file"` plus
`single_file="name.safetensors"` for a single checkpoint. Then verify the size:

```bash
python -c "from claudali import registry; e=registry.get('your-id'); \
f=registry.resolve_remote_files(e); print(len(f), sum(x.size for x in f)/1e9, 'GB')"
```

Update `approx_gb` to the measured value.

**Add a sampler** — one entry in `SAMPLERS` in `engine/pipelines.py`, mapping to
a diffusers scheduler class name and its kwargs.

**Add an overlay type** — a `Literal` member on `Overlay.type` in `spec.py`, a
`_draw_*` function in `compose.py`, and a branch in `apply_overlays`.

## Testing without a GPU

Everything up to the sampler is deterministic and needs no weights, no CUDA and
no network. `tests/test_claudali.py` covers the spec contract, the compiler,
control maps, compositing, postprocessing and diagnostics — 34 tests, ~1.5 s.

```bash
.venv\Scripts\python -m pytest -q            # or: pip install pytest
```

Several tests are regression locks on bugs that actually happened: an explicit
`steps: 30` being overwritten by the intent default, `horizon` being inverted,
and the gradient `direction` being backwards. Do not delete them to make a
change pass.

For anything visual, **write the image out and look at it** rather than
trusting an assertion about pixel statistics:

```bash
python -c "
import json
from claudali.spec import load_spec
from claudali.control.maps import build_depth_map
spec = load_spec(json.load(open('examples/surreal-lighthouse.json', encoding='utf-8')))
build_depth_map(spec).save('depth.png')"
```

`installer status`, `installer models --list` and `installer uninstall --dry-run`
are all non-destructive and safe to run any time.

## Gotchas found the hard way

- **The HF tree API lists redundant weights.** SDXL repos ship both a diffusers
  folder layout *and* single-file convenience copies at the repo root. Taking
  everything made sdxl-base a 21 GB download instead of 7.14 GB. The fix is
  `WEIGHT_PREFIXES` in `registry.py`: only `diffusion_pytorch_model*` and
  `model*` safetensors are canonical. Do not relax this filter.
- **HuggingFace drops connections regularly.** Both `_http_json` and
  `download_file` retry with backoff. Over a 22 GB install this is load-bearing,
  not defensive padding.
- **FastAPI raises `RequestValidationError`, not `ValidationError`,** for a bad
  request body. Both handlers exist in `api.py`; removing the first one silently
  reverts every spec error to FastAPI's default shape.
- **compel is what makes attention weights real.** Without it, `(phrase)1.15` is
  passed to CLIP as literal parentheses and a number — the weight does nothing
  and the punctuation costs tokens. If compel fails to construct, the render
  proceeds but a note says the weights were ignored. Do not remove that note.
- **`from_pipe` shares weights.** img2img and inpainting derive from the loaded
  txt2img pipeline at no extra disk, download or load cost. This is why SDXL base
  can inpaint without a dedicated inpainting checkpoint.
- **Decimal vs binary bytes.** `human_bytes` uses decimal (÷1000) so the
  installer's "7.1 GB" matches the catalogue's "7.14 GB". Two numbers for one
  file makes users distrust the tool.
- **`horizon` is a y coordinate, not a proportion of sky.** 0 is the top edge.
  A horizon near the *bottom* means the sky dominates. This was inverted in the
  first draft of the compiler.

## Deliberate non-features

Do not add these without a reason that survives the argument against them:

- **No distilled/turbo sampler tier.** The user explicitly chose full steps for
  every render, accepting the wall-clock cost to keep quality and variation
  count. Adding an LCM or Lightning path would quietly change what images look
  like.
- **No ControlNet + init/inpaint combination.** It raises a clear error. The
  pipeline class exists in diffusers but the path is untested here.
- **No depth estimation from a photo.** `control.source: "file"` with
  `mode: "depth"` uses the file as-is. Estimating would mean shipping MiDaS.
- **No auto-retry on bad diagnostics.** See design rule 6.
- **`post.seamless` is a cross-fade heuristic**, good for organic textures and
  visibly wrong for anything structured. `post.transparent_bg` is a corner
  flood-fill, not segmentation. Both are documented as such; neither should be
  quietly upgraded to something with a model dependency.

## Commands

```powershell
.\install.ps1 [-ModelProfile minimal|standard|full] [-RecreateVenv] [-Cpu]
.\scripts\start.ps1 [-BindHost 127.0.0.1] [-Port 8188] [-NoBrowser]
.\uninstall.ps1 [-Models] [-Env] [-Outputs] [-All] [-DryRun]

python -m installer install --profile standard
python -m installer models --add juggernaut-xl
python -m installer status

.venv\Scripts\python -m claudali serve|compile|render|doctor|models
```

## Keeping docs in step

`README.md` is for users, this file is for future sessions, `docs/` holds
reference material. When behaviour changes, fix the specific sentences it made
wrong — in particular the profile sizes in README and `installer/__main__.py`
help text, which are measured values and drift when the catalogue changes.
`docs/vocabulary.md` is generated by `scripts/gen_vocab_docs.py`; re-run it after
editing any vocabulary table.
