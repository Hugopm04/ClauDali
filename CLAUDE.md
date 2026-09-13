# CLAUDE.md — working on ClauDali

Orientation for future Claude sessions in this repo. Read this before changing
anything; it records decisions whose reasons are not visible in the code.

## Open work

`PLAN.md` is an agreed plan (2026-09-13) for four tasks: silence three library
warnings, exact pause/resume, a cleanup pass, and optional max-quality stages.
Every decision in it was made with Hugo, so do not re-ask them. Read it before
starting any of that work, and keep its status table current.

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

The target machine has a **GTX 1660 Ti, 6 GB VRAM, Turing sm_75**. Four
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
4. **The cuDNN fp16 convolution fault.** Measured on this machine with driver
   591.86 and cuDNN 9.1: an fp16 `conv2d` returns NaN for exactly a quarter of
   its values, from finite inputs and finite weights, so **every image decodes
   black**. fp32 is clean; disabling cuDNN is clean.

   **Which shapes are hit is not predictable.** Measured failures and passes sit
   next to each other with no rule joining them:

   | Shape | Where it runs | Result |
   |---|---|---|
   | b1 256→128 k3 256×256 | VAE decoder's last block | NaN |
   | b2 2560→1280 k1 24×42 | UNet up-block 0 shortcut, 1344×768 | NaN |
   | b2 2560→1280 k1 32×32 | the same conv at 1024×1024 | clean |
   | b2 2560→1280 k1 42×24 | the same conv at 768×1344 | clean |
   | b1/b3/b4/b8 2560→1280 k1 24×42 | any batch but 2 | clean |
   | b2 2560→1280 k3 24×42 | 3×3 instead of 1×1 | clean |

   So it is not about narrowing, not about kernel size, not about resolution and
   not about batch — it is the combination, and it is the cuDNN kernel picked for
   that combination. `benchmark` and `deterministic` change nothing.

   The remedy is therefore to **stop using cuDNN for the whole process** rather
   than to dodge one shape. `pipelines.apply_cudnn_workaround()` probes a table
   of known-bad shapes once, sets `torch.backends.cudnn.enabled = False` if any
   fails, and **re-probes to confirm the fallback is actually clean**. PyTorch's
   own convolution path costs nothing here: measured at 0.93× cuDNN's time over
   a real sampler step, because a 1660 Ti has no tensor cores for cuDNN to use.
   `CLAUDALI_CUDNN` is `auto|on|off`.

   The fp32 VAE decode (`CLAUDALI_VAE_UPCAST`) is now the **second** line of
   defence: in `auto` it fires only if the fault survives disabling cuDNN. Do not
   replace the measurement with a check on the card's name, and do not trim the
   probe table down to one shape — a single-shape probe is exactly what let this
   fault through the first time. It passed at 1024×1024 and failed at 1344×768.

Do not "optimise" any of these away without checking `claudali doctor` output on
the actual machine.

## Module map

Dependencies point downward. Nothing below imports anything above it.

```
claudali/
  config.py       Paths and settings. Redirects HF_HOME into models/ AT IMPORT
                  TIME -- must stay import-safe and stdlib-only.
  spec.py         The SceneSpec pydantic models. The contract. extra="forbid".
  tokens.py       Counts CLIP tokens and packs a prompt into 75-token chunks.
                  Lazy tokenizer, fitted fallback. Stdlib at import time.
  compiler.py     spec -> weighted prompts. Reads vocabulary/*.yaml.
  vocabulary/     YAML tables: media, movements, lighting, camera, palettes,
                  negatives, intents, subjects. Data, not code -- edit freely.
  control/maps.py Procedural depth/edge/region maps from composition.layers.
                  numpy + Pillow only, no model loading.
  registry.py     Model catalogue; resolves file lists from the HF tree API.
  engine/
    quiet.py      Silences three specific, harmless library warnings by text
                  match on their loggers. Stdlib only.
    pipelines.py  Loads, configures and caches diffusers pipelines. All the
                  VRAM and fp16 handling lives here.
    render.py     Runs the sampler. Chooses txt2img / img2img / inpaint.
    regional.py   Opt-in masked cross-attention for per-layer prompts.
                  UNTESTED on hardware; falls back loudly.
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
   |                                  steps, cfg, sampler, size, token count,
   |                                  chunk anchors, regions, warnings, notes)
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
   Never ignore it quietly. (See `render()` raising on control + init.) This is
   the rule most easily broken by omission rather than by code:
   `composition.layers[].prompt` was read by *nothing* for months, and the spec
   even documented it as "reserved". If a field exists, something must consume
   it or say why it did not.
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
8. **Warnings and notes are different channels.** A warning means the spec
   probably wants changing. A note means the compiler decided something on the
   caller's behalf and is saying so. They were one list, and collapsing them
   trained the eye to skip both. An inferred default goes in `notes`.

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

**A subject was misjudged as having, or not having, a body** — add the word to
`person_words` in `claudali/vocabulary/subjects.yaml`. Data, no code change.
Resist writing a smarter rule; see the gotcha below on why. The same file holds
`plural_words`, for the warning that a one-body crop cannot hold several
subjects.

**A subject keeps disappearing from a long prompt** — read `tokens.chunks` in
the compiled result first. Past one chunk the subject is restated per chunk, and
`anchors` shows what was restated. If it is the whole of `subject.primary`, the
setting is being reinforced along with the subject: set `subject.anchor` to two
or three words naming the subject alone.

**Add an overlay type** — a `Literal` member on `Overlay.type` in `spec.py`, a
`_draw_*` function in `compose.py`, and a branch in `apply_overlays`.

## Testing without a GPU

Everything up to the sampler is deterministic and needs no weights, no CUDA and
no network. `tests/test_claudali.py` covers the spec contract, the compiler,
control maps, compositing, postprocessing, diagnostics and the engine's
device decisions — 61 tests, ~1.5 s.

```bash
.venv\Scripts\python -m pytest -q            # or: pip install pytest
```

Several tests are regression locks on bugs that actually happened: an explicit
`steps: 30` being overwritten by the intent default, `horizon` being inverted,
the gradient `direction` being backwards, `auto` VAE upcasting ignoring the
measured card, the cuDNN workaround not firing on a card that needs it, a long
prompt leaving four of its five CLIP chunks with no mention of the subject, the
token estimator reading 31% low, and an occupation such as "alchemist" not
counting as a person. Do not delete them to make a change pass.

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

- **A long prompt loses its subject, and the image looks fine without it.**
  This is the single most expensive failure mode in the whole tool, because
  nothing errors. CLIP reads 75 content tokens at a time; compel encodes chunk
  after chunk and concatenates, and cross-attention is then a softmax over the
  *whole* concatenated sequence, so every chunk votes. A subject named only in
  chunk one is outvoted by every later chunk, and the later chunks are setting
  and style.

  Measured, not theorised. `"a photorealistic ancient forest with multiple
  small ethereal fairies"` compiled to 306 tokens; "fairies" appeared at tokens
  46, 58 and 73, all inside chunk one. Chunks two to five were rainforest, god
  rays, a 50mm lens, bokeh and a green palette. All four variations rendered as
  beautiful, empty forests.

  The fix is `_assemble` in `compiler.py` plus `tokens.pack`: pack fragments
  into groups of at most 75 tokens, reserving room for the anchor, and restate
  `subject.anchor` at the head of every group after the first. **The packing
  deliberately does not try to line up with compel's window boundaries**, and
  it does not need to — with consecutive anchors at most 75 tokens apart, any
  75-token window must contain one, or two neighbouring anchors would straddle
  a gap the packing forbids. That pigeonhole argument is the whole guarantee;
  `test_the_subject_appears_in_every_clip_chunk` locks it. Do not "simplify" it
  into anchoring at fixed token offsets.

- **Do not estimate CLIP tokens by counting words.** The old guard was
  `len(prompt.split()) * 1.3`, which read 212 for the 306-token prompt above,
  so the over-length warning never fired on the one prompt that needed it. A
  compiled prompt is mostly commas and `(phrase)1.15` weight syntax, and every
  bracket, comma and digit run is a token that is not a word. `tokens.py` uses
  CLIP's own tokenizer when any model is installed and a **fitted** fallback
  otherwise, accurate to 3.7% worst case over the example corpus. The two
  constants in `estimate_tokens` were measured; re-fit them, do not tune them
  by eye. The compiled result says which method was used, because a number that
  is sometimes exact and sometimes guessed is misleading unless it says so.

- **A "photoreal" negative preset can delete a fantasy subject.** The
  `photoreal` intent used to inherit `cgi`, which is `"cgi, 3d render, video
  game screenshot, plastic skin, uncanny valley"`. SDXL's idea of a
  photorealistic fairy, dragon or robot is built out of illustrative and
  rendered space, so banning that space removes the subject and leaves the
  setting. Split into `cgi_surface` (kept) and `cgi_medium` (dropped);
  `cgi` survives as the union for callers who name it. Do not merge them back.

- **Shot keys describe where a *body* is cropped.** `medium` is "waist up" and
  `close_up` is "head and shoulders". A forest has no waist, so SDXL resolves
  the instruction against whatever is nearest and returns a macro shot of
  undergrowth — which is the other half of why the fairy render was pictures of
  moss. `scene`, `tableau`, `group` and `tight` are the bodiless equivalents.
  The `body` and `single` flags in `camera.yaml` drive warnings only.

- **The person-word list will be wrong sometimes, and that is priced in.**
  `vocabulary/subjects.yaml` decides whether a subject has a body. The first
  version did not know "alchemist" was a person, so it called `full_body` on
  one a mistake. It gates two warnings and whether the `anatomy` negatives are
  inherited — never the prompt text — and the compiler says out loud when it
  drops them. Fix a misjudgement by adding the word, not by writing a cleverer
  rule: a suffix rule generous enough to catch "-ist" also makes people out of
  "mist" and "water".

- **Three library warnings are silenced on purpose, and only those three.**
  diffusers' empty "kept in float32: []" list, transformers' `Siglip2ImageProcessorFast`
  rename, and the tokenizer's "longer than the specified maximum (131 > 77)",
  the last only while compel encodes. `engine/quiet.py` explains why each is
  false here. They are matched by message text on the one logger that emits
  each. Do not widen that to logger levels: that would also hide the next real
  warning those libraries print.
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
- **compel needs `CompelForSDXL`, built with the encoders on the GPU.** Two
  separate traps, both of which silently cost every attention weight. The bare
  `Compel` class accepts a list of two encoders but its padding helper reads an
  `empty_z` attribute the multi-encoder provider does not have, so any prompt
  whose positive and negative differ in token length raises. And each provider
  captures its encoder's device **once, at construction**: under CPU offload the
  encoders are on the CPU at that moment, so the provider builds token ids there
  while the offload hook has already moved the weights to the GPU, and the call
  dies at `index_select`. Passing `device=` does not help, because it is not
  forwarded to the providers. `_build_compel` moves the encoders to the execution
  device, constructs, and moves them back.
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
- **Regional prompting exists but is unproven.** `composition.regional` patches
  the UNet's cross-attention processors to apply each layer's prompt inside its
  own mask. It was written to request, it is opt-in, it has never been run on
  the GPU, and every failure path falls back to a global render with a note. It
  is deliberately *not* a `control.mode`, because ControlNet conditions geometry
  through a side network while this decides which text applies where; they act
  at different points and can combine. Do not promote it to a default, and do
  not delete the fallback, until somebody renders with it and looks.
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
