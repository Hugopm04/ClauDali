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

The target machine has a **GTX 1660 Ti, 6 GB VRAM, Turing sm_75**, in a laptop
with 16.6 GB of RAM of which other programs hold ~9 GB. Five consequences
shaped the architecture, and none of them are negotiable:

1. **No tensor cores.** fp16 buys memory headroom, not speed. Measured on this
   laptop: ~13–14 minutes per 1344×768 image at 32 steps (23–26 s per step),
   with model offload, attention slicing and cuDNN off. This is why the API is a
   job queue and not a blocking call, and why the queue has exactly one worker.
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
5. **The VAE decode is where quality and memory trade.** With cuDNN off, GPU
   convolutions go through im2col: the column buffer for one 3×3 convolution at
   1344×768 is 4.75 GB in fp16 and 9.5 GB in fp32. Measured decodes of one
   1344×768 image:

   | Decode | Time | Memory |
   |---|---|---|
   | GPU fp16, tiled, fp16-fix VAE | 25.5 s | 1.76 GB VRAM |
   | GPU fp16, untiled | 18.2 s | 6.39 GB, over the card; shared memory absorbed it |
   | GPU fp32, untiled | out of memory | wanted 8.86 GiB |
   | CPU fp32, untiled, checkpoint VAE | 29.0 s | ~5.8 GB of RAM |

   Tiling blends overlapping tiles with per-tile GroupNorm statistics, and on the
   GPU it applies to every SDXL size: diffusers tiles above the VAE's
   `sample_size`, 512 px for the fp16-fix VAE (1024 px for SDXL base's own).
   So `engine/decode.py`'s `auto` prefers the untiled CPU decode, and RAM decides:
   a loaded pipeline leaves 0.1–1.8 GB free here. Measured 2026-09-13 with SDXL
   base loaded: `auto` at 768×768 saw 0.1 GB free and fell back to the GPU;
   forced `cpu` completed at 768×768 with 1.1 GB free, paging, no NaN. On this
   laptop `auto` falls back almost every time, and `cpu` is the way to get the
   untiled decode. `auto` as the default was agreed with Hugo; do not change it
   without asking.

   The CPU decode also buys a better decoder. On a 768×768 crop of a real
   render, in fp32 on the CPU, SDXL base's own VAE round-tripped at 50.2 dB PSNR
   and the fp16-fix VAE the GPU path uses at 43.6 dB. The cost can come later
   than the decode, though: after a forced CPU decode short of RAM, the next
   render in the same process waited ~2.5 minutes for its first step and ran a
   768×768 step in 90 s instead of ~12 s, most likely paging the offloaded
   weights back in.

Do not "optimise" any of these away without checking `claudali doctor` output on
the actual machine.

## Module map

Dependencies point downward. Nothing below imports anything above it.

```
claudali/
  config.py       Paths and settings. Redirects HF_HOME into models/ AT IMPORT
                  TIME -- must stay import-safe and stdlib-only.
  sysinfo.py      Free and total RAM, via ctypes on Windows and /proc on Linux.
                  None when unmeasurable. Stdlib only.
  spec.py         The SceneSpec pydantic models. The contract. extra="forbid".
  tokens.py       Counts CLIP tokens and packs a prompt into 75-token chunks.
                  Lazy tokenizer, fitted fallback. Stdlib at import time.
  compiler.py     spec -> weighted prompts. Reads vocabulary/*.yaml.
  vocabulary/     YAML tables: media, movements, lighting, camera, palettes,
                  negatives, intents, subjects. Data, not code -- edit freely.
  control/maps.py Procedural depth/edge/region maps from composition.layers.
                  numpy + Pillow only, no model loading.
  registry.py     Model catalogue; resolves file lists from the HF tree API.
                  Says why the optional refiner or upscaler cannot run.
  engine/
    quiet.py      Silences three specific, harmless library warnings by text
                  match on their loggers. Stdlib only.
    checkpoint.py Pause and exact resume: the controller, resume state,
                  checkpoint files (with the latents waiting between stages),
                  fingerprint, and the two pipeline hooks. Stdlib at import time.
    pipelines.py  Loads, configures and caches diffusers pipelines -- the base,
                  the refiner, and the checkpoint's own VAE for CPU decoding.
                  All the VRAM, precision and offload handling lives here.
    decode.py     The VAE decode policy (auto/cpu/gpu/gpu_tiled) and the
                  decode and encode that follow it. Stdlib at import time.
    upscale.py    Lanczos, or Real-ESRGAN through spandrel (imported lazily),
                  before the hi-res pass.
    render.py     Runs each variation through its stages -- base, then the
                  optional refiner and hi-res pass -- batched by model.
                  denoise() runs one stage. Chooses txt2img / img2img / inpaint.
    regional.py   Opt-in masked cross-attention for per-layer prompts.
                  UNTESTED on hardware; falls back loudly.
  diagnostics.py  Measurements over a finished image. Reports, never enforces.
  compose.py      Post-diffusion: Pillow overlays and exact postprocessing.
  bundle.py       Writes the output bundle as the render goes (BundleWriter):
                  images, previews, contact sheet, sidecar, diagnostics, status.
  jobs.py         Single-worker queue: progress, ETA, cancellation, pause with
                  or without holding the queue, resume at the front, and the
                  queue saved to runs/queue.json across a server restart.
  api.py          FastAPI app. Also serves web/index.html.
  web/index.html  The UI: one file, vanilla JS, no build step.
  __main__.py     CLI: serve / compile / render / resume / doctor / models.

installer/
  progress.py     Dependency-free progress bars. Stdlib only, deliberately.
  download.py     Resumable HTTP downloads with byte-level progress.
  core.py         Install stages: venv, torch, deps, models, verify.
  uninstall.py    Tiered removal with confirmation and reclaimed-space reporting.
```

**Import-cost rule:** importing `claudali.api`, `claudali.jobs` or
`claudali.bundle` must never pull in torch. Torch is imported *inside functions*
in `engine/`, including `engine/checkpoint.py`, which all three import. This
keeps the server start, the CLI, and every test that does not render fast. If
you add a top-level `import torch` anywhere outside a function, you have broken
this, and `test_the_server_modules_do_not_import_torch` will say so.

## Data flow

```
SceneSpec
   |
   |-- compiler.compile_spec ------> CompiledPrompt (prompt, negatives, model,
   |                                  steps, cfg, sampler, size, token count,
   |                                  chunk anchors, regions, warnings, notes,
   |                                  precision, vae_decode, refiner, hires, stages)
   |-- control.maps.build_control_image -> depth / canny PNG   (optional)
   |
   v
engine.render.render  ---> stages per variation: base [-> refiner] [-> hires]
   |                        with a refiner: all bases, all refiners, all hires
   |                        each stage: denoise -> latents, held between stages
   |                        hires: decode -> upscale -> encode -> img2img
   |                        last stage: decode.decode by the vae_decode plan
   |                        on_checkpoint(ResumeState) before each stage
   |                        on_variation(RenderedImage) as each image lands
   |                        raises RenderPaused / RenderAborted when asked
   |                        returns RenderResult (images + seeds + warnings + notes + device)
   v
bundle.BundleWriter   ---> compose.finish (postprocess, then overlays), then
                           outputs/<stamp>_<slug>_<jobid>/ written as it goes:
                             images, previews, contact sheet, spec.json,
                             result.json (status, diagnostics), checkpoint/
```

The caller owns the disk: `jobs.py` and the CLI hand `BundleWriter` methods to
`render` as its callbacks, and save the checkpoint a pause raises.

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
   trained the eye to skip both. An inferred default goes in `notes`. The render
   result, the bundle, the CLI and the UI keep the two apart as well, and the
   engine sorts its own messages the same way: something asked for that did not
   happen is a warning (compel unavailable, a prompt truncated without it), a
   workaround chosen for the caller is a note (cuDNN disabled).

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

Update `approx_gb` to the measured value. `PROFILES` and `QUALITY_MODELS` are
explicit lists, so a new entry is in no profile until you add it to one. Only
`kind="checkpoint"` is offered as a base model; the refiner and upscaler have
kinds of their own for that reason.

**Add a sampler** — one entry in `SAMPLERS` in `engine/pipelines.py`, mapping to
a diffusers scheduler class name and its kwargs. A test builds every entry and
fails if a class needs a package `requirements.txt` does not install, which is
why `lms` (scipy) is gone.

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
control maps, compositing, postprocessing, diagnostics, the engine's device
decisions, the silenced warnings, pause/resume (checkpoints, the queue, the
bundle writer, and an exact resume on a tiny random SDXL pipeline on the CPU),
and the quality options: the decode and offload plans, the max preset, and
render() itself taking tiny base and refiner pipelines through all three stages,
paused in the refiner and resumed. 103 tests, ~15 s. Importing diffusers for the
tiny pipelines is most of it.

What needs the GPU and a person looking -- the web UI, decode quality, timings,
the refiner, Real-ESRGAN, fp32 -- is a checklist in `tests/manual/README.md`,
with its specs beside it. They are deliberately not in `examples/`: several
warn by design, and every example must compile without warnings.

```bash
.venv\Scripts\python -m pytest -q            # or: pip install pytest
```

Several tests are regression locks on bugs that actually happened: an explicit
`steps: 30` being overwritten by the intent default, `horizon` being inverted,
the gradient `direction` being backwards, `auto` VAE upcasting ignoring the
measured card, the cuDNN workaround not firing on a card that needs it, a long
prompt leaving four of its five CLIP chunks with no mention of the subject, the
token estimator reading 31% low, an occupation such as "alchemist" not counting
as a person, a resume that re-ran every step instead of continuing, a torch
import creeping into the server modules, a saved spec whose written-out defaults
beat its intent when loaded back, a sampler inheriting the previous one's Karras
sigmas, `lms` needing a package nothing installs, and a staged resume re-running
stages it had finished. Do not delete them to make a change pass.

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
  proceeds with a warning that the weights were ignored, and another saying how
  much of a long prompt diffusers truncated at 77 tokens. Do not remove either.
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
- **Exact resume rests on two hooks into diffusers' loop, and a test locks them.**
  diffusers cannot start a sampler partway. `checkpoint.resume_into` wraps
  `pipe.prepare_latents` on the instance: the original runs, so the generator
  is consumed as usual, and then the saved latents are swapped in. From inside
  that wrapper it sets `pipe._interrupt` to a `_SkipUntil`, which makes
  `if self.interrupt: continue` skip the steps already done. At the first step
  left to run, `_SkipUntil` restores the scheduler state and generator state.
  It has to be inside the loop, after `set_timesteps` and `set_begin_index`
  have reset exactly that state. Both hooks are removed in `finally`, because
  the pipeline is cached. `test_a_paused_render_resumes_bit_for_bit` runs a
  tiny random SDXL pipeline on the CPU. It checks `torch.equal` on the latents
  and counts UNet calls, since a resume that silently re-ran every step with
  the same seed would also match. If a diffusers upgrade breaks it, fix the
  hooks; do not loosen the test. The pipeline is called with
  `output_type="latent"` and decoded in `engine/decode.py`, whose GPU path
  `decode_latents` is a line-for-line mirror of the pipeline's own decode. That
  is what lets a render paused after its last step, or handed to a later stage,
  need no second path. The same hooks work in img2img, which the refiner and
  hi-res stages are: `test_a_render_paused_between_stages_resumes_bit_for_bit`
  pauses inside the refiner stage of a tiny render and compares final pixels.
- **A pause must never overwrite a step checkpoint with a boundary one.** `render`
  calls `on_checkpoint` before each stage of each variation, which is what lets
  a killed process resume. It skips that call for a stage resuming mid-image. A
  crash during the resume then still resumes from the saved step, not from the
  stage's first step.
- **With a refiner, the stages are batched, and a phase must let go of the last
  one's pipeline.** The refiner is a second 6 GB model and the cache has one slot:
  every variation's base stage runs, the base is released, every refiner stage
  runs, and the base is loaded again for the hi-res passes. The latents waiting
  in between live on the CPU and in the checkpoint (`ResumeState.staged`, format
  2). Releasing the cache slot frees nothing while `_Run` still references the
  old pipeline, its derived pipelines or its embeddings, so `_acquire` clears
  them before loading: miss one and two UNets share 16 GB of RAM. Without a
  refiner there is one phase and each variation runs base then hi-res back to
  back, so its image lands sooner. Because the waiting latents are on the CPU,
  the GPU decode moves them to the VAE's device and dtype itself: diffusers'
  offload hook on `vae.decode` moves the VAE, never its input, and the first
  hardware check of the hi-res path failed on exactly that.
  `test_latents_waiting_on_the_cpu_decode_on_the_gpu_path` locks it.
- **`render.quality: "max"` resolves in the compiler, never in the spec.** A
  validator that filled in precision or the refiner would mark them set, and a
  spec saved with `exclude_unset` would come back with the preset spelled out
  as explicit choices. `_resolve_quality` supplies what the spec leaves unset,
  explicit fields win (`refiner.enabled: false` under `max` keeps the rest), and
  the result is frozen in `CompiledPrompt`, so a resume keeps it. An explicit
  refiner or Real-ESRGAN that cannot run is kept, so `render` refuses it; under
  the preset it is left out with a warning.
- **The checkpoint's own VAE is not SDXL base's VAE.** Juggernaut ships a VAE of
  its own inside its single file: every tensor key matches SDXL's, but weights
  differ by up to 55. Measured round-trip PSNR on a 512 px crop was 63.4 dB for
  SDXL base's VAE and 49.8 dB for Juggernaut's, both clean. The CPU decode reads
  just those tensors from the file through safetensors and diffusers' own
  converter; loading the whole 7 GB file for them would not fit beside a loaded
  pipeline. The GPU decode uses the fp16-fix VAE, which is SDXL base's.
- **compel under sequential offload cannot move the encoders.** `_build_compel`
  moves them to the GPU for construction (see the compel gotcha above), but
  sequential offload leaves placeholder weights that cannot be moved, and fp32
  on a small card selects sequential offload. There the encoders stay put and
  compel is handed `device=` instead, which compel 2.4 does forward to each
  provider. Unmeasured, like fp32 itself.
- **A hi-res pass can round to zero steps.** img2img runs `int(steps *
  strength)` steps, so `hires.steps: 2, strength: 0.3` is none, and diffusers
  fails obscurely. The compiler warns and `render` refuses with the arithmetic.
- **`from_pipe` shares weights.** img2img and inpainting derive from the loaded
  txt2img pipeline at no extra disk, download or load cost. This is why SDXL base
  can inpaint without a dedicated inpainting checkpoint.
- **A saved spec holds only what was set.** `BundleWriter.create` and the queue
  save use `model_dump(exclude_unset=True)`, and `SceneSpec` unmarks the size it
  fills in from the aspect. A full dump writes `steps: 30` and `cfg: 6.5` out as
  if chosen, so a painterly spec loaded back from history rendered at 30 steps
  and CFG 6.5 instead of 34 and 7.5, and its stored size beat any new aspect.
  Older bundles still hold full dumps.
- **A sampler is built from the checkpoint's scheduler, never the current one.**
  `from_config` carries over every setting the new class accepts, so building
  from whatever the last render left gave `euler` Karras sigmas after a
  `dpmpp_2m_karras` job: same spec, same seed, a different image depending on the
  job before. `LoadedPipeline.base_scheduler` is the one loaded, and
  `apply_sampler` sets the sampler per render, not per load.
- **The web form edits only what it shows.** `FORM_FIELDS` in `web/index.html`
  maps each input to a spec path. A loaded spec is kept whole; an input still
  showing what loading wrote leaves the spec's value alone; everything no input
  shows is listed under "Also in this spec". `buildSpec` used to rebuild the spec
  from the form, which dropped `subject.anchor`, `overlays`, `post`, `init` and
  the rest without a word. A new input is one `FORM_FIELDS` entry; give a select
  a blank default option so leaving it alone writes nothing.
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
- **The refiner, Real-ESRGAN and full-size hi-res are untested on hardware, and
  fp32 is unmeasured.** None of the refiner or Real-ESRGAN weights were on the
  machine when they were written, and a 2016×1152 hi-res pass or an fp32 load
  did not fit the GPU test budget. The CPU stage-loop test locks the logic, not
  the models. Keep them opt-in, and label them so until somebody renders with
  them and looks.
- **No refiner with init or inpainting, and no ControlNet after the base
  stage.** The refiner with `init` raises. The refiner and hi-res stages run
  uncontrolled, and a note says so: a hi-res ControlNet pass would need the
  control image rebuilt at the new size, which is untested territory.
- **`post.seamless` is a cross-fade heuristic**, good for organic textures and
  visibly wrong for anything structured. `post.transparent_bg` is a corner
  flood-fill, not segmentation. Both are documented as such; neither should be
  quietly upgraded to something with a model dependency.

## Commands

```powershell
.\install.ps1 [-ModelProfile minimal|standard|full] [-QualityModels] [-RecreateVenv] [-Cpu]
.\scripts\start.ps1 [-BindHost 127.0.0.1] [-Port 8188] [-NoBrowser]
.\uninstall.ps1 [-Models] [-Env] [-Outputs] [-All] [-DryRun]

python -m installer install --profile standard [--with-quality-models]
python -m installer models --add juggernaut-xl
python -m installer status

.venv\Scripts\python -m claudali serve|compile|render|doctor|models
.venv\Scripts\python -m claudali resume <bundle dir> [--force]
```

## Keeping docs in step

`README.md` is for users, this file is for future sessions, `docs/` holds
reference material. When behaviour changes, fix the specific sentences it made
wrong — in particular the profile sizes in README and `installer/__main__.py`
help text, which are measured values and drift when the catalogue changes.
`docs/vocabulary.md` is generated by `scripts/gen_vocab_docs.py`; re-run it after
editing any vocabulary table.
