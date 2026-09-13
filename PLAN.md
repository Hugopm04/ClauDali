# Work plan: warnings, pause/resume, cleanup, quality options

Written 2026-09-13 at commit `de25160`, for the Claude sessions that will carry
this out. Every decision below was made with Hugo in the planning session. **Do
not re-ask them.** Where this plan picks a default Hugo was not asked about, it
says so; change it only with a reason.

Line numbers are as of `de25160`. Before editing any file, compare its mtime
with this plan's date and re-read it if newer (Hugo's global CLAUDE.md rule).

---

## 0. How to use this plan

1. Read sections 1–5 in full. Sections 6–9 are one per task; read the one you
   are doing. Section 11 is a map of the code relevant to this work, so you do
   not need to re-read whole modules to orient yourself.
2. Do the tasks in order: **1 → 2 → 3 → 4**. Task 4 builds on task 2's render
   refactor, and task 3 touches the same bundle, render and UI code as task 2.
3. **One commit per task, on `main`, no push.** Hugo chose that for tasks 1–3.
   Task 4 grew out of a task-3 question; commit it separately and mention the
   split in your report.
4. **Update the status table below as you go**, not at the end. Record GPU
   minutes spent: the budget is shared across every session (section 3.3).
5. Keep docs in step with each change (Hugo's global rule). Each task section
   lists the doc spots it makes wrong.
6. When everything here is done, delete `PLAN.md` and its pointer in `CLAUDE.md`
   in the last commit.

### Status

| Task | State | Commit | GPU min used | Notes |
|---|---|---|---|---|
| Planning session: VAE decode benchmark | done | — | ~1.5 | Section 4.2 |
| 1. Silence three library warnings | done | `463a4df` | 0 | Verified on CPU against the real libraries: all three printed without `quiet`, none with it; the tokenizer warning still prints outside compel. Section 10 test 1 then confirmed none of the three on stderr during a Juggernaut load and render |
| 2. Exact pause/resume (engine, CLI, API, UI) | done | see `git log` | ~3.5 | Section 10 test 1 passed in full, see below. **Pushed** at Hugo's request, not only committed |
| Section 10 test 1 (after task 2) | done | — | (in row 2) | Juggernaut, 768×768, 4 steps, dpmpp_2m_karras, cuDNN disabled by the probe. Uninterrupted vs paused after step 2, saved to disk, resumed through `render()` in a reopened bundle: **latents `torch.equal`, pixels identical**. `decode_latents` vs the pipeline's own decode of the same latents: **identical, max pixel diff 0**. Steps took ~10–13 s at 768². **Available RAM: 6.9 GB before loading, 1.29 GB after the pipeline load and a render**, of 16.56 GB total. So task 4's `auto` decode (~3.3 GB needed at 768², ~5.8 GB at 1344×768) will fall back to GPU tiled on this laptop, as 4.2 predicted. Remaining GPU budget: **~9.5 min** |
| 3. Cleanup | not started | | | |
| 4. Quality options (decode, fp32, refiner, hi-res, max preset) | not started | | | |

### Found while executing (read before task 3)

Facts discovered by the session that did tasks 1 and 2, 2026-09-13. None of
them was acted on unless the row above says so.

- **The `lms` sampler cannot load.** `LMSDiscreteScheduler` needs `scipy`, which
  is not installed and not in `requirements.txt`. `sampler: "lms"` fails at
  pipeline load. Task 3 should either add `scipy` to the requirements (Hugo
  re-runs the installer) or drop `lms` from `SAMPLERS` and the docs. Every other
  sampler's state was probed on CPU and is tensors, lists of tensors/None and
  primitives only.
- **The test suite takes ~5 s warm and ~20 s cold**, not the "~1.5 s" in
  CLAUDE.md. Loading the CLIP tokenizer dominates. Fix the number in task 3's
  doc pass along with the test count.
- `.venv\Scripts\python` scripts run from the scratchpad need
  `PYTHONPATH=<project root>`, since `sys.path[0]` is the script's directory.

**What task 2 changed that tasks 3 and 4 build on.** Section 11's line numbers
for `render.py`, `bundle.py`, `jobs.py`, `__main__.py`, `api.py` and
`web/index.html` are stale; re-grep. The rest:

- `render.py` now has `denoise(pipe, call_kwargs, seed, controller=, resume=,
  on_step=)` returning latents, and `decode_latents(pipe, latents)`, a mirror of
  the pipeline's decode. **Task 4's decode policy replaces the body of
  `decode_latents`**; keep today's GPU path as one branch of it. The refiner and
  hi-res stages can call `denoise` on any SDXL pipeline, and a pause inside them
  already works through `checkpoint.resume_into`. `ResumeState.stage` exists and
  is always `"base"`. `render()`'s loop is per variation. The refiner's "all
  bases first, then all refiners" batching needs a second loop, and the
  checkpoint needs to hold the saved base latents of finished-base variations.
  That is a format change, so bump `checkpoint.FORMAT_VERSION`.
- `render()._fingerprint` lists what a resume compares. **Task 4 must add
  precision, vae_decode, refiner and hires settings to it**, or a resume under
  different quality settings will not be refused.
- `render()` still merges `compiled.warnings + compiled.notes` into `notes` (task
  3 item 2). `BundleWriter._absorb` dedupes notes across sessions of a resumed
  job; split the channels there too.
- `write_bundle` is gone; `BundleWriter.create/open`, then `add_variation`,
  `record_checkpoint`, and one of `complete/pause/abort/cancel/fail`.
- UI: new `watchJob`, `showJobActions`, `loadPaused`, `describeCheckpoint`, and
  `esc()` for HTML escaping. `buildSpec`/`applySpec` are untouched, so task 3
  item 1 is still wholly to do.
- Deviations from section 7, all small and deliberate. (1) A job's `error` stays
  a string; a refused resume adds `error_type: "resume_mismatch"` and a
  `mismatch` list, rather than turning `error` into an object for every job.
  (2) The bundle status list gained `cancelled`, distinct from `aborted`:
  aborted is resumable from the variation start, cancelled is not. (3)
  Cancelling the job that holds the queue clears the owner but leaves the queue
  held; the UI offers "Release queue". (4) `POST /api/render` returns 503 when
  the queue is full instead of blocking the request thread. (5) `claudali serve`
  runs `uvicorn.Server` itself, so the shutdown can see uvicorn's second-Ctrl+C
  flag through `api.force_exit_requested`.
- **Verified: uvicorn 0.52 on Windows runs the lifespan shutdown and waits for
  it.** A real `serve` process was stopped with Ctrl+Break, which goes through
  the same `handle_exit` as Ctrl+C. It saved the held queue, restored it on
  start, and served the pause endpoints. A second SIGINT sets `force_exit`, and
  uvicorn skips the lifespan shutdown only if that is set *before* the shutdown
  starts.
- `JobQueue(autostart=False)`, `_take_next(timeout=0)` and `_settle(job)` are
  how the tests drive the queue without a worker thread or torch.
- Test count and time in CLAUDE.md were updated by task 2 (81 tests, ~35 s
  cold), so task 3's doc pass need not.

Suggested session split, if context runs short: session A = tasks 1 and 2;
session B = task 3; session C = task 4. Task 2 is the biggest; if it must be
split, finish engine + CLI + tests first, then API + UI, and commit once at the
end of the task.

---

## 1. What prompted this

Hugo ran `.venv\Scripts\python -m claudali render examples/forest-fairies.json`,
saw four warnings, then **stopped it with Ctrl+C** and never saw images. He asked
for three things, adding a standing rule of thumb:

1. Look at the warnings and fix them if needed.
2. Add a way to stop and continue a generation, if possible and sensible.
3. A code, documentation and architecture cleanup: remove or fix old, obsolete
   or wrong code, docs and architecture.
4. **Rule of thumb: always provide the best accuracy/quality generation as an
   option, even at very high time cost.**

The warnings he saw, all harmless (task 1):

```
There are modules in UNet2DConditionModel that should be kept in float32: []. Casting directly with `to()` can lead to inconsistent results; ...
[transformers] `Siglip2ImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Siglip2ImageProcessor` instead.
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (131 > 77). Running this sequence through the model will result in indexing errors   (x2)
```

**Why no images:** `bundle.write_bundle` runs only after every variation has
finished (`jobs.py:233`, `__main__.py:94`), and nothing handles
`KeyboardInterrupt`. A Ctrl+C at variation 4 of 4 loses all four. No bundle
newer than 2026-09-12 16:22 exists in `outputs/`.

---

## 2. Decisions made with Hugo

| Topic | Decision | Implications |
|---|---|---|
| Warnings | **Silence exactly these three**, with narrow log filters, each commented with why it is safe. Every other warning still prints. | Task 1. Draft code in Appendix A. |
| Pause/resume depth | **Exact mid-image resume.** Pause at any step; latents, sampler state and RNG saved to disk; the resumed image is bit-identical to an uninterrupted one; survives closing the terminal or rebooting. **Finished variations are always saved immediately.** | Task 2. |
| Where | **CLI, HTTP API and web UI.** | |
| CLI Ctrl+C | **First Ctrl+C pauses** (checkpoint at the current step, clean exit); **a second aborts immediately.** Continue with `claudali resume <bundle dir>`. | |
| Server pause | Hugo, verbatim: *"Give the option to pause and resume next job or to pause until resumed. Ctr + C does the second one."* So there are **two pause modes**: pause and let the next queued job run, or pause and hold the whole queue until resumed. **Ctrl+C on the server = pause and hold.** | Default chosen here, not asked: a resumed job goes to the **front** of the queue. |
| Server Ctrl+C persistence | **The paused job and the not-yet-started queue survive the restart, held.** The next `serve` lists them all, held until resumed. | Queue persisted to disk. |
| Resume mismatch | **Refuse, with an explicit override.** If anything that affects the pixels changed (model file, library versions, precision, cuDNN state…), stop and list what changed; `--force` / `"force": true` continues and records the mismatch in the bundle notes. | |
| Exact-resume test | **In the default test suite** (tiny random SDXL pipeline on CPU, +10–20 s). | Appendix B. |
| Speed | Measured ~13 min/image vs the documented 2–4. **Only correct the docs to the measured numbers.** Do not investigate or change speed settings. | Likely cause, for the record only: attention slicing is on by default and diffusers warns it seriously slows SDPA (`pipeline_utils.py:2082`). **Do not act on this unless Hugo asks.** |
| GPU testing | *"No long tests, just short ones. The total process of testing should take less than 15 min."* | Section 3.3. No full fairy render; Hugo runs that himself. |
| VAE decode default | **CPU fp32, untiled, with the checkpoint's own VAE; auto-fallback.** If free RAM is below what the decode needs, fall back to GPU tiled and say so in the notes. | Task 4. Section 4.2 has the measurements. |
| Web UI dropping fields | **Merge form edits onto the loaded spec + add a `subject.anchor` field.** The form edits only what it shows; everything else in a loaded spec is kept and listed as "also in this spec"; "Use this JSON" renders exactly that JSON. | Task 3. |
| Best-quality options | Hugo, verbatim: *"Add all of the above, making it optional to use each of them. Add the new downloads to the download script and an option to download them, don't download them by yourself."* "All of the above" = (a) make the top-precision paths actually work (fp32 on 6 GB via sequential offload), (b) a one-switch max preset, (c) an SDXL refiner stage and a hi-res pass. | Task 4. **Never download model weights yourself**, and do not `pip install` new packages either (see 3.2). |
| fp32 precision | **Build it, label it unmeasured.** Needs ~13–14 GB of weights in RAM on a 15.4 GB machine; will page. Document the RAM requirement and that its speed was not measured. | |
| Refiner | **Official "ensemble of experts" handoff, stages batched:** base denoises the first ~80% of every variation, is unloaded, then the refiner finishes each. One UNet in RAM at a time. | |
| Hi-res pass | **Both upscalers, selectable:** `lanczos` (no download) and a Real-ESRGAN x4 model via `spandrel` (installer option), then SDXL img2img at low strength. | |
| Max preset | A spec field, `render.quality: "max"`, turning on **all four**: fp32 precision, untiled fp32 VAE decode, refiner (when installed, else a warning), hi-res pass. | Explicit fields still win. |
| Regional prompting | **Keep as is.** Opt-in, falls back loudly, documented as unproven. | No change. |
| `build_region_masks` | **Wire it in as `init.mask_layer`**: inpaint a region by naming a layer role instead of drawing a mask. | Task 3. |
| Commits | **One commit per task on `main`**, no push. | |

About Hugo: he writes briefly in English with typos. He answers precise
questions quickly, told the planning session to ask about every genuine doubt,
and limits GPU time. Most doubts are settled above. Ask only when something
truly blocks you.

---

## 3. Constraints and environment

### 3.1 Machine (measured 2026-09-13)

| | |
|---|---|
| Machine | **Laptop.** AMD Ryzen 7 4800H, 8 cores / 16 threads; torch uses 8 threads |
| RAM | **15.4 GB total, ~6.4 GB free at idle** (other apps hold ~9 GB). Pagefile 12.1 GB, 4.7 GB free |
| GPU | NVIDIA GeForce GTX 1660 Ti, 6 GB, sm_75, driver 591.86, cuDNN 9.1 (90100) |
| Measured GPU state | `fp16_conv_broken=True`, `cudnn_disabled=True`, `fp16_conv_broken_without_cudnn=False`: ClauDali disables cuDNN for the process. **All GPU convolutions therefore use PyTorch's im2col path** (slow, memory-hungry; see 4.2) |
| OS / shells | Windows 11 Home; PowerShell 5.1 is the primary shell; Git Bash is also available |

Versions in `.venv`: torch 2.6.0+cu124, diffusers 0.40.0, transformers 5.16.1,
compel 2.4.0, accelerate 1.14.0, safetensors 0.8.0, tokenizers 0.23.2,
numpy 2.5.2, pillow 12.3.0, pydantic 2.13.5, fastapi 0.141.1, uvicorn 0.52.4,
pytest 9.1.1. **Not installed:** spandrel, invisible-watermark, psutil, einops.

Installed models (`models/weights/`): `sdxl-base`, `sdxl-vae-fp16-fix`,
`controlnet-depth-sdxl`, `controlnet-canny-sdxl`, `juggernaut-xl` (single file),
`dreamshaper-xl`, i.e. the `full` profile. **Not installed, and must not be
downloaded by you:** the SDXL refiner and any Real-ESRGAN model.

### 3.2 Hard rules

- **This folder is inside Hugo's Nextcloud sync root** (`Mis proyectos/`; see
  the parent `CLAUDE.md`). **Deleting a file deletes it from the cloud.**
  `models/` holds weights that carry Nextcloud file IDs. **Never delete
  anything under `models/`.** Deleting your own scratch files is fine.
- **No model downloads** and **no `pip install`**. Add new dependencies to
  `requirements.txt` / `pyproject.toml` so the installer picks them up when
  Hugo re-runs it. Code paths that need them must import them lazily and fail
  with a message saying how to install. Querying the HuggingFace **tree API**
  for file names and sizes is metadata only and is fine: it is what CLAUDE.md's
  "Add a model" step does (`registry.resolve_remote_files`).
- Existing design rules in `CLAUDE.md` still hold. The ones this work touches
  most: nothing silently dropped; explicit beats inferred (`model_fields_set`);
  warnings ≠ notes; the import-cost rule (no top-level `import torch` outside
  functions: `api`, `jobs` and `bundle` must stay torch-free at import); never
  write to `~/.cache`.
- `test_examples_compile_without_warnings` must keep passing: new spec fields
  must not add warnings to the bundled examples.

### 3.3 GPU test budget

**Total under 15 minutes across all sessions, short runs only.** ~1.5 min went
on the decode benchmark, leaving **~13 min**. Suggested use (section 10):
~6 min after task 2, ~5 min after task 4, ~2 min slack. Log what you spend in
the status table. Loading a pipeline costs 30–90 s per process, so batch checks
into one script per session.

### 3.4 Tool quirks that cost time in the planning session

- **The Grep/Glob tools respect `.gitignore`, and `.gitignore` has an unanchored
  `models/`**, so every `.venv/Lib/site-packages/*/models/` directory
  (`diffusers/models`, `transformers/models`) is **silently skipped**. Search
  those by explicit file path. Task 3 anchors the pattern (`/models/`), which
  removes the trap.
- A Grep over all of `site-packages` times out after 20 s. Narrow the path.
- PowerShell 5.1: no `&&`. `2>&1` on a native exe wraps stderr lines as
  `NativeCommandError` noise; pipe through `ForEach-Object { "$_" }` or use Git
  Bash. Set `$env:PYTHONIOENCODING='utf-8'` before running Python that prints
  paths: the project path contains `gráficos`.
- Run Python as `.venv\Scripts\python`; tests with
  `.venv\Scripts\python -m pytest -q`.
- The ctypes `K32GetProcessMemoryInfo` call needs `argtypes` set, or it raises
  `OverflowError` on 64-bit.

---

## 4. Measured facts

### 4.1 Render timings (from `outputs/*/result.json`)

All juggernaut-xl, 32 steps, 1344×768 (16:9), dpmpp_2m_karras, model CPU
offload, attention slicing on, cuDNN disabled, VAE tiling on:

| Bundle | Variations | Duration |
|---|---|---|
| `2026-09-12_140504_…_d0c60b8c` | 4 | 2794.6 s |
| `2026-09-12_144100_…_verified` | 1 | 860.8 s |
| `2026-09-12_150913_…_1ade6793` | 1 | 818.4 s |
| `2026-09-12_162229_…_6bffbed0` | 4 | 3176.8 s |

That is **~13–14 min per image**, roughly 23–26 s per step, and **47–53 min for
four**. The docs say 2–4 min per 1024×1024 image; that is wrong on this machine.
1344×768 and 1024×1024 have nearly the same pixel count.

### 4.2 VAE decode benchmark (random latent 1×4×96×168 = one 1344×768 image)

| Decode | Time | Memory | Result |
|---|---|---|---|
| GPU fp16, **tiled** (today's default), fix VAE | 25.5 s | 1.76 GB VRAM peak | ok |
| GPU fp16, untiled, fix VAE | 18.2 s | 6.39 GB "VRAM" peak | ok, but over the card's 6 GB: the driver's shared-memory fallback absorbed it. Fragile |
| GPU fp32, untiled, base VAE | — | tried to allocate **8.86 GiB** | OOM |
| **CPU fp32, untiled, base VAE** | **29.0 s** | **RAM peak 6.73 GB (~5.8 GB above the 0.9 GB baseline)** | ok, no NaN |

What this means:

- **Tiling is on for every non-square aspect bucket.** SDXL VAE
  `sample_size=1024` gives a tile threshold of 128 latent pixels
  (`autoencoder_kl.py:133,139,200`). 1344/8 = 168 > 128. Only 1:1 at 1024 is
  decoded whole. Tiled decode blends overlapping tiles (`tile_overlap_factor`
  0.25), and GroupNorm statistics differ per tile. That is the quality
  compromise.
- **GPU convolutions without cuDNN use im2col.** The column buffer for a 3×3
  conv with 256 input channels at 1344×768 is 256·9·1,032,192 elements =
  **4.75 GB in fp16, 9.5 GB in fp32**, which is exactly the 8.86 GiB OOM above.
  It also explains why the CPU decode costs about the same time as the GPU one.
- **RAM is the real limit for CPU decode.** ~5.6 GB per megapixel, linear in
  pixels (one measurement; treat as an estimate). With a pipeline loaded under
  model offload, SDXL's fp16 weights (~7 GB) also sit in RAM. **On this laptop
  the "auto" RAM check may fall back to GPU tiled almost every time.** Measure
  free RAM after a pipeline load in the task-4 GPU test and **tell Hugo what
  auto actually does**. `CLAUDALI_VAE_DECODE=cpu` (forced, accepts paging) is
  the obvious knob to offer; do not change the agreed default without asking.
- A hi-res image of 2016×1152 (1.5× of 16:9) would need ~13 GB for a CPU fp32
  untiled decode. It will not fit next to a loaded pipeline, so hi-res decodes
  will be tiled in practice.

Appendix C has the benchmark script.

### 4.3 The fairy spec (`examples/forest-fairies.json`)

`claudali compile`: model juggernaut-xl, 1344×768, 32 steps, cfg 5.5,
dpmpp_2m_karras. Positive **279 tokens, 4 CLIP chunks**, subject restated 3×.
Negative **129 content tokens (131 with BOS/EOS), 2 chunks**. That 131 is the
number in the warning.

---

## 5. Warning diagnosis (inputs to task 1)

| # | Message | Emitted at | Logger | Why harmless |
|---|---|---|---|---|
| 1 | `There are modules in UNet2DConditionModel that should be kept in float32: []` | `diffusers/models/modeling_utils.py:1500,1521-1524` | `diffusers.models.modeling_utils` | `fp32_modules = self._keep_in_fp32_modules or []` is never None, yet the guard is `fp32_modules is not None`, so it fires on every `.to(dtype)`. Printed while `from_single_file` loads Juggernaut. **The empty list means nothing needed keeping.** Drop only when the list is `[]` |
| 2 | `` `Siglip2ImageProcessorFast` is deprecated `` | `transformers/utils/import_utils.py:2644-2652` (`logger.warning_once`) | `transformers.utils.import_utils` | diffusers 0.40 imports every pipeline in `diffusers.pipelines` on first access (verified: even `from diffusers import StableDiffusionXLPipeline` loads `z_image`). `pipelines/z_image/pipeline_z_image_omni.py:20` imports the old name. **Cannot be avoided by importing different classes**; the planning session checked |
| 3 | `Token indices sequence length is longer than the specified maximum sequence length for this model (131 > 77)` ×2 | `transformers/tokenization_utils_base.py:2949-2966` | `transformers.tokenization_utils_base` | compel tokenizes the whole prompt to chunk it itself. `CompelForSDXL` builds both providers with `truncate_long_prompts=False` (`compel/convenience_wrappers.py:93-105`; the bare `Compel` default is `True`, `compel.py:24`). One warning per tokenizer instance (two encoders), triggered by the 131-token negative. **Only true outside compel**, so silence it only while compel encodes |

Do **not** "fix" #3 by raising `tokenizer.model_max_length` on the pipeline's
tokenizers: compel reads `model_max_length` to size its chunks. `tokens.py:124`
does raise it, but only on ClauDali's own counting tokenizer, never fed to a
model.

---

## 6. Task 1: silence the three warnings

**Design.** New stdlib-only module `claudali/engine/quiet.py` (full draft in
Appendix A):

- `install()` attaches permanent `logging.Filter`s to exactly two loggers:
  #1, dropped only when the message contains `should be kept in float32: []`,
  and #2. Idempotent.
- `compel_tokenization()` is a context manager that adds the #3 filter only for
  the duration of a compel call. Re-entrant.
- Filters match message text on the named logger. Logger-level filters run
  before handlers and before propagation, and transformers' `get_logger` returns
  stdlib loggers (`transformers/utils/logging.py`), so this works whether or not
  the library has configured its root logger yet.

**Hook points.**
- `claudali/engine/pipelines.py`: `from . import quiet` and `quiet.install()`
  at module import. `pipelines.py` is imported by `api.py`, `render.py` and the
  doctor command before any diffusers import happens inside functions.
- `claudali/engine/render.py:78`: wrap `loaded.compel(...)` in
  `with quiet.compel_tokenization():`.
- `claudali/engine/regional.py:134`: same around `loaded.compel(...)`.

**Tests** (no torch). Attach a collecting `logging.Handler` directly to each
named logger, not via `caplog`: transformers may set `propagate=False` on its
root logger once another test has imported it. Assert:
- #1 with `[]` is dropped; the same text with a non-empty list is kept.
- #2 is dropped; a different message on that logger is kept.
- #3 is kept outside `compel_tokenization()`, dropped inside, kept again after.

**Docs.** `CLAUDE.md` module map: add `quiet.py`. `CLAUDE.md` gotchas: a short
entry, "three library warnings are silenced on purpose; do not widen to logger
levels", pointing at the module docstring. README: nothing, users no longer see
them.

**Verify on GPU** in the section-10 test 1: no line containing any of the three
messages on stderr during a Juggernaut load and render.

---

## 7. Task 2: exact pause/resume

### 7.1 Why it is possible: the diffusers loop

`StableDiffusionXLPipeline.__call__` (`pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`):

| Line | What |
|---|---|
| 818 | `interrupt` property returns `self._interrupt` |
| 1059 | `self._interrupt = False` at call start |
| 1103 | `retrieve_timesteps` → `scheduler.set_timesteps` (resets scheduler state) |
| 1109 | `latents = self.prepare_latents(..., generator, latents)` |
| 1194-1195 | `scheduler.set_begin_index(0)` **after** `prepare_latents` |
| 1197-1199 | `for i, t in enumerate(timesteps): if self.interrupt: continue` (no UNet call, no callback) |
| 1233 | `latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)` (ancestral/SDE samplers draw noise from `generator` here) |
| 1239-1248 | `callback_on_step_end(self, i, t, {"latents": latents})`, return value replaces `latents` |
| 1260-1293 | decode: `needs_upcasting = vae.dtype == fp16 and vae.config.force_upcast` → `upcast_vae()`; latents mean/std or `/ scaling_factor`; `vae.decode`; cast back |
| 1296-1300 | watermark (None here: invisible-watermark not installed), `image_processor.postprocess` |
| 1303 | `maybe_free_model_hooks()` |

The other three pipelines have the same shape:

| Pipeline | `_interrupt=False` | `prepare_latents` | `if self.interrupt` | callback | Notes |
|---|---|---|---|---|---|
| ControlNet XL (`controlnet/pipeline_controlnet_sd_xl.py`) | 1250 | def 885, call 1356 (same signature as txt2img) | 1452 | 1538 | **No `set_begin_index` call.** `controlnet_keep[i]` (1379-1385, 1484-1490) indexes by `i`, so skipping by `continue` keeps it correct |
| img2img XL (`…/pipeline_stable_diffusion_xl_img2img.py`) | 1221 | def 686: `(image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, add_noise=True)`; a 4-channel `image` is taken as latents (709); `add_noise = denoising_start is None` (1280) | 1385 | 1428 | `get_timesteps` (647) calls `set_begin_index` before `prepare_latents` (655/683) |
| inpaint XL (`…/pipeline_stable_diffusion_xl_inpaint.py`) | 1363 | **not read.** Returns a tuple `(latents[, noise][, image_latents])`, and the loop re-noises the unmasked region from `noise` every step | 1604 | not read | **Verify before relying on it** |

`DPMSolverMultistepScheduler` (`schedulers/scheduling_dpmsolver_multistep.py`)
keeps stepping state in `model_outputs` (list, length `solver_order`),
`lower_order_nums`, `_step_index`, `_begin_index`; `init_noise_sigma = 1.0`
(308); `set_timesteps` resets all of them (489-496); `_init_step_index` (1192)
only runs while `_step_index is None`. Other schedulers keep different fields,
which is why state capture should be generic.

### 7.2 The mechanism

**Capture** (inside `callback_on_step_end` at step `i`, when a pause is
requested): clone `callback_kwargs["latents"]` to CPU keeping its dtype; capture
the scheduler's stepping state; capture `generator.get_state()` (render already
uses a CPU generator, `render.py:222`). `next_step = i + 1`. Raise
`RenderPaused(state)`. Raising from the callback is how cancel already works
(`jobs.py:221`).

**Resume at step `k`** for one variation:
1. Wrap `pipe.prepare_latents` with an **instance attribute**: call the original
   (so the generator is consumed exactly as before and img2img/inpaint still get
   their `noise`/`image_latents`), then substitute the saved latents: element 0
   if the result is a tuple, the whole value otherwise.
2. In the same wrapper, set `pipe._interrupt = _SkipUntil(k, restore)`. This
   runs after line 1059's reset, so it survives. `if self.interrupt:` calls
   `__bool__`, which returns True for the first `k` evaluations. On evaluation
   `k` it calls `restore()` and returns False from then on.
3. `restore()` writes the saved scheduler state back and calls
   `generator.set_state(...)`. It runs **after** `set_begin_index(0)`, so the
   restored `_step_index` wins and `_init_step_index` never runs.
4. After the call, in `finally`: `del pipe.prepare_latents` and
   `pipe._interrupt = False`. The pipeline is cached and shared across jobs.
5. `k == total` (paused after the last step) needs no special case: every step
   is skipped and the latents go straight to decode.

```python
class _SkipUntil:
    """Truthy for the first `count` checks of `if self.interrupt:`, then restores state once."""
    def __init__(self, count, restore):
        self.remaining, self.restore = count, restore
    def __bool__(self):
        if self.remaining > 0:
            self.remaining -= 1
            return True
        if self.restore is not None:
            restore, self.restore = self.restore, None
            restore()
        return False
```

**Scheduler state, generically.** Walk `vars(scheduler)`, skip `config` and
`_internal_dict`, keep values that are None, bool, int, float, str, tensor, or
lists/tuples of those. Save tensors on CPU with their original device recorded;
restore onto that device. `sigmas` and `timesteps` are recomputed identically by
`set_timesteps`, so either skip them or compare them as a sanity check.

**Checkpoint files**, kept loadable with `torch.load(weights_only=True)` (the
torch 2.6 default):
- `<bundle>/checkpoint/state.json` holds metadata only, readable without torch
  by the API and UI: format version, job id, stage, variation, next step,
  seeds, completed variation indices, reason (`pause`/`abort`/`shutdown`),
  paused_at, the fingerprint, and the frozen `compiled` prompt dict.
- `<bundle>/checkpoint/state.pt` holds the tensors and primitives only:
  latents, scheduler state, generator state.
- Delete `checkpoint/` when the job completes. Size is ~0.5 MB at 1344×768
  (latents 108 KB fp16 plus solver history); it syncs to Nextcloud harmlessly.

**Always resume from the frozen compiled prompt** stored in the checkpoint,
never by recompiling: vocabulary edits between pause and resume must not change
the image. `CompiledPrompt` has `to_dict` but no `from_dict` (`compiler.py:131-149`);
add one.

**Fingerprint** (compared on resume; mismatch → refuse unless forced):
diffusers, transformers, torch, compel versions (`importlib.metadata`, no torch
import needed); model id plus checkpoint file size and mtime (or the manifest's
`installed_at`); controlnet id; dtype/precision; `torch.backends.cudnn.enabled`
after the probe; attention slicing; VAE decode mode (affects the final image);
sampler, steps, cfg, width, height, clip_skip; a hash of the control image and
the init/mask images; the GPU name. Offload mode does not change numerics:
leave it out. With force, list the differences in the bundle notes.

**Pause timing.** A pause is honoured at the next step boundary, up to ~25 s
away on this card, so print or show "pausing after the current step". Also check
the flag before each variation's pipeline call, so a pause requested during
model load or decode takes effect at the next variation boundary
(`next_step = 0`, no latents).

### 7.3 Render and bundle refactor

- Recommended now, even though task 4 needs it more: **call the pipeline with
  `output_type="latent"` and decode in a ClauDali helper** that reproduces
  `pipeline_stable_diffusion_xl.py:1260-1300` exactly (upcast semantics and
  tiling as configured today). Then "paused after the last step", refiner
  handoff and task 4's decode policy all slot in without another refactor. Keep
  output identical: the task-2 GPU test compares against a normal decode.
- `render()` gains a controller (`pause_requested`, `abort_requested`), an
  optional resume state, and an `on_variation(RenderedImage)` callback, and
  raises `RenderPaused`. Record `stage` in the checkpoint from the start
  (`"base"` only in task 2) so task 4 needs no format change.
- **`bundle.py` becomes incremental.** A `BundleWriter` creates the directory
  and `spec.json` up front, writes each variation (image, preview,
  diagnostics) the moment it lands, rewrites `result.json` each time with a
  `status` (`running`/`paused`/`done`/`aborted`/`error`) and `checkpoint` info,
  and rebuilds the contact sheet as images arrive. Replace `write_bundle`'s
  single end-of-job write; do not keep two paths.
- Progress and ETA: keep `(step, total, variation, variations)`. On resume, the
  ETA should count only remaining steps.

### 7.4 CLI (`claudali/__main__.py`)

- `render spec.json [--out DIR]`: install a SIGINT handler. The first press sets
  the pause flag, prints "Pausing after the current step… (Ctrl+C again to
  abort)", and restores `signal.default_int_handler`, so the second press raises
  `KeyboardInterrupt` in the main thread (where the pipeline runs) and aborts at
  the next Python bytecode.
- On pause: bundle status `paused`; print the variation/step and
  `Resume with: claudali resume "<dir>"`. Default chosen here: exit code **0**.
- On abort: bundle status `aborted`, finished images kept, `state.json` written
  without latents. Default chosen here: `resume` on an aborted bundle restarts the
  interrupted variation from step 0 and says so in a note. Exit code **130**.
- New `resume <bundle dir> [--force]`. Mismatch → print the differences, exit
  non-zero.
- Keep the SIGINT logic in a small function so a test can drive it without
  sending signals.

### 7.5 Queue, server and API

`claudali/jobs.py` today: `queue.Queue` of ids (102), statuses
queued/running/done/error/cancelled (33-38), cancel via the `JobCancelled` raise
in the progress callback (221), bundle written at the end (233), one daemon
worker, jobs in memory only.

- Replace `queue.Queue` with a `collections.deque` + `threading.Condition` so
  a resumed job can go to the **front**.
- Statuses: add `pausing` and `paused`. Queue flag `held`.
- `pause(job_id, hold_queue)`: running → pausing → paused when the checkpoint
  lands. Queued → paused immediately, with no checkpoint. With `hold_queue`, the
  worker takes no new job until the queue is released or that job resumes.
- `resume(job_id, force)`: paused → queued at the front; releases the hold if
  that job caused it. A mismatch found when the worker starts the job → status
  `error` with `error.type = "resume_mismatch"` and the list of differences; the
  checkpoint stays intact so a forced resume can follow.
- Resuming a paused bundle found on disk (e.g. left by the CLI) creates a job
  bound to that bundle.
- Default chosen here: **cancel a paused job** keeps its finished images,
  deletes the checkpoint, and says so in the response.
- **Server Ctrl+C = pause and hold.** In the FastAPI lifespan shutdown
  (`api.py:47-54`): request a hold-pause of the running job, wait for its
  checkpoint (up to one step plus the save), then persist to
  `runs/queue.json`: `{held: true, jobs: [{id, status, spec, bundle_dir,
  created_at}]}`. On startup, restore those jobs as paused/queued with the queue
  held. If a second Ctrl+C kills the process first, the bundle's `state.json`
  still allows a variation-level resume. **Unverified:** that uvicorn 0.52 on
  Windows runs lifespan shutdown on Ctrl+C and waits for it. Check it.
- `stats()` (159-168) should add `cancelled`, `paused`, `held`.

Endpoints to add, and document in `docs/api.md`:

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs/{id}/pause` body `{"hold_queue": false}` | Pause at the next step boundary |
| `POST /api/jobs/{id}/resume` body `{"force": false}` | Requeue at the front |
| `POST /api/resume` body `{"bundle": "<dir>", "force": false}` | Resume a paused bundle left on disk |
| `GET /api/paused` | Paused jobs and paused bundles on disk (reads `checkpoint/state.json`, no torch) |
| `POST /api/queue/release` | Un-hold the queue without resuming the paused job |

### 7.6 Web UI (`claudali/web/index.html`)

The status card has only Cancel (217, 513-517). `pollJob` (435-463) handles
running/queued/else. Add **Pause (next job runs)**, **Pause & hold queue**,
**Resume**, and a "Paused" card listing paused jobs and bundles with Resume
buttons. On a mismatch error, show the differences and offer "Resume anyway"
(force). Show "queue held" in the queue chip (628-635).

### 7.7 Tests (default suite)

- **Exact resume with a tiny random SDXL pipeline on CPU** (Appendix B sketch):
  for `dpmpp_2m_karras` (multistep history) and `euler_a` (draws noise from the
  generator each step), run uninterrupted, then pause at step 2 of 5 and resume;
  assert `torch.equal` on the latents. This test also locks the diffusers loop
  shape: if an upgrade moves `_interrupt` or `prepare_latents`, it fails.
- Checkpoint round trip (`weights_only=True` load); scheduler-state filter;
  fingerprint diff listing; force path.
- `JobQueue`: pause queued, resume goes to front, hold semantics,
  `queue.json` persistence round trip. No torch.
- CLI SIGINT handler: first press → pause flag, second → `KeyboardInterrupt`.
- `BundleWriter`: incremental writes and statuses, using PIL images. No torch.

### 7.8 Docs for task 2

README ("Three ways to use it": CLI `resume`, API table rows; "Output": the
`checkpoint/` folder and `status`); `docs/api.md` (statuses at 79-80, new
endpoints, bundle `status`/`checkpoint`); `docs/refine-loop.md` (pausing long
batches); `CLAUDE.md` (module map: `engine/checkpoint.py`; data flow; a gotcha
explaining `_SkipUntil` + the `prepare_latents` wrapper and that the tiny test
locks it; test count).

---

## 8. Task 3: cleanup

Each item: where it is, what to do, and why.

1. **UI drops spec fields** (Hugo: merge + anchor field). `buildSpec`
   (`index.html:287-333`) builds a spec from scratch out of form fields;
   `applySpec` (335-365) loads a spec into the form, losing everything the form
   lacks: `subject.anchor/count/secondary`, `scene.season/background/foreground`,
   `style.artists`, `camera.aperture`, `lighting.mood/accents`,
   `palette.colors`, `negative.*`, `overlays`, `post`, `init`, `raw`,
   `composition.regional`, `control.source/image/start/end`, `render.clip_skip`,
   `name`, `notes`. Examples (519-525), history (569-575) and "Use this JSON"
   (532-535) all go through `applySpec`, so loading `forest-fairies` in the UI
   and rendering **silently drops `subject.anchor`**, the field the chunking fix
   depends on. Fix: keep the loaded spec; `buildSpec` starts from a deep copy
   and overwrites only the form fields the user changed (track dirty inputs, or
   compare against the values `applySpec` wrote: fields with form defaults such
   as aspect `1:1`, control `none`, strength 0.6 and variations 1 must not clobber
   loaded values). "Use this JSON" renders that JSON as-is. Show an "also in
   this spec: …" list of preserved paths. Add a `subject.anchor` input in the
   Subject fieldset (103-110). Violates design rule 1 today.
2. **Warnings and notes merged again** (design rule 8). `render.py:138`
   concatenates `compiled.warnings + compiled.notes` into `notes`; the bundle has
   only `notes`; the CLI prints them all as `note:` (`__main__.py:104-105`); the
   UI merges them (`index.html:388`, 420-423). Carry separate `warnings` and
   `notes` through `RenderResult`, the bundle and `result.json`, the CLI and the
   UI. Classify engine messages: *requested behaviour did not happen* → warning
   (compel unavailable, a regional install failure); *decided on the caller's
   behalf* → note (cuDNN disabled, VAE upcast).
3. **`init.mask_layer`** (Hugo: wire it in). Add
   `mask_layer: Optional[str]` to `InitImage` (`spec.py:224-230`), mutually
   exclusive with `mask`; a `SceneSpec` validator requires the role to exist in
   `composition.layers`. `_select_task` (`render.py:116-121`) treats it as
   inpaint. `_load_init_images` (99-113) takes `build_region_masks(spec)[role]`
   (`control/maps.py:208-225`) and applies `mask_blur`. The maps docstring
   ("Handed to the inpainting path") becomes true. Note in the docs: bboxes are
   normalised, so the inpaint spec must carry the same layers and aspect as the
   original. Tests: selects inpaint; the mask equals the region mask; an
   unknown role is rejected. Docs: `docs/scene-spec.md` init table (244-252),
   `docs/refine-loop.md:139-140` (show the spec usage instead of the function
   name).
4. **Dead settings** in `config.py`: `keep_pipeline_warm`/`CLAUDALI_KEEP_WARM`
   (108-110, never read) and `extra` (117, never read). `default_model`/
   `CLAUDALI_DEFAULT_MODEL` (112) is only echoed by `/api/health`
   (`api.py:146`); the compiler hardcodes `"sdxl-base"` (`compiler.py:646`) and
   every intent names a model anyway. Remove all three and the health field.
   `field` may become an unused import.
5. **Dead diagnostic**: `diagnostics.py:176-177` checks `np.isfinite` on a
   `uint8` array, so it can never fire. Remove it; the black-frame flag already
   covers NaN renders.
6. **Unused** `registry.installed()` (220-221, and in `__all__`). Remove.
7. **Version drift**: `pyproject.toml` has `compel>=2.0.2`, but
   `requirements.txt` explains 2.4 is the floor (`CompelForSDXL`). Set `>=2.4`.
8. `pipelines.py:62-75`: `notes: list[str] = None  # type: ignore` plus
   `__post_init__` → `field(default_factory=list)`.
9. `render.py:149`: `spec.render.model or compiled.model`. `compiled.model`
   already applies that fallback (`compiler.py:646`); use it alone.
10. **Silent truncation without compel.** When compel failed to build
    (`pipelines.py:487-492`) the render sends plain prompts (`render.py:69-73`),
    and diffusers **truncates at 77 tokens**. The note mentions ignored weights,
    not the lost text. Add a warning with the token count whenever
    `compiled.tokens.chunks > 1` on that path (design rule 1).
11. `api.py:23`: `from contextlib import asynccontextmanager` sits after
    third-party imports. Move it into the stdlib group.
12. **`.gitignore`**: anchor `models/`, `outputs/`, `runs/` → `/models/`,
    `/outputs/`, `/runs/`, so directories with those names deeper down (every
    `site-packages/*/models`) stop vanishing from tools. Check `git status`
    before and after: nothing tracked should change.
13. **Docs that are wrong** (fix surgically):
    - Timings → measured (section 4.1): README "Performance, honestly"
      (263-276); `CLAUDE.md` hardware point 1 ("~2–4 minutes per 1024×1024
      image at 30 steps"); `docs/refine-loop.md:22-24` ("2–4 minutes") and
      160-162 ("Eight variations at four minutes is half an hour"). State the
      conditions: this laptop's GTX 1660 Ti, 32 steps, 1344×768, model offload,
      cuDNN off.
    - README `CLAUDALI_DTYPE` row (289) and Troubleshooting (334) present
      `float32` as a simple fix, but an fp32 SDXL UNet (~10 GB) cannot run under
      model offload on 6 GB. In task 3 correct the claim; task 4 makes fp32
      actually work by selecting sequential offload automatically.
    - `docs/api.md:233` health example shows torch 2.4.0 (installed: 2.6.0).
      Optional.
    - `CLAUDE.md` "61 tests, ~1.5 s" → the new count and time.
14. Leave alone: regional prompting (Hugo); `config.FONTS_DIR` (`assets/fonts`,
    absent but harmless); `SPEC_VERSION`; `device_report`'s name-based
    `needs_fp16_vae_fix`, which is reported only.

---

## 9. Task 4: quality options

Everything here is **optional per render**. `render.quality: "max"` turns it
all on; explicit fields win. Nothing here downloads anything.

### 9.1 Proposed spec shape (defaults chosen here)

```jsonc
"render": {
  "quality":    "standard",   // "standard" | "max"
  "precision":  "auto",       // "auto" (= CLAUDALI_DTYPE) | "fp16" | "fp32"
  "vae_decode": "auto"        // "auto" | "cpu" | "gpu" | "gpu_tiled"; default from CLAUDALI_VAE_DECODE
},
"refiner": { "enabled": false, "model": "sdxl-refiner", "handoff": 0.8,
             "aesthetic_score": 6.0, "negative_aesthetic_score": 2.5 },
"hires":   { "enabled": false, "scale": 1.5, "upscaler": "lanczos",   // or "realesrgan-x4"
             "strength": 0.3, "steps": null }                          // null = compiled steps
```

Resolve the preset in the compiler (it already uses `model_fields_set`,
`compiler.py:641-653`) and say what it enabled in `notes`. The preset changes no
steps, cfg or sampler. Under `max`: refiner not installed → warning plus how to
install; Real-ESRGAN not installed or spandrel missing → `lanczos` plus a note.

### 9.2 VAE decode policy (default: CPU fp32 untiled, auto-fallback)

- `auto`: estimate the CPU decode need at ~5.6 GB/megapixel plus a margin
  (section 4.2). If available physical RAM is at least that, decode on CPU in
  fp32 with the checkpoint's **own** VAE; otherwise GPU tiled (today's path)
  **with a note**. RAM probe, stdlib: Windows `GlobalMemoryStatusEx.ullAvailPhys`
  via ctypes; Linux `/proc/meminfo` `MemAvailable`; unknown → GPU tiled plus a
  note, because a missing measurement must not silently pick the risky path.
- `cpu`: forced, paging accepted. If allocation still fails
  (`RuntimeError: DefaultCPUAllocator: not enough memory` / `MemoryError`),
  fall back to **CPU fp32 tiled** with a warning: it keeps fp32 and the
  original VAE.
- `gpu`: untiled on the fix VAE; tiled on OOM with a note. `gpu_tiled`:
  today's behaviour.
- **`CLAUDALI_VAE_DECODE` replaces `CLAUDALI_VAE_TILING`** (`config.py:90`,
  `pipelines.py:369-376`). Keep `CLAUDALI_VAE_UPCAST` semantics for the GPU
  paths.
- The checkpoint's own fp32 VAE on CPU (~335 MB), cached with the pipeline:
  diffusers layout → `AutoencoderKL.from_pretrained(path, subfolder="vae",
  torch_dtype=torch.float32, variant=…)`; single file (Juggernaut) →
  `AutoencoderKL.from_single_file(path, config=<sdxl-base dir>, subfolder="vae",
  torch_dtype=torch.float32)`. **That signature is unverified.** Weights on
  disk are fp16 variants, so this is an fp32 *computation* on upcast weights.
  That is still the reference path.
- Decode steps: mirror `pipeline_stable_diffusion_xl.py:1274-1300` (latents
  mean/std if present, else `/ scaling_factor`; `vae.decode`;
  `image_processor.postprocess`). Under model offload, check that calling
  `pipe.vae.decode` directly still triggers accelerate's hook (the GPU path).
- Pure decision function `plan_vae_decode(mode, megapixels, available_gb)`, unit
  tested without torch.
- **Report to Hugo what `auto` does on his laptop** (section 4.2 caveat).

### 9.3 fp32 precision

- `precision: "fp32"` (or `CLAUDALI_DTYPE=float32`): skip the fp16-fix VAE (the
  gate at `pipelines.py:326` already does this), and select **sequential
  offload automatically** when total VRAM is below ~12 GB, with a note.
  Label it **unmeasured**: ~13–14 GB of weights in RAM on a 15.4 GB machine.
- Precision becomes per job, so `load_pipeline`'s cache key (507-518, today
  model + controlnet) must include dtype and offload mode. Pass dtype/offload
  as parameters instead of reading `SETTINGS.torch_dtype` (528),
  `SETTINGS.dtype` (326) and `SETTINGS.offload` (358-365) inside. Switching
  precision reloads the pipeline; say so in a note.
- The cuDNN probe only tests fp16 shapes. The process-wide cuDNN state still
  applies to fp32 convolutions; nothing to change, just do not assume the probe
  covers fp32.

### 9.4 Refiner (official handoff, stages batched)

- Catalogue: `sdxl-refiner`, repo `stabilityai/stable-diffusion-xl-refiner-1.0`,
  layout `diffusers`, licence CreativeML Open RAIL++-M. Measure `approx_gb` with
  `resolve_remote_files` (metadata only; expect ~6 GB with fp16 variants). Use a
  **new `kind` such as `"refiner"`** (the `ModelEntry.kind` Literal at
  `registry.py:66`) so the UI's checkpoint dropdown (`index.html:617` filters
  `kind === "checkpoint"`) does not offer it as a base model.
- **`PROFILES["full"] = list(CATALOG)` (`registry.py:203`)** would silently grow
  `full` with every new entry. Make `full` an explicit list.
- Flow per job: base stage for **all** variations with
  `denoising_end=handoff`, `output_type="latent"`, latents saved into
  `checkpoint/` as each finishes; `pipelines.release()`; load
  `StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_dir, …)` (its
  config has `requires_aesthetics_score=True` and no `text_encoder`; pass the
  fix VAE when fp16); for each variation call it with `image=<base latents>`
  (4-channel = latents, no encode, no added noise with `denoising_start` set),
  `denoising_start=handoff`, the same `num_inference_steps`, cfg, aesthetic
  scores, a generator seeded from the variation seed, `output_type="latent"`;
  then decode.
- **compel for the refiner:** `CompelForSDXL` expects two encoders. Build a
  single-provider `Compel(tokenizer=pipe.tokenizer_2,
  text_encoder=pipe.text_encoder_2,
  returned_embeddings_type=PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  requires_pooled=True, truncate_long_prompts=False)` with the same
  move-encoders-to-the-execution-device trick as `_build_compel`
  (`pipelines.py:448-492`). Verify the API against compel 2.4.
- Pause/resume: the same mechanism works in the img2img pipeline; the
  checkpoint's `stage` is `"refiner"`.
- Restrictions (raise or warn, never drop silently): refiner + `init` → raise
  "not supported yet"; refiner + ControlNet → note that the refiner stage runs
  uncontrolled; refiner with a fine-tune (juggernaut-xl, dreamshaper-xl) →
  warning that the refiner was trained on SDXL base outputs and may wash out
  the fine-tune's look.
- Cannot be run here (not downloaded): label it **untested on hardware** in
  CLAUDE.md "Deliberate non-features", as regional is.

### 9.5 Hi-res pass (lanczos or Real-ESRGAN)

- Per variation, after its final decode: upscale → target = `(w·scale,
  h·scale)` rounded to multiples of 8 → encode (the same CPU/GPU policy as
  decode; pass *scaled* latents as a 4-channel `image` so the pipeline skips its
  own encode) → base img2img (`derive_pipeline(..., "img2img")`) at `strength`,
  compel embeds, variation seed, `output_type="latent"` → decode. Batched after
  the refiner stage, which means reloading the base when the refiner ran.
- **Memory at hi-res:** attention slicing `"auto"` slices to half the heads. At
  2016×1152 the first transformer level has 9072 tokens, ~165 MB per head in
  fp16, next to a 5 GB UNet on a 6 GB card, so OOM is likely. Setting
  `pipe.set_attention_slice(1)` for this stage is numerically equivalent; restore
  it afterwards. Label full-size hi-res **unmeasured**.
- Regional prompting: remove its handle before the hi-res stage, since its
  masks are sized for the base resolution (`regional.grid_for` would decline
  them anyway), and note it.
- Real-ESRGAN: catalogue entry with a new `kind` (e.g. `"upscaler"`), layout
  `single_file` (`resolve_remote_files` for `single_file` bypasses
  `SKIP_EXTENSIONS`, which lists `.pth`). **Repo and file are unverified.**
  Candidates: `ai-forever/Real-ESRGAN` (`RealESRGAN_x4.pth`),
  `lllyasviel/Annotators` (`RealESRGAN_x4plus.pth`). Prefer a `.safetensors`
  copy if one exists (`.pth` is a pickle). Real-ESRGAN is BSD-3-Clause; confirm
  per repo. Load with `spandrel`
  (`ModelLoader().load_from_file(path)`, `.model.eval()`, `.scale`), run in
  512 px tiles with overlap on GPU, then Lanczos down to the target size.
  `spandrel` → `requirements.txt` and `pyproject.toml`, lazy import.
- Postprocess and overlays still run last, in the bundle's `finish`. Overlays
  use normalised coordinates, so they scale.

### 9.6 Installer: an option to download, never auto-download

- `python -m installer install --with-quality-models` and
  `install.ps1 -QualityModels` add `sdxl-refiner` and the Real-ESRGAN entry on
  top of any profile. `python -m installer models --add sdxl-refiner …` already
  works for any catalogue entry.
- Include them in the preflight disk estimate when the flag is set
  (`installer/core.py:117`). Update the help-text sizes (`installer/__main__.py:88`,
  `install.ps1` `.PARAMETER ModelProfile`) and the README profile table
  (92-96).

### 9.7 UI

A "Quality" fieldset: preset select; precision; VAE decode; refiner (checkbox,
handoff); hi-res (checkbox, scale, upscaler, strength). It must follow task 3's
merge semantics.

### 9.8 Docs for task 4

README: Install (flag), Configuration table (`CLAUDALI_VAE_DECODE`, the
`CLAUDALI_DTYPE` row), a short "Quality options" section with honest costs
(measured decode numbers; fp32, refiner and full-size hi-res unmeasured),
Performance. `docs/scene-spec.md`: `render.quality/precision/vae_decode`,
`refiner`, `hires`. `docs/api.md`: bundle stage info. `CLAUDE.md`: module map
(new modules, e.g. `engine/decode.py`, `engine/upscale.py`, `sysinfo.py`: keep
torch imports inside functions), hardware section (the im2col memory fact and
the decode measurements), gotchas, deliberate non-features (refiner, hi-res at
full size and Real-ESRGAN untested on hardware; fp32 unmeasured).

---

## 10. GPU test plan (≤ ~13 min remaining)

Use `sdxl-base` (diffusers layout, faster load) unless the check needs
Juggernaut. The float32 warning appeared during Juggernaut's single-file load,
so test 1 should load Juggernaut once.

**Test 1, after task 2 (~6 min), one script, one process:**
1. Load juggernaut-xl through `load_pipeline`, capturing stderr: none of the
   three warnings appears (task 1).
2. Small render, e.g. `width=height=768`, 4 steps, fixed seed, `dpmpp_2m_karras`:
   uninterrupted latents vs pause at step 2 + resume → `torch.equal`. That checks
   GPU determinism with cuDNN off. If they are not equal, stop and report before
   claiming exactness.
3. The ClauDali decode helper vs the pipeline's own decode of the same latents:
   identical pixels.
4. Record available RAM right after the pipeline load (input to task 4).

**Test 2, after task 4 (~5 min):**
1. The `auto` decode inside a real 2-step 768×768 render: which path it took and
   why (report to Hugo).
2. A forced `cpu` decode in the same process: completes, no NaN.
3. Hi-res smoke test at tiny size (512×512 base, 2 steps → 1.5×, 2 steps,
   `lanczos`) to exercise the code path end to end.
4. **Not testable here:** refiner and Real-ESRGAN (not downloaded), fp32 (RAM
   and time). Say so in the report and the docs.

---

## 11. Code map for this work (at `de25160`)

Enough detail to avoid re-reading whole files. Verify against the file before
editing.

- **`engine/render.py`**: `ProgressCallback(step, total, variation, variations)`
  (26); `_seeds_for`, consecutive seeds from a fixed seed (53-63);
  `_encode_prompts`, compel call (78) with fallback to plain prompts (66-96);
  `_load_init_images` (99-113); `_select_task` (116-121); `render()` (124-255):
  ControlNet + init raises (143-147), `load_pipeline` (149), `derive_pipeline`
  (152), regional install (160) and removal in `finally` (243-245), kwargs
  (172-208), `effective_steps` for img2img/inpaint = `int(steps*strength)`
  (216-218), per-variation loop with a CPU generator (222), callback (227-235),
  `pipe(...)` **with no `output_type`, so the pipeline decodes** (237-241).
- **`engine/pipelines.py`**: `SAMPLERS` (40-56); single-slot `_CACHE` + `_LOCK`
  (58-59); `device_report` runs the cuDNN probe (77-106); probe shapes (114-118);
  `_plan_cudnn`, pure and tested (172-223); `apply_cudnn_workaround`,
  `lru_cache` (226-243); `_plan_vae_precision` (259-287); `_load_vae`, fp16-fix
  only when fp16 (323-346); `_apply_memory_strategy`: offload, **attention
  slicing on (367-368)**, VAE tiling+slicing (369-376); `_apply_vae_precision`
  sets `force_upcast` via `register_to_config` (403-435); `_build_compel`, the
  encoders-to-device trick (448-492); `load_pipeline`, cache key model +
  controlnet, re-applies the sampler on reuse (495-597); a single file loads
  with `config=<sdxl-base dir>` (552-561); ControlNet via `from_pipe` (567-580);
  `derive_pipeline` via AutoPipeline `from_pipe` (600-615); `release()`
  (636-639).
- **`jobs.py`**: see 7.5. `Job.progress` (68-75); `to_dict` (77-95); `QUEUE`
  built from `SETTINGS.max_queue/job_retention` (239).
- **`bundle.py`**: `VariationRecord` (35-43), `Bundle` (46-66), previews
  (73-89), contact sheet (92-128), `bundle_directory` =
  `outputs/<stamp>_<slug>_<jobid8>` (131-134), `write_bundle`, all at the end
  (137-195); `list_bundles` sorted by name, newest first (206-220).
- **`__main__.py`**: `cmd_compile` prints warnings and notes separately
  (43-70); `cmd_render`, no interrupt handling, prints everything as `note:`
  (73-106); `cmd_doctor` (109-142); parser (163-190).
- **`api.py`**: lifespan (47-54); error shape `{"error": {type, message,
  detail}}` (73-77); both validation handlers (102-113, and **both are needed**,
  per CLAUDE.md); health (126-148); compile/control-preview/render/jobs/history
  (214-277); `_safe_path` allows `outputs/` and `runs/` only (284-303); upload
  to `runs/uploads` (317-334); examples (353-364).
- **`spec.py`**: all models `extra="forbid"` (40-47). `Render` (233-251):
  model/width/height/steps 30/cfg 6.5/sampler/seed/variations 1–16/clip_skip.
  `InitImage` (224-230). `SceneSpec` fills width/height from the aspect bucket
  (351-358).
- **`compiler.py`**: `CompiledPrompt` with separate `warnings`/`notes`
  (102-149, `to_dict` only); `compile_spec` (398-664); explicit-vs-intent via
  `model_fields_set` (641-653).
- **`control/maps.py`**: `build_region_masks` (208-225), used only by tests
  today; `control_repo_for_mode` (259-265).
- **`engine/regional.py`**: `_encode` compel call (134); `install` (264-323);
  `RegionalHandle.remove` (78-91).
- **`diagnostics.py`**: `failure_flags` (153-190).
- **`config.py`**: `Settings` (78-125): see task 3 item 4 for the dead fields;
  `torch_dtype` maps float16/bfloat16/float32 (119-125); HF cache redirect at
  import (36-50).
- **`registry.py`**: `SKIP_EXTENSIONS` (34-50), `WEIGHT_PREFIXES` (58),
  `ModelEntry` (61-93), `CATALOG` (100-192), `PROFILES` (195-204),
  `resolve_checkpoint` (236-257), `resolve_remote_files` (326-348).
- **`installer/`**: `core.install` stages (340-402), `download_models`
  (290-303), `add_models` (405-413), preflight disk check (117-124),
  `download.download_file` resumable with retries; `__main__` subcommands
  install/models/uninstall/status/verify. `install.ps1` forwards
  `-ModelProfile/-RecreateVenv/-Cpu/-SkipTorch/-SkipModels`.
- **`web/index.html`**: one file, vanilla JS. Helpers `$`, `fileURL`, `api()`
  (245-283); `buildSpec`/`applySpec` (287-365); `showCompiled` (383-392);
  `showBundle` (394-424); `pollJob` every 1.2 s (435-463); actions (467-543);
  history (555-578); boot loads vocabulary, health, models, examples (582-636).
- **`tests/test_claudali.py`**: 61 tests. Helpers `minimal()` (33-36),
  `layered_spec()` (169-181), `fairy_spec()` (416-447), `_cudnn_plan()`
  (350-355). All examples must compile without warnings (156-161).

---

## Appendix A: `claudali/engine/quiet.py` draft (task 1)

Written and then removed in the planning session. Not yet run.

```python
"""Three library warnings that are wrong in ClauDali's context, and nothing else.

Each was traced to the line that emits it and is silenced by matching its text
on the one logger that logs it, so every other message from those libraries
still prints. Lowering a logger's level instead would have hidden the next real
warning along with these.

1. ``diffusers.models.modeling_utils``: *"There are modules in
   UNet2DConditionModel that should be kept in float32: []"*. diffusers tests
   ``fp32_modules is not None`` on a value that is never None, so it fires on
   every dtype cast, and the empty list is the proof there was nothing to keep.
   Dropped only when the list is empty; a real list still prints.
2. ``transformers.utils.import_utils``: *"`Siglip2ImageProcessorFast` is
   deprecated"*. diffusers 0.40 imports every pipeline it ships, and its
   Z-Image pipeline still uses a class name transformers 5 renamed. No ClauDali
   code path touches it.
3. ``transformers.tokenization_utils_base``: *"Token indices sequence length is
   longer than the specified maximum sequence length (131 > 77)"*. compel
   tokenizes a whole prompt in order to split it into 77-token windows itself --
   ``CompelForSDXL`` builds both of its providers with
   ``truncate_long_prompts=False`` -- so nothing over 77 tokens ever reaches a
   text encoder and the promised indexing error cannot happen. Dropped only
   while compel is encoding, because the same message from anywhere else would
   be true.

Stdlib only: installing the filters must not import either library.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

_FLOAT32_LOGGER = "diffusers.models.modeling_utils"
_SIGLIP_LOGGER = "transformers.utils.import_utils"
_TOKENS_LOGGER = "transformers.tokenization_utils_base"


class _Drop(logging.Filter):
    """Drop a record whose message contains every one of ``fragments``."""

    def __init__(self, *fragments: str) -> None:
        super().__init__()
        self.fragments = fragments

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not all(fragment in message for fragment in self.fragments)


_EMPTY_FLOAT32 = _Drop("that should be kept in float32: []")
_SIGLIP = _Drop("`Siglip2ImageProcessorFast` is deprecated")
_LONG_SEQUENCE = _Drop("Token indices sequence length is longer than the specified maximum")


def install() -> None:
    """Attach the two process-wide filters. Idempotent."""
    for name, rule in ((_FLOAT32_LOGGER, _EMPTY_FLOAT32), (_SIGLIP_LOGGER, _SIGLIP)):
        logger = logging.getLogger(name)
        if rule not in logger.filters:
            logger.addFilter(rule)


@contextlib.contextmanager
def compel_tokenization() -> Iterator[None]:
    """Silence the over-length tokenizer warning while compel encodes a prompt."""
    logger = logging.getLogger(_TOKENS_LOGGER)
    if _LONG_SEQUENCE in logger.filters:
        # Already inside a compel call; the outer one removes the filter.
        yield
        return
    logger.addFilter(_LONG_SEQUENCE)
    try:
        yield
    finally:
        logger.removeFilter(_LONG_SEQUENCE)


__all__ = ["compel_tokenization", "install"]
```

## Appendix B: tiny SDXL pipeline for the exact-resume test (sketch, not run)

Pattern taken from diffusers' own SDXL tests. Prompt embeddings are passed
directly, so no tokenizer, text encoder, weights or network is needed.
`projection_class_embeddings_input_dim` must equal
`addition_time_embed_dim * 6 + pooled_dim`, which the pipeline checks in
`_get_add_time_ids`.

```python
import pytest

torch = pytest.importorskip("torch")
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline,
                       UNet2DConditionModel)


def tiny_sdxl(scheduler):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=2, sample_size=16,
        in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4), use_linear_projection=True,
        addition_embed_type="text_time", addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,  # 6 * 8 + pooled dim 32
        cross_attention_dim=64, norm_num_groups=1,
    )
    vae = AutoencoderKL(
        block_out_channels=[32, 64], in_channels=3, out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 2,
        up_block_types=["UpDecoderBlock2D"] * 2, latent_channels=4,
    )
    pipe = StableDiffusionXLPipeline(
        vae=vae, text_encoder=None, text_encoder_2=None, tokenizer=None,
        tokenizer_2=None, unet=unet, scheduler=scheduler, add_watermarker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def call_kwargs():
    g = torch.Generator().manual_seed(123)
    return dict(
        prompt_embeds=torch.randn(1, 8, 64, generator=g),
        pooled_prompt_embeds=torch.randn(1, 32, generator=g),
        negative_prompt_embeds=torch.randn(1, 8, 64, generator=g),
        negative_pooled_prompt_embeds=torch.randn(1, 32, generator=g),
        num_inference_steps=5, guidance_scale=5.0, height=32, width=32,
        output_type="latent",
    )
# Test: run once uninterrupted through ClauDali's per-variation denoise helper;
# run again with a controller that pauses at step 2 (capturing RenderPaused.state);
# resume from that state; assert torch.equal on the final latents.
# Parametrize over DPMSolverMultistepScheduler(algorithm_type="dpmsolver++",
# use_karras_sigmas=True) and EulerAncestralDiscreteScheduler().
```

Exercise the production helper that `render()` uses, not a copy of the
mechanism, or the test locks nothing.

## Appendix C: the decode benchmark (condensed)

```python
# run with .venv\Scripts\python from the project root
import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from claudali.config import WEIGHTS_DIR   # redirects HF caches first
from diffusers import AutoencoderKL

latent = torch.randn(1, 4, 96, 168, generator=torch.Generator().manual_seed(0)) / 0.13025

def vae(kind, dtype):
    if kind == "base":
        return AutoencoderKL.from_pretrained(str(WEIGHTS_DIR / "sdxl-base"), subfolder="vae",
                                             torch_dtype=dtype, variant="fp16", local_files_only=True)
    return AutoencoderKL.from_pretrained(str(WEIGHTS_DIR / "sdxl-vae-fp16-fix"),
                                         torch_dtype=dtype, local_files_only=True)

def run(kind, device, dtype, tiled):
    model = vae(kind, dtype).to(device)
    if tiled:
        model.enable_tiling()
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        model.decode(latent.to(device, dtype), return_dict=False)
    torch.cuda.synchronize() if device == "cuda" else None
    print(kind, device, dtype, tiled, round(time.perf_counter() - start, 1), "s",
          torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else "")

run("base", "cpu", torch.float32, False)
torch.backends.cudnn.enabled = False          # as ClauDali does on this card
run("fix", "cuda", torch.float16, True)
run("fix", "cuda", torch.float16, False)
run("base", "cuda", torch.float32, False)     # OOMs on 6 GB: im2col wants 8.86 GiB
```

Peak process RAM was read with `kernel32.K32GetProcessMemoryInfo`
(`PeakWorkingSetSize`); set `argtypes` or it raises `OverflowError`.
