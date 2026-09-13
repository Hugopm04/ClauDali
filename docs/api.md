# HTTP API reference

Base URL: `http://127.0.0.1:8188`. Interactive documentation is served at
`/docs`, and the OpenAPI document at `/openapi.json`.

The API is **asynchronous by necessity**. A full-step SDXL render takes minutes,
which exceeds the default timeout of essentially every HTTP client, so renders
are submitted to a queue and polled. There is exactly one worker: two concurrent
renders on one GPU finish later than either would alone.

There is **no authentication**. The server binds to localhost by default. Only
expose it to a network you trust.

---

## Error format

Every error ClauDali raises itself has the same shape:

```json
{
  "error": {
    "type": "invalid_spec",
    "message": "the scene spec failed validation",
    "detail": [
      { "field": "composition.aspect", "message": "aspect must be one of [...]", "type": "value_error" }
    ]
  }
}
```

`field` paths match the spec you wrote — `composition.aspect`, not
`body.composition.aspect`. `type` is stable and safe to branch on:
`invalid_spec`, `not_found`.

---

## Rendering

### `POST /api/render`

Queue a render. Body is a [scene spec](scene-spec.md). Returns immediately.

```bash
curl -X POST http://127.0.0.1:8188/api/render \
     -H "Content-Type: application/json" \
     -d '{"intent":"photoreal","subject":{"primary":"a stoneware coffee cup"}}'
```

```json
{
  "id": "9f2c1a...",
  "status": "queued",
  "progress": 0.0,
  "total_steps": 32,
  "total_variations": 1,
  "name": "a stoneware coffee cup",
  "created_at": "2026-09-08T14:33:55"
}
```

Returns 503 when the queue already holds `CLAUDALI_MAX_QUEUE` waiting jobs.

### `GET /api/jobs/{id}`

Poll a job. Polling once every 1–2 seconds is plenty.

```json
{
  "id": "9f2c1a...",
  "status": "running",
  "progress": 0.42,
  "step": 13,
  "total_steps": 32,
  "variation": 0,
  "total_variations": 4,
  "eta_s": 218.4
}
```

`status` is `queued`, `running`, `pausing`, `paused`, `done`, `error` or
`cancelled`. Once the job has started, the response also contains `bundle` —
see below — which grows as each image lands, so finished variations can be
looked at before the last one is done. `bundle_dir` is its directory,
`checkpoint` says where a paused job will carry on from, and `resumable` says
whether it can be resumed.

### `DELETE /api/jobs/{id}`

Cancel. A queued or paused job ends immediately; a running one stops at its next
sampler step. A cancelled job keeps the images it finished; its checkpoint is
deleted, so it cannot be resumed. Returns 409 if the job already finished.

### `GET /api/jobs?limit=50`

Recent jobs without their bundles, plus queue statistics.

---

## Pausing and resuming

A running job can be paused at any sampler step and resumed later, **bit for
bit**: the latents, the sampler's history and the random stream are saved to the
bundle's `checkpoint/` folder, so the resumed image is identical to an
uninterrupted one, even after a restart. A pause is honoured when the current
step ends, which on a 6 GB card can take ~25 s; until then the job is `pausing`.

### `POST /api/jobs/{id}/pause`

Body `{"hold_queue": false}` (optional). With `hold_queue: false` the next queued
job starts as soon as this one has paused. With `true` the whole queue is held:
nothing starts until this job resumes or the queue is released. A queued job
can be paused too, and simply waits. Returns the job; 409 if it has finished.

### `POST /api/jobs/{id}/resume`

Body `{"force": false}` (optional). Puts a paused job back **at the front** of the
queue, and releases the hold if this job caused it. When the worker picks it up
it first checks that nothing which shapes the pixels has changed since the
pause: the model's files, the torch, diffusers, transformers and compel
versions, precision, cuDNN state, attention slicing, VAE settings, sampler,
size, and the control and init images. If something has, the job ends in
`error` with `"error_type": "resume_mismatch"` and a `mismatch` list naming each
difference; its checkpoint is untouched. Resuming again with `"force": true`
carries on anyway and records the differences in the bundle notes.

### `POST /api/resume`

Body `{"bundle": "<bundle directory>", "force": false}`. Resumes a bundle found on
disk that no job in this server holds: one paused from the command line, or left
behind when a process died mid-render. The directory must be under `outputs/`.
Returns the new job.

### `GET /api/paused`

```json
{
  "jobs": [ { "id": "9f2c1a...", "status": "paused", "checkpoint": { "...": "see below" } } ],
  "bundles": [
    {
      "directory": "C:\\...\\outputs\\2026-09-13_101500_forest-fairies_4be1c2d0",
      "job_id": "4be1c2d0...", "name": "forest fairies", "status": "paused",
      "images": 1, "preview": "...\\001_seed1848.preview.jpg",
      "checkpoint": { "...": "see below" }
    }
  ],
  "held": false
}
```

A bundle whose `status` is still `running` was interrupted: nothing is running it.

### `POST /api/queue/release`

Lets queued jobs start again without resuming the job that held the queue.

### Stopping the server

Ctrl+C on the server pauses the running job at its next step and holds the queue.
The paused job and every job not yet started are written to `runs/queue.json`,
and the next start restores them, with the queue held. Ctrl+C a second time
stops waiting; that job then resumes from the start of the variation it was on.

---

## The bundle

A finished job carries a `bundle`, which is also written to disk as
`result.json`:

```json
{
  "job_id": "9f2c1a...",
  "directory": "C:\\...\\outputs\\2026-09-08_143355_coffee-cup_9f2c1a2b",
  "spec": { "...": "exactly what you submitted, with defaults resolved" },
  "compiled": {
    "prompt": "(professional photograph, ...)1.10, (a stoneware coffee cup)1.10, ...",
    "negative_prompt": "jpeg artifacts, ...",
    "model": "juggernaut-xl",
    "steps": 32, "cfg": 5.5, "sampler": "dpmpp_2m_karras",
    "width": 896, "height": 1152,
    "fragments": [ { "source": "medium", "key": "photograph", "text": "...", "weight": 1.1 } ],
    "tokens": { "tokens": 239, "chunks": 4, "exact": true, "source": "clip-tokenizer" },
    "negative_tokens": { "tokens": 112, "chunks": 2, "exact": true, "source": "clip-tokenizer" },
    "anchors": ["a stoneware coffee cup", "a stoneware coffee cup", "a stoneware coffee cup"],
    "regions": [],
    "warnings": [],
    "notes": []
  },
  "variations": [
    {
      "index": 0,
      "seed": 1848,
      "image": "...\\001_seed1848.png",
      "preview": "...\\001_seed1848.preview.jpg",
      "diagnostics": { "...": "see below" }
    }
  ],
  "contact_sheet": "...\\contact-sheet.jpg",
  "control_image": null,
  "notes": [],
  "duration_s": 184.2,
  "task": "txt2img",
  "device": { "gpu": "NVIDIA GeForce GTX 1660 Ti", "vram_gb": 6.0, "...": "..." },
  "status": "done",
  "checkpoint": null,
  "error": null
}
```

`compiled.fragments` is the audit trail: every phrase in the prompt with the spec
field it came from and the weight applied.

`status` is `running`, `paused`, `done`, `aborted`, `cancelled` or `error`, and
`result.json` is rewritten with it each time an image lands. `aborted` means the
job was stopped hard, by a second Ctrl+C on the command line: it can still be
resumed, from the start of the variation it was on. `cancelled` cannot.
`duration_s` counts rendering time across every session of a resumed job.
Bundles written before pausing existed have no `status` and are complete.

While a job can be resumed, `checkpoint` summarises where it will carry on:

```json
{
  "stage": "base", "variation": 1, "next_step": 13, "steps": 32,
  "variations": 4, "completed": [0], "mid_image": true,
  "reason": "pause", "paused_at": "2026-09-13T10:22:41"
}
```

`variation` is zero-based. `mid_image` is false at a variation boundary, where
the variation restarts from step 0 with its original seed, which is still exact.
`reason` is `pause`, `shutdown`, `abort`, or `running` for a process that died.

### Diagnostics

Per variation:

```json
{
  "size": { "width": 896, "height": 1152 },
  "exposure": {
    "mean": 118.4, "std": 52.1, "p01": 8.0, "median": 121.0, "p99": 243.0,
    "dynamic_range": 235.0, "shadow_clip_pct": 0.31, "highlight_clip_pct": 1.02
  },
  "detail": { "laplacian_variance": 412.7, "edge_density_pct": 6.14 },
  "color": { "colorfulness": 38.2, "mean_saturation": 74.9 },
  "palette": [ { "hex": "#c9b79a", "rgb": [201,183,154], "coverage_pct": 34.2 } ],
  "composition": {
    "center_x": 0.48, "center_y": 0.54, "off_center": 0.045,
    "quadrants": { "top_left": 18.2, "top_right": 21.0, "bottom_left": 30.4, "bottom_right": 30.4 }
  },
  "flags": []
}
```

`flags` names specific, known failure modes rather than judging taste — a solid
black frame (the fp16 VAE failure), a diverged sampler, a flat dynamic range,
heavy clipping. An empty list means nothing recognisable went wrong, not that the
image is good.

`composition` weights by *local contrast* rather than brightness, because
brightness alone would call an empty sky the subject.

---

## Working with a spec before rendering

### `POST /api/compile`

Compile to prompts without rendering. Costs milliseconds. This is the cheapest
feedback loop available and should be the first thing you reach for when a
result surprises you.

```json
{
  "prompt": "...", "negative_prompt": "...",
  "model": "dreamshaper-xl", "steps": 34, "cfg": 7.5,
  "width": 1216, "height": 832,
  "fragments": [ "..." ],
  "tokens": { "tokens": 228, "chunks": 4, "exact": true, "source": "clip-tokenizer" },
  "anchors": ["a lighthouse of stacked clocks"],
  "regions": [],
  "warnings": ["camera: lens specified with non-optical medium 'woodcut'"],
  "notes": []
}
```

`warnings` and `notes` mean different things and are worth reading differently.
A **warning** says the spec probably wants changing: a field that cannot take
effect, a shot that contradicts the subject, a prompt long enough that its
detail is spread thin. A **note** says the compiler decided something on your
behalf and is telling you so, such as leaving out an intent's implicit `anatomy`
negatives because the subject names no person. Both are shown in the web UI.

`tokens` is the measurement that matters most when a subject fails to appear.
`chunks` above one means CLIP read the prompt in several passes, and `anchors`
lists the subject restated at the head of each of them — see
[scene-spec.md](scene-spec.md) for why that is the difference between a forest
with fairies in it and an empty forest. `exact` is `false`, and `source` is
`estimate`, only when no model has been installed yet and there is no tokenizer
to count with.

### `POST /api/control-preview`

Returns the procedural ControlNet map as a PNG (`image/png`). Lets you check a
layer stack before spending GPU time on it. 400 if `control.mode` is `"none"`.

---

## Discovery

Designed so a caller can learn the format without reading documentation.

| Endpoint | Returns |
|---|---|
| `GET /api/schema` | Full JSON Schema for a scene spec |
| `GET /api/vocabulary` | Every key per table, what each expands to, intents, samplers, aspect buckets, available fonts |
| `GET /api/models` | Catalogue with installed state, disk usage, licences, profiles |
| `GET /api/examples` | The bundled example specs |
| `GET /api/health` | Device report, queue stats, active settings |
| `GET /api/history?limit=30` | Previously written bundles, newest first |

`GET /api/health` is the fastest way to explain a slow render:

```json
{
  "status": "ok",
  "device": {
    "torch": "2.4.0+cu124", "cuda_available": true,
    "gpu": "NVIDIA GeForce GTX 1660 Ti", "vram_gb": 6.0,
    "compute_capability": "7.5", "needs_fp16_vae_fix": true,
    "cudnn": 90100, "fp16_conv_broken": true, "cudnn_disabled": true,
    "fp16_conv_broken_without_cudnn": false, "vae_upcast": "auto",
    "offload": "model", "dtype": "float16"
  },
  "queue": {
    "queued": 0, "running": 1, "pausing": 0, "paused": 0, "done": 12, "error": 0,
    "cancelled": 0, "held": false, "worker_alive": true
  }
}
```

`fp16_conv_broken` is measured on the card, not inferred from its name: it is
true when a half-precision convolution returns NaNs, which turns every image
solid black. `cudnn_disabled` says ClauDali turned cuDNN off for the process to
work around it, which is the normal outcome and costs no measurable speed on a
card without tensor cores. `fp16_conv_broken_without_cudnn` is the re-measure
that confirms it worked; when that is true as well, the VAE also decodes in fp32
if `vae_upcast` is `auto` or `always`.

---

## Files

### `GET /api/file?path=<absolute path>`

Serves an image from a bundle. Paths outside `outputs/` and `runs/` return 403 —
bundle manifests contain absolute paths, so this endpoint must exist, and must
therefore be constrained.

### `POST /api/upload`

Multipart upload of an init image or mask. Returns the path to put in
`init.image` or `init.mask`.

```json
{ "path": "C:\\...\\runs\\uploads\\3f8a....png", "width": 1216, "height": 832 }
```

---

## A complete session

```python
import time, requests

BASE = "http://127.0.0.1:8188"

spec = {
    "intent": "painterly",
    "subject": {"primary": "a lighthouse built from stacked antique clocks"},
    "scene": {"setting": "a black basalt cliff", "time": "blue hour"},
    "style": {"medium": "oil_impasto", "movement": "surrealism"},
    "composition": {"aspect": "3:2", "rule": "thirds"},
    "render": {"variations": 4, "seed": 1848},
}

# Check the prompt before spending minutes on it.
print(requests.post(f"{BASE}/api/compile", json=spec).json()["prompt"])

job = requests.post(f"{BASE}/api/render", json=spec).json()
while True:
    status = requests.get(f"{BASE}/api/jobs/{job['id']}").json()
    if status["status"] in {"done", "error", "cancelled", "paused"}:
        break
    print(f"{status['progress']*100:.0f}%  eta {status.get('eta_s')}s")
    time.sleep(2)

for variation in status["bundle"]["variations"]:
    print(variation["seed"], variation["preview"], variation["diagnostics"]["flags"])
```

Then look at the previews, pick a seed, and change one field.
