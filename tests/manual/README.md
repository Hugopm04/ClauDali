# Manual tests

What `pytest` cannot check: the web UI in a real browser, render times, image
quality, and every path that needs the GPU or a model that was not installed
when it was written. Each section says how long it takes, the commands to run,
and what you should see. The JSON specs are in this folder.

Run everything from the project root, in PowerShell:

```powershell
cd "C:\Users\Hugo\Desktop\Mis proyectos\Juegos y gráficos\ClauDali"
```

Times below are estimates from what has been measured on the GTX 1660 Ti
laptop: 23–26 s per step at 1344×768, ~10–13 s per step at 768×768, ~40 s to
load SDXL base, 30–90 s for Juggernaut.

## What has never been tested

| Change | State | Section |
|---|---|---|
| Web UI: form merge, "Also in this spec", pause and resume buttons, the Quality section | Checked in Node against the page's script, never clicked in a browser | 2 |
| VAE decode `auto` / `cpu` | Both ran on the GPU once at 768×768 (below). Image quality never compared | 4 |
| Hi-res pass, Lanczos | Smoke test at 512→768 only (below). Full size never run | 5 |
| Hi-res pass, Real-ESRGAN | Never run: model and spandrel not installed | 5 |
| Refiner | Never run: model not installed. Only the CPU stage-loop test | 6 |
| Pause and resume inside the refiner and hi-res stages | CPU test only | 3, 7 |
| fp32 precision, sequential offload | Never run | 8 |
| `render.quality: "max"` end to end | Never run | 9 |

## Already measured (2026-09-13, SDXL base, one process)

- Loading SDXL base took ~40 s.
- **`auto` decode, 768×768, 2 steps:** at decode time 0.1 GB of RAM was free
  against a ~4.3 GB estimate, so it decoded on the GPU and said so in a note.
  The image was normal (no diagnostic flags).
- **Forced `cpu` decode, same process:** completed with 1.1 GB free, so the
  machine paged. fp32, untiled, no NaN, no flags.
- **Hi-res smoke test, 512×512 → 768×768, Lanczos, 2 base steps and 1 hi-res
  step:** ran end to end: base stage, decode (`auto` chose the CPU: ~2.5 GB
  needed, 3.2 GB free), upscale, encode (`auto` fell back to the GPU), one
  img2img step, final decode (`auto` chose the CPU: ~4.3 GB needed, 4.6 GB free).
  The picture is no verdict on quality: two steps at 512×512 is outside what SDXL
  does, and it came out as concentric frames.
- **That run was slow, most likely from paging.** The forced CPU decode before
  it had left too little RAM, and the offloaded weights went to the pagefile. The
  next render then waited ~2.5 minutes before its first step, and its one
  768×768 hi-res step took 90 s where ~12 s is normal. A CPU decode short of RAM
  can cost more time after it than it takes itself.
- **The GPU encode the hi-res pass uses** (fp16-fix VAE, fp16, cuDNN off), checked
  on its own: 5.4 s and 1.06 GB of VRAM at 768×768, and its latents decode back
  to the source photo at 68.7 dB PSNR.
- **A bug that check found, now fixed:** latents waiting between stages sit on the
  CPU, and the GPU decode handed them to the VAE as they were, which fails
  ("Input type (float) and bias type (Half)"). Every hi-res pass whose first
  decode `auto` sends to the GPU, the usual case on this laptop, would have
  stopped there.
- **Which VAE decodes better**, measured on the CPU in fp32 on a 768×768 crop of
  a real render: SDXL base's own VAE reconstructs it at 50.2 dB PSNR, the
  fp16-fix VAE the GPU path uses at 43.6 dB. So the CPU decode gains a better
  decoder as well as losing the tiles. On a 2-step 768×768 image the two decodes
  differed by 5.8/255 on average.

---

## 0. Before you start (5 minutes, plus downloads)

```powershell
.venv\Scripts\python -m pip install -r requirements.txt       # adds spandrel
.venv\Scripts\python -m pytest -q                              # expect: 103 passed
.venv\Scripts\python -m claudali doctor                        # note ram_available_gb, cudnn_disabled
python -m installer models --list                              # the two new models, "quality models 6.30 GB"
python -m installer models --add sdxl-refiner realesrgan-x4    # 6.3 GB; only sections 5 (Real-ESRGAN), 6 and 9 need it
```

Close what you can before the render sections: free RAM decides the `auto`
decode, and fp32 needs all of it.

## 1. Compile every spec, no GPU (1 minute)

```powershell
Get-ChildItem tests\manual\*.json | ForEach-Object {
  "== $($_.Name)"
  .venv\Scripts\python -m claudali compile $_.FullName | Select-String "^(quality|size|warning|note)"
}
```

Check the `quality` line of each:

| Spec | Expect |
|---|---|
| `decode-*.json` | `VAE decode gpu_tiled` / `cpu` / `auto`, stages `base` |
| `hires-lanczos.json`, `hires-realesrgan.json` | stages `base -> hires` |
| `refiner-on.json` | stages `base -> refiner`. Before installing the refiner, a warning ending "rendering this spec stops with an error" |
| `fp32-small.json` | `float32` |
| `fairies-max.json` | `float32`, `VAE decode cpu`, stages `base -> refiner -> hires` once the refiner is installed; a warning that the refiner may wash out Juggernaut's look; a note beginning "render.quality 'max' turned on" |
| `ui-roundtrip.json` | `float16`, `gpu_tiled`, stages `base -> hires`: `max` with explicit opt-outs |

## 2. Web UI (20 minutes)

```powershell
.\scripts\start.ps1
```

Open <http://127.0.0.1:8188>. Keep the server's terminal visible.

**2.1 The page loads.** The header chips show the GPU, "queue idle" and the
number of checkpoints. A **Quality** section sits under Render. Its VAE decode
list starts with "server default (auto)" and has a grey line under it with RAM
numbers. If the refiner is not installed, its "on" option says "(not installed)".

**2.2 The decode hint follows the form.** Switch Aspect between 1:1 and 16:9:
the "~X GB" in the hint changes. Choose "Force CPU fp32 untiled": the hint says
"expect paging" if free RAM is short. Choose "GPU tiled": the hint disappears.

**2.3 The max preset.** Set VAE decode back to "server default". Choose Preset
"max": Precision becomes fp32, VAE decode "Force CPU", Hi-res pass "on", and
Refiner "on" only if it is installed. Choose "standard" again: those go back to
their defaults. Now set VAE decode to "GPU tiled" first and choose "max": VAE
decode stays "GPU tiled".

**2.4 Compile only**, with max chosen: the status line ends in
"float32 · decode cpu · base → hires" (or "→ refiner → hires"). The Compiled
prompt card lists the "turned on" note, and a warning if the refiner is missing.

**2.5 Show JSON** writes `render.quality`, `precision`, `vae_decode` and
`hires.enabled`, and nothing for inputs you did not touch.

**2.6 Use this JSON.** Open "Raw spec JSON", paste the contents of
`tests/manual/ui-roundtrip.json`, press "Use this JSON". The form shows intent
painterly, anchor "clock lighthouse", preset max, precision fp16, decode GPU
tiled, refiner off, hi-res "by preset", scale 1.25, strength 0.35. "Also in this
spec" lists `name`, `notes`, `subject.secondary`, `scene.season`,
`style.artists`, `overlays` and `post.grain`, and none of the Quality inputs.
Change only Steps to 4 and press Show JSON: `render.steps` is 4 and everything
else, overlays included, is as pasted.

**2.7 Render it** (about 5 minutes). The status reads "Rendering variation 1/1"
then "Hi-res pass on variation 1/1", and the progress bar never moves backwards.
The result is 1520×1040 with "TEMPUS" at the bottom, and its card says
"decoded gpu fp16 tiled". The notes mention the hi-res pass.

**2.8 Pause and resume.** Load `tests/manual/pause-hires.json` the same way and
Render. At step 3 of variation 1 press **Pause (next job runs)**: "Pausing after
the current step…", then "Paused" with "variation 1/2, step 3/8", and a Paused
card. Press **Resume**: it continues from step 4. During a "Hi-res pass on"
status press **Pause & hold queue**: the queue chip says "queue held" and the
Paused card offers "Release queue". Press Resume: it finishes. Both images land.

**2.9 Cancel a paused job.** Render `pause-hires.json` again, pause it, press
**Cancel job**: status "Cancelled", finished images stay in the bundle.

**2.10 History.** Click an old entry from 2026-09-12: the form fills, and "Also
in this spec" lists only fields that differ from a default.

**2.11 Stopping the server keeps the job.** Render `pause-hires.json`, and
during the render press Ctrl+C once in the server terminal. It logs "Pausing the
running job…" then "Saved 1 paused or queued job(s)". Start the server again:
the Paused card shows the job and "queue held". Resume it: it completes.

**2.12 A refused resume.** Pause a job, stop the server with Ctrl+C, then:

```powershell
$env:CLAUDALI_ATTENTION_SLICING = "0"; .\scripts\start.ps1
```

Resume the job: "Resume refused", with `attention_slicing: True -> False` listed
and a **Resume anyway** button, which finishes it. Then close that terminal, or
run `Remove-Item Env:CLAUDALI_ATTENTION_SLICING`, before going on.

## 3. The HTTP API (10 minutes, the server from section 2)

```powershell
$api = "http://127.0.0.1:8188"
Invoke-RestMethod "$api/api/memory"
(Invoke-RestMethod "$api/api/compile" -Method Post -ContentType "application/json" -InFile tests\manual\fairies-max.json) |
  Select-Object precision, vae_decode, stages, refiner, hires
$job = Invoke-RestMethod "$api/api/render" -Method Post -ContentType "application/json" -InFile tests\manual\pause-hires.json
Invoke-RestMethod "$api/api/jobs/$($job.id)" | Select-Object status, stage, step, total_steps, variation, progress, eta_s
```

Repeat the last line during the run: `stage` goes `base`, `hires`, `base`,
`hires`, and `progress` only rises. During a hi-res stage:

```powershell
Invoke-RestMethod "$api/api/jobs/$($job.id)/pause" -Method Post -ContentType "application/json" -Body '{"hold_queue": false}'
(Invoke-RestMethod "$api/api/paused").jobs.checkpoint     # stage hires, stages base+hires, steps 4
Invoke-RestMethod "$api/api/jobs/$($job.id)/resume" -Method Post -ContentType "application/json" -Body '{}'
```

When it is done, `bundle.variations[0].decode` holds `mode`, `device`,
`precision`, `tiled`, `need_gb` and `free_gb`.

## 4. VAE decode: quality and time (20 minutes)

Stop the server first: renders from the command line share the GPU.

```powershell
.venv\Scripts\python -m claudali render tests\manual\decode-gpu-tiled.json
.venv\Scripts\python -m claudali render tests\manual\decode-cpu.json
.venv\Scripts\python -m claudali render tests\manual\decode-auto.json
```

Note each "image(s) in N s" line. Then compare the two forced decodes and write
a picture of where they differ, 8x amplified:

```powershell
.venv\Scripts\python -c "import glob,numpy as np;from PIL import Image;a,b=[np.asarray(Image.open(sorted(glob.glob(f'outputs/*_{n}_*/001_seed2026.png'))[-1]),dtype=int) for n in ('decode-gpu-tiled','decode-cpu')];d=np.abs(a-b);print('max',d.max(),'mean',round(d.mean(),3));Image.fromarray(np.clip(d*8,0,255).astype('uint8')).save('decode-diff.png')"
```

- Open both PNGs at 100% and look at the sky for lines or bands.
- In `decode-diff.png`, a grid of lines is tile seams; noise spread everywhere
  is fp16 against fp32.
- In each bundle's `result.json`, `variations[0].decode` should say
  `gpu`/`tiled: true`, `cpu`/`tiled: false`, and for `auto` whichever it chose
  with the `free_gb` it saw. The auto run's notes explain the choice.

## 5. Hi-res pass (about 30 minutes each; full size never run)

```powershell
.venv\Scripts\python -m claudali render tests\manual\hires-off.json
.venv\Scripts\python -m claudali render tests\manual\hires-lanczos.json
.venv\Scripts\python -m claudali render tests\manual\hires-realesrgan.json   # needs realesrgan-x4 + spandrel
```

- The progress line shows `base` for 20 steps, then `hi-res` for 6.
- Watch for an out-of-memory error at 2016×1152. If one comes, note the stage
  and step it came in.
- Time one hi-res step against one base step. The estimate is ~2.25x.
- Compare the 2016×1152 results with the 1344×768 `hires-off` image at the same
  framing: engraving on the watch, wood grain, dust. Real-ESRGAN against Lanczos:
  sharper, or invented texture?

## 6. Refiner (about 30 minutes; never run)

```powershell
.venv\Scripts\python -m claudali render tests\manual\refiner-off.json
.venv\Scripts\python -m claudali render tests\manual\refiner-on.json
```

- `refiner-on` shows `base` for 24 steps, then loads the refiner, then `refiner`
  for ~6 steps.
- In Task Manager, memory should drop when the base unloads, before the refiner
  loads. Two models at once would push the machine into heavy swapping.
- Compare the faces, hands, net and skin texture with `refiner-off`.

## 7. Pause and resume across stages (30 minutes)

Run each spec twice: once straight through, once paused and resumed. Then
compare. Identical images are the point: max difference 0.

```powershell
.venv\Scripts\python -m claudali render tests\manual\pause-hires.json
.venv\Scripts\python -m claudali render tests\manual\pause-hires.json
#   press Ctrl+C once when the progress line shows "hi-res  variation 1/2":
#   it prints "Paused in the hi-res stage at variation 1/2, step N/4" and a resume command
.venv\Scripts\python -m claudali resume "outputs\<the folder it printed>"
.venv\Scripts\python -c "import glob,numpy as np;from PIL import Image;d=sorted(glob.glob('outputs/*_pause-hires_*'))[-2:];[print(i,np.abs(np.asarray(Image.open(glob.glob(f'{d[0]}/{i:03d}_seed*.png')[0]),dtype=int)-np.asarray(Image.open(glob.glob(f'{d[1]}/{i:03d}_seed*.png')[0]),dtype=int)).max()) for i in (1,2)]"
```

Then the same with `pause-refiner.json` (needs the refiner), pressing Ctrl+C
when the line shows `refiner  variation 2/2`, and `pause-hires` replaced by
`pause-refiner` in the compare line. Also try Ctrl+C twice in a row: it aborts,
and `resume` restarts that stage from its first step with the same seed.

## 8. fp32 (10–20 minutes; never run, may page hard)

```powershell
.venv\Scripts\python -m claudali render tests\manual\fp16-small.json
.venv\Scripts\python -m claudali render tests\manual\fp32-small.json
```

- The fp32 notes include "uses sequential CPU offload".
- Watch Task Manager's memory during the load and write down the peak.
- Seconds per step, fp16 against fp32.
- If it dies with an out-of-memory error, that answers whether fp32 fits this
  laptop at all.

## 9. The whole max preset (hours; run overnight)

```powershell
.venv\Scripts\python -m claudali render tests\manual\fairies-standard.json
.venv\Scripts\python -m claudali render tests\manual\fairies-max.json
```

`fairies-max` warns that the refiner may wash out Juggernaut's look; that is by
design. Pause with Ctrl+C whenever you need the machine, and resume later. Judge
it against `fairies-standard`: are the three fairies there, how much detail do
they carry, and is it worth the time?

## 10. What to send back

Fill in what you ran:

| Test | Result (ok / problem) | Total time | s per step | Decode path | Notes |
|---|---|---|---|---|---|
| 1 compile | | | | | |
| 2 web UI, by step number | | | | | |
| 3 API | | | | | |
| 4 decode gpu-tiled / cpu / auto | | | | | max and mean pixel difference |
| 5 hires off / lanczos / realesrgan | | | base vs hi-res | | |
| 6 refiner off / on | | | | | RAM when models swap |
| 7 pause hires / refiner | | | | | max difference per image |
| 8 fp16 / fp32 small | | | | | peak RAM |
| 9 fairies standard / max | | | | | |

For any problem, the bundle's `result.json` (its `warnings`, `notes`, `status`
and `error`) and the terminal output are what is needed.
