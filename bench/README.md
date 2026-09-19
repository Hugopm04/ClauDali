# The ClauDali benchmark

Two experiments sharing one runner, one state file and one report.

- The **settings benchmark** holds the picture still and varies the tool's
  options, to find out what each one costs and what it buys.
- The **topic survey** holds the options still and varies the picture, to find
  out what the three checkpoints are actually good at.

Mixing them would confound both, so they are separate phases of one ordered
list. 1,816 items, about **1,526 GPU hours** if it is ever run to the end —
which it does not have to be. See [Stopping early](#stopping-early-is-the-point).

---

## Running it

Everything below works from a **plain terminal with no virtual environment
activated** — PowerShell or cmd, freshly opened. Every command is the project's
own interpreter called by its full path, so nothing has to be set up first.

```powershell
cd "C:\Users\Hugo\Desktop\Mis proyectos\Juegos y gráficos\ClauDali"

.\.venv\Scripts\python.exe -m bench plan --check   # once, and after editing topics
.\.venv\Scripts\python.exe -m bench run            # start, or carry on
```

That is the whole loop. **To stop: press Ctrl+C once.** To carry on: run
`... -m bench run` again. It picks up exactly where it left off, including
mid-image.

The other commands, none of which render anything:

```powershell
.\.venv\Scripts\python.exe -m bench status         # where it stands
.\.venv\Scripts\python.exe -m bench estimate       # what is left, by phase and wave
.\.venv\Scripts\python.exe -m bench report         # write the gallery and the CSV
```

If you would rather not type the path every time, one line per session:

```powershell
$py = "$PWD\.venv\Scripts\python.exe"
& $py -m bench run
```

### Useful flags

```powershell
& $py -m bench run --limit 5              # stop after five items
& $py -m bench run --phase settings       # only that phase
& $py -m bench run --wave w01             # only that survey wave
& $py -m bench run --dry-run              # list what would run, render nothing
& $py -m bench run --force                # resume across a library or model upgrade
```

---

## Stopping and resuming

**One Ctrl+C.** The render stops at the end of the sampler step it is inside,
writes an exact checkpoint into its bundle, and the runner records it as paused
and exits. You lose nothing: not the images already finished, not the steps
already run in the image in progress.

**Two Ctrl+C.** Aborts instead. The variation in flight is lost; the ones
already saved stay, and a later run restarts that variation from step 0 with
the same seed.

**Killed outright** — a crash, a power cut, Task Manager. Nothing is lost that
was written. Every session starts by reading the bundles on disk and believing
them over the state file, because bundles are written *during* an item and the
state file only between them.

Resuming is always the same command:

```powershell
& $py -m bench run
```

It is bit-for-bit. A paused render resumes to the same pixels it would have
produced uninterrupted; `tests/test_claudali.py` locks that with two tests.

One caveat worth knowing when you read the timings: **an item that was paused
and resumed reloaded its model, so its `duration_s` contains an extra load.**
The per-session numbers in `result.json`'s `sessions` list are what to compare;
`timing` describes the most recent session only. Items that ran straight
through, which is most of them, are unaffected.

---

## The progress bar

```
[#########.........................]   23.4187%   item 312/1816
  base     variation 2/4  step 17/32    24.3 s/step   juggernaut-xl 1024x1024
  1d 17h 42m elapsed  ·  52d 03h 18m left  ·  finish Tue 04 Nov 09:20
```

The percentage is of the **whole benchmark**, measured in megapixel-steps
rather than items, so an item that renders four variations at 1024² counts for
more than one at 640². Four decimals, because at this scale a single sampler
step is about 0.0005% and two decimals would sit still for ten minutes and
look like a hang.

The ETA is **self-calibrating**. It starts from the rates measured in
`CLAUDE.md` and then refits from your own completed items, keyed by resolution
and precision, using the median of the last dozen readings so it follows the
machine as it warms up rather than averaging that away. `bench status` prints
what it has learned.

---

## Stopping early is the point

1,526 hours is about 191 nights at eight hours each, and you are not expected
to finish it. The list is ordered so that **stopping at any point leaves a
coherent result rather than a half-filled matrix.**

| Phase | Items | Est. | What it answers |
|---|---:|---:|---|
| features | 22 | 3 h | Do the never-run code paths run at all? |
| settings | 153 | 44 h | What does each option cost and buy? |
| accuracy | 21 | 19 h | Where does prompt adherence break? |
| survey | 1,620 | 1,460 h | What is each model good at? |

Features run first because they are cheap and they are the ones that fail;
finding out in hour two beats finding out in week five. Settings run before the
survey because what they find decides the settings the survey then holds still
— running the survey first risks 1,460 hours on a decode path the benchmark
then shows was the wrong one.

### What the first hour will probably tell you

Two measurements taken while building this, both on the 640x640 six-step smoke
items, give an idea of what the features phase is for:

| Item | Time | s/step | Peak VRAM | Free RAM |
|---|---:|---:|---:|---|
| plain txt2img | 132 s | 13.8 | 5.88 GB | 9.7 → 1.7 GB |
| the same with a depth ControlNet | >25 min | ~49 | 12.35 GB | 8.5 → 0.0 GB |

**ControlNet on this card is pathologically slow.** A second network beside an
already-offloaded 6 GB UNet pushes the allocation past the card, Windows absorbs
the overflow into shared system memory, and the render crawls. 12.35 GB of
"peak VRAM" on a 6 GB card is that spill, not a bad reading.

So expect the three `control-*` items to dominate the features phase, and treat
the full-quality ControlNet items at 1024² as genuinely open-ended — that is the
question they exist to answer. If they are still going after an hour, Ctrl+C is
the right response; the phase after them is not blocked by it.

### The survey's twelve waves

Inside the survey every item costs the same, so ordering by time would be
meaningless. It is ordered by information instead:

| Wave | Items | What you have when it finishes |
|---|---:|---|
| w01 | 90 | A complete **family-level** survey: one rich prompt per sub-case on the model its intent would pick. |
| w02 | 90 | …plus the **minimal control** for each, so you know what the style vocabulary was worth. |
| w03–w04 | 360 | …plus prompts 2 and 3, making every **sub-case** verdict defensible rather than a hunch. |
| w05–w08 | 540 | The same four waves on the second checkpoint. |
| w09–w12 | 540 | And on the third. |

So w01 alone (about 81 hours) is a finished piece of work. w01–w04 is a
defensible sub-case survey of one model. Nothing is ever half-done.

---

## What gets measured

Beyond the images, every bundle's `result.json` now records:

- `timing.stages` — per stage: runs, total, mean, and **seconds per step
  measured** rather than divided out of the total.
- `timing.steps` — count, mean, median, min, max, and **`first_s` on its own**,
  because the first step of a stage is where paging shows up. After a CPU
  decode short of RAM it has been twelve times the others on this machine.
- `timing.spans` — every load, prompt encode, stage, decode, upscale and
  encode, with the decode's device, precision and tiling recorded on it.
- `memory` — RAM free at the **start** (the number that explains the decode
  path chosen), at the end, and the **low-water mark** from a background
  sampler taking a reading every two seconds, plus peak VRAM allocated and
  reserved.
- `sessions` — one entry per render call, so a paused-and-resumed item's
  numbers stay separable.

`bench report` flattens all of it into `bench/report/measurements.csv`, one row
per item.

---

## Reading the results

```powershell
& $py -m bench report
start "" "bench\report\index.html"
```

You get an index, a page per phase, and the CSV. Each page groups the images
that are meant to be compared side by side — the same scene across an axis's
values, or the same topic across models — with each one's timings, memory,
decode path and diagnostic flags underneath, and the sentence saying what that
group is testing for.

Each image has a 1–5 score box and a notes field. **They save in the browser as
you type, but press "Export scores" before closing the tab**: a `file://` page's
storage is per-browser and easy to lose, and the exported `bench-scores.json`
is the copy that counts.

---

## Editing the topics

`bench/topics/*.yaml`, one file per family. 18 families, 90 sub-cases, 3
prompts each.

Each prompt renders twice: a **rich** spec with the family's and sub-case's
style, camera and lighting keys applied, and a **minimal** spec that is nothing
but the subject line and the intent. The pair is the control — where they
disagree you learn whether the vocabulary helped, and a weak rich result can
be blamed on the topic or on the style key rather than being ambiguous between
them.

A `rich` block may set any key in `RICH_KEYS` (`bench/catalog.py`), and
`key: null` **removes** one the family set. That matters: a family sets
`lens: 85mm` because most of its sub-cases are photographs, and the painterly
one has to take it off again or the compiler rightly complains that the spec
asks for two incompatible things.

After editing, always:

```powershell
& $py -m bench plan --check
```

`--check` compiles all 1,816 specs against the real vocabulary and refuses to
write a broken plan. It is worth the minute: an unknown vocabulary key does not
fail, it passes through as free text with a warning, which would quietly turn a
style test into a text test. Only 12 warnings are expected, all of them the
refiner telling you it may wash out Juggernaut's look, which is by design.

**Ids are built from family, sub-case and prompt index**, so inserting a prompt
in the middle of a list renumbers everything after it and orphans those items'
state. Append to the end of a `prompts` list instead.

---

## Layout

```
bench/
  topics/*.yaml     the survey's subject matter -- 18 families, 90 sub-cases
  catalog.py        builds the ordered item list for all four phases
  model.py          Item, the self-calibrating cost model, the run state
  runner.py         spawns workers, follows them, keeps the books
  worker.py         renders; the only module here that reaches torch
  progress.py       the whole-run bar and its ETA
  report.py         the HTML gallery and the CSV
  __main__.py       plan / run / status / report / estimate

  plan.json         generated: the ordered list
  specs/            generated: one SceneSpec per item
  runs/             generated: one bundle per item -- the images
  state.json        generated: what has run
  calibration.json  generated: measured s/step, per resolution
  report/           generated: index.html, <phase>.html, measurements.csv
  logs/             generated: each worker's stderr
  assets/           generated: the source image the init and inpaint items load
```

Everything generated is in `.gitignore`. What defines the benchmark — the topic
files and the code — is committed; the gigabytes it produces are not.

## Process model

The settings phase runs **one process per item**, because its numbers are the
point and a forced CPU decode leaves the machine paging: `CLAUDE.md` records a
768×768 step going from 12 s to 90 s in the next render in the same process.

The other phases run **consecutive same-model items in one process**, up to
twelve, which skips a 7 GB reload between them and saves roughly six hours over
the survey. Load time is recorded per item either way, so a warm item is
visible in the data as `load_s` near zero rather than being silently averaged
in.
