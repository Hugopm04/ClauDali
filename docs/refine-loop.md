# The refine loop

How to get a good image out of ClauDali, written for whoever is driving it —
a person at the UI, a script, or a language model with no eyes on the terminal.

The premise: **a first render is a question, not an answer.** ClauDali is built
so that the second render is informed rather than another guess.

---

## The loop

```
compile  ->  look at the prompt        (milliseconds, no GPU)
control  ->  look at the composition   (milliseconds, no GPU)
render   ->  look at the previews      (minutes)
            read the diagnostics
            change ONE thing
            re-render with the same seed
```

The two cheap steps come first for a reason. On a 6 GB card a render costs
about a quarter of an hour; a compile costs nothing. Most bad renders are visible as bad
prompts before any pixels exist.

---

## Step 1 — Compile before you render

```bash
curl -X POST localhost:8188/api/compile -H "Content-Type: application/json" -d @spec.json
```

Read the `prompt` field and ask:

- **How many chunks is it?** Read `tokens.chunks`. CLIP reads 75 tokens at a
  time, and every chunk votes in cross-attention, so a four-chunk prompt is
  four prompts averaged together. The compiler restates the subject in each one
  (`anchors` shows where), which is what stops a long prompt rendering the
  setting and forgetting the subject. Set `subject.anchor` to two or three words
  naming the subject alone; without it `primary` is used and the setting gets
  restated too.
- **Is the subject buried?** CLIP weighs early tokens most. If your subject is
  behind four style fragments, cut the style, not the subject.
- **Are there warnings or notes?** A warning means the spec wants changing; a
  note means the compiler decided something for you. An unknown vocabulary key
  means you got free text where you expected curated phrasing — often a typo
  (`rembrant` for `rembrandt`).
- **Does the framing assume a body?** `shot: medium` is "waist up" and
  `close_up` is "head and shoulders". On a landscape, an object or a swarm of
  small creatures those resolve as a tight crop of whatever is nearest. Use
  `scene`, `tableau`, `group` or `tight`.
- **Is the prompt fighting itself?** `medium: photograph` with
  `movement: cubism` compiles to a request for a photorealistic cubist image.
  SDXL will pick one, probably not the one you meant.
- **Do the negatives contradict the positives?** `intent: painterly` adds
  `photographic` to the negatives. If you then ask for `medium: photograph`,
  you have asked for and against the same thing.

## Step 2 — Preview the composition

If you are using `control`, look at the map before rendering:

```bash
curl -X POST localhost:8188/api/control-preview -H "Content-Type: application/json" \
     -d @spec.json --output control.png
```

A depth map should read as a scene: near things bright, far things dark, shapes
distinguishable. If the map is ambiguous, the render will be too. Common fixes:

- Shapes overlapping at similar depths merge — separate their `depth` values.
- A `rect` reads as a flat plate. Use `column` for uprights, `ellipse` or `blob`
  for volumes.
- Everything mid-grey means the depth range is too narrow. Push the near layer
  toward 0.9 and the far one toward 0.15.

## Step 3 — Render several variations

Ask for 4 or 6, not 1. On this hardware a single render costs the same minutes
whether you learn much from it or not, and a batch tells you whether a problem
is in your spec or just in one unlucky seed.

Set an explicit `seed`. Variations then use consecutive seeds, so the batch is
a neighbourhood you can return to exactly.

## Step 4 — Read the result, not just the image

Open `contact-sheet.jpg` first — one look tells you which variation to pursue.
Then read `result.json` for what the preview hides.

| Signal | What it means | Likely fix |
|---|---|---|
| `flags` contains the black-frame message | Something upstream produced NaNs | Run `claudali doctor`: if `fp16_conv_broken` and not `cudnn_disabled`, set `CLAUDALI_CUDNN=off`; if cuDNN is already off, add `CLAUDALI_VAE_UPCAST=always`; else check `sdxl-vae-fp16-fix` is installed |
| `exposure.dynamic_range` < 25 | Flat, muddy image | Raise `steps`, or `palette.contrast: "high"` |
| `exposure.highlight_clip_pct` > 8 | Blown highlights | Lower `render.cfg` by 1–2 |
| `detail.laplacian_variance` much lower than siblings | That variation is soft | Prefer a sharper seed; or `camera.focus: "tack_sharp"` |
| `composition.off_center` > 0.2 when you asked for `centered` | The model ignored the framing | Add layers and `control.mode: "depth"` |
| `color.mean_saturation` very low, unintentionally | Washed out | `palette.name`, or `post.saturation: 1.2` |
| `palette` dominated by one grey | Fog, haze, or an empty frame | Strengthen `lighting.key`, add a foreground layer |

Diagnostics never gate a render. They tell you what happened; the judgement is
yours.

## Step 5 — Change one thing

The discipline that makes the loop work. With a fixed seed, one change is
attributable; three changes are a new guess.

Rough order of leverage, strongest first:

1. `intent` — changes the model, CFG and negatives all at once
2. `style.medium` — the strongest single aesthetic lever
3. `lighting.key` — the strongest lever on mood
4. `composition.layers` + `control` — the only way to *enforce* framing
5. `render.cfg` — 4–6 loose and natural, 7–9 literal and contrasty
6. `palette`, `camera`, `detail` — refinements

## Step 6 — Fix locally rather than rerolling

When one region is wrong and the rest is right, do not re-render. Inpaint:

```json
{
  "subject": { "primary": "a stoneware coffee cup, clean undamaged rim" },
  "init": {
    "image": "outputs/.../001_seed1848.png",
    "mask":  "runs/uploads/rim-mask.png",
    "strength": 0.6,
    "mask_blur": 12
  }
}
```

White in the mask means regenerate. `mask_blur` feathers the join; without it
there is a visible seam. `strength` around 0.5–0.7 changes the region while
keeping its lighting; above 0.85 it stops matching its surroundings.

If the region is one of your layers, name its `role` instead of drawing a mask.
Keep the original spec's `composition` as it was: layer boxes are normalised to
the frame, so the mask lands on the right pixels only with the same layers and
aspect.

```json
"init": { "image": "outputs/.../001_seed1848.png", "mask_layer": "lighthouse", "strength": 0.6 }
```

---

## Working notes

**Fix the seed early.** Without it every render changes two things: your edit
and the noise. You cannot tell which caused what.

**Prefer subtraction.** Long specs produce muddy images. If a render is
cluttered, remove `details` and `descriptors` before adding more.

**Do not stack negatives.** Past roughly 40 tokens, negative conditioning mostly
adds noise. If something unwanted keeps appearing, it is usually being *implied*
by the positive prompt.

**Let the model do what models do.** ControlNet placement, exact colour, and
legible text are ClauDali's job. Texture, light, and material are SDXL's. Fight
whichever one is losing, not both.

**Batch overnight.** On a GTX 1660 Ti four variations take most of an hour. Queue
them, walk away, judge the contact sheet in one sitting. The queue is designed
for throughput, not latency.

**Pause rather than cancel.** Each variation is saved the moment it finishes, so
a long batch can be judged while it runs. When you need the machine back, pause:
Ctrl+C in the terminal, or Pause in the UI or the API. The render stops at the
end of its current step and resumes later with an image identical to the one it
would have made. A variation that already looks wrong is the time to cancel.

---

## A worked example

```jsonc
// 1. First attempt: too vague.
{ "intent": "painterly", "subject": { "primary": "a lighthouse" } }
// -> compiles fine, renders as generic stock art. No sense of place.

// 2. Add specificity and light. Fix the seed.
{
  "intent": "painterly",
  "subject": { "primary": "a lighthouse built from stacked antique clocks",
               "details": ["salt-worn brass", "one lit window"] },
  "scene": { "setting": "a black basalt cliff", "time": "blue hour" },
  "lighting": { "key": "stormlight" },
  "render": { "seed": 1848, "variations": 4 }
}
// -> much better, but the lighthouse drifts to the centre in 3 of 4.

// 3. Enforce the framing. Nothing else changes.
{
  "...": "as above",
  "composition": {
    "aspect": "3:2", "horizon": 0.62,
    "layers": [
      { "role": "far_ridge",  "shape": "horizon", "bbox": [0.0, 0.52, 1.0, 0.64], "depth": 0.18 },
      { "role": "lighthouse", "shape": "column",  "bbox": [0.62, 0.14, 0.74, 0.66], "depth": 0.75 },
      { "role": "cliff",      "shape": "blob",    "bbox": [0.0, 0.55, 0.55, 1.0],  "depth": 0.92 }
    ]
  },
  "control": { "mode": "depth", "strength": 0.55, "end": 0.8 }
}
// -> placement now holds across all four seeds. `end: 0.8` lets the last
//    20% of steps soften the constraint so it does not look traced.
```

Three rounds, one variable at a time, each one informed by the last.
