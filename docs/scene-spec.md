# The scene spec

The complete field reference. A live, machine-readable version is always
available from a running server at `GET /api/schema`.

Two rules govern the whole format:

- **Unknown section or field names are rejected.** `"lightning"` instead of
  `"lighting"` returns a 422 naming the field. A misspelled key is always a bug,
  and silently ignoring it is how you spend four minutes rendering something
  that ignored half your request.
- **Unknown *values* are accepted.** `"medium": "cyanotype"` is not in the
  vocabulary, so it passes through as free text and a warning says so. Inventing
  style words is legitimate; you just get no curated phrasing for them.

The smallest valid spec:

```json
{ "subject": { "primary": "a stoneware coffee cup" } }
```

---

## Top level

| Field | Type | Default | Notes |
|---|---|---|---|
| `version` | string | `"1"` | Spec format version |
| `name` | string | — | Human label; used in the output directory name |
| `intent` | enum | `"photoreal"` | `photoreal`, `painterly`, `game_asset`, `render3d`, `graphic` |
| `notes` | string | — | Free text, carried into the sidecar, never used in the prompt |

`intent` is the highest-leverage field. It sets the default model, sampler, CFG,
step count, medium, lighting and negatives — see
[vocabulary.md](vocabulary.md#intents). Anything you set explicitly wins.

## `subject` — required

| Field | Type | Notes |
|---|---|---|
| `primary` | string, required | The main subject in plain words |
| `details` | string[] | Specific attributes to enforce |
| `secondary` | string[] | Supporting elements |
| `count` | int 1–99 | Prefixed to the subject (`"3 lighthouses"`) |
| `action` | string | What the subject is doing |

`primary` carries a small attention weight bump. It is the one thing the image
must not lose.

## `scene`

| Field | Type | Example |
|---|---|---|
| `setting` | string | `"a black basalt cliff"` |
| `time` | string | `"blue hour"` |
| `weather` | string | `"a storm pulling away"` |
| `season` | string | `"late autumn"` |
| `background` | string | `"soft out-of-focus kitchen"` |
| `foreground` | string | `"wet grass in the near field"` |

## `style`

| Field | Type | Notes |
|---|---|---|
| `medium` | vocabulary key | The artefact the image pretends to be |
| `movement` | vocabulary key | Aesthetic tradition; additive on top of medium |
| `artists` | string[] | Rendered as `"in the style of X"` |
| `descriptors` | string[] | Free-text style words |
| `detail` | enum | `minimal`, `moderate`, `high`, `intricate` |

## `camera`

| Field | Type | Notes |
|---|---|---|
| `shot` | vocabulary key | Framing; applies to any medium |
| `lens` | vocabulary key | `85mm`, `tilt_shift`, `anamorphic`… |
| `aperture` | string | `"f/1.8"` |
| `angle` | vocabulary key | `low_angle`, `dutch`, `top_down`… |
| `focus` | vocabulary key | `shallow_dof`, `deep_focus`, `bokeh`… |

Combining an aperture with a non-optical medium (a woodcut has no f-stop) is
kept as requested but produces a warning.

## `lighting`

| Field | Type | Notes |
|---|---|---|
| `key` | vocabulary key | The lighting setup — the strongest lever on mood |
| `mood` | string | Free text |
| `accents` | string[] | Secondary sources |

## `palette`

| Field | Type | Notes |
|---|---|---|
| `name` | vocabulary key | Named palette |
| `colors` | hex[] | Explicit colours, `"#rrggbb"`. Also used by `post.palette_lock` |
| `contrast` | enum | `low`, `normal`, `high` |

## `composition`

| Field | Type | Default | Notes |
|---|---|---|---|
| `aspect` | enum | `"1:1"` | Resolves to an SDXL training bucket |
| `rule` | enum | — | `thirds`, `centered`, `golden_spiral`, `symmetry`, `diagonal` |
| `horizon` | float 0–1 | — | **y coordinate**: 0 is the top edge |
| `layers` | Layer[] | `[]` | The layer stack — see below |

`horizon` catches people out. It is a position, not a proportion: `0.62` puts the
horizon low in the frame, so the *sky* dominates.

### `composition.layers[]`

The layer stack is what turns composition from a request into a constraint. Each
layer contributes a shape at a known depth; `control.mode` then renders the stack
into a map that ControlNet enforces.

| Field | Type | Default | Notes |
|---|---|---|---|
| `role` | string, required | — | Label, e.g. `"lighthouse"`. Also keys the region masks |
| `shape` | enum | `"rect"` | `rect`, `ellipse`, `horizon`, `column`, `blob`, `line` |
| `bbox` | float[4] | `[.25,.25,.75,.75]` | `[x0, y0, x1, y1]` normalised, `x0 < x1` |
| `depth` | float 0–1 | `0.5` | 0 far, 1 near the camera |
| `prompt` | string | — | Per-region description (reserved) |

Shapes are not equivalent. `column` shades across x only, so it reads as a
cylinder. `ellipse` gets a dome falloff, so it reads as volume rather than a flat
plate. `blob` is an irregular noise-warped shape, stable across renders because
its seed derives from its `role`. `horizon` is a flat distant band.

```json
"layers": [
  { "role": "far_ridge",  "shape": "horizon", "bbox": [0.0, 0.52, 1.0, 0.64], "depth": 0.18 },
  { "role": "lighthouse", "shape": "column",  "bbox": [0.62, 0.14, 0.74, 0.66], "depth": 0.75 },
  { "role": "cliff",      "shape": "blob",    "bbox": [0.0, 0.55, 0.55, 1.0],  "depth": 0.92 }
]
```

Preview the result before rendering with `POST /api/control-preview`.

## `control`

| Field | Type | Default | Notes |
|---|---|---|---|
| `mode` | enum | `"none"` | `none`, `depth`, `canny`, `scribble` |
| `source` | enum | `"procedural"` | `procedural` builds from layers; `file` uses your image |
| `image` | path | — | Required when `source: "file"` |
| `strength` | float 0–2 | `0.6` | ControlNet conditioning scale |
| `start` / `end` | float 0–1 | `0` / `1` | Step fractions between which control applies |

`scribble` routes through the canny ControlNet — SDXL has no widely available
scribble model. With `source: "file"` and `mode: "depth"`, your file is used as
the depth map directly; ClauDali does not estimate depth from a photograph.

Lower `end` (say `0.7`) to let the model relax the constraint in the final steps,
which usually reads as more natural.

## `init` — img2img and inpainting

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | path, required | — | The source image |
| `strength` | float 0–1 | `0.55` | How far to depart from the source |
| `mask` | path | — | Presence of a mask switches to inpainting. White = regenerate |
| `mask_blur` | int | `8` | Feather radius; prevents a visible seam |

Cannot currently be combined with `control` — that raises a clear error rather
than ignoring one of them.

## `render`

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | by intent | Catalogue id or a custom checkpoint filename |
| `width` / `height` | int | from `aspect` | Multiples of 8 |
| `steps` | int 1–150 | by intent | |
| `cfg` | float 0–30 | by intent | Guidance scale |
| `sampler` | string | `dpmpp_2m_karras` | See `GET /api/vocabulary` |
| `seed` | int | random | |
| `variations` | int 1–16 | `1` | |
| `clip_skip` | int 1–4 | — | |

With a fixed `seed`, variations use consecutive seeds (`seed`, `seed+1`, …) so a
batch explores a neighbourhood you can return to exactly, rather than rendering
the same image repeatedly.

## `negative`

| Field | Type | Default | Notes |
|---|---|---|---|
| `presets` | string[] | `["artifacts"]` | Vocabulary keys |
| `extra` | string[] | `[]` | Free text |
| `disable_defaults` | bool | `false` | Skip the intent's implicit negatives |

Negative prompts are kept deliberately short. Past roughly 40 tokens, SDXL's
negative conditioning mostly adds noise and eats budget the positive prompt
needs.

## `overlays[]` — composited after diffusion

Anything that must be **legible** goes here. Diffusion models have no concept of
a glyph, so text they generate is always wrong.

| Field | Type | Default | Applies to |
|---|---|---|---|
| `type` | enum | `"text"` | `text`, `rect`, `line`, `ellipse`, `image`, `gradient`, `vignette` |
| `text` | string | — | text |
| `font` | string | system default | Family name or path |
| `size` | int | `48` | text, in pixels |
| `color` | hex | `"#ffffff"` | all |
| `stroke` / `stroke_width` | hex / int | — / `0` | text outline |
| `xy` | float[2] | `[0.5, 0.5]` | Normalised anchor point |
| `anchor` | enum | `"c"` | `nw n ne w c e sw s se` |
| `bbox` | float[4] | — | shapes, gradients, images |
| `direction` | enum | `"down"` | gradient falloff: `down`, `up`, `left`, `right` |
| `opacity` | float 0–1 | `1.0` | all |
| `rotation` | float | `0` | all |
| `width` | int | `2` | shape stroke width |
| `fill` | bool | `false` | shapes |
| `align` / `line_spacing` | enum / float | `left` / `1.2` | multi-line text |
| `image` | path | — | `type: "image"` |

A bottom scrim behind a title wants `direction: "up"` — opaque at the bottom
edge, clearing upward.

## `post` — deterministic finishing

| Field | Type | Default | Notes |
|---|---|---|---|
| `pixelate` | int 1–64 | — | Downscale/upscale factor for true pixel-grid alignment |
| `seamless` | bool | `false` | Cross-fade the wrapped seams so the image tiles |
| `contrast` / `saturation` | float 0–3 | `1.0` | |
| `sharpen` | float 0–2 | `0` | Unsharp mask |
| `palette_lock` | bool | `false` | Snap every pixel to `palette.colors` |
| `palette_size` | int 2–256 | — | Quantise to N colours (when no explicit palette) |
| `grain` | float 0–1 | `0` | Applied last, so it is not itself quantised |
| `transparent_bg` | bool | `false` | Cut the background to alpha |

Two honest limitations: `seamless` is a cross-fade heuristic — good for organic
textures like stone or bark, visibly wrong for anything with structure.
`transparent_bg` is a corner flood-fill, not segmentation; it works well on the
plain backgrounds `game_asset` already asks for and poorly on busy ones.

Order is fixed and deliberate: geometry, then colour, then grain, then alpha.

## `raw` — the escape hatch

| Field | Type | Notes |
|---|---|---|
| `prepend` | string | Inserted before the compiled prompt |
| `append` | string | Added after the compiled prompt |
| `prompt` | string | **Replaces** the compiled prompt entirely |
| `negative_prompt` | string | Replaces the compiled negatives |

`prompt` and `negative_prompt` discard the compiler's work and say so in the
warnings. A rich vocabulary should never become a cage: when you already know the
exact prompt you want, you must be able to say it.
