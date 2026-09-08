# Vocabulary reference

Every key ClauDali understands, and the prompt fragment it expands to.

**This file is generated.** Edit `claudali/vocabulary/*.yaml` and re-run
`python scripts/gen_vocab_docs.py`.

Keys are optional everywhere. An unrecognised value is not an error: it is
passed through as free text and reported in the compiled result's `warnings`,
so `"medium": "cyanotype"` works even though there is no `cyanotype` entry — it
just gets no curated phrasing or implied negatives.

`weight` is the attention multiplier applied in compel syntax. Values stay
within roughly 0.9–1.25; past about 1.3 SDXL starts to distort colour.


## Intents

`intent` is the one field worth choosing deliberately: it sets the
default model, sampler, CFG, medium, lighting and negatives.

| Intent | Model | Steps | CFG | Default medium | Purpose |
|---|---|---|---|---|---|
| `photoreal` | `juggernaut-xl` | 32 | 5.5 | `photograph` | Photographs of things that could exist: objects, people, places. |
| `painterly` | `dreamshaper-xl` | 34 | 7.5 | `oil_impasto` | Paintings, surrealism, illustration, anything that reads as made by hand. |
| `game_asset` | `sdxl-base` | 30 | 7.0 | `—` | Sprites, tiles, textures, icons: images destined for a game engine. |
| `render3d` | `sdxl-base` | 32 | 6.5 | `render3d` | Rendered 3D scenes: materials, global illumination, product shots. |
| `graphic` | `sdxl-base` | 30 | 8.0 | `vector_flat` | Posters, covers, diagrams, UI: layout-led images finished with overlays. |

## Aspect buckets

SDXL was trained on fixed aspect buckets. Named aspects always resolve
to one, because rendering off-bucket costs coherence.

| `composition.aspect` | Resolution |
|---|---|
| `1:1` | 1024 × 1024 |
| `5:4` | 1152 × 896 |
| `4:5` | 896 × 1152 |
| `3:2` | 1216 × 832 |
| `2:3` | 832 × 1216 |
| `16:9` | 1344 × 768 |
| `9:16` | 768 × 1344 |
| `21:9` | 1536 × 640 |
| `9:21` | 640 × 1536 |

## Media

Spec field: `style.medium` — 32 keys

| Key | Expands to | Weight |
|---|---|---|
| `photograph` | professional photograph, photographic, realistic detail, natural colour | 1.1 |
| `film_35mm` | 35mm film photograph, kodak portra, fine film grain, halation, analog | 1.1 |
| `polaroid` | polaroid instant photo, soft focus, milky highlights, faded emulsion, white border | 1.05 |
| `cinematic_still` | cinematic film still, anamorphic, colour graded, shallow depth of field, movie frame | 1.1 |
| `oil_impasto` | oil painting, thick impasto brushwork, visible palette knife strokes, canvas weave | 1.15 |
| `oil_classical` | classical oil painting, glazed layers, subtle varnish sheen, museum canvas | 1.12 |
| `watercolor` | watercolour painting, wet-on-wet bleeds, granulating pigment, paper tooth, soft edges | 1.15 |
| `gouache` | gouache painting, matte opaque pigment, flat washes, visible brush ends | 1.12 |
| `ink_linework` | pen and ink drawing, confident linework, cross-hatching, high contrast, white paper | 1.15 |
| `charcoal` | charcoal drawing, smudged tonal shading, grainy paper, deep blacks | 1.12 |
| `pastel` | soft pastel drawing, chalky pigment, blended tones, textured paper | 1.1 |
| `acrylic` | acrylic painting, bold saturated colour, crisp brush edges, matte finish | 1.1 |
| `digital_painting` | digital painting, painterly brushwork, confident value structure, artstation quality | 1.1 |
| `concept_art` | production concept art, design sheet clarity, strong silhouette, cinematic staging | 1.1 |
| `matte_painting` | digital matte painting, vast scale, atmospheric perspective, film production quality | 1.1 |
| `pixel_art` | pixel art, crisp aligned pixels, limited palette, clean dithering, sprite art | 1.2 |
| `voxel` | voxel art, cubic blocks, isometric construction, clean geometry, toy-like | 1.15 |
| `low_poly` | low poly 3d, flat shaded triangles, faceted geometry, clean edges | 1.15 |
| `render3d` | octane render, physically based materials, ray traced global illumination, 3d | 1.1 |
| `clay_render` | clay render, matte grey material, soft studio ambient occlusion, no textures | 1.15 |
| `cel_shaded` | cel shaded illustration, flat colour zones, clean ink outline, anime production art | 1.15 |
| `comic` | comic book art, bold inked outlines, halftone dots, dynamic panel composition | 1.15 |
| `woodcut` | woodcut print, carved gouge marks, stark black and white, rough register | 1.2 |
| `engraving` | copperplate engraving, fine parallel hatching, antique print, ink on rag paper | 1.18 |
| `stained_glass` | stained glass window, leaded came outlines, luminous saturated panes, backlit | 1.2 |
| `mosaic` | mosaic, tessellated tesserae, grout lines, byzantine gold ground | 1.2 |
| `blueprint` | technical blueprint, white line drawing on cyan ground, orthographic, dimension marks | 1.2 |
| `vector_flat` | flat vector illustration, clean geometric shapes, solid fills, generous negative space | 1.18 |
| `isometric` | isometric illustration, 2:1 dimetric projection, clean parallel edges, tidy scene | 1.15 |
| `airbrush` | airbrush illustration, smooth gradients, glossy highlights, retro 1980s poster art | 1.1 |
| `collage` | paper collage, torn edges, layered cut-outs, mixed printed textures, visible seams | 1.18 |
| `tapestry` | woven tapestry, thread texture, muted dyed wool, medieval millefleur ground | 1.2 |

## Movements

Spec field: `style.movement` — 26 keys

| Key | Expands to | Weight |
|---|---|---|
| `surrealism` | surrealist, dreamlike impossible juxtaposition, uncanny scale, symbolic imagery | 1.15 |
| `magic_realism` | magic realism, ordinary world with one impossible detail, calm matter-of-fact staging | 1.1 |
| `impressionism` | impressionist, broken colour, visible directional strokes, luminous atmosphere | 1.15 |
| `post_impressionism` | post-impressionist, bold non-naturalistic colour, rhythmic swirling strokes | 1.15 |
| `expressionism` | expressionist, distorted emotional forms, aggressive colour, raw gestural marks | 1.15 |
| `baroque` | baroque, dramatic diagonal composition, tenebrism, opulent drapery, theatrical | 1.12 |
| `romanticism` | romanticist, sublime landscape, awe of nature, turbulent sky, small human figure | 1.12 |
| `symbolism` | symbolist, allegorical figures, mystic pallor, ornamental melancholy | 1.12 |
| `art_nouveau` | art nouveau, whiplash organic curves, botanical ornament, decorative border, mucha | 1.15 |
| `art_deco` | art deco, symmetrical geometric ornament, stepped forms, gold and black, streamlined | 1.15 |
| `bauhaus` | bauhaus, primary colours, pure geometric forms, functional grid, sans-serif rigour | 1.18 |
| `constructivism` | russian constructivism, bold diagonal geometry, red black cream, propaganda poster | 1.18 |
| `ukiyo_e` | ukiyo-e woodblock print, flat colour fields, confident contour, hokusai wave pattern | 1.18 |
| `cubism` | cubist, fragmented simultaneous viewpoints, faceted planes, muted analytic palette | 1.2 |
| `dada` | dada, absurd photomontage, irreverent juxtaposition, typographic fragments | 1.18 |
| `minimalism` | minimalist, radical negative space, few elements, quiet precision | 1.15 |
| `brutalism` | brutalist, raw board-marked concrete, monolithic mass, heavy shadow, austere | 1.15 |
| `gothic` | gothic, soaring pointed arches, tracery, candle gloom, solemn verticality | 1.12 |
| `cyberpunk` | cyberpunk, rain-slick neon streets, dense signage, chrome and grime, night city | 1.15 |
| `solarpunk` | solarpunk, green rooftop gardens, warm sunlight, art nouveau technology, optimistic | 1.15 |
| `dieselpunk` | dieselpunk, riveted steel, interwar machinery, soot and brass, industrial art deco | 1.15 |
| `vaporwave` | vaporwave, magenta and cyan gradient, grid horizon, roman bust, retro glitch | 1.2 |
| `psychedelia` | psychedelic, kaleidoscopic symmetry, vibrating complementary colour, liquid forms | 1.18 |
| `hyperrealism` | hyperrealist, forensic surface detail, pore-level texture, uncanny clarity | 1.15 |
| `folk` | folk art, naive flattened perspective, decorative repetition, hand-made warmth | 1.15 |
| `noir` | film noir, venetian blind shadows, wet asphalt, cigarette haze, moral gloom | 1.15 |

## Lighting

Spec field: `lighting.key` — 25 keys

| Key | Expands to | Weight |
|---|---|---|
| `golden_hour` | golden hour sunlight, long warm raking shadows, amber rim on every edge | 1.1 |
| `blue_hour` | blue hour twilight, deep cyan sky, cool ambient light, warm artificial accents | 1.1 |
| `overcast` | soft overcast daylight, huge diffuse source, gentle wraparound shadows, low contrast | 1.05 |
| `harsh_noon` | harsh midday sun, short hard shadows, blown highlights, high contrast | 1.05 |
| `rembrandt` | rembrandt lighting, single key at 45 degrees, triangle of light on the shadowed cheek | 1.15 |
| `split_light` | split lighting, hard side key, half the face in darkness, severe | 1.15 |
| `rim_light` | strong rim light, bright separation edge, subject carved out of a dark ground | 1.15 |
| `backlit` | backlit, subject against the source, glowing translucent edges, deep frontal shadow | 1.12 |
| `silhouette` | silhouette, subject rendered as pure dark shape against a luminous background | 1.2 |
| `studio_three_point` | three point studio lighting, soft key, controlled fill, clean rim, seamless backdrop | 1.1 |
| `softbox` | large softbox key, creamy gradient falloff, catchlight in the eyes, editorial | 1.1 |
| `hard_flash` | direct on-camera flash, hard shadow behind the subject, blown foreground, snapshot | 1.12 |
| `chiaroscuro` | chiaroscuro, extreme light against deep shadow, single dramatic source, caravaggio | 1.18 |
| `volumetric` | volumetric light shafts, visible beams through haze, atmospheric scattering | 1.15 |
| `god_rays` | crepuscular god rays breaking through cloud, radiant shafts, dust motes | 1.15 |
| `candlelight` | candlelight, small warm flickering source, rapid falloff, orange skin, deep gloom | 1.15 |
| `firelight` | firelight from below, dancing warm glow, upward shadows, ember sparks | 1.15 |
| `moonlight` | moonlight, cold blue key, silver highlights, deep desaturated shadow | 1.12 |
| `neon` | neon lighting, saturated magenta and cyan sources, wet reflective specular highlights | 1.15 |
| `bioluminescent` | bioluminescent glow, soft cyan-green emissive light from within the subject, dark ground | 1.18 |
| `underwater_caustics` | underwater caustics, rippling refracted light patterns, blue-green depth haze | 1.18 |
| `high_key` | high key lighting, bright airy exposure, minimal shadow, white on white | 1.15 |
| `low_key` | low key lighting, predominantly dark frame, small pools of light, moody | 1.15 |
| `practical_glow` | practical lights in frame, lamps and signage as visible sources, warm pools | 1.1 |
| `stormlight` | storm light, bruised sky, one break of sun on the subject, dramatic weather | 1.15 |

## Shot size

Spec field: `camera.shot` — 12 keys

| Key | Expands to | Weight |
|---|---|---|
| `extreme_wide` | extreme wide shot, subject small in a vast environment, establishing scale | 1.12 |
| `wide` | wide shot, full environment visible around the subject | 1.1 |
| `full_body` | full body shot, head to feet in frame | 1.12 |
| `medium` | medium shot, subject from the waist up | 1.1 |
| `close_up` | close-up, head and shoulders filling the frame | 1.12 |
| `extreme_close_up` | extreme close-up, a single detail filling the entire frame | 1.15 |
| `macro` | macro photography, life-size magnification, razor-thin plane of focus | 1.15 |
| `aerial` | aerial view from high above, drone perspective, landscape spread below | 1.15 |
| `birds_eye` | bird's eye view, looking straight down, flattened graphic composition | 1.18 |
| `worms_eye` | worm's eye view, looking steeply up, towering converging verticals | 1.18 |
| `over_shoulder` | over the shoulder framing, foreground figure out of focus | 1.1 |
| `establishing` | establishing shot, the whole location legible at once | 1.1 |

## Lens

Spec field: `camera.lens` — 11 keys

| Key | Expands to | Weight |
|---|---|---|
| `14mm` | 14mm ultra wide lens, exaggerated perspective, curved edges | 1.0 |
| `24mm` | 24mm wide angle lens, expansive framing, mild distortion | 1.0 |
| `35mm` | 35mm lens, natural reportage framing | 1.0 |
| `50mm` | 50mm lens, human-eye perspective, no distortion | 1.0 |
| `85mm` | 85mm portrait lens, flattering compression, creamy background separation | 1.0 |
| `135mm` | 135mm telephoto, strong compression, isolated subject | 1.0 |
| `200mm` | 200mm telephoto, heavily compressed planes, distant subject pulled close | 1.0 |
| `fisheye` | fisheye lens, 180 degree circular distortion | 1.0 |
| `tilt_shift` | tilt-shift lens, sliver of focus, miniature-model effect | 1.0 |
| `anamorphic` | anamorphic lens, oval bokeh, horizontal blue flare | 1.0 |
| `macro_lens` | macro lens, extreme magnification, paper-thin depth of field | 1.0 |

## Camera angle

Spec field: `camera.angle` — 6 keys

| Key | Expands to | Weight |
|---|---|---|
| `eye_level` | eye level camera, neutral and direct | 1.0 |
| `low_angle` | low camera angle looking up, subject made imposing | 1.12 |
| `high_angle` | high camera angle looking down, subject made small | 1.12 |
| `dutch` | dutch tilt, canted horizon, unease | 1.15 |
| `top_down` | top-down orthographic view, flat lay | 1.15 |
| `profile` | strict profile view, subject seen exactly from the side | 1.15 |

## Focus

Spec field: `camera.focus` — 6 keys

| Key | Expands to | Weight |
|---|---|---|
| `shallow_dof` | shallow depth of field, subject sharp against a dissolved background | 1.12 |
| `bokeh` | creamy circular bokeh, out-of-focus highlight orbs | 1.12 |
| `deep_focus` | deep focus, everything from foreground to horizon sharp | 1.12 |
| `tack_sharp` | tack sharp, crisp micro-detail, no motion blur | 1.1 |
| `soft_focus` | soft focus, gentle diffusion, glowing highlights | 1.1 |
| `motion_blur` | motion blur, long exposure streaks, sense of speed | 1.15 |

## Composition rule

Spec field: `composition.rule` — 5 keys

| Key | Expands to | Weight |
|---|---|---|
| `thirds` | rule of thirds composition, subject off-centre on a third line | 1.0 |
| `centered` | centred symmetrical composition, subject dead centre | 1.1 |
| `golden_spiral` | golden spiral composition, eye led in a curve to the subject | 1.0 |
| `symmetry` | perfect bilateral symmetry, mirrored halves | 1.15 |
| `diagonal` | strong diagonal composition, dynamic leading lines | 1.1 |

## Palettes

Spec field: `palette.name` — 17 keys

| Key | Expands to | Weight |
|---|---|---|
| `teal_amber` | teal and amber colour grade, cool shadows against warm highlights | 1.0 |
| `neon_noir` | neon noir palette, magenta and cyan against near-black, wet specular colour | 1.0 |
| `earth_tones` | earth tone palette, ochre umber and olive, muted natural pigments | 1.0 |
| `monochrome` | monochrome, pure greyscale, tonal range only | 1.0 |
| `sepia` | sepia toned, warm brown monochrome, aged print | 1.0 |
| `pastel` | soft pastel palette, chalky desaturated pinks blues and mints | 1.0 |
| `ice_blue` | cold palette, glacial blues and whites, frost and pale steel | 1.0 |
| `autumn` | autumn palette, rust burnt orange and deep gold against damp brown | 1.0 |
| `jewel_tones` | jewel tone palette, emerald sapphire and garnet, saturated and rich | 1.0 |
| `desaturated` | desaturated palette, muted low-chroma colour, restrained and grey-leaning | 1.0 |
| `candy` | bright candy palette, high chroma pinks yellows and turquoise, playful | 1.0 |
| `forest` | forest palette, deep greens moss and bark, dappled shade | 1.0 |
| `desert` | desert palette, sand terracotta and pale sky, sun-bleached | 1.0 |
| `cyber_magenta` | cyber palette, electric magenta and acid green on charcoal | 1.0 |
| `high_contrast_bw` | stark black and white, no midtones, graphic contrast | 1.0 |
| `gameboy` | four-tone green monochrome palette, handheld lcd look | 1.0 |
| `ember` | ember palette, black ground with molten orange and red glow | 1.0 |

## Contrast

Spec field: `palette.contrast` — 3 keys

| Key | Expands to | Weight |
|---|---|---|
| `low` | low contrast, compressed tonal range, soft and hazy | 1.0 |
| `normal` | — | 1.0 |
| `high` | high contrast, deep blacks and bright highlights, punchy | 1.1 |

## Detail level

Spec field: `style.detail` — 4 keys

| Key | Expands to | Weight |
|---|---|---|
| `minimal` | minimal detail, simplified shapes, restrained | 1.0 |
| `moderate` | moderate detail | 1.0 |
| `high` | highly detailed, rich surface texture | 1.1 |
| `intricate` | intricate detail, dense ornamental complexity, rewarding close inspection | 1.15 |

## Negative presets

Spec field: `negative.presets[]` — 16 keys

| Key | Expands to | Weight |
|---|---|---|
| `artifacts` | jpeg artifacts, compression noise, banding, moire, oversharpened halos | 1.0 |
| `lowres` | low resolution, blurry, out of focus, pixelated, upscaling artifacts | 1.0 |
| `anatomy` | deformed hands, extra fingers, fused fingers, extra limbs, malformed face, crossed eyes | 1.0 |
| `text` | text, letters, words, watermark, signature, caption, logo, subtitles | 1.0 |
| `watermark` | watermark, stock photo watermark, signature, copyright notice | 1.0 |
| `photo_flaws` | overexposed, underexposed, harsh flash, chromatic aberration, lens dirt, tilted horizon | 1.0 |
| `painting_flaws` | muddy colour, flat lighting, unfinished, sketchy underdrawing showing through | 1.0 |
| `render_flaws` | clipping geometry, z-fighting, untextured, default grey material, aliased edges | 1.0 |
| `clutter` | cluttered background, busy composition, distracting elements, visual noise | 1.0 |
| `amateur` | amateur, poorly composed, awkward crop, cheap looking | 1.0 |
| `cgi` | cgi, 3d render, video game screenshot, plastic skin, uncanny valley | 1.0 |
| `cartoon` | cartoon, anime, illustration, drawing, painting | 1.0 |
| `photographic` | photograph, photorealistic, dslr, camera | 1.0 |
| `smooth` | smooth gradients, airbrushed, anti-aliased, soft blur | 1.0 |
| `frame` | picture frame, border, matte, passepartout, torn edges | 1.0 |
| `duplicates` | duplicate subject, cloned elements, repeated faces, mirrored artifacts | 1.0 |

## Palette colours

Used by `post.palette_lock` and available as explicit hex in
`palette.colors`.

| Palette | Colours |
|---|---|
| `teal_amber` | `#0b3d47` `#12707f` `#e0a458` `#f6e7cb` `#08161a` |
| `neon_noir` | `#0a0612` `#2d1b4e` `#ff2e88` `#00e5ff` `#f2f0ff` |
| `earth_tones` | `#3d2f22` `#7a5c3e` `#b08954` `#9aa07a` `#e8dcc4` |
| `monochrome` | `#000000` `#3a3a3a` `#767676` `#b4b4b4` `#ffffff` |
| `sepia` | `#1c1208` `#4a3520` `#8a6a44` `#c9a878` `#f0e2cc` |
| `pastel` | `#f7d6e0` `#c9e4de` `#f2e8cf` `#c6def1` `#faedcb` |
| `ice_blue` | `#0c2233` `#1d4e6b` `#5b9bb5` `#a8d4e0` `#f2fbff` |
| `autumn` | `#2b1a12` `#7c3a12` `#c1651c` `#e0a42e` `#6b6b3a` |
| `jewel_tones` | `#0f2a23` `#0d5c46` `#123a6b` `#7a1230` `#d4af37` |
| `desaturated` | `#2e2f33` `#55585e` `#83868c` `#a9a49c` `#d6d2cb` |
| `candy` | `#ff5d8f` `#ffd166` `#06d6a0` `#118ab2` `#fff3f8` |
| `forest` | `#0f1c14` `#1f3d22` `#3f6b3a` `#7fa05a` `#d9d6a1` |
| `desert` | `#e6cfa8` `#c98d5f` `#8c4a2f` `#7fa8b5` `#f5efe2` |
| `cyber_magenta` | `#101014` `#2a0e3a` `#ff00a0` `#9dff00` `#e8e8f0` |
| `high_contrast_bw` | `#000000` `#ffffff` |
| `gameboy` | `#0f380f` `#306230` `#8bac0f` `#9bbc0f` |
| `ember` | `#0a0503` `#3d1108` `#a32a0d` `#f2711c` `#ffd08a` |
