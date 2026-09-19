# Compile every spec, no GPU (1 minute)
(.venv) PS C:\Users\Hugo\Desktop\Mis proyectos\Juegos y gráficos\ClauDali> Get-ChildItem tests\manual\*.json | ForEach-Object {
>>   "== $($_.Name)"
>>   .venv\Scripts\python -m claudali compile $_.FullName | Select-String "^(quality|size|warning|note)"
>> }
== decode-auto.json

size     : 1344x768
quality  : standard, float16, VAE decode auto, stages base
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== decode-cpu.json
size     : 1344x768
quality  : standard, float16, VAE decode cpu, stages base
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== decode-gpu-tiled.json
size     : 1344x768
quality  : standard, float16, VAE decode gpu_tiled, stages base
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== fairies-max.json
size     : 1344x768
quality  : max, float32, VAE decode cpu, stages base -> refiner -> hires
warning: the refiner was trained on SDXL base's own outputs and 'juggernaut-xl' is a fine-tune, so the refiner may wash out its look. Compare against 
refiner.enabled false, or render.model 'sdxl-base'.
note   : render.quality 'max' turned on fp32 precision, a forced CPU fp32 untiled VAE decode, the SDXL refiner and a 1.5x hi-res pass to 2016x1152. Setting 
any of them explicitly opts out of it. Expect several times the standard render time.
== fairies-standard.json
size     : 1344x768
quality  : standard, float16, VAE decode auto, stages base
== fp16-small.json
size     : 768x768
quality  : standard, float16, VAE decode cpu, stages base
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== fp32-small.json
size     : 768x768
quality  : standard, float32, VAE decode cpu, stages base
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== hires-lanczos.json
size     : 1344x768
quality  : standard, float16, VAE decode gpu_tiled, stages base -> hires
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== hires-off.json
size     : 1344x768
quality  : standard, float16, VAE decode gpu_tiled, stages base
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== hires-realesrgan.json
size     : 1344x768
quality  : standard, float16, VAE decode gpu_tiled, stages base -> hires
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== pause-hires.json
size     : 768x768
quality  : standard, float16, VAE decode gpu_tiled, stages base -> hires
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== pause-refiner.json
size     : 768x768
quality  : standard, float16, VAE decode gpu_tiled, stages base -> refiner
note   : intent 'photoreal' would add the 'anatomy' negatives, but the subject names no person, so they were left out. Add 'anatomy' to negative.presets to 
force them.
== refiner-off.json
size     : 1024x1024
quality  : standard, float16, VAE decode gpu_tiled, stages base
== refiner-on.json
size     : 1024x1024
quality  : standard, float16, VAE decode gpu_tiled, stages base -> refiner
== ui-roundtrip.json
size     : 1216x832
quality  : max, float16, VAE decode gpu_tiled, stages base -> hires
note   : render.quality 'max' turned on a 1.25x hi-res pass to 1520x1040. Setting any of them explicitly opts out of it. Expect several times the standard 
render time.

# 2. Web UI (20 minutes)

## 2.1 The page loads.

The websites loads properly and the quality sections seems as expected.

<fieldset>
      <legend>Quality</legend>
      <label>Preset</label>
      <select id="quality">
        <option value="">standard</option>
        <option value="max">max: fp32, CPU fp32 decode, refiner, hi-res (very slow)</option>
      </select>
      <label>VAE decode</label>
      <select id="vae-decode">
        <option value="">server default (auto)</option>
        <option value="auto">Auto: CPU fp32 untiled if RAM allows, else GPU tiled</option>
        <option value="cpu">Force CPU fp32 untiled (best quality; may swap and take several minutes)</option>
        <option value="gpu">GPU untiled</option>
        <option value="gpu_tiled">GPU tiled (fastest, lowest memory)</option>
      </select>
      <div id="decode-hint" class="muted" style="font-size:11.5px;margin-top:4px">1024×1024 needs ~6.9 GB of free RAM on the CPU, 7.1 GB free: auto decodes on the CPU in fp32 untiled right now</div>
      <label>Precision</label>
      <select id="precision">
        <option value="">server default (float16)</option>
        <option value="fp16">fp16</option>
        <option value="fp32">fp32 (~14 GB of RAM, sequential offload on 6 GB; unmeasured)</option>
      </select>
      <div class="row">
        <div><label>Refiner</label>
          <select id="refiner"><option value="">by preset</option><option value="true">on</option><option value="false">off</option></select></div>
        <div><label>Refiner handoff</label><input id="refiner-handoff" type="number" step="0.05" min="0.05" max="0.95" placeholder="0.8"></div>
      </div>
      <div class="row">
        <div><label>Hi-res pass</label>
          <select id="hires"><option value="">by preset</option><option value="true">on</option><option value="false">off</option></select></div>
        <div><label>Upscaler</label>
          <select id="hires-upscaler"><option value="">by preset</option><option value="lanczos">Lanczos</option><option value="realesrgan-x4">Real-ESRGAN x4</option></select></div>
      </div>
      <div class="row">
        <div><label>Hi-res scale</label><input id="hires-scale" type="number" step="0.05" min="1.05" max="2" placeholder="1.5"></div>
        <div><label>Hi-res strength</label><input id="hires-strength" type="number" step="0.05" min="0.05" max="1" placeholder="0.3"></div>
      </div>
    </fieldset>

## 2.2 The decode hint follows the form.

Tested and works accordingly to the description given.

## 2.3 The max preset.

Tested and works accordingly to the description given.

## 2.4 Compile only
Seems to be what was expected:
<div class="card" id="card-status">
  <h2>Status</h2>
  <div id="status-text" class="muted">Compiled</div>
  <div class="bar"><div id="progress" style="width: 0%;"></div></div>
  <div id="status-detail" class="muted" style="font-size:12px">juggernaut-xl · 1344×768 · 32 steps · cfg 5.5 · float32 · decode cpu · base → refiner → hires</div>
  <ul class="err-list hidden" id="mismatch-list"></ul>
  <div class="job-actions hidden" id="job-actions">
    <button id="btn-pause" title="Stop at the end of the current step; the next queued job runs">Pause (next job runs)</button>
    <button id="btn-pause-hold" title="Stop at the end of the current step and start nothing else until this job resumes">Pause &amp; hold queue</button>
    <button id="btn-resume" class="primary">Resume</button>
    <button id="btn-resume-force" title="Continue although what shapes the pixels changed; the bundle notes record it">Resume anyway</button>
    <button id="btn-cancel">Cancel job</button>
  </div>
</div>

## 2.5 Show JSON
I truly don't know if this was the desired output:
<section id="form">
  <fieldset>
    <legend>Intent</legend>
    <select id="intent"><option value="photoreal">photoreal</option><option value="painterly">painterly</option><option value="game_asset">game asset</option><option value="render3d">render3d</option><option value="graphic">graphic</option></select>
    <div class="muted" id="intent-desc" style="font-size:11.5px;margin-top:6px">Photographs of things that could exist: objects, people, places. → juggernaut-xl, cfg 5.5, 32 steps</div>
  </fieldset>

  <fieldset>
    <legend>Subject</legend>
    <label>Primary</label>
    <textarea id="subject" placeholder="a lighthouse on a black basalt cliff"></textarea>
    <label>Anchor (the subject alone, restated in every CLIP chunk)</label>
    <input id="anchor" placeholder="the lighthouse">
    <label>Details (comma separated)</label>
    <input id="details" placeholder="salt-worn paint, one lit window">
    <label>Action</label>
    <input id="action" placeholder="beam sweeping across the water">
  </fieldset>

  <fieldset>
    <legend>Scene</legend>
    <label>Setting</label>
    <input id="setting" placeholder="north atlantic coast">
    <div class="row">
      <div><label>Time</label><input id="time" placeholder="blue hour"></div>
      <div><label>Weather</label><input id="weather" placeholder="storm approaching"></div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Style</legend>
    <div class="row">
      <div><label>Medium</label><select id="medium"><option value="">—</option><option value="acrylic">acrylic</option><option value="airbrush">airbrush</option><option value="blueprint">blueprint</option><option value="cel_shaded">cel shaded</option><option value="charcoal">charcoal</option><option value="cinematic_still">cinematic still</option><option value="clay_render">clay render</option><option value="collage">collage</option><option value="comic">comic</option><option value="concept_art">concept art</option><option value="digital_painting">digital painting</option><option value="engraving">engraving</option><option value="film_35mm">film 35mm</option><option value="gouache">gouache</option><option value="ink_linework">ink linework</option><option value="isometric">isometric</option><option value="low_poly">low poly</option><option value="matte_painting">matte painting</option><option value="mosaic">mosaic</option><option value="oil_classical">oil classical</option><option value="oil_impasto">oil impasto</option><option value="pastel">pastel</option><option value="photograph">photograph</option><option value="pixel_art">pixel art</option><option value="polaroid">polaroid</option><option value="render3d">render3d</option><option value="stained_glass">stained glass</option><option value="tapestry">tapestry</option><option value="vector_flat">vector flat</option><option value="voxel">voxel</option><option value="watercolor">watercolor</option><option value="woodcut">woodcut</option></select></div>
      <div><label>Movement</label><select id="movement"><option value="">—</option><option value="art_deco">art deco</option><option value="art_nouveau">art nouveau</option><option value="baroque">baroque</option><option value="bauhaus">bauhaus</option><option value="brutalism">brutalism</option><option value="constructivism">constructivism</option><option value="cubism">cubism</option><option value="cyberpunk">cyberpunk</option><option value="dada">dada</option><option value="dieselpunk">dieselpunk</option><option value="expressionism">expressionism</option><option value="folk">folk</option><option value="gothic">gothic</option><option value="hyperrealism">hyperrealism</option><option value="impressionism">impressionism</option><option value="magic_realism">magic realism</option><option value="minimalism">minimalism</option><option value="noir">noir</option><option value="post_impressionism">post impressionism</option><option value="psychedelia">psychedelia</option><option value="romanticism">romanticism</option><option value="solarpunk">solarpunk</option><option value="surrealism">surrealism</option><option value="symbolism">symbolism</option><option value="ukiyo_e">ukiyo e</option><option value="vaporwave">vaporwave</option></select></div>
    </div>
    <label>Descriptors (comma separated)</label>
    <input id="descriptors" placeholder="weathered, monumental">
  </fieldset>

  <fieldset>
    <legend>Light &amp; colour</legend>
    <div class="row">
      <div><label>Lighting</label><select id="lighting"><option value="">—</option><option value="backlit">backlit</option><option value="bioluminescent">bioluminescent</option><option value="blue_hour">blue hour</option><option value="candlelight">candlelight</option><option value="chiaroscuro">chiaroscuro</option><option value="firelight">firelight</option><option value="god_rays">god rays</option><option value="golden_hour">golden hour</option><option value="hard_flash">hard flash</option><option value="harsh_noon">harsh noon</option><option value="high_key">high key</option><option value="low_key">low key</option><option value="moonlight">moonlight</option><option value="neon">neon</option><option value="overcast">overcast</option><option value="practical_glow">practical glow</option><option value="rembrandt">rembrandt</option><option value="rim_light">rim light</option><option value="silhouette">silhouette</option><option value="softbox">softbox</option><option value="split_light">split light</option><option value="stormlight">stormlight</option><option value="studio_three_point">studio three point</option><option value="underwater_caustics">underwater caustics</option><option value="volumetric">volumetric</option></select></div>
      <div><label>Palette</label><select id="palette"><option value="">—</option><option value="autumn">autumn</option><option value="candy">candy</option><option value="cyber_magenta">cyber magenta</option><option value="desaturated">desaturated</option><option value="desert">desert</option><option value="earth_tones">earth tones</option><option value="ember">ember</option><option value="forest">forest</option><option value="gameboy">gameboy</option><option value="high_contrast_bw">high contrast bw</option><option value="ice_blue">ice blue</option><option value="jewel_tones">jewel tones</option><option value="monochrome">monochrome</option><option value="neon_noir">neon noir</option><option value="pastel">pastel</option><option value="sepia">sepia</option><option value="teal_amber">teal amber</option></select></div>
    </div>
    <div class="row">
      <div><label>Contrast</label><select id="contrast"><option value="">—</option><option value="high">high</option><option value="low">low</option><option value="normal">normal</option></select></div>
      <div><label>Detail</label><select id="detail"><option value="">—</option><option value="high">high</option><option value="intricate">intricate</option><option value="minimal">minimal</option><option value="moderate">moderate</option></select></div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Camera</legend>
    <div class="row">
      <div><label>Shot</label><select id="shot"><option value="">—</option><option value="aerial">aerial</option><option value="birds_eye">birds eye</option><option value="close_up">close up</option><option value="establishing">establishing</option><option value="extreme_close_up">extreme close up</option><option value="extreme_wide">extreme wide</option><option value="full_body">full body</option><option value="group">group</option><option value="macro">macro</option><option value="medium">medium</option><option value="over_shoulder">over shoulder</option><option value="scene">scene</option><option value="tableau">tableau</option><option value="tight">tight</option><option value="wide">wide</option><option value="worms_eye">worms eye</option></select></div>
      <div><label>Lens</label><select id="lens"><option value="">—</option><option value="135mm">135mm</option><option value="14mm">14mm</option><option value="200mm">200mm</option><option value="24mm">24mm</option><option value="35mm">35mm</option><option value="50mm">50mm</option><option value="85mm">85mm</option><option value="anamorphic">anamorphic</option><option value="fisheye">fisheye</option><option value="macro_lens">macro lens</option><option value="tilt_shift">tilt shift</option></select></div>
    </div>
    <div class="row">
      <div><label>Angle</label><select id="angle"><option value="">—</option><option value="dutch">dutch</option><option value="eye_level">eye level</option><option value="high_angle">high angle</option><option value="low_angle">low angle</option><option value="profile">profile</option><option value="top_down">top down</option></select></div>
      <div><label>Focus</label><select id="focus"><option value="">—</option><option value="bokeh">bokeh</option><option value="deep_focus">deep focus</option><option value="motion_blur">motion blur</option><option value="shallow_dof">shallow dof</option><option value="soft_focus">soft focus</option><option value="tack_sharp">tack sharp</option></select></div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Composition</legend>
    <div class="row">
      <div><label>Aspect</label><select id="aspect"><option value="1:1">1:1</option><option value="5:4">5:4</option><option value="4:5">4:5</option><option value="3:2">3:2</option><option value="2:3">2:3</option><option value="16:9">16:9</option><option value="9:16">9:16</option><option value="21:9">21:9</option><option value="9:21">9:21</option></select></div>
      <div><label>Rule</label><select id="rule"><option value="">—</option><option value="centered">centered</option><option value="diagonal">diagonal</option><option value="golden_spiral">golden spiral</option><option value="symmetry">symmetry</option><option value="thirds">thirds</option></select></div>
    </div>
    <label>Horizon (blank = none, 0 top → 1 bottom)</label>
    <input id="horizon" type="number" step="0.01" min="0" max="1" placeholder="0.62">
    <label>Control mode</label>
    <div class="row">
      <select id="control-mode">
        <option value="none">none</option>
        <option value="depth">depth</option>
        <option value="canny">canny</option>
        <option value="scribble">scribble</option>
      </select>
      <input id="control-strength" type="number" step="0.05" min="0" max="2" value="0.6" title="ControlNet strength">
    </div>
    <details style="margin-top:8px">
      <summary>Layers (JSON) — drives the procedural control map</summary>
      <textarea id="layers" placeholder="[{&quot;role&quot;:&quot;subject&quot;,&quot;shape&quot;:&quot;column&quot;,&quot;bbox&quot;:[0.6,0.2,0.75,0.7],&quot;depth&quot;:0.8}]"></textarea>
    </details>
  </fieldset>

  <fieldset>
    <legend>Render</legend>
    <label>Model</label>
    <select id="model"><option value="">auto (by intent)</option><option value="sdxl-base">sdxl-base</option><option value="juggernaut-xl">juggernaut-xl</option><option value="dreamshaper-xl">dreamshaper-xl</option></select>
    <div class="row-3">
      <div><label>Steps</label><input id="steps" type="number" min="1" max="150" placeholder="auto"></div>
      <div><label>CFG</label><input id="cfg" type="number" step="0.5" min="0" max="30" placeholder="auto"></div>
      <div><label>Variations</label><input id="variations" type="number" min="1" max="16" value="1"></div>
    </div>
    <div class="row">
      <div><label>Sampler</label><select id="sampler"><option value="">auto</option><option value="ddim">ddim</option><option value="dpmpp_2m">dpmpp 2m</option><option value="dpmpp_2m_karras">dpmpp 2m karras</option><option value="dpmpp_sde_karras">dpmpp sde karras</option><option value="euler">euler</option><option value="euler_a">euler a</option><option value="heun">heun</option><option value="unipc">unipc</option></select></div>
      <div><label>Seed (blank = random)</label><input id="seed" type="number" min="0"></div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Quality</legend>
    <label>Preset</label>
    <select id="quality">
      <option value="">standard</option>
      <option value="max">max: fp32, CPU fp32 decode, refiner, hi-res (very slow)</option>
    </select>
    <label>VAE decode</label>
    <select id="vae-decode">
      <option value="">server default (auto)</option>
      <option value="auto">Auto: CPU fp32 untiled if RAM allows, else GPU tiled</option>
      <option value="cpu">Force CPU fp32 untiled (best quality; may swap and take several minutes)</option>
      <option value="gpu">GPU untiled</option>
      <option value="gpu_tiled">GPU tiled (fastest, lowest memory)</option>
    </select>
    <div id="decode-hint" class="muted" style="font-size:11.5px;margin-top:4px">1024×1024 needs ~6.9 GB of free RAM on the CPU, 5 GB free: auto falls back to GPU tiled right now. Force CPU for the untiled decode.</div>
    <label>Precision</label>
    <select id="precision">
      <option value="">server default (float16)</option>
      <option value="fp16">fp16</option>
      <option value="fp32">fp32 (~14 GB of RAM, sequential offload on 6 GB; unmeasured)</option>
    </select>
    <div class="row">
      <div><label>Refiner</label>
        <select id="refiner"><option value="">by preset</option><option value="true">on</option><option value="false">off</option></select></div>
      <div><label>Refiner handoff</label><input id="refiner-handoff" type="number" step="0.05" min="0.05" max="0.95" placeholder="0.8"></div>
    </div>
    <div class="row">
      <div><label>Hi-res pass</label>
        <select id="hires"><option value="">by preset</option><option value="true">on</option><option value="false">off</option></select></div>
      <div><label>Upscaler</label>
        <select id="hires-upscaler"><option value="">by preset</option><option value="lanczos">Lanczos</option><option value="realesrgan-x4">Real-ESRGAN x4</option></select></div>
    </div>
    <div class="row">
      <div><label>Hi-res scale</label><input id="hires-scale" type="number" step="0.05" min="1.05" max="2" placeholder="1.5"></div>
      <div><label>Hi-res strength</label><input id="hires-strength" type="number" step="0.05" min="0.05" max="1" placeholder="0.3"></div>
    </div>
  </fieldset>

  <div class="actions">
    <button id="btn-compile">Compile only</button>
    <button id="btn-control">Preview control</button>
    <button id="btn-render" class="primary wide">Render</button>
    <button id="btn-example">Load example</button>
    <button id="btn-json">Show JSON</button>
  </div>
  <div id="also-in-spec" class="muted hidden" style="font-size:11.5px;margin-top:10px"></div>

  <details style="margin-top:12px" open="">
    <summary>Raw spec JSON (editable — Use this JSON renders it as written)</summary>
    <textarea id="json-spec" style="min-height:220px;margin-top:8px"></textarea>
    <button id="btn-apply-json" style="margin-top:6px">Use this JSON</button>
  </details>
</section>

## 2.6 Use this JSON
Tested it and seems perfect.

## ...

## 2.10 History
Looks perfect to me!
