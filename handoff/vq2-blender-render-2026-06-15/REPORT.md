# VQ2 Blender render — validation + pilot (Worker REPORT)

**Session:** P5 VISION-DETECTOR — Worker (VQ2 Blender render: validate + pilot dataset)
**Machine:** ShadowPC — Blender **5.1.2**, bundled Python **3.13.9**, GPU **NVIDIA RTX 2000 Ada** (OptiX/CUDA)
**Date:** 2026-06-15
**Branch:** `vq2-blender-render-2026-06-15`

---

## HEADLINE

**BlenderBackend renders verified-correct.** The bpy/Cycles backend — previously DOC-VERIFIED but
NEVER RENDER-VERIFIED — now renders correctly-labeled frames on real hardware:

- **Intrinsics self-check: 0.0000 px** (max reprojection 3.6e-05 px, gate ≤1 px) on every run
  (vq1_faithful EEVEE, appearance_broad Cycles). The Blender camera reproduces `K` essentially exactly.
- **Overlay + PnP: labels land on the rendered gate's inner corners.** The 4 keypoints (canonical order
  0=LL/1=LR/2=UR/3=UL) sit on the inner-square corners across ranges and perspective tilt; the deployed
  PnP estimator recovers sane ranges (4.2–28 m, in the 2–30 m envelope) from the on-disk labels.

Three real **bugs/version-gaps** were found by actually running it (none catchable by the laptop suite,
which never imports the bpy leaves) and fixed — see below.

---

## PHASE 0 — RENDER VALIDATION ✅

### Setup that worked (full steps in `src/racer/vision/blender_gen/RUN_GUIDE.md`)
1. Blender 5.1.2 already installed at `C:\Program Files\Blender Foundation\Blender 5.1\`.
2. `blender_python -m pip install --user opencv-python-headless scipy albumentations` (numpy ships with
   Blender). Lands in `…\AppData\Roaming\Python\Python313\site-packages`.
3. **Isolated-Python gotcha:** Blender runs embedded Python in isolated mode (`sys.flags.isolated==1`,
   `no_user_site==1`) → it **strips PYTHONPATH and disables user-site**, so pip'd deps are invisible
   (`No module named 'cv2'`). Fix: set `PYTHONPATH` to that site-packages dir; `render_entry.py` now
   re-honours `os.environ['PYTHONPATH']` (still readable in isolated mode) onto `sys.path`. No machine
   path is hard-coded.

### Bugs found by render-verification (fixed)
1. **`bpy_camera.py` wrong relative imports** (`from .. import contract` / `from ..intrinsics …`) —
   resolved to `racer.vision` instead of `racer.vision.blender_gen`, so the backend **crashed on
   construction** (`ImportError: cannot import name 'contract'`). Every sibling leaf uses single-dot.
   The laptop suite never caught it because it never imports `bpy_camera` (top-level `import bpy` fails
   first on a Blender-less box). **Fixed to single-dot.** → *This was the load-bearing find: the camera
   leaf could not even load before this.*
2. **EEVEE engine id drift**: 4.2 = `BLENDER_EEVEE_NEXT`; **5.x renamed it `BLENDER_EEVEE`** (no `_NEXT`
   in the enum). `--eevee`/preset hardcodes the 4.2 id → `enum "BLENDER_EEVEE_NEXT" not found`. Added
   `bpy_render._resolve_engine` to map an unavailable EEVEE id onto whatever EEVEE the build exposes.
3. **Compositor output node removed in 5.x**: glare/bloom used `CompositorNodeComposite`, which **5.x
   deleted** (`Node type … undefined`) → render crash. Glare is non-load-bearing (never touches geometry
   or labels); made the whole glare build fall back to `NodeGroupOutput` and otherwise degrade to "no
   bloom" instead of aborting. Bloom/lens-flare now works on 5.1 (visible in samples).

### Evidence (committed under `handoff/vq2-blender-render-2026-06-15/samples/`)
- `eevee_vq1_faithful/` — faithful VQ1 orange-red gate, keypoints on inner corners, PnP 4.2 m / 10.4 m.
- `cycles_appearance_broad/` — photoreal, hue-randomized gates (teal/green — colour-as-nuisance working),
  dark-arena + coloured lighting, clutter confusers, lens flare, motion blur; overlays + masks.

---

## PHASE 1 — PILOT DATASET + MASKS

### Segmentation masks (banked hedge) — added, tested, green
New pure-Python `masks.py`: per-frame **gate-ring instance mask** rasterized from the already-known
projected OUTER + INNER polygons (the annulus = `fillPoly(outer) AND NOT fillPoly(inner)`, painted
far→near so the z-order matches the render; each gate carves its own opening so see-through gates are
preserved). Built from the **post-augment** labelled gates, so masks stay aligned with geometric warps
and consistent with the keypoint rows. Wired behind `generate_dataset(emit_masks=True)` /
`render_entry --masks` → `masks/<split>/*.png`. Verified on the laptop ProceduralBackend AND through the
Blender backend (annulus aligns with the rendered gate; opening is background). **Laptop suite: 30 passed**
(was 28; +`test_gate_ring_mask_is_annulus_under_keypoints`, +`test_generate_dataset_emits_masks`).

### Overlay/validation tool — added (`viz_overlay.py`)
Reusable, pure-Python (`-m racer.vision.blender_gen.viz_overlay`): denormalizes YOLO-pose rows, draws
bbox + colour-coded corners + skeleton, runs `estimate_gate_pose` per 4-visible gate, writes `*_overlay.png`
and a range summary. This is the arbiter used for the Phase 0 visual check; reusable on any dataset dir.

### Render throughput (measured, RTX 2000 Ada, 640×360, per-frame rebuild + augment + masks)
| engine | samples | sec/frame | frames/hr |
|--------|---------|-----------|-----------|
| **Cycles (GPU, OptiX denoise)** | 96 | **~2.4** | **~1500** |
| EEVEE (preview) | 96 TAA | ~2 (warm) | preview-fast |

### Pilot composition + location
Rendering with **Cycles GPU** (throughput practical — no EEVEE fallback needed), distinct seeds, `--masks`:
| arm | preset | train | val | seed |
|-----|--------|-------|-----|------|
| appearance_broad | appearance_broad | 4000 | 400 | 0 |
| negatives | negatives | 2000 | 200 | 101 |
| terminal_approach | terminal_approach | 1000 | 100 | 202 |

- **Location (LOCAL, gitignored):** `handoff/vq2-blender-render-2026-06-15/pilot/<arm>/`
- **Driver + log:** `run_pilot.ps1`, `pilot_render.log` (per-arm timing + frames/hr).
- **Status:** _launched in background; ~7.7k frames ≈ ~5 h at ~1500 fph. Final per-arm counts/throughput
  appended on completion._  <!-- PILOT_STATUS -->

Reproduce: see RUN_GUIDE §3 (one `render_entry` invocation per arm).

---

## DEFERRED (named, not done — per brief)
Full recipe mix (45/20/25/5/3/2); dedicated dark-arena lighting pass; HDRI fetcher (use_hdri=true but
`hdri_dir=null` → sky fallback today); rotational blur / vignette / CA / flash; the Adroit YOLO-pose
train (`cluster/yolo_train_v3.sbatch`) on a merged `data.yaml`; the 40-real-frame sim-to-real eval.

## NEW PARKED ITEMS (surface up)
- **Glare on 5.x is wired to `NodeGroupOutput` but not validated as feeding the render result** — bloom is
  visible in samples, so it works, but the compositor-group output path on 5.x deserves a deliberate check
  before relying on glare quantitatively. Low priority (cosmetic).
- **`hdri_dir=null`** → appearance_broad's `use_hdri=true` silently falls back to procedural sky; a real
  HDRI set would widen the domain. (Deferred HDRI fetcher.)
- **opencv-python-headless** chosen (no GUI libs needed for `--background`); note if a future viz step
  needs full `opencv-python`.

---

# UPDATE — Photoreal productionization (same session, post-pilot)

The pilot look (flat procedural sky + primitive box clutter) was rejected as not photoreal. Rebuilt the
appearance around **real CC0 assets**, validated, and productionized into the generator. The verified
**label/intrinsics path was never touched** — re-validated at every step.

### The new look (validated, signed off)
- **HDRI environments** (Polyhaven CC0): image-based lighting + real photographic backgrounds.
- **PBR floor** (concrete / asphalt pit-lane / brushed concrete), placed BELOW the flight line so **gates
  float in the air** with ground visible below (matches real racing).
- **Real photoscanned props** (boxes, chairs, tables, barrels, crates, jerrycans, plants, tyre…) +
  **procedural clothing-tinted "mannequin" people**, scattered OFF the gate corridor (sides/background).
- **Plain solid vivid gate** (VQ1-red or hue-randomized), small emission lift for brightness, true
  dimensions. Clean renders — **no in-render motion blur / glare** (those are albumentations' job).

### Real bugs found + fixed (each by actually rendering)
1. **HDRI orientation**: equirect HDRIs assume up=+Z but our world is the optical frame (up=−Y), so the
   zenith rendered forward and **every back wall was a ceiling**. Fixed with a −90° X env-mapping rotation.
2. **Frame-drop ~90%**: the occlusion raycast counted a gate hitting its OWN angled frame as occlusion →
   off-axis gates self-dropped. Fixed to only count PROP/PERSON hits. 16/16 frames now kept.
3. **Throughput ~25 s/frame → ~2 s/frame**: props were re-parsed from glTF every frame; now imported ONCE
   and spawned as mesh-sharing duplicates (PropCache). ~1500 frames/hr.

### Dimensions + intrinsics RE-VERIFIED (it's a training set)
- Camera vs K: **0.000036 px** (fx=fy=320, cx=320, cy=180, 640×360).
- Gate **true to spec**: inner 1.5 m / outer 2.72 m / depth 0.26 m. Analytically-projected corners land
  on the rendered edges at 3/5/10 m; projected widths == `fx·size/range` to the pixel (proof:
  `preview/21_verify_dims_*`).

### 8-keypoint labels (inner + outer)
YOLO-pose now emits **8 keypoints**: inner 0..3 (gate opening / PnP) then outer 4..7 (gate frame).
`kpt_shape [8,3]`, `flip_idx [1,0,3,2,5,4,7,6]`. Inner 4 byte-identical to the old block (PnP unchanged).
**Forks the keypoint contract from the 4-kpt synthetic/VQ1 writer** — flagged.

### Assets + datasets
- `assets.py`: CC0 fetcher (`python -m racer.vision.blender_gen.assets`) + `AssetLibrary` registry.
  Local under `assets_vq2/` (gitignored). RUN_GUIDE §7.
- **Base set: 2000 clean images** (`--no-augment`, 8-kpt + masks) = all-hue `appearance_broad` 1400 (incl.
  ~20% negatives) + VQ1-red `vq1_faithful` 600. Location `handoff/.../sets/{allhue,vq1red}/` (gitignored).
  Albumentations multiply over these is the agreed NEXT step (deferred by request).

### Tests: 31 passed (+8-keypoint emission, +mask tests). Sign-off look in `preview/16..20_combinedV2_*`.

## NEW PARKED (surface up)
- **People are procedural mannequins** (no CC0 humans exist on Polyhaven). Fine at distance; a real human
  asset pack would improve close-ups.
- **Outer-corner visibility is in-frame only** (not prop-occlusion-tested like the inner PnP corners).
- **Albumentations multiplier** (base→N variants, keypoint-aware, regen masks/bbox) — agreed next step.
- **Thinner-gate question**: visual outer is the true 2.72 m (matches the bbox label); a thinner look would
  require changing the labeled outer too — kept TRUE.
