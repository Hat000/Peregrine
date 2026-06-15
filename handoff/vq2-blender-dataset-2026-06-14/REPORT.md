# VQ2 photoreal Blender dataset generator — build report (2026-06-14)

**Branch:** `vq2-blender-dataset` · **Session:** VQ2 photoreal Blender dataset generator (laptop
authoring; runs on ShadowPC). Deliverable = runnable, parameterized photoreal gate-course dataset
generator producing YOLO-pose training data with appearance domain-randomization for VQ2
generalization.

## What shipped
A modular generator under `src/racer/vision/blender_gen/` — the **photoreal upgrade to
`racer.vision.synthetic`**, same label contract, real Cycles pixels + wide domain randomization.

```
blender_gen/
  contract.py     FROZEN SSOT: K, keypoint order, gate geom, VQ1 gate red, label fmt, frame xforms
  intrinsics.py   K <-> Blender (lens/sensor/fit) math + reproduction check        [pure, tested]
  geometry.py     camera viewpoints on the REAL course -> per-gate optical poses    [pure, tested]
  labels.py       YOLO-pose row writer (byte-identical to synthetic.py)             [pure, tested]
  augment.py      keypoint-AWARE albumentations pipeline + OpenCV fallback          [pure, tested]
  config.py       scenario-preset schema + strict validation + loader              [pure, tested]
  dataset.py      RenderBackend protocol + ProceduralBackend + YOLO writer          [pure, tested]
  backends/blender.py   bpy/Cycles BlenderBackend (composes the 4 bpy leaves)       [ShadowPC]
  bpy_camera.py   reproduce K, place camera (optical-frame), self-check             [ShadowPC, bpy]
  bpy_scene.py    gate ring mesh + optical-pose placement + background              [ShadowPC, bpy]
  bpy_materials.py gate/bg materials, lighting, world/HDRI (appearance DR)          [ShadowPC, bpy]
  bpy_render.py   Cycles/Eevee, motion blur, glare/exposure, render->BGR            [ShadowPC, bpy]
  render_entry.py `blender --background --python` entrypoint (intrinsics self-check)
  presets/*.json  vq1_faithful, appearance_broad, hard_visual, long_range, terminal_approach
scripts/gen_blender_dataset.py       laptop launcher (procedural backend)
scripts/visualize_blender_labels.py  label-overlay verify tool (procedural OR Blender output)
tests/test_blender_gen.py            23 laptop tests (no Blender)
handoff/vq2-blender-dataset-2026-06-14/{REPORT,RUN_GUIDE,DATASET_CONTRACT}.md
```

## Key design decisions (why the labels are provably correct)
1. **Optical-frame rendering.** Each frame renders in the OpenCV camera frame: camera at origin,
   `matrix_world = Rx(π)` so Blender-world ≡ optical frame. A gate placed at its
   `(R_cam_gate, t_cam_gate)` projects to **exactly** `project_gate_corners(R_cam_gate,
   t_cam_gate)` — the same projection the deployed estimator inverts. The renderer never reads
   labels back.
2. **Backend protocol.** Geometry/labels/augment/layout are pure Python and backend-independent.
   `ProceduralBackend` (numpy/cv2) runs the WHOLE pipeline on the laptop (and in CI); the bpy
   `BlenderBackend` is the photoreal drop-in. A procedural dataset and a Cycles dataset differ
   only in pixels — byte-identical labels.
3. **Geometry from the real course.** Viewpoints are sampled on the VQ1 course via the canonical
   `navigator.gates_from_track_records(corner_to_center=True)` — the exact gate world poses the
   deployed stack uses — so range distribution, the 26 m descent, gate facing and down-course
   co-visibility match deployment.
4. **Intrinsics reproduce K.** `sensor_fit=HORIZONTAL`, `sensor_width=36 mm`, `lens=18 mm`,
   centred principal point ⇒ `shift=0` reproduces `K` to floating point (test) and is re-checked
   ≤ 1 px against the live Blender camera at render start (hard-abort otherwise).
5. **No-flip augmentation.** Keypoint-aware geometric augs are orientation-preserving only; flips
   are left to ultralytics' `flip_idx` (a mirror is a reflection that would break corner identity).

## Gate red — EXTRACTED, no flag needed
Sampled from the real VQ1 sim frames (`handoff/shadowpc-followups-2026-06-05/task2_frames`, ranges
1.8/5/10 m): median of the saturated warm ring = **RGB (255, 50, 0)**, OpenCV-HSV (6, 255, 255) —
a vivid **orange-red**, not pure red. `vq1_faithful` anchors here; `appearance_broad`/others
randomize hue/sat/val widely around it to kill the gambling-red overfit.

## Tests
`tests/test_blender_gen.py` — **28 passed** (`.venv … pytest`, ~21 s), no Blender required. Covers:
intrinsics reproduce K (< 1e-9 px) + lens/FoV + off-centre rejection; corner projection == the
`frames.py` chain (< 1e-9 px); label row == `synthetic.to_yolo_pose_label`; label→PnP round-trip;
albumentations keypoint alignment (known translate) + photometric-preserves-corners + geometric
bbox; all 5 presets validate + round-trip + reject bad keys/range/engine; viewpoint envelope +
co-visibility + 3-corner clips in `terminal_approach`; full procedural dataset end-to-end +
label→PnP recovery + determinism. Adjacent suites (`test_synthetic/frames/gate_pose`, 53) still
green; package imports clean. End-to-end laptop run + viz overlay confirmed keypoints land on the
gate inner corners.

## bpy modules — verification status
The 4 bpy leaves were authored + adversarially API-checked against the bundled Blender RST docs
(they cannot execute on the laptop — no Blender). All 5 bpy files **compile** (AST) on the laptop.

- **bpy_camera.py** — built + verified by the Workflow, **zero defects found**; every class/attr/enum
  confirmed against the Blender 4.2+ RST (camera data, `world_to_camera_view` NDC→pixel mapping,
  `Matrix.Rotation(pi,4,'X')`). Carries the live `projection_max_error_px` self-check (≤ 1 px).
- **bpy_materials.py** — built + verified, **no API errors**; Principled v2 sockets accessed via a
  name-with-fallback helper, `ShaderNodeTexSky` uses `MULTIPLE_SCATTERING` (verified `NISHITA` is
  *gone* in 4.x), HDRI/sky/flat-colour world fallback chain. I extended it post-verify with the
  **dark-arena spotlight** path (guarded SPOT cone controls) + configurable ambient.
- **bpy_render.py** — built + verified; the verifier **caught + fixed a real crash** (`Action.fcurves`
  removed in Blender 4.4 → added a legacy/layered-action fallback for the motion-blur keyframes) and
  made the compositor Glare + 4.2↔5.x compositor-tree access version-tolerant.
- **bpy_scene.py** — its verify stage dropped on a transient socket error, so **I authored it
  directly** (gate annulus-prism mesh in gate frame, optical-pose placement, background/clutter),
  matching the sibling modules' verified API patterns + the backend's exact interface.

Residual (verify on ShadowPC, all guarded so a miss degrades, never crashes): the Cycles-addon GPU
prefs (`compute_device_type`/`get_devices`), `CyclesRenderSettings` names, the `CompositorNodeGlare`
control names/enums, and the Principled/Background dynamic socket names — none are bundled in the
core RST. The intrinsics self-check + viz overlay are the live arbiters.

## The VQ2 appearance-design PLAN (the strategic deliverable) — `VQ2_VISION_TRAINING_PLAN.md`
A parallel deep-research pass (MonoRace, the real DCL/A2RL gate, sim-to-real DR, Blender recipes)
reframed the whole effort. Headline, evidence-grounded:

- **MonoRace (the A2RL winner, near-1:1 prior art) used 2D composites + heavy augmentation, NOT
  photoreal 3D**, and **decoupled detection from color** (orange-LED gates, but the net keys on
  edges/shape). ⇒ **photorealism is not the dominant transfer lever — appearance diversity + motion
  blur + a geometry-keyed label + hard negatives are.** Photoreal Blender is the *base* (it buys us
  correct gate geometry across the pose space, which composites can't), not the goal.
- **The real gate is a SOLID purple structural frame, not LED/emissive**; the orange is the *drone's*
  LEDs + sim styling. **Gate color is the #1 unknown ⇒ randomize hard (color = nuisance axis).**
- **The venue is a DARK indoor arena with bright (often colored) spotlights** (Sept SoCal + Nov
  Columbus both indoor; A2RL Abu Dhabi). ⇒ dark-arena lighting dominant, not daylight.
- **~30–40 % hard negatives** beat fancy augmentation (YOLOv11 DR study).
- **Offline boresight IoU calibration** was MonoRace's biggest accuracy lever — *exactly* our banked
  ε_vert ≈ −0.25 m problem, and `frames.BoresightCorrection` already exists to apply it.

**Findings folded into the code (this session):** added **hard-negative generation** (`negatives`
preset + `negative_fraction`, empty-label background frames — tested end-to-end), a **dark-arena /
colored-spotlight** lighting mode + config axes, and **retuned `appearance_broad`/`hard_visual`** to
color-as-nuisance + dark-arena. Remaining plan steps (HDRI fetcher, rotational motion-blur, vignette/
CA, decals, the sim-to-real eval harness on the 40 real frames, auto-label flywheel) are sequenced in
the plan §6. **Flagged for the commander (plan §7): detector keypoint-vs-segmentation choice; real-
frame sourcing; ratifying color-as-nuisance.**

## How to run — see `RUN_GUIDE.md`
Laptop dry-run: `python scripts/gen_blender_dataset.py --preset appearance_broad --out … ` then
`python scripts/visualize_blender_labels.py --dataset …`. ShadowPC photoreal:
`blender --background --python src/racer/vision/blender_gen/render_entry.py -- --preset … --out …`.

## Open items / flags for the commander
- **bpy is doc-verified, not render-verified.** The intrinsics self-check + viz overlay (in
  `render_entry.py` / the run guide) are the live arbiters on ShadowPC. First photoreal run must
  pass the 5-point validation checklist before bulk rendering.
- **HDRI:** presets set `use_hdri: true` with `hdri_dir: null` (procedural sky fallback). To use
  real HDRIs, drop `.hdr/.exr` files in a folder on ShadowPC and set `hdri_dir` in the preset.
- **Gate depth parallax** at terminal range (< 2 m) is < 0.4° (keypoints on the Z=0 plane); flagged
  in `contract.py`, immaterial vs trained corner σ, but noted for the terminal-lock regime.
- **Round-2 appearance gap** (3D-scanned arena) is exactly what `appearance_broad`/`hard_visual`
  target; if real Round-2 reference imagery appears, add an HDRI/material set tuned to it.

## MEMORY-DELTA (≤10 lines, for the commander to triage/bank)
1. **VQ2 photoreal dataset generator BUILT + laptop-green (28 tests)** — branch `vq2-blender-dataset`. `src/racer/vision/blender_gen/`: pure geometry/labels/intrinsics/augment/config/dataset (PnP-exact labels, <1e-9px vs frames.py) + a `RenderBackend` protocol (ProceduralBackend runs the whole pipeline on the laptop; bpy/Cycles BlenderBackend for ShadowPC). Renders in the OPTICAL frame so labels == project_gate_corners. Reuses navigator.gates_from_track_records.
2. **STRATEGIC REFRAME (MonoRace, the A2RL winner, near-1:1):** won with 2D composites + heavy aug, NOT photoreal 3D; detection DECOUPLED from color (keys on edges/shape). ⇒ photorealism is the base not the lever; **appearance DIVERSITY + motion blur + geometry-keyed label + ~30% HARD NEGATIVES** are what transfer. Full plan: handoff/vq2-blender-dataset-2026-06-14/VQ2_VISION_TRAINING_PLAN.md.
3. **Real gate = SOLID purple structural frame, NOT LED/emissive** (A2RL×DCL, same 1.5m spec); orange is the DRONE's LEDs + sim styling. **Gate color = #1 unknown → randomize hard (nuisance axis).** Venue = DARK indoor arena + colored spotlights (Sept SoCal + Nov Columbus both indoor).
4. **VQ1 gate red EXTRACTED = RGB(255,50,0)** (orange-red, HSV~6) from task2_frames; now only the `vq1_faithful` regression anchor — bulk arms drop the red prior.
5. **Hard negatives are first-class** (YOLOv11 DR study ~35%): added `negatives` preset + `negative_fraction` (empty-label background frames). Recommended mix: appearance_broad 45 / hard_visual 20 / negatives 25 / terminal 5 / long_range 3 / vq1_faithful 2.
6. **MonoRace's biggest ACCURACY lever = offline IoU extrinsic calibration = our banked ε_vert ≈ −0.25m boresight** (frames.BoresightCorrection already exists to apply it). Auto-label flywheel + sim-to-real eval on the 40 real VQ1 frames are the validation path.
7. **OPEN for commander (plan §7):** detector keypoint(YOLO-pose) vs MonoRace segmentation+geometric-corner — generator can also emit masks to hedge; real-frame sourcing; ratify color-as-nuisance.
8. **bpy = doc-verified (vs bundled RST), NOT render-verified** — ShadowPC intrinsics self-check (≤1px, hard-abort) + viz overlay are the live arbiters. bpy_render verify caught a real 4.4 `Action.fcurves` crash + fixed it. Not merged; branch pushed for the commander's gate.
