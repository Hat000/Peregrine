# VQ2 photoreal dataset — ShadowPC run guide

Authoring is done on the laptop; **rendering runs on ShadowPC** (Blender + CUDA GPU). The pure
geometry/label/augment pipeline is shared and already laptop-tested; ShadowPC only adds the Cycles
pixels.

## 0. One-time Blender setup (ShadowPC)
Blender ships its own Python, which must be able to `import numpy, cv2, scipy, albumentations` and
this repo's `racer`. Install the wheels into Blender's Python once:

```powershell
# find Blender's python (adjust version/path)
& "C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe" -m pip install `
    numpy scipy opencv-python albumentations
```
`racer` is added to `sys.path` automatically by `render_entry.py` (it locates the repo `src`).
Confirm GPU: Edit → Preferences → System → Cycles Render Devices shows CUDA/OptiX enabled (the
generator also enables it programmatically).

## 1. Render a preset
```powershell
& "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" --background --python `
    src\racer\vision\blender_gen\render_entry.py -- `
    --preset appearance_broad --out data\vq2\appearance_broad --n-train 4000 --n-val 400 --seed 0
```
First thing it prints is the **intrinsics self-check** (`max reprojection error … px`). If that is
> 1 px it ABORTS — the camera is misconfigured and labels would be wrong; fix `bpy_camera.py`
before rendering anything.

Flags: `--engine BLENDER_EEVEE_NEXT` (or `--eevee`) for a fast preview pass; `--samples N` to
override Cycles samples; `--track path\to\track_map.json` for a different course; `--image-ext jpg`.

### Presets (what each is for)
| preset | regime | ~mix share |
|--------|--------|-----------:|
| `appearance_broad` | color-as-nuisance, dark-arena, HDRI, +20% negatives | 45 % |
| `hard_visual`      | flares, motion blur, occlusion, low light, +25% negatives | 20 % |
| `negatives`        | NO-gate background/confusers (empty labels) | 25 % |
| `terminal_approach`| 2–8 m, 3-corner clips | 5 % |
| `long_range`       | gates 15–30 m, small target | 3 % |
| `vq1_faithful`     | extracted orange-red gates, arena light | 2 % (regression anchor) |

Render each arm to its own dir, then point `yolo pose train` at the merged set. The mix is grounded
in `VQ2_VISION_TRAINING_PLAN.md` (read it first — it explains *why* this distribution): photoreal is
the base, but **appearance diversity + motion blur + a geometry-keyed label + ~30 % hard negatives**
are the real sim-to-real levers (per MonoRace, the A2RL winner). If obtainable, fold in ~10–15 % real
frames.

**Appearance knobs that matter most** (in each preset's `appearance` block): `lighting_mode`
(`dark_arena` is the real-venue look — low ambient + colored spotlights; `auto` biases to it),
`gate_hue_jitter` (keep wide — color is a nuisance axis, not the orange-red), `use_hdri` + `hdri_dir`
(point at a folder of indoor-biased Poly Haven CC0 HDRIs on ShadowPC), `negative_fraction`.

## 2. VERIFY the labels (do this every run — the #1 guard)
```powershell
.venv\Scripts\python.exe scripts\visualize_blender_labels.py --dataset data\vq2\appearance_broad --n 24
```
Writes `…/_viz/*_overlay.png`: bbox + 4 keypoints (green=visible, yellow=occluded, red=off-frame)
+ corner indices + a PnP range readout. **The keypoints must sit on the rendered gate's inner
corners.** If they float off the gate, STOP — the camera/scene transform is wrong (re-run the
intrinsics self-check; it is the discriminator).

## 3. Output + cost
- Layout / format: see `DATASET_CONTRACT.md`. 640×360 PNG + 17-field YOLO-pose `.txt` + `data.yaml`.
- Render time (rough, single modern GPU): Cycles 64–112 samples at 640×360 ≈ **0.1–0.5 s/frame**
  with denoise → ~1–4 k frames per 10 min. `--eevee` is ~5–10× faster for preview. Disk ≈ 0.1–0.3
  MB/frame PNG. Tune `render.samples` per preset (low for bulk, high for the hero set).

## 4. Laptop dry-run (no Blender)
The procedural backend runs the WHOLE pipeline minus Cycles — use it to validate a preset and the
labels before committing GPU time on ShadowPC:
```powershell
.venv\Scripts\python.exe scripts\gen_blender_dataset.py --preset appearance_broad --out data\runs\vq2_proc --n-train 200 --n-val 20
.venv\Scripts\python.exe scripts\visualize_blender_labels.py --dataset data\runs\vq2_proc --n 16
```

## 5. ShadowPC validation checklist (first photoreal run)
1. Intrinsics self-check ≤ 1 px (printed at start; hard-aborts otherwise).
2. `_viz` overlays: keypoints on the inner corners across near (2–4 m, expect 3-corner clips),
   mid (~8–15 m) and far (20–30 m) frames.
3. Gate colour: `vq1_faithful` looks like the real VQ1 orange-red; `appearance_broad` spans hues.
4. A few `terminal_approach` frames show the outer boundary leaving frame with the inner ring + a
   clipped (v=0) corner.
5. Spot-check a JPG-compressed / motion-blurred `hard_visual` frame is still gate-detectable.
