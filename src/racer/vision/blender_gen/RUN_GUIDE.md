# VQ2 Blender render — ShadowPC RUN GUIDE

How to render the VQ2 photoreal gate dataset with the **Blender (Cycles/EEVEE) backend** on ShadowPC.
This is the setup that was **render-verified on 2026-06-15** (Blender 5.1.2, Python 3.13, NVIDIA RTX 2000
Ada). The pure-Python `ProceduralBackend` needs none of this — it is the laptop fallback. This guide is
only for the photoreal `BlenderBackend` path (`render_entry.py`).

---

## 0. What you need

- **Blender 4.2 LTS or newer** (verified on **5.1.2**). Installed here at
  `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`.
- A CUDA/OptiX GPU for Cycles (optional — falls back to CPU). Verified GPU: **NVIDIA RTX 2000 Ada**.
- Python deps **importable by BLENDER's own Python**: `opencv-python(-headless)`, `scipy`,
  `albumentations`. `numpy` already ships with Blender.

> ⚠️ Blender ships its **own** Python (`...\5.1\python\bin\python.exe`, here **3.13.9**) and ignores
> your system/conda Python entirely.

---

## 1. Install the deps into Blender's Python

Program Files is not writable without admin, so `pip` installs to the **per-user** site-packages — that
is fine, we point Blender at it in step 2.

```powershell
$py = "C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe"
& $py -m pip install --user opencv-python-headless scipy albumentations
```

This installs to `C:\Users\<you>\AppData\Roaming\Python\Python313\site-packages`.
(Use `opencv-python-headless`, not `opencv-python` — we render `--background`, no GUI needed.)

---

## 2. The isolated-Python gotcha (why a plain install "doesn't work")

Blender launches its embedded interpreter in **isolated mode** (`sys.flags.isolated == 1`,
`no_user_site == 1`). That mode **strips `PYTHONPATH` from `sys.path` AND disables user-site**, so the
packages you just installed are invisible even though `numpy` (bundled) imports fine. Symptom:

```
ModuleNotFoundError: No module named 'cv2'   # also scipy, albumentations
```

**Fix (already wired into `render_entry.py`):** set `PYTHONPATH` to the site-packages dir before
launching. `render_entry.py` reads `os.environ['PYTHONPATH']` (still readable in isolated mode) and
re-appends it to `sys.path`. No machine path is hard-coded — it lives in the env var:

```powershell
$env:PYTHONPATH = "C:\Users\$env:USERNAME\AppData\Roaming\Python\Python313\site-packages"
```

Verify deps import inside Blender:

```powershell
$bl = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
& $bl --background --python-expr "import cv2, scipy, albumentations; print('deps OK')"
```

---

## 3. Render

```powershell
$bl   = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$repo = "C:\Users\Shadow\Peregrine"
$env:PYTHONPATH = "C:\Users\$env:USERNAME\AppData\Roaming\Python\Python313\site-packages"

& $bl --background --python "$repo\src\racer\vision\blender_gen\render_entry.py" -- `
    --preset appearance_broad --out D:\vq2\appearance_broad `
    --n-train 4000 --n-val 400 --seed 0 --masks
```

Args after the standalone `--` go to `render_entry.py`:

| flag | meaning |
|------|---------|
| `--preset` | `vq1_faithful` \| `appearance_broad` \| `hard_visual` \| `long_range` \| `terminal_approach` \| `negatives`, or a path |
| `--out` | dataset dir (`images/`, `labels/`, `data.yaml`, plus `masks/` with `--masks`) |
| `--n-train` / `--n-val` | frame counts per split |
| `--seed` | RNG seed (use **distinct** seeds per arm) |
| `--eevee` | fast preview (maps to the build's EEVEE engine) |
| `--engine` | `CYCLES` \| `BLENDER_EEVEE_NEXT` (override the preset) |
| `--samples` | override Cycles/EEVEE samples |
| `--masks` | **also** emit gate instance masks → `masks/<split>/*.png` |
| `--seg` | **also** emit 2-class YOLO-seg polygons → `seg/<split>/*.txt` |
| `--max-intrinsics-err-px` | abort threshold (default 1.0 — **do not raise to mask a real failure**) |

### Which gates get a label at all (two rules, both measured)
A gate is labelled only if it passes **both**:

1. **enough of it is on screen** — clipped outer extent ≥ `MIN_LABEL_AREA_PX` (200 px², `geometry._has_labellable_area`). Blind to anything in front of the gate;
2. **enough of what is on screen is not hidden** — its *rendered* silhouette ≥ `MIN_UNOCCLUDED_AREA_PX` (64 px, `geometry.has_visible_silhouette`), measured with the Workbench object-id pass so props, people, the floor, nearer gates and the frame edge have all already taken their bite.

Rule 2 needs the id pass, so **the structure pass runs on every frame whether or not you pass
`--seg`** (+0.26 s/frame, +7.6 % on a 128-sample Cycles frame). That is deliberate: otherwise the
*pose* labels would silently depend on an unrelated flag. Verified on disk — the same seed with and
without `--seg` writes byte-identical `labels/`.

Neither rule is a corner count. `>= 3 in-frame corners` was retired in 2026-07 after it was found
rendering visible gates into the image and handing them to training as **background**; it came back
a fourth time disguised as an occlusion test in `bpy_photoreal.occlude_blocked_keypoints` (it counted
off-frame corners too, so it re-killed 30 % / 39 % of the labels the area rule had just rescued).
The raycast there still marks individual corners `V_OCC` — that is real information the pose loss
uses — but it no longer drops gates.

The run prints an **occlusion census** at the end:
```
[vq2] occlusion census (pre-augment gates): hidden=23 (21.1%), labelled=86 (78.9%)
```
A large `hidden` share means gates are being dropped — look at an overlay before trusting the run.
Any `unmeasured` means the id pass failed and those frames were written with **no** occlusion check.

### What `--seg` / `--masks` actually label (read before mixing datasets)
Under the **Blender** backend both are the gate's **rendered silhouette**, measured per gate with a
Workbench object-id pass (`bpy_idmask`) taken right after the beauty render: class 0 `gate_frame` is
the true projected outer boundary — *including the inner side walls the 0.26 m frame depth exposes at
close range and off-axis* — and class 1 `gate_opening` is the genuinely see-through hole, measured
with an invisible opening-plane proxy so an oblique gate whose walls close the hole emits **no**
opening rather than an invented quad. `--seg`/`--masks` add only the **opening** pass on top of the
structure pass that every run already does (see above); total ~2–3 extra Workbench renders per frame.
Pose labels are byte-identical with or without it.

Under the **procedural** backend there is no 3D scene, so both fall back to the old flat-quad
approximation (projected inner/outer squares, clipped). That is a *different target*: it omits the
side walls and over-states the opening. The run prints a loud warning and stamps `seg/SOURCE.txt`
with `rendered-silhouette` or `flat-quad-fallback` — **check that file before merging arms.**

Look at the result, do not trust a census:
```powershell
python scripts\seg_overlay_sheet.py --data <out> --split train --limit 24 --cols 4
```
It reads image + seg file back **off disk**, draws the polygons, and prints the gates/frame
signature. A collapse toward 1.0 gates/frame means gates are being dropped again.

### The built-in safety gate
Before rendering a single frame, `render_entry` runs the **intrinsics self-check**: it projects known
gate corners through the live Blender camera and through `K`, and **aborts (rc=2)** if they disagree by
> 1 px. A passing run prints e.g. `max reprojection error = 0.0000 px`. It **cannot** silently emit
mislabeled frames. If it aborts, fix `bpy_camera`, don't raise the threshold.

---

## 4. Verify the labels (overlay + PnP)

Pure-Python, runs on the laptop or here. Draws the 4 keypoints + bbox + skeleton on each render and runs
the deployed PnP estimator to recover gate range:

```powershell
$env:PYTHONPATH = "$repo\src"
python -m racer.vision.blender_gen.viz_overlay --data D:\vq2\appearance_broad --split train --out D:\vq2\appearance_broad_overlay --limit 30
```

Corners are colour-coded in canonical order: **0=LL red, 1=LR green, 2=UR blue, 3=UL yellow**. They must
land on the rendered gate's **inner-square** corners; the printed range must be sane (2–30 m).

---

## 5. Blender 5.x version notes (verified the hard way)

The backend modules say "Blender 4.2+". Three things differ on **5.1** and are handled in-code so 4.2 and
5.x both work:

1. **EEVEE engine id**: 4.2 = `BLENDER_EEVEE_NEXT`; **5.x renamed it back to `BLENDER_EEVEE`** (the enum
   is `CYCLES / BLENDER_EEVEE / BLENDER_WORKBENCH`, no `_NEXT`). `bpy_render._resolve_engine` maps an
   unavailable EEVEE id onto whichever EEVEE the running build exposes.
2. **Compositor output node**: 4.2 ends on `CompositorNodeComposite`; **5.x removed it** (uses
   `NodeGroupOutput`). The glare/bloom builder falls back to the group output and, on any unsupported
   layout, degrades to "no bloom" instead of crashing the render.
3. **Isolated Python** (section 2).

---

## 6. Throughput (measured here)

640×360, per-frame scene rebuild + augment + masks, on the RTX 2000 Ada:

| engine | samples | ~sec/frame | ~frames/hr |
|--------|---------|-----------|-----------|
| Cycles (GPU, OptiX denoise) | 96 | ~2.4 | ~1500 |
| EEVEE (preview) | 96 (TAA) | ~2 (warm) | faster preview |

Datasets are large → keep them **local** (a gitignored dir or another drive); commit only this guide, the
mask/overlay code, a few sample frames, and the session REPORT.

---

## 7. Photoreal real-asset pipeline (the 2026-06-15 look)

The `photoreal` presets (`vq1_faithful`, `appearance_broad`) dress each frame with **real CC0 assets**
instead of the procedural sky/walls: an HDRI environment (image-based lighting + a photographic
background), a PBR-textured floor placed BELOW the flight line (gates float in the air), real
photoscanned props + procedural clothing-tinted "mannequin" people scattered OFF the gate corridor, and
a plain solid vivid gate. Renders are CLEAN (no in-render motion blur / glare — those are albumentations'
job). Driven by `appearance.photoreal=true` + an on-disk asset library.

### 7a. Fetch the assets once (needs network)
```powershell
$py = "C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe"
$env:PYTHONPATH = "$repo\src"
& $py -m racer.vision.blender_gen.assets          # -> <repo>/assets_vq2/{hdris,textures,models}
```
Pulls a curated, reproducible CC0 set from polyhaven.com (≈26 indoor HDRIs, 3 PBR floor texsets, 21
props). Idempotent. `assets_vq2/` is gitignored (large). The render path then reads it with no network.

### 7b. Render (same entrypoint)
```powershell
& $bl --background --python "$repo\src\racer\vision\blender_gen\render_entry.py" -- `
    --preset appearance_broad --out D:\vq2\allhue --n-train 1400 --seed 1000 --masks --no-augment
```
- `--no-augment` renders **clean base** frames (the in-line albumentations pass off) — use this when the
  augmentation will be applied later as an offline multiplier. Omit it for the normal 1-render-1-augment.
- The HDRI is rotated **−90° about X** so its vertical aligns with the optical frame (else the back wall
  renders as a ceiling — a real bug that was here). Handled in `bpy_photoreal.setup_hdri_world`.

### 7c. Labels: 8 keypoints
The YOLO-pose label carries **8 keypoints**: inner 0..3 (the gate opening / PnP corners) then outer 4..7
(the gate frame). `kpt_shape: [8, 3]`, `flip_idx: [1,0,3,2,5,4,7,6]`. The inner 4 are byte-identical to
the legacy 4-corner block, so 4-corner PnP still parses them directly (`viz_overlay` draws inner filled,
outer as rings).

### 7d. Photoreal throughput (RTX 2000 Ada, Cycles 160 + OptiX denoise)
| stage | cost |
|-------|------|
| prop import (once, at backend init) | ~10–15 s (cached; per-frame spawns are duplicates) |
| per frame (HDRI + floor + ~12 props + people + raycast + render) | **~2 s** → **~1500 frames/hr** |

> Props are imported ONCE then spawned as mesh-sharing duplicates — re-importing glTF every frame was a
> ~25 s/frame trap. The occlusion raycast only counts PROP/PERSON hits (never a gate hitting its own
> angled frame), else ~90% of frames were wrongly dropped.
