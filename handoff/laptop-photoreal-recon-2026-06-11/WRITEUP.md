# Photoreal Synthetic Pipeline — Dataset Recon
**Session:** laptop-photoreal-recon-2026-06-11 | **Model:** sonnet-4.6

---

## 1. Dataset Location

The v2 training set **is NOT a separate stored artifact on disk** — it is re-generated at job
start from source code if `data/mix_v2/data.yaml` doesn't exist yet.

- **Generation script:** `scripts/gen_synthetic_dataset.py` → calls `src/racer/vision/synthetic.py`
- **Adroit path:** `/scratch/network/fl3689/peregrine/data/mix_v2/`
- **SLURM job:** `cluster/yolo_train.sbatch` (the v2 real run, labeled as such in comments)
- **Generation command (from sbatch):**
  ```
  python scripts/gen_synthetic_dataset.py data/mix_v2 --mix --n-train 6000 --n-val 800
  ```
  Flags: `--mix` (one curriculum level drawn per image, weights `[1,2,2,3,3]`), no `--hard`,
  no `--max-gates` override → default `max_gates=3`, `high_roll_prob=0.0`, `edge_prob=0.0`.

---

## 2. Dataset Inventory

### (a) Partial-gate / off-frame-corner scenes — YES, built-in from L1

Three explicit edge-case categories are generated at **every curriculum level**:

| Case | Mechanism | Visibility flag |
|---|---|---|
| **Gate transit** (drone very close, outer ring bleeds off frame) | `cv2.fillPoly` auto-clips at frame edge; ring just bleeds | V_VIS on inner corners that remain in frame |
| **Corner clipping** (one corner exits the frame) | Detected by `_corner_visibility` bounds check | `V_OFF = 0` |
| **In-frame occlusion** (random rectangle covers a corner) | `_draw_occluder` at L2+, 35% chance | `V_OCC = 1` |

Samples require **≥3 corners with V_VIS** AND gate centre in-frame — so 3-corner
(one V_OFF or V_OCC) images ARE kept and labelled. Full 4-corner clipping (all corners off)
is rejected. There is no direct count of the clipping/transit fraction, but:

- Gate poses are biased **close** (depth drawn as `min + range × U²`, skewing toward 1 m),
  so transit scenes are common.
- Off-frame corners arise naturally from the full ±0.8 rad yaw / ±0.5 rad pitch / ±0.7 rad
  roll envelope combined with close approach distances.
- `edge_prob=0.0` in the v2 run → NO deliberate edge-biasing. Off-frame clips come only from
  natural geometry, not the targeted oversampling introduced for v3.

**Rough fraction:** no explicit per-label statistics available without running a count on
Adroit, but given the close-range skew and full angular envelope, 3-corner frames are
expected to be ~15–30% of labels (comparable to the proportion visible in the `eval_detector`
tails described in the pipeline doc).

### (b) Per-keypoint visibility flags — YES, fully populated

Every label row carries the ultralytics YOLO-pose format with explicit `v` per corner:
- `v=2` (V_VIS): clearly visible in-frame
- `v=1` (V_OCC): in-frame but under an occluder rectangle
- `v=0` (V_OFF): off-frame (coordinate clamped to [0,1])

These flags are correctly set for all three categories above (`_corner_visibility` +
`_in_ring` cross-gate occlusion check in `render_scene`). The `flip_idx=[1,0,3,2]`
entry in `data.yaml` keeps horizontal-flip augmentation correct. Implementation in
`to_yolo_pose_label` (synthetic.py:387–401).

### (c) Hue / appearance randomization range

**Gate color** (synthetic.py:114–119):
- One channel randomly chosen as "vivid" → drawn from `[120, 255]`
- Other two channels drawn from `[0, 160]`
- Effective hue randomization: **full 360° hue range** (any of R/G/B dominant), but only
  saturated gate colors (no pastels, no near-white, no near-black gates).
- No HSV parameterization, no explicit "±15° hue" logic — the RGB channel-selection approach
  spans the full color wheel coarsely.

**Background** (synthetic.py:100–111):
- Base BGR drawn from `[0, 130]` (dark-to-mid range); gradient overlay ±50 counts.
- At L2+: 0–6 random colored rectangles, colors drawn uniformly from `[0, 255]³`.

**L2 gate shading:** random directional gradient (full 0–2π direction angle, lo factor 0.35–0.75) — models directional lighting but not point-source, shadows, or specular.

**L3 chaos (albumentations or OpenCV fallback):** ONE primary effect per image (motion blur,
JPEG artifacts, downscale, Gaussian noise, ISO noise) + optional lighting effect (random shadow,
sun flare, fog, coarse dropout) + brightness/contrast ±20% + rare channel-shuffle (5%).

**No HDRI, no PBR, no physically-accurate shading at any level.**

### (d) Scene count + composition

| Split | Images (= scenes) | Train labels | Val labels |
|---|---|---|---|
| train | 6000 | 6000+ (multi-gate scenes have >1 row/file) | — |
| val | 800 | — | 800+ |

**Curriculum mix** (weights `[1,2,2,3,3]`):
- L1 (clean): ~11% of scenes
- L2 (realistic): ~44% of scenes
- L3 (chaos): ~44% of scenes

**Multi-gate composition** (max_gates=3, natural distribution via `_sample_n_gates`):
- L1: pool `[1,1,1,1,2]` → ~80% single-gate, ~20% two-gate
- L2/L3: pool `[1,1,2,2,3]` → ~40% single, ~40% two-gate, ~20% three-gate

**Gate appearance variants:** continuous (random color each render). No discrete gate skin
library, no physical material model, no texture maps.

**Backgrounds:** procedural only — flat dark base + gradient + colored rectangles. No real
photos, no HDRI environment, no course geometry.

---

## 3. Gap Assessment vs Photoreal Spec

The v2 dataset is a strong **geometric curriculum** — it covers the full pose envelope,
partial-gate/clipping/occlusion cases, and basic appearance variation — but it is entirely
**procedural/synthetic with no photorealism layer**. Against the Blender/Cycles photoreal spec
(HDRI environment lighting, PBR materials, motion blur, occlusion, hue randomization wider
than ±15°, off-frame partial gates with visibility loss), the gaps are: **(1) lighting model**
— v2 uses affine gradient shading and random RGB rectangles; Cycles would provide physically
correct shadows, reflections, and environment-dependent tone that the sim camera captures; the
detector has never seen metallic/textured gate materials or sky/arena HDRI backgrounds; **(2)
background realism** — procedural flat gradients vs real-looking environments (arena floor,
crowd, sky) are a large domain gap that will hurt on any VQ2 background not represented; **(3)
motion blur** is only in the L3 `OneOf` pool (~18% exposure), not correlated with speed/gate
distance as it would be in flight; **(4) hue randomization** in v2 covers the full hue wheel
but only with saturated "vivid channel dominant" gate colors — pastels, off-white, and near-
black gates (which could appear with aged/painted surfaces or different sim skins) are absent;
**(5) off-frame partial gates** arise only from natural geometry with no edge-bias (`edge_prob=0`
in v2), so severely cropped gates — the most challenging case for the PnP fallback — are
underrepresented relative to what Blender targeted oversampling would provide; **(6) the
visibility-flag infrastructure is already correct** (V_VIS/V_OCC/V_OFF fully populated) so a
Blender pipeline can reuse `to_yolo_pose_label` and `render_scene`'s label logic without
changes to the YOLO pose schema.

---

## MEMORY-DELTA:
1. v2 curriculum dataset: `data/mix_v2/`, 6000 train + 800 val, re-generated at job start from `scripts/gen_synthetic_dataset.py --mix --n-train 6000 --n-val 800` (no `--hard`, default max_gates=3, edge_prob=0).
2. Partial-gate scenes: YES built-in (transit + corner-clipping + in-frame occlusion at all levels); ~15–30% estimated 3-corner frames; edge_prob=0 so severely-cropped edges underrepresented.
3. Visibility flags: fully populated (V_VIS=2/V_OCC=1/V_OFF=0) in every label; YOLO-pose schema correct; flip_idx set.
4. Hue range: full 360° hue wheel via vivid-channel selection, but only saturated gates; no pastels/near-white/near-black.
5. No HDRI, no PBR, no physically-accurate lighting at any level; biggest gaps vs photoreal spec are lighting model, background realism, and edge-case partial gate density.
6. Blender pipeline CAN reuse existing label infrastructure (`to_yolo_pose_label`, visibility flags, data.yaml schema) — only the renderer needs replacing.
