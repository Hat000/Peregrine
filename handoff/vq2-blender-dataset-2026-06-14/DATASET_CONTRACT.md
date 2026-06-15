# VQ2 photoreal dataset — FROZEN YOLO-pose contract

The training harness (standard `yolo pose train`) targets exactly this. It is byte-identical to
the procedural `racer.vision.synthetic` contract, so a model can train on a mix of both. The SSOT
in code is `src/racer/vision/blender_gen/contract.py`.

## Class
- Single class. `nc = 1`, `names: {0: gate}`.

## Keypoints
- `kpt_shape: [4, 3]` — 4 inner-square corners, each `(x, y, visibility)`.
- **Canonical order (identical to `gate_pose` IPPE_SQUARE object-point order):**

  | idx | corner | gate-frame (X-right, Y-DOWN) | head-on image position |
  |-----|--------|------------------------------|------------------------|
  | 0   | LL     | (−h, +h)                     | lower-left             |
  | 1   | LR     | (+h, +h)                     | lower-right            |
  | 2   | UR     | (+h, −h)                     | upper-right            |
  | 3   | UL     | (−h, −h)                     | upper-left             |

  `h = inner_size / 2 = 0.75 m`. The detector's keypoint `i` **is** gate-frame corner `i`; PnP
  consumes them in this order. Re-ordering silently produces garbage poses.
- `flip_idx: [1, 0, 3, 2]` — horizontal flip swaps LL↔LR and UR↔UL (used by ultralytics'
  train-time `fliplr`; our own augmentation never flips, see below).

## Visibility flags
- `2` = visible, `1` = occluded-in-frame (covered by a nearer gate's ring), `0` = off-frame.
- Off-frame corners are clamped into `[0, 1]` and carry `v = 0`. A gate is labelled only when
  ≥ 3 corners are visible **and** its centre is in frame — exactly the 3-/4-corner cases the
  PnP + P3P fallback consume.

## Label row
- One ultralytics YOLO-pose row per labelled gate:
  `class cx cy w h x0 y0 v0 x1 y1 v1 x2 y2 v2 x3 y3 v3` — **17 fields**, all of `cx cy w h x* y*`
  normalized to `[0, 1]` (bbox from the **outer** 2.72 m square, clipped to frame).
- Multiple gates per image → multiple rows (co-visibility down the descending course).

## Hard negatives
- Frames with **no gate** (background / arena structure / confusers only) are written with an
  **empty `.txt` label file** — ultralytics treats these as background/negative examples. Driven by
  `ScenarioPreset.negative_fraction` (the `negatives` preset = 1.0; recommended overall mix ≈ 25–30 %).
- A negative image still has a label file (empty), so `images/` and `labels/` counts stay equal.

## Image
- `640 × 360`, BGR on disk (`cv2.imwrite` → PNG default, JPG optional). Layout:
  ```
  <out>/images/{train,val}/000000.png
  <out>/labels/{train,val}/000000.txt
  <out>/data.yaml
  ```

## Geometry guarantees (why the labels are correct)
- Corners are projected through `racer.frames` + `gate_pose.project_gate_corners` in the camera
  optical frame — the **same** projection the deployed estimator inverts. Camera intrinsics
  reproduce `K = [[320,0,320],[0,320,180],[0,0,1]]` to floating point (test) and ≤ 1 px live
  (ShadowPC self-check).
- The renderer never feeds labels back: geometry is computed once (pure Python) and the Cycles /
  procedural backends only paint pixels under it.

## Augmentation note (no flips)
- The keypoint-aware albumentations pipeline applies only **orientation-preserving** warps
  (small translate/scale/rotate/shear/perspective) + photometric/sensor effects. It never flips
  or 90°-rotates: a mirror is a reflection (not a proper camera pose) and would break the
  keypoint→gate-corner identity. Left/right flip augmentation is left to ultralytics' `flip_idx`.
