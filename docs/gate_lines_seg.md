# The line-based gate path: segmentation front-end

**Status: NOT wired into flight.** The keypoint path is the only emitter. This is the independent
second path, kept behind so there is always a fallback.

## Why this exists

A gate corner that leaves the frame is unlearnable in stock ultralytics: augmentation zeros and
clamps it, the pose loss masks `v=0`, and inference clips the prediction back to the border. So on a
cropped gate the model returns corners *pinned to the image edge* — measured **12.3 px** of pure
error against homography truth, versus 0.73 px in-frame. That is the close-range approach, the
frames right before a pass.

A **line** has no such failure mode: it is fixed by the in-frame pixels it passes through, so it
survives cropping. The chain `mask → lines → homography → centre` was validated against task2 ground
truth at **13.7 px median**, with full coverage close in (7/7 at 0–4 m) where the keypoint path is
weakest — at 1.8 m the keypoint path returns a 23.9 m pose error and this returns ~0.13 m.

The measured weak link is **not** the geometry, it is the **HSV colour mask**: on failure-mined
frames the solver disagrees with the keypoint centre by ~100 px, and that disagreement is *not*
gate-mismatch (multi-gate frames disagree less). Replacing the mask with a learned one is the whole
remaining job.

## The target: two classes, both convex quads

| class | name | geometry |
|---|---|---|
| 0 | `gate_frame` | the 2.72 m outer square |
| 1 | `gate_opening` | the 1.5 m inner square |

Deliberately **not** one annulus polygon. A ring needs a bridged polygon that rasterises with a
seam, and it forces the solver to recover the opening by contour-hierarchy hole-finding
(`RETR_CCOMP`) — the fragile step. Two nested convex instances hand the solver both squares
directly: 8 edge lines, no hole logic. `gate_opening` is genuinely visible; `gate_frame` is amodal
over its own opening, which segmentation handles well and which makes its outer boundary — the thing
actually fitted — unambiguous.

## Labels cost nothing

Every existing 8-keypoint label already pins the gate plane, so both squares re-project through it.
Off-frame corners are stored *clamped with `v=0`*, so their coordinates are junk and are **never
read**: the homography is fitted to the usable keypoints only and the true squares are
reconstructed, then clipped to the image.

Measured on the real corpus: only **29.9%** of gate rows carry all 8 corners, but **59.1%** carry all
4 inner, and the rest are mostly 3-inner-plus-outers — all ≥4 usable points, which is all the
homography needs. Conversion yield: **1716/1728 = 99.3%**.

```bash
python scripts/build_seg_dataset.py --src C:/Users/Shadow/vq2_darkred_partial_2026-07-06 --out C:/Users/Shadow/vq2_seg_2026-07-22
```

Two traps the builder handles, both of which fail **silently**:
- The pose file lists mix datasets that each number frames from `000000`. Keying outputs by stem
  collided on **437 of 1572** frames, pairing images with another dataset's geometry. Frames are
  re-keyed `<dataset>__<stem>` and uniqueness is asserted.
- Ultralytics finds a label by substituting `/images/` → `/labels/` on the **image** path. Pointing
  the new lists at the original images would have trained against their **pose** labels with no
  error anywhere. The builder mirrors an image tree (hardlinks).

## Curriculum

**Stage 1 — pose-derived labels, no hand-labelling.** Start from COCO-pretrained `yolo11n-seg`: the
gate is large, high-contrast and low-variety, so the small model is the right capacity and leaves
GPU headroom — the detector shares one RTX 2000 Ada with the simulator.

```bash
python scripts/train_gate_seg.py --data C:/Users/Shadow/vq2_seg_2026-07-22/data.yaml
```

Augmentation follows the pose model's hard-won settings: keep photometric aug (it was load-bearing
there — good-fix 80% → 22% without it), keep mosaic because it manufactures exactly the cropped
gates this path serves, but **no vertical flip** (gates are near-upright; a flip invents a pose the
course never shows) and `overlap_mask=False` because `gate_opening` sits inside `gate_frame` by
construction.

**Stage 2 — failure-mined frames.** The 998-frame inbox is mined from flights where the *keypoint*
path failed, i.e. exactly this path's target distribution. Convert what is labelled, eyeball the
overlays, fix in the labeler, fine-tune from stage 1 at ~0.2× LR.

**Select on the solver, not on mAP.** mask-mAP scores region overlap, but this model feeds
`homography_from_lines`, which fits the mask *boundary* — a blobby mask can score well and still be
too ragged to fit. The pose model taught this from the other side: val-mAP mispicked `best.pt` and
the ep49 `last.pt` was the real champion.

```bash
python scripts/eval_gate_seg.py --weights runs/segment/gate_seg/weights/best.pt --kpt-weights models/vq2_partial_m_2026-07-06_fp16_384x640.engine
```

Reports **coverage** against the HSV baseline (currently 822/998 = 82.4% on the inbox) and
**agreement** with trusted 4-corner keypoint centres. Agreement is a cross-check, not truth — the
two paths share no front-end, so agreement is real evidence, but a disagreement says only that they
differ.

## Labelling

`tools/gate_labeler` — press **`M`** to shade the seg target derived from the current clicks
(orange = `gate_frame`, green = `gate_opening`, both clipped to frame). The polygons come from the
same clicks as the pose label, so there is nothing extra to draw; the preview exists because a wrong
click that still looks plausible as 8 keypoints can produce a badly wrong mask.

## Known gap

`homography_from_lines` only uses a square when **both** its extremes are present in a pencil, so a
gate cropped on **two** edges gets no fit. Two generalisations were written, unit-tested green, and
measured net-negative on the 998-frame inbox — see the `KNOWN GAP` block in that function before
attempting a third. The short version: the sign is not recoverable from the lines alone; it needs
the interior direction, which only the mask knows. Pass that in.
